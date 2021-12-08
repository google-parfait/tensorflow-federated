# Copyright 2020, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Evaluation for Federated Reconstruction.

Since a trained `tff.learning.reconstruction.Model` consists of only global
variables, evaluation for models trained using Federated Reconstruction involves
(1) reconstructing local variables on client data and (2) evaluation of model
global variables and reconstructed local variables, computing loss and metrics.
Generally (1) and (2) should use disjoint parts of data for a given client.

`build_federated_evaluation`: builds a `tff.Computation` that
  performs (1) similarly to the Federated Reconstruction training algorithm and
  then (2) with the reconstructed local variables.
"""

import collections
import functools
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process as measured_process_lib
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning.reconstruction import keras_utils
from tensorflow_federated.python.learning.reconstruction import reconstruction_utils
from tensorflow_federated.python.learning.reconstruction import training_process


def build_federated_evaluation(
    model_fn: training_process.ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: training_process.LossFn,
    metrics_fn: Optional[training_process.MetricsFn] = None,
    reconstruction_optimizer_fn: training_process.OptimizerFn = functools
    .partial(tf.keras.optimizers.SGD, 0.1),
    dataset_split_fn: Optional[reconstruction_utils.DatasetSplitFn] = None,
    broadcast_process: Optional[measured_process_lib.MeasuredProcess] = None,
) -> computation_base.Computation:
  """Builds a `tff.Computation` for evaluating a reconstruction `Model`.

  The returned computation proceeds in two stages: (1) reconstruction and (2)
  evaluation. During the reconstruction stage, local variables are reconstructed
  by freezing global variables and training using `reconstruction_optimizer_fn`.
  During the evaluation stage, the reconstructed local variables and global
  variables are evaluated using the provided `loss_fn` and `metrics_fn`.

  Usage of returned computation:
    eval_comp = build_federated_evaluation(...)
    metrics = eval_comp(reconstruction_utils.get_global_variables(model),
                        federated_data)

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.reconstruction.Model`. This method must *not* capture
      Tensorflow tensors or variables and use them. Must be constructed entirely
      from scratch on each invocation, returning the same pre-constructed model
      each call will result in an error.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      reconstruct and evaluate the model. The loss will be applied to the
      model's outputs during the evaluation stage. The final loss metric is the
      example-weighted mean loss across batches (and across clients).
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. The metrics will be applied to the model's outputs
      during the evaluation stage. Final metric values are the example-weighted
      mean of metric values across batches (and across clients). If None, no
      metrics are applied.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` used to reconstruct the local variables
      with the global ones frozen.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a single
      TF dataset and producing two TF datasets. The first is iterated over
      during reconstruction, and the second is iterated over during evaluation.
      This can be used to preprocess datasets to e.g. iterate over them for
      multiple epochs or use disjoint data for reconstruction and evaluation. If
      None, split client data in half for each user, using one half for
      reconstruction and the other for evaluation. See
      `reconstruction_utils.build_dataset_split_fn` for options.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)` and have empty state. If
      set to default None, the server model is broadcast to the clients using
      the default `tff.federated_broadcast`.

  Raises:
    TypeError: if `broadcast_process` does not have the expected signature or
      has non-empty state.

  Returns:
    A `tff.Computation` that accepts global model parameters and federated data
    and returns example-weighted evaluation loss and metrics.
  """
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  with tf.Graph().as_default():
    model = model_fn()
    global_weights = reconstruction_utils.get_global_variables(model)
    model_weights_type = type_conversions.type_from_tensors(global_weights)
    batch_type = computation_types.to_type(model.input_spec)
    metrics = [keras_utils.MeanLossMetric(loss_fn())]
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    federated_output_computation = (
        keras_utils.federated_output_computation_from_metrics(metrics))
    # Remove unneeded variables to avoid polluting namespace.
    del model
    del global_weights
    del metrics

  if dataset_split_fn is None:
    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)

  if broadcast_process is None:
    broadcast_process = optimizer_utils.build_stateless_broadcaster(
        model_weights_type=model_weights_type)
  if not optimizer_utils.is_valid_broadcast_process(broadcast_process):
    raise TypeError(
        'broadcast_process type signature does not conform to expected '
        'signature (<state@S, input@S> -> <state@S, result@C, measurements@S>).'
        ' Got: {t}'.format(t=broadcast_process.next.type_signature))
  if optimizer_utils.is_stateful_process(broadcast_process):
    raise TypeError(f'Eval broadcast_process must be stateless, has state '
                    f'{broadcast_process.initialize.type_signature.result!r}')

  @computations.tf_computation(model_weights_type,
                               computation_types.SequenceType(batch_type))
  def client_computation(incoming_model_weights: computation_types.Type,
                         client_dataset: computation_types.SequenceType):
    """Reconstructs and evaluates with `incoming_model_weights`."""
    client_model = model_fn()
    client_global_weights = reconstruction_utils.get_global_variables(
        client_model)
    client_local_weights = reconstruction_utils.get_local_variables(
        client_model)
    metrics = [keras_utils.MeanLossMetric(loss_fn())]
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    client_loss = loss_fn()
    reconstruction_optimizer = reconstruction_optimizer_fn()

    @tf.function
    def reconstruction_reduce_fn(num_examples_sum, batch):
      """Runs reconstruction training on local client batch."""
      with tf.GradientTape() as tape:
        output = client_model.forward_pass(batch, training=True)
        batch_loss = client_loss(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, client_local_weights.trainable)
      reconstruction_optimizer.apply_gradients(
          zip(gradients, client_local_weights.trainable))
      return num_examples_sum + output.num_examples

    @tf.function
    def evaluation_reduce_fn(num_examples_sum, batch):
      """Runs evaluation on client batch without training."""
      output = client_model.forward_pass(batch, training=False)
      # Update each metric.
      for metric in metrics:
        metric.update_state(y_true=output.labels, y_pred=output.predictions)
      return num_examples_sum + output.num_examples

    @tf.function
    def tf_client_computation(incoming_model_weights, client_dataset):
      """Reconstructs and evaluates with `incoming_model_weights`."""
      recon_dataset, eval_dataset = dataset_split_fn(client_dataset)

      # Assign incoming global weights to `client_model` before reconstruction.
      tf.nest.map_structure(lambda v, t: v.assign(t), client_global_weights,
                            incoming_model_weights)

      recon_dataset.reduce(tf.constant(0), reconstruction_reduce_fn)
      eval_dataset.reduce(tf.constant(0), evaluation_reduce_fn)

      eval_local_outputs = keras_utils.read_metric_variables(metrics)
      return eval_local_outputs

    return tf_client_computation(incoming_model_weights, client_dataset)

  @computations.federated_computation(
      computation_types.at_server(model_weights_type),
      computation_types.at_clients(computation_types.SequenceType(batch_type)))
  def server_eval(server_model_weights: computation_types.FederatedType,
                  federated_dataset: computation_types.FederatedType):
    broadcast_output = broadcast_process.next(broadcast_process.initialize(),
                                              server_model_weights)
    client_outputs = intrinsics.federated_map(
        client_computation, [broadcast_output.result, federated_dataset])
    aggregated_client_outputs = federated_output_computation(client_outputs)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            broadcast=broadcast_output.measurements,
            eval=aggregated_client_outputs))
    return measurements

  return server_eval
