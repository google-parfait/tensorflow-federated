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
"""Evaluation for Federated Reconstruction.

Since a trained `tff.learning.models.ReconstructionModel` consists of only
global variables, evaluation for models trained using Federated Reconstruction
involves (1) reconstructing local variables on client data and (2) evaluation of
model global variables and reconstructed local variables, computing loss and
metrics. Generally (1) and (2) should use disjoint parts of data for a given
client.

`build_fed_recon_eval`: builds a `tff.learning.templates.LearningProcess` that
performs (1) similarly to the Federated Reconstruction training algorithm and
then (2) with the reconstructed local variables.
"""

import collections
from typing import Any, Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process as measured_process_lib
from tensorflow_federated.python.learning.algorithms import fed_recon
from tensorflow_federated.python.learning.metrics import keras_finalizer as metrics_finalizers_lib
from tensorflow_federated.python.learning.metrics import sum_aggregation_factory
from tensorflow_federated.python.learning.models import reconstruction_model
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors
from tensorflow_federated.python.learning.templates import finalizers
from tensorflow_federated.python.learning.templates import learning_process as learning_process_lib


_AggregationProcess = aggregation_process.AggregationProcess


def build_fed_recon_eval(
    model_fn: fed_recon.ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: fed_recon.LossFn,
    metrics_fn: Optional[fed_recon.MetricsFn] = None,
    reconstruction_optimizer_fn: optimizer_base.Optimizer = sgdm.build_sgdm(
        learning_rate=0.1
    ),
    dataset_split_fn: Optional[
        reconstruction_model.ReconstructionDatasetSplitFn
    ] = None,
    model_distributor: Optional[distributors.DistributionProcess] = None,
    metrics_aggregation_process: Optional[
        aggregation_process.AggregationProcess
    ] = None,
) -> learning_process_lib.LearningProcess:
  """Builds a `tff.Computation` for evaluating a reconstruction `Model`.

  The returned computation proceeds in two stages: (1) reconstruction and (2)
  evaluation. During the reconstruction stage, local variables are reconstructed
  by freezing global variables and training using `reconstruction_optimizer_fn`.
  During the evaluation stage, the reconstructed local variables and global
  variables are evaluated using the provided `loss_fn` and `metrics_fn`.

  Usage of returned computation:
    eval_comp = build_federated_evaluation(...)
    metrics = eval_comp(
      tff.learning.models.ReconstructionModel.get_global_variables(model),
      federated_data)

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.ReconstructionModel`. This method must *not* capture
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
    reconstruction_optimizer_fn: A `tff.learning.optimizers.Optimizer` used to
      reconstruct the local variables with the global ones frozen.
    dataset_split_fn: A `tff.learning.models.ReconstructionDatasetSplitFn`
      taking in a single TF dataset and producing two TF datasets. The first is
      iterated over during reconstruction, and the second is iterated over
      during evaluation. This can be used to preprocess datasets to e.g. iterate
      over them for multiple epochs or use disjoint data for reconstruction and
      evaluation. If None, split client data in half for each user, using one
      half for reconstruction and the other for evaluation. See
      `tff.learning.models.ReconstructionModel.build_dataset_split_fn` for
      options.
    model_distributor: An optional `tff.learning.templates.DistributionProcess`
      that distributes the model weights on the server to the clients. If set to
      `None`, the distributor is constructed via
      `tff.learning.templates.build_broadcast_process`.
    metrics_aggregation_process: An optional `tff.templates.AggregationProcess`
      which aggregates the local unfinalized metrics at clients to server and
      finalizes the metrics at server. The `tff.templates.AggregationProcess`
      accumulates unfinalized metrics across round in the state, and produces a
      `tuple` of current round metrics and total rounds metrics in the result.
      If `None`, the `tff.templates.AggregationProcess` created by the
      `SumThenFinalizeFactory` with metric finalizers defined in the model is
      used.

  Raises:
    TypeError: if `model_distributor` does not have the expected signature.

  Returns:
    A `tff.learning.templates.LearningProcess` that accepts global model
    parameters and federated data and returns example-weighted evaluation loss
    and metrics.
  """
  batch_type = None

  @tensorflow_computation.tf_computation
  def build_initial_model_weights():
    model = model_fn()
    if not isinstance(model, reconstruction_model.ReconstructionModel):
      raise TypeError(
          '`model_fn` must return an instance of '
          f'`tff.learning.models.ReconstructionModel`. Got a: {type(model)}'
      )
    nonlocal batch_type
    batch_type = computation_types.tensorflow_to_type(model.input_spec)
    return reconstruction_model.ReconstructionModel.get_global_variables(model)

  if dataset_split_fn is None:
    dataset_split_fn = (
        reconstruction_model.ReconstructionModel.build_dataset_split_fn(
            split_dataset=True
        )
    )

  model_weights_type = build_initial_model_weights.type_signature.result
  dataset_type = computation_types.SequenceType(batch_type)

  if model_distributor is None:
    model_distributor = distributors.build_broadcast_process(model_weights_type)

  # Metric finalizer functions that will be populated while tracing
  # `client_update` and used later in the federated computation.
  metric_finalizers: collections.OrderedDict[
      str, metrics_finalizers_lib.KerasMetricFinalizer
  ] = collections.OrderedDict()

  @tensorflow_computation.tf_computation(model_weights_type, dataset_type)
  def client_computation(
      incoming_model_weights: Any,
      client_dataset: tf.data.Dataset,
  ):
    """Reconstructs and evaluates with `incoming_model_weights`."""
    client_model = model_fn()
    client_global_weights = (
        reconstruction_model.ReconstructionModel.get_global_variables(
            client_model
        )
    )
    client_local_weights = (
        reconstruction_model.ReconstructionModel.get_local_variables(
            client_model
        )
    )
    loss_metric = tf.keras.metrics.MeanMetricWrapper(loss_fn(), name='loss')
    if metrics_fn is None:
      metrics = [loss_metric]
    else:
      metrics = metrics_fn() + [loss_metric]
    nonlocal metric_finalizers
    for metric in metrics:
      if metric.name in metric_finalizers:
        raise ValueError(
            f'Duplicate metric name detected: {metric.name}. '
            f'Already saw metrics {list(metric_finalizers.keys())}'
        )
      metric_finalizers[metric.name] = (
          metrics_finalizers_lib.create_keras_metric_finalizer(metric)
      )
    # To be used to calculate batch loss for model updates.
    client_loss = loss_fn()

    reconstruction_optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        reconstruction_optimizer_fn,
        client_local_weights.trainable,
        disjoint_init_and_next=False,
    )

    @tf.function
    def reconstruction_reduce_fn(state, batch):
      """Runs reconstruction training on local client batch."""
      num_examples_sum, optimizer_state = state
      with tf.GradientTape() as tape:
        output = client_model.forward_pass(batch, training=True)
        batch_loss = client_loss(
            y_true=output.labels, y_pred=output.predictions
        )

      gradients = tape.gradient(batch_loss, client_local_weights.trainable)
      updated_optimizer_state, updated_weights = reconstruction_optimizer.next(
          optimizer_state,
          tuple(client_local_weights.trainable),
          tuple(gradients),
      )
      if not isinstance(
          reconstruction_optimizer, keras_optimizer.KerasOptimizer
      ):
        # TFF optimizers require assigning the updated tensors back into the
        # model variables. (With Keras optimizers we don't need to do this,
        # because Keras optimizers mutate the model variables within the `next`
        # step.)
        tf.nest.map_structure(
            lambda a, b: a.assign(b),
            client_local_weights.trainable,
            list(updated_weights),
        )

      return num_examples_sum + output.num_examples, updated_optimizer_state

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
      tf.nest.map_structure(
          lambda v, t: v.assign(t),
          client_global_weights,
          incoming_model_weights,
      )

      # If needed, do reconstruction, training the local variables while keeping
      # the global ones frozen.
      if client_local_weights.trainable:
        # Ignore output number of examples used in reconstruction, since this
        # isn't included in `client_weight`.
        def initial_state_reconstruction_reduce():
          trainable_tensor_specs = tf.nest.map_structure(
              lambda v: tf.TensorSpec(v.shape, v.dtype),
              client_local_weights.trainable,
          )
          # We convert the trainable specs to tuple, as the data iteration
          # pattern might try to stack the tensors in a list.
          initial_num_examples = tf.constant(0)
          return initial_num_examples, reconstruction_optimizer.initialize(
              tuple(trainable_tensor_specs)
          )

        recon_dataset.reduce(
            initial_state=initial_state_reconstruction_reduce(),
            reduce_func=reconstruction_reduce_fn,
        )

      eval_dataset.reduce(tf.constant(0), evaluation_reduce_fn)

      eval_local_outputs = (
          reconstruction_model.ReconstructionModel.read_metric_variables(
              metrics
          )
      )
      return eval_local_outputs

    return tf_client_computation(incoming_model_weights, client_dataset)

  if metrics_aggregation_process is None:
    metrics_aggregation_process = (
        sum_aggregation_factory.SumThenFinalizeFactory(
            metric_finalizers
        ).create(client_computation.type_signature.result)
    )
  else:
    py_typecheck.check_type(
        metrics_aggregation_process,
        _AggregationProcess,
        'metrics_aggregation_process',
    )

  @federated_computation.federated_computation
  def client_initialize():
    return metrics_aggregation_process.initialize()

  @federated_computation.federated_computation(
      client_initialize.type_signature.result,
      computation_types.FederatedType(model_weights_type, placements.CLIENTS),
      computation_types.FederatedType(dataset_type, placements.CLIENTS),
  )
  def client_work(state, model_weights, client_dataset):
    unfinalized_metrics = intrinsics.federated_map(
        client_computation, (model_weights, client_dataset)
    )
    metrics_output = metrics_aggregation_process.next(
        state, unfinalized_metrics
    )
    current_round_metrics, total_rounds_metrics = metrics_output.result
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            eval=collections.OrderedDict(
                current_round_metrics=current_round_metrics,
                total_rounds_metrics=total_rounds_metrics,
            )
        )
    )
    # Return empty result as no model update will be performed for evaluation.
    empty_client_result = intrinsics.federated_value(
        client_works.ClientResult(update=(), update_weight=()),
        placements.CLIENTS,
    )
    return measured_process_lib.MeasuredProcessOutput(
        metrics_output.state,
        empty_client_result,
        measurements,
    )

  client_work = client_works.ClientWorkProcess(
      initialize_fn=client_initialize, next_fn=client_work
  )

  # The evaluation will *not* send model updates back, only metrics; so the type
  # is simply an empty tuple.
  empty_client_work_result_type = computation_types.FederatedType(
      client_works.ClientResult(update=(), update_weight=()), placements.CLIENTS
  )
  empty_model_update_type = empty_client_work_result_type.member.update  # pytype: disable=attribute-error
  empty_model_update_weight_type = (
      empty_client_work_result_type.member.update_weight  # pytype: disable=attribute-error
  )
  empty_model_aggregator = mean.MeanFactory().create(
      empty_model_update_type, empty_model_update_weight_type
  )

  # Identity finalizer does not update the server model state.
  identity_finalizer = finalizers.build_identity_finalizer(
      model_weights_type, update_type=empty_model_update_weight_type
  )

  return composers.compose_learning_process(
      build_initial_model_weights,
      model_distributor,
      client_work,
      empty_model_aggregator,
      identity_finalizer,
  )
