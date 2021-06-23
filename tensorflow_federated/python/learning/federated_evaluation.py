# Copyright 2019, The TensorFlow Federated Authors.
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
"""A simple implementation of federated evaluation."""

import collections
from typing import Callable, Optional

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import optimizer_utils


# Convenience aliases.
SequenceType = computation_types.SequenceType


def build_federated_evaluation(
    model_fn: Callable[[], model_lib.Model],
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    use_experimental_simulation_loop: bool = False,
) -> computation_base.Computation:
  """Builds the TFF computation for federated evaluation of the given model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENTS)` and have empty state. If
      set to default None, the server model is broadcast to the clients using
      the default tff.federated_broadcast.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the evaluation metrics
    as aggregated by `tff.learning.Model.federated_output_computation`.
  """
  if broadcast_process is not None:
    if not isinstance(broadcast_process, measured_process.MeasuredProcess):
      raise ValueError('`broadcast_process` must be a `MeasuredProcess`, got '
                       f'{type(broadcast_process)}.')
    if optimizer_utils.is_stateful_process(broadcast_process):
      raise ValueError(
          'Cannot create a federated evaluation with a stateful '
          'broadcast process, must be stateless, has state: '
          f'{broadcast_process.initialize.type_signature.result!r}')
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO(b/124477628): Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_utils.weights_type_from_model(model)
    batch_type = computation_types.to_type(model.input_spec)

  @computations.tf_computation(model_weights_type, SequenceType(batch_type))
  @tf.function
  def client_eval(incoming_model_weights, dataset):
    """Returns local outputs after evaluting `model_weights` on `dataset`."""
    with tf.init_scope():
      model = model_fn()
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          incoming_model_weights)

    def reduce_fn(num_examples, batch):
      model_output = model.forward_pass(batch, training=False)
      if model_output.num_examples is None:
        # Compute shape from the size of the predictions if model didn't use the
        # batch size.
        return num_examples + tf.shape(
            model_output.predictions, out_type=tf.int64)[0]
      else:
        return num_examples + tf.cast(model_output.num_examples, tf.int64)

    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    num_examples = dataset_reduce_fn(
        reduce_fn=reduce_fn,
        dataset=dataset,
        initial_state_fn=lambda: tf.zeros([], dtype=tf.int64))
    return collections.OrderedDict(
        local_outputs=model.report_local_outputs(), num_examples=num_examples)

  @computations.federated_computation(
      computation_types.at_server(model_weights_type),
      computation_types.at_clients(SequenceType(batch_type)))
  def server_eval(server_model_weights, federated_dataset):
    if broadcast_process is not None:
      # TODO(b/179091838): Zip the measurements from the broadcast_process with
      # the result of `model.federated_output_computation` below to avoid
      # dropping these metrics.
      broadcast_output = broadcast_process.next(broadcast_process.initialize(),
                                                server_model_weights)
      client_outputs = intrinsics.federated_map(
          client_eval, (broadcast_output.result, federated_dataset))
    else:
      client_outputs = intrinsics.federated_map(client_eval, [
          intrinsics.federated_broadcast(server_model_weights),
          federated_dataset
      ])
    model_metrics = model.federated_output_computation(
        client_outputs.local_outputs)
    statistics = collections.OrderedDict(
        num_examples=intrinsics.federated_sum(client_outputs.num_examples))
    return intrinsics.federated_zip(
        collections.OrderedDict(eval=model_metrics, stat=statistics))

  return server_eval
