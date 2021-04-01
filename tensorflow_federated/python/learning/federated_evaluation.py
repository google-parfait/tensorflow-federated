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
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce


def build_federated_evaluation(
    model_fn,
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    use_experimental_simulation_loop: bool = False):
  """Builds the TFF computation for federated evaluation of the given model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    broadcast_process: a `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)` and have empty state. If
      set to default None, the server model is broadcast to the clients using
      the default tff.federated_broadcast.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the evaluation metrics
    as aggregated by `tff.learning.Model.federated_output_computation`.
  """
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO(b/124477628): Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = model_utils.weights_type_from_model(model)
    batch_type = computation_types.to_type(model.input_spec)

  @computations.tf_computation(model_weights_type,
                               computation_types.SequenceType(batch_type))
  @tf.function
  def client_eval(incoming_model_weights, dataset):
    """Returns local outputs after evaluting `model_weights` on `dataset`."""
    with tf.init_scope():
      model = model_fn()
    model_weights = model_utils.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          incoming_model_weights)

    def reduce_fn(prev_loss, batch):
      model_output = model.forward_pass(batch, training=False)
      return prev_loss + tf.cast(model_output.loss, tf.float64)

    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    dataset_reduce_fn(
        reduce_fn=reduce_fn,
        dataset=dataset,
        initial_state_fn=lambda: tf.constant(0, dtype=tf.float64))
    return collections.OrderedDict(local_outputs=model.report_local_outputs())

  @computations.federated_computation(
      computation_types.FederatedType(model_weights_type, placements.SERVER),
      computation_types.FederatedType(
          computation_types.SequenceType(batch_type), placements.CLIENTS))
  def server_eval(server_model_weights, federated_dataset):
    if broadcast_process is not None:
      # TODO(b/179091838): Confirm that the process has no state.
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
    return model.federated_output_computation(client_outputs.local_outputs)

  return server_eval
