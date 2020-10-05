# Copyright 2020, The TensorFlow Federated Authors.
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
"""An implementation of the Federated Averaging algorithm.

This is intended to be a minimal stand-alone implementation of Federated
Averaging, suitable for branching as a starting point for algorithm
modifications; see `tff.learning.build_federated_averaging_process` for a
more full-featured implementation.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.examples.stateful_clients.stateful_fedavg_tf import build_server_broadcast_message
from tensorflow_federated.python.examples.stateful_clients.stateful_fedavg_tf import client_update
from tensorflow_federated.python.examples.stateful_clients.stateful_fedavg_tf import server_update
from tensorflow_federated.python.examples.stateful_clients.stateful_fedavg_tf import ServerState


def _initialize_optimizer_vars(model, optimizer):
  """Creates optimizer variables to assign the optimizer's state."""
  model_weights = model.weights
  model_delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  # Create zero gradients to force an update that doesn't modify.
  # Force eagerly constructing the optimizer variables. Normally Keras lazily
  # creates the variables on first usage of the optimizer. Optimizers such as
  # Adam, Adagrad, or using momentum need to create a new set of variables shape
  # like the model weights.
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (tf.zeros_like(x), v), tf.nest.flatten(model_delta),
      tf.nest.flatten(model_weights.trainable))
  optimizer.apply_gradients(grads_and_vars)
  assert optimizer.variables()


def build_federated_averaging_process(
    model_fn,
    client_state_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    client_state_fn: A no-arg function that returns a
      `stateful_fedavg_tf.ClientState`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for server update.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  dummy_model = model_fn()

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model_weights=model.weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0,
        total_iters_count=0)

  server_state_type = server_init_tf.type_signature.result

  model_weights_type = server_state_type.model_weights

  client_state_type = tff.framework.type_from_tensors(client_state_fn())

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      client_state_type.iters_count)
  def server_update_fn(server_state, model_delta, total_iters_count):
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta,
                         total_iters_count)

  @tff.tf_computation(server_state_type)
  def server_message_fn(server_state):
    return build_server_broadcast_message(server_state)

  server_message_type = server_message_fn.type_signature.result
  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

  @tff.tf_computation(tf_dataset_type, client_state_type, server_message_type)
  def client_update_fn(tf_dataset, client_state, server_message):
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    return client_update(model, tf_dataset, client_state, server_message,
                         client_optimizer)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)

  federated_client_state_type = tff.type_at_clients(client_state_type)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type,
                             federated_client_state_type)
  def run_one_round(server_state, federated_dataset, client_states):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `stateful_fedavg_tf.ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.
      client_states: A federated `stateful_fedavg_tf.ClientState`.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_states, server_message_at_client))

    weight_denom = client_outputs.client_weight
    round_model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=weight_denom)
    total_iters_count = tff.federated_sum(
        client_outputs.client_state.iters_count)
    server_state = tff.federated_map(
        server_update_fn, (server_state, round_model_delta, total_iters_count))
    round_loss_metric = tff.federated_mean(
        client_outputs.model_output, weight=weight_denom)

    return server_state, round_loss_metric, client_outputs.client_state

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round)
