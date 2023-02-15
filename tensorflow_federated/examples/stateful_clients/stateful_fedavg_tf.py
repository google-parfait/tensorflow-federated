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
"""An implementation of the FedAvg algorithm with stateful clients.

The TF functions for sever and client udpates.
"""

import collections
from typing import Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


def get_model_weights(
    model: Union[tff.learning.models.VariableModel, 'KerasModelWrapper']
) -> Union[tff.learning.models.ModelWeights, ModelWeights]:
  """Gets the appropriate ModelWeights object based on the model type."""
  if isinstance(model, tff.learning.models.VariableModel):
    return tff.learning.models.ModelWeights.from_model(model)
  else:
    # Using simple_fedavg custom Keras wrapper.
    return model.weights


class KerasModelWrapper:
  """A standalone keras wrapper to be used in TFF."""

  def __init__(self, keras_model, input_spec, loss):
    """A wrapper class that provides necessary API handles for TFF.

    Args:
      keras_model: A `tf.keras.Model` to be trained.
      input_spec: Metadata of dataset that desribes the input tensors, which
        will be converted to `tff.Type` specifying the expected type of input
        and output of the model.
      loss: A `tf.keras.losses.Loss` instance to be used for training.
    """
    self.keras_model = keras_model
    self.input_spec = input_spec
    self.loss = loss

  def forward_pass(self, batch_input, training=True):
    """Forward pass of the model to get loss for a batch of data.

    Args:
      batch_input: A `collections.abc.Mapping` with two keys, `x` for inputs and
        `y` for labels.
      training: Boolean scalar indicating training or inference mode.

    Returns:
      A scalar tf.float32 `tf.Tensor` loss for current batch input.
    """
    preds = self.keras_model(batch_input['x'], training=training)
    loss = self.loss(batch_input['y'], preds)
    return ModelOutputs(loss=loss)

  @property
  def weights(self):
    return ModelWeights(
        trainable=self.keras_model.trainable_variables,
        non_trainable=self.keras_model.non_trainable_variables,
    )

  def from_weights(self, model_weights):
    tf.nest.map_structure(
        lambda v, t: v.assign(t),
        self.keras_model.trainable_variables,
        list(model_weights.trainable),
    )
    tf.nest.map_structure(
        lambda v, t: v.assign(t),
        self.keras_model.non_trainable_variables,
        list(model_weights.non_trainable),
    )


def keras_evaluate(model, test_data, metric):
  metric.reset_states()
  for batch in test_data:
    preds = model(batch['x'], training=False)
    metric.update_state(y_true=batch['y'], y_pred=preds)
  return metric.result()


@attr.s(eq=False, frozen=True, slots=True)
class ClientState:
  """Structure for state on the client.

  Fields:
  -   `client_index`: The client index integer to map the client state back to
      the database hosting client states in the driver file.
  -   `iters_count`: The number of total iterations a client has computed in
      the total rounds so far.
  """

  client_index = attr.ib()
  iters_count = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput:
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
       variables.
  -   `client_weight`: Weight to be used in a weighted mean when
       aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics`,
      reflecting the
       results of training on the input dataset.
  -   `client_state`: The updated `ClientState`.
  """

  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()
  client_state = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState:
  """Structure for state on the server.

  Fields:
  -   `model_weights`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  -   'round_num': Current round index
  -   `total_iters_count`: The total number of iterations run on seen clients
  """

  model_weights = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()
  total_iters_count = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage:
  """Structure for tensors broadcasted by server during federated optimization.

  Fields:
  -   `model_weights`: A dictionary of model's trainable tensors.
  -   `round_num`: Round index to broadcast. We use `round_num` as an example to
       show how to broadcast auxiliary information that can be helpful on
       clients. It is not explicitly used, but can be applied to enable
       learning rate scheduling.
  """

  model_weights = attr.ib()
  round_num = attr.ib()


@tf.function
def server_update(
    model, server_optimizer, server_state, weights_delta, total_iters_count
):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.models.VariableModel`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.
    total_iters_count: A scalar to update `ServerState.total_iters_count`.

  Returns:
    An updated `ServerState`.
  """
  # Initialize the model with the current state.
  model_weights = get_model_weights(model)
  tf.nest.map_structure(
      lambda v, t: v.assign(t), model_weights, server_state.model_weights
  )
  tf.nest.map_structure(
      lambda v, t: v.assign(t),
      server_optimizer.variables(),
      server_state.optimizer_state,
  )

  # Apply the update to the model.
  neg_weights_delta = [-1.0 * x for x in weights_delta]
  server_optimizer.apply_gradients(
      zip(neg_weights_delta, model_weights.trainable), name='server_update'
  )

  # Create a new state based on the updated model.
  return tff.structure.update_struct(
      server_state,
      model_weights=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1,
      total_iters_count=total_iters_count,
  )


@tf.function
def build_server_broadcast_message(server_state):
  """Build `BroadcastMessage` for broadcasting.

  This method can be used to post-process `ServerState` before broadcasting.
  For example, perform model compression on `ServerState` to obtain a compressed
  state that is sent in a `BroadcastMessage`.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model_weights, round_num=server_state.round_num
  )


@tf.function
def client_update(
    model, dataset, client_state, server_message, client_optimizer
):
  """Performans client local training of `model` on `dataset`.

  Args:
    model: A `tff.learning.models.VariableModel`.
    dataset: A 'tf.data.Dataset'.
    client_state: A 'ClientState'.
    server_message: A `BroadcastMessage` from server.
    client_optimizer: A `tf.keras.optimizers.Optimizer`.

  Returns:
    A 'ClientOutput`.
  """
  model_weights = get_model_weights(model)
  initial_weights = server_message.model_weights
  tf.nest.map_structure(
      lambda v, t: v.assign(t), model_weights, initial_weights
  )

  num_examples = tf.constant(0, dtype=tf.int32)
  loss_sum = tf.constant(0, dtype=tf.float32)
  iters_count = tf.convert_to_tensor(client_state.iters_count)
  for batch in dataset:
    with tf.GradientTape() as tape:
      outputs = model.forward_pass(batch)
    grads = tape.gradient(outputs.loss, model_weights.trainable)
    client_optimizer.apply_gradients(zip(grads, model_weights.trainable))
    batch_size = tf.shape(batch['x'])[0]
    num_examples += batch_size
    loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)
    iters_count += 1

  weights_delta = tf.nest.map_structure(
      lambda a, b: a - b, model_weights.trainable, initial_weights.trainable
  )

  client_weight = tf.cast(num_examples, tf.float32)
  return ClientOutput(
      weights_delta,
      client_weight,
      loss_sum / client_weight,
      ClientState(
          client_index=client_state.client_index, iters_count=iters_count
      ),
  )
