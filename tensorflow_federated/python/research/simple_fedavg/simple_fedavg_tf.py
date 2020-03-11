# Lint as: python3
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

import collections
import attr
import tensorflow as tf
import tensorflow_federated as tff


ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')


class KerasModelWrapper(object):
  """A standalone keras wrapper to be used in TFF."""

  def __init__(self, keras_model, sample_batch, loss):
    """A wrapper class that provides necessary API handles for TFF.

    Args:
      keras_model: A `tf.keras.Model` to be trained.
      sample_batch: A `collections.Mapping` with two keys, `x` for inputs and
        `y` for labels.
      loss: A `tf.keras.losses.Loss` instance to be used for training.
    """
    self.keras_model = keras_model
    self.input_spec = self._get_input_spec(sample_batch)
    self.loss = loss
    # Eagerly construct the model variables now. Keras will lazily wait until
    # the first call, but TFF needs to statically know the shape of the model
    # during computation tracing.
    self._eagerly_construct_model_vars(sample_batch)

  def forward_pass(self, batch_input, training=True):
    """Forward pass of the model to get loss for a batch of data.

    Args:
      batch_input: A `collections.Mapping` with two keys, `x` for inputs and `y`
        for labels.
      training: Boolean scalar indicating training or inference mode.

    Returns:
      A scalar tf.float32 `tf.Tensor` loss for current batch input.
    """
    preds = self.keras_model(batch_input['x'], training=training)
    loss = self.loss(batch_input['y'], preds)
    return loss

  def _eagerly_construct_model_vars(self, sample_batch):
    """Forces construction of variables."""
    self.keras_model(sample_batch['x'])

  def _get_input_spec(self, sample_batch):
    """Gets the input data type to specify in TFF."""

    def _tensor_spec_with_undefined_batch_dim(tensor):
      # Remove the batch dimension and leave it unspecified.
      spec = tf.TensorSpec(
          shape=[None] + tensor.shape.dims[1:], dtype=tensor.dtype)
      return spec

    tf_dataset_type = tf.nest.map_structure(
        _tensor_spec_with_undefined_batch_dim, sample_batch)
    return tf_dataset_type

  @property
  def weights(self):
    return ModelWeights(
        trainable=self.keras_model.trainable_variables,
        non_trainable=self.keras_model.non_trainable_variables)

  def from_weights(self, model_weights):
    tff.utils.assign(self.keras_model.trainable_variables,
                     list(model_weights.trainable))
    tff.utils.assign(self.keras_model.non_trainable_variables,
                     list(model_weights.non_trainable))


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  model_output = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model_weights`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  -   'round_num': Current round index
  """
  model_weights = attr.ib()
  optimizer_state = attr.ib()
  round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
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
def server_update(model, server_optimizer, server_state, weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `KerasModelWrapper` or `tff.learning.Model`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
      creates variables, they must have already been created.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: A nested structure of tensors holding the updates to the
      trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  # Initialize the model with the current state.
  model_weights = model.weights
  tff.utils.assign(model_weights, server_state.model_weights)
  tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

  # Apply the update to the model.
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
      tf.nest.flatten(model_weights.trainable))
  server_optimizer.apply_gradients(grads_and_vars, name='server_update')

  # Create a new state based on the updated model.
  return tff.utils.update_state(
      server_state,
      model_weights=model_weights,
      optimizer_state=server_optimizer.variables(),
      round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
  """Builds `BroadcastMessage` for broadcasting.

  This method can be used to post-process `ServerState` before broadcasting.
  For example, perform model compression on `ServerState` to obtain a compressed
  state that is sent in a `BroadcastMessage`.

  Args:
    server_state: A `ServerState`.

  Returns:
    A `BroadcastMessage`.
  """
  return BroadcastMessage(
      model_weights=server_state.model_weights,
      round_num=server_state.round_num)


@tf.function
def client_update(model, dataset, server_message, client_optimizer):
  """Performans client local training of `model` on `dataset`.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    server_message: A `BroadcastMessage` from server.
    client_optimizer: A `tf.keras.optimizers.Optimizer`.

  Returns:
    A 'ClientOutput`.
  """
  model_weights = model.weights
  initial_weights = server_message.model_weights
  tff.utils.assign(model_weights, initial_weights)

  num_examples = tf.constant(0, dtype=tf.int32)
  loss_sum = tf.constant(0, dtype=tf.float32)
  for batch in dataset:
    with tf.GradientTape() as tape:
      loss = model.forward_pass(batch)
    grads = tape.gradient(loss, model_weights.trainable)
    grads_and_vars = zip(grads, model_weights.trainable)
    client_optimizer.apply_gradients(grads_and_vars)
    batch_size = tf.shape(batch['x'])[0]
    num_examples += batch_size
    loss_sum += loss * tf.cast(batch_size, tf.float32)

  # TODO(b/142341957): This control_dependency should not be needed, but is
  # currently necessary to work around a TF bug with how tf.function handles
  # tf.data.Datasets.
  with tf.control_dependencies([num_examples]):
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)

  client_weight = tf.cast(num_examples, tf.float32)
  return ClientOutput(weights_delta, client_weight, loss_sum / client_weight)
