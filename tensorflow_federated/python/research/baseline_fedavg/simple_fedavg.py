# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import attr
import tensorflow as tf

import tensorflow_federated as tff
from tensorflow_federated.python.tensorflow_libs import tensor_utils


@attr.s(cmp=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `weights_delta_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  weights_delta_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


@attr.s(cmp=False, frozen=True)
class ServerState(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `model`: A dictionary of model's trainable variables.
  -   `optimizer_state`: Variables of optimizer.
  """
  model = attr.ib()
  optimizer_state = attr.ib()


def _create_optimizer_vars(model, optimizer):
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
      tf.nest.flatten(_get_weights(model).trainable))
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  return optimizer.variables()


def _get_weights(model):
  model_weights = collections.namedtuple('ModelWeights',
                                         'trainable non_trainable')
  return model_weights(
      trainable=tensor_utils.to_var_dict(model.trainable_variables),
      non_trainable=tensor_utils.to_var_dict(model.non_trainable_variables))


@tf.function
def server_update(model, server_optimizer, server_optimizer_vars, server_state,
                  weights_delta):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `tff.learning.TrainableModel`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_optimizer_vars: A list of previous variables of server_optimzer.
    server_state: A `tff.learning.framework.ServerState` namedtuple, the state
      to be updated.
    weights_delta: An update to the trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  tf.nest.map_structure(tf.assign, (_get_weights(model), server_optimizer_vars),
                        (server_state.model, server_state.optimizer_state))

  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
      tf.nest.flatten(_get_weights(model).trainable))

  server_optimizer.apply_gradients(grads_and_vars, name='server_update')

  return tff.utils.update_state(
      server_state,
      model=_get_weights(model),
      optimizer_state=server_optimizer_vars)


@tf.function
def client_update(model, dataset, initial_weights):
  """Updates client model.

  Args:
    model: A `tff.learning.Model`.
    dataset: A 'tf.data.Dataset'.
    initial_weights: A `tff.learning.Model.weights` from server.

  Returns:
    A 'ClientOutput`.
  """
  tf.nest.map_structure(tf.assign, _get_weights(model), initial_weights)

  @tf.function
  def reduce_fn(num_examples_sum, batch):
    """Runs `tff.learning.Model.train_on_batch` on local client batch."""
    output = model.train_on_batch(batch)
    return num_examples_sum + tf.shape(output.predictions)[0]

  num_examples_sum = dataset.reduce(
      initial_state=tf.constant(0), reduce_func=reduce_fn)

  weights_delta = tf.nest.map_structure(tf.subtract,
                                        _get_weights(model).trainable,
                                        initial_weights.trainable)
  aggregated_outputs = model.report_local_outputs()

  weights_delta_weight = tf.cast(num_examples_sum, tf.float32)

  return ClientOutput(
      weights_delta, weights_delta_weight, aggregated_outputs,
      tensor_utils.to_odict({
          'num_examples': num_examples_sum,
      }))


def build_server_init_fn(model_fn, server_optimizer_fn):
  """Returns initial `tff.learning.framework.ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.learning.framework.ServerState` namedtuple.
  """

  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer)
    return ServerState(
        model=_get_weights(model), optimizer_state=server_optimizer_vars)

  @tff.federated_computation
  def server_init_tff():
    """Returns initial `tff.learning.framework.ServerState."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return server_init_tff


def build_server_update_fn(model_fn, server_optimizer_fn, server_state_type,
                           model_weights_type):
  """Build the server update function.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    server_state_type: type_signature of server state.
    model_weights_type: type_signature of model weights.

  Returns:
    A function for the server update.
  """

  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_tf(server_state, model_delta):
    """Build the server update function.

    Args:
      server_state: The server_state.
      model_delta: The model difference from clients

    Returns:
      A function for the server update.
    """
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer)

    return server_update(model, server_optimizer, server_optimizer_vars,
                         server_state, model_delta)

  return server_update_tf


def build_client_update_fn(model_fn, tf_dataset_type, model_weights_type):
  """Build the client update function.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    tf_dataset_type: type_signature of dataset.
    model_weights_type: type_signature of model weights.

  Returns:
    A function for the client update.
  """

  @tff.tf_computation(tf_dataset_type, model_weights_type)
  def client_delta_tf(tf_dataset, initial_model_weights):
    """Performs client local model optimization.

    Args:
      tf_dataset: a `tf.data.Dataset` that provides training examples.
      initial_model_weights: a `model_utils.ModelWeights` containing the
        starting weights.

    Returns:
        A `ClientOutput` structure.
    """
    model = model_fn()

    return client_update(model, tf_dataset, initial_model_weights)

  return client_delta_tf


def build_run_one_round_fn(server_update_fn, client_update_fn,
                           dummy_model_for_metadata,
                           federated_server_state_type, federated_dataset_type):
  """Build build_run_one_round_fn function.

  Args:
    server_update_fn: A function for updates in the server.
    client_update_fn: A function for updates in the clients.
    dummy_model_for_metadata: A dummy `tff.learning.TrainableModel`.
    federated_server_state_type: type_signature of federated server state.
    federated_dataset_type: type_signature of federated dataset.

  Returns:
    A function for the procedure of federated learning.
  """

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `tff.learning.framework.ServerState` named tuple.
      federated_dataset: A federated `tf.Dataset` with placement tff.CLIENTS.

    Returns:
      A tuple of updated `tff.learning.framework.ServerState` and the result of
    `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)

    client_outputs = tff.federated_map(client_update_fn,
                                       (federated_dataset, client_model))

    weight_denom = client_outputs.weights_delta_weight
    round_model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=weight_denom)

    server_state = tff.federated_apply(server_update_fn,
                                       (server_state, round_model_delta))

    aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
        client_outputs.model_output)
    aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  return run_one_round


def build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.utils.IterativeProcess`.
  """

  dummy_model_for_metadata = model_fn()

  server_init_tff = build_server_init_fn(model_fn, server_optimizer_fn)

  federated_server_state_type = server_init_tff.type_signature.result
  server_state_type = federated_server_state_type.member

  tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)
  federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

  server_update_fn = build_server_update_fn(model_fn, server_optimizer_fn,
                                            server_state_type,
                                            server_state_type.model)

  client_update_fn = build_client_update_fn(model_fn, tf_dataset_type,
                                            server_state_type.model)

  run_one_round_tff = build_run_one_round_fn(server_update_fn, client_update_fn,
                                             dummy_model_for_metadata,
                                             federated_server_state_type,
                                             federated_dataset_type)

  return tff.utils.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round_tff)
