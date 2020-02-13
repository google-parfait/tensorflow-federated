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
"""Server optimizer with adaptive learning rate for federated learning.

At each round, clients sum up the gradients' l2 norm layer-wisely and send to
the server. Server optimizer uses the average of gradients norm from clients to
normalize the learning rate of each layer.

"""

import collections

import attr
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.flars import flars_optimizer
from tensorflow_federated.python.tensorflow_libs import tensor_utils


@attr.s(eq=False, frozen=True)
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
  -   `optimizer_output`: structure containing tensors used by the optimizer.
  """
  weights_delta = attr.ib()
  weights_delta_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: A dictionary of model's trainable variables.
  -   `optimizer_state`: the list of variables of the optimizer.
  """
  model = attr.ib()
  optimizer_state = attr.ib()

  @classmethod
  def from_tff_result(cls, anon_tuple):
    return cls(
        model=tff.learning.framework.ModelWeights.from_tff_result(
            anon_tuple.model),
        optimizer_state=list(anon_tuple.optimizer_state))


def _create_optimizer_vars(model, optimizer):
  """Generate variables for optimizer."""
  model_weights = _get_weights(model)
  delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  flat_trainable_weights = tf.nest.flatten(model_weights.trainable)
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
      tf.nest.flatten(model_weights.trainable))
  optimizer.update_grads_norm(
      flat_trainable_weights,
      [tf.constant(1, dtype=w.dtype) for w in flat_trainable_weights])
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
                  weights_delta, grads_norm):
  """Updates `server_state` based on `weights_delta`.

  Args:
    model: A `tff.learning.TrainableModel`.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_optimizer_vars: A list of previous variables of server_optimzer.
    server_state: A `ServerState` namedtuple, the state to be updated.
    weights_delta: An update to the trainable variables of the model.
    grads_norm: Summation of the norm of gradients from clients.

  Returns:
    An updated `ServerState`.
  """
  model_weights = _get_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t),
                        (model_weights, server_optimizer_vars),
                        (server_state.model, server_state.optimizer_state))

  # Zero out the weight if there are any non-finite values.
  weights_delta, _ = (tensor_utils.zero_all_if_any_non_finite(weights_delta))

  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
      tf.nest.flatten(model_weights.trainable))

  server_optimizer.update_grads_norm(
      tf.nest.flatten(model_weights.trainable), grads_norm)
  server_optimizer.apply_gradients(grads_and_vars, name='server_update')

  return tff.utils.update_state(
      server_state, model=model_weights, optimizer_state=server_optimizer_vars)


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
  model_weights = _get_weights(model)
  tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                        initial_weights)
  flat_trainable_weights = tuple(tf.nest.flatten(model_weights.trainable))

  @tf.function
  def reduce_fn(state, batch):
    """Train on local client batch, summing the gradients and gradients norm."""
    flat_accumulated_grads, flat_accumulated_grads_norm, batch_weight_sum = state

    # Unliked the FedAvg client update, we need to capture the gradients during
    # training so we can send back the norms to the server.
    with tf.GradientTape() as tape:
      output = model.forward_pass(batch)

    model_weights = _get_weights(model)
    flat_grads = tape.gradient(output.loss,
                               tf.nest.flatten(model_weights.trainable))

    grads_and_vars = list(
        zip(flat_grads, tf.nest.flatten(model_weights.trainable)))

    # Support keras model only for now.
    model._model._keras_model.optimizer.apply_gradients(grads_and_vars)  # pylint: disable=protected-access

    batch_weight = tf.cast(tf.shape(output.predictions)[0], dtype=tf.float32)

    flat_accumulated_grads = tuple(
        accumulator + batch_weight * grad
        for accumulator, grad in zip(flat_accumulated_grads, flat_grads))

    flat_accumulated_grads_norm = tuple(
        norm_accumulator + batch_weight * tf.norm(grad)
        for norm_accumulator, grad in zip(flat_accumulated_grads_norm,
                                          flat_grads))

    return (flat_accumulated_grads, flat_accumulated_grads_norm,
            batch_weight_sum + batch_weight)

  def _zero_initial_state():
    """Create a tuple of (tuple of gradient accumulators, batch weight sum)."""
    return (
        tuple(tf.zeros_like(w, dtype=w.dtype) for w in flat_trainable_weights),
        tuple(tf.constant(0, dtype=w.dtype) for w in flat_trainable_weights),
        tf.constant(0, dtype=tf.float32),
    )

  flat_grads_sum, flat_grads_norm_sum, batch_weight_sum = dataset.reduce(
      initial_state=_zero_initial_state(), reduce_func=reduce_fn)

  grads_sum = tf.nest.pack_sequence_as(
      _get_weights(model).trainable, flat_grads_sum)
  weights_delta = tf.nest.map_structure(
      lambda gradient: -1.0 * gradient / batch_weight_sum, grads_sum)

  flat_grads_norm_sum = tf.nest.map_structure(
      lambda grad_norm: grad_norm / batch_weight_sum, flat_grads_norm_sum)

  aggregated_outputs = model.report_local_outputs()

  weights_delta, has_non_finite_delta = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  # Zero out the weight if there are any non-finite values.
  if has_non_finite_delta > 0:
    weights_delta_weight = tf.constant(0.0)
  else:
    weights_delta_weight = batch_weight_sum

  return ClientOutput(
      weights_delta,
      weights_delta_weight,
      model_output=aggregated_outputs,
      optimizer_output=collections.OrderedDict([
          ('num_examples', batch_weight_sum),
          ('flat_grads_norm_sum', flat_grads_norm_sum),
      ]))


def build_server_init_fn(model_fn, server_optimizer_fn):
  """Builds a `tff.tf_computation` that returns initial `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
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

  return server_init_tf


def build_server_update_fn(model_fn, server_optimizer_fn, server_state_type,
                           model_weights_type, grads_norm_type):
  """Builds a `tff.tf_computation` that updates `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    server_state_type: type_signature of server state.
    model_weights_type: type_signature of model weights.
    grads_norm_type: type_signature of the norm of gradients from clients.

  Returns:
    A `tff.tf_computation` that updates `ServerState`.
  """

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      grads_norm_type)
  def server_update_tf(server_state, model_delta, grads_norm):
    """Updates the `server_state`.

    Args:
      server_state: The `ServerState`.
      model_delta: The model difference from clients.
      grads_norm: Summation of the norm of gradients from clients.

    Returns:
      The updated `ServerState`.
    """
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer)

    return server_update(model, server_optimizer, server_optimizer_vars,
                         server_state, model_delta, grads_norm)

  return server_update_tf


def build_client_update_fn(model_fn, tf_dataset_type, model_weights_type):
  """Builds a `tff.tf_computation` for local model optimization.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    tf_dataset_type: type_signature of dataset.
    model_weights_type: type_signature of model weights.

  Returns:
    A `tff.tf_computation` for local model optimization.
  """

  @tff.tf_computation(tf_dataset_type, model_weights_type)
  def client_delta_tf(tf_dataset, initial_model_weights):
    """Performs client local model optimization.

    Args:
      tf_dataset: a `tf.data.Dataset` that provides training examples.
      initial_model_weights: a `model_utils.ModelWeights` containing the
        starting weights.

    Returns:
      A `ClientOutput`.
    """
    return client_update(model_fn(), tf_dataset, initial_model_weights)

  return client_delta_tf


def build_run_one_round_fn(server_update_fn, client_update_fn,
                           dummy_model_for_metadata,
                           federated_server_state_type, federated_dataset_type):
  """Builds a `tff.federated_computation` for a round of training.

  Args:
    server_update_fn: A function for updates in the server.
    client_update_fn: A function for updates in the clients.
    dummy_model_for_metadata: A dummy `tff.learning.TrainableModel`.
    federated_server_state_type: type_signature of federated server state.
    federated_dataset_type: type_signature of federated dataset.

  Returns:
    A `tff.federated_computation` for a round of training.
  """

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_model = tff.federated_broadcast(server_state.model)

    client_outputs = tff.federated_map(client_update_fn,
                                       (federated_dataset, client_model))

    weight_denom = client_outputs.weights_delta_weight
    round_model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=weight_denom)

    round_grads_norm = tff.federated_mean(
        client_outputs.optimizer_output.flat_grads_norm_sum,
        weight=weight_denom)

    server_state = tff.federated_map(
        server_update_fn, (server_state, round_model_delta, round_grads_norm))

    aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
        client_outputs.model_output)
    aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  return run_one_round


def build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=lambda: flars_optimizer.FLARSOptimizer(learning_rate=1.0
                                                              )):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.utils.IterativeProcess`.
  """
  dummy_model_for_metadata = model_fn()

  type_signature_grads_norm = tff.NamedTupleType([
      weight.dtype for weight in tf.nest.flatten(
          _get_weights(dummy_model_for_metadata).trainable)
  ])

  server_init_tf = build_server_init_fn(model_fn, server_optimizer_fn)
  server_state_type = server_init_tf.type_signature.result
  server_update_fn = build_server_update_fn(model_fn, server_optimizer_fn,
                                            server_state_type,
                                            server_state_type.model,
                                            type_signature_grads_norm)

  tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)
  client_update_fn = build_client_update_fn(model_fn, tf_dataset_type,
                                            server_state_type.model)

  federated_server_state_type = tff.FederatedType(server_state_type, tff.SERVER)
  federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
  run_one_round_tff = build_run_one_round_fn(server_update_fn, client_update_fn,
                                             dummy_model_for_metadata,
                                             federated_server_state_type,
                                             federated_dataset_type)

  return tff.utils.IterativeProcess(
      initialize_fn=tff.federated_computation(
          lambda: tff.federated_value(server_init_tf(), tff.SERVER)),
      next_fn=run_one_round_tff)
