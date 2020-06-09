# Copyright 2018, The TensorFlow Federated Authors.
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
"""Common building blocks for federated optimization algorithms."""

import abc
import collections
from typing import Callable, List

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils


# Type aliases.
_ModelConstructor = Callable[[], model_lib.Model]
_OptimizerConstructor = Callable[[], tf.keras.optimizers.Optimizer]


@attr.s(eq=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: a dictionary of updates to the model's trainable
      variables.
  -   `weights_delta_weight`: weight to use in a weighted mean when aggregating
      `weights_delta`.
  -   `model_output`: a structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  weights_delta_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib()


class ClientDeltaFn(object, metaclass=abc.ABCMeta):
  """Represents a client computation that produces an update to a model."""

  @abc.abstractproperty
  def variables(self):
    """Returns all the variables of this object.

    Note this only includes variables that are part of the state of this object,
    and not the model variables themselves.

    Returns:
      An iterable of `tf.Variable` objects.
    """
    pass

  @abc.abstractmethod
  def __call__(self, dataset, initial_weights):
    """Defines the complete client computation.

    Typically implementations should be decorated with `tf.function`.

    Args:
      dataset: a `tf.data.Dataset` producing batches than can be fed to
        `tff.learning.Model.forward_pass`.
      initial_weights: a dictionary of initial values for all trainable and
        non-trainable model variables, keyed by name. This will be supplied by
        the server in Federated Averaging.

    Returns:
      An `optimizer_utils.ClientOutput` namedtuple.
    """
    pass


@attr.s(eq=False, frozen=True)
class ServerState(object):
  """Represents the state of the server carried between rounds.

  Attributes:
    model: a `ModelWeights` structure, containing Tensors or Variables.
    optimizer_state: a list of Tensors or Variables, in the order returned by
      `optimizer.variables()`
    delta_aggregate_state: state (possibly empty) of the delta_aggregate_fn.
    model_broadcast_state: state (possibly empty) of the model_broadcast_fn.
  """
  model = attr.ib()
  optimizer_state = attr.ib()
  delta_aggregate_state = attr.ib()
  model_broadcast_state = attr.ib()


def state_with_new_model_weights(
    server_state: ServerState,
    trainable_weights: List[np.ndarray],
    non_trainable_weights: List[np.ndarray],
) -> ServerState:
  """Returns a `ServerState` with updated model weights.

  Args:
    server_state: a server state object returned by an iterative training
      process like `tff.learning.build_federated_averaging_process`.
    trainable_weights: a list of `numpy` arrays in the order of the original
      model's `trainable_variables`.
    non_trainable_weights: a list of `numpy` arrays in the order of the original
      model's `non_trainable_variables`.

  Returns:
    A new server `ServerState` object which can be passed to the `next` method
    of the iterative process.
  """
  # TODO(b/123092620): This can be simplified if TFF stops exposing
  # AnonymousTuple.
  py_typecheck.check_type(server_state, anonymous_tuple.AnonymousTuple)
  leaf_types = (int, float, np.ndarray, tf.Tensor)

  def assert_weight_lists_match(old_value, new_value):
    """Assert two flat lists of ndarrays or tensors match."""
    if isinstance(new_value, leaf_types) and isinstance(old_value, leaf_types):
      if (old_value.dtype != new_value.dtype or
          old_value.shape != new_value.shape):
        raise TypeError('Element is not the same tensor type. old '
                        f'({old_value.dtype}, {old_value.shape}) != '
                        f'new ({new_value.dtype}, {new_value.shape})')
    elif (isinstance(new_value, collections.Sequence) and
          isinstance(old_value, anonymous_tuple.AnonymousTuple)):
      if anonymous_tuple.name_list(old_value):
        raise TypeError('`tff.learning` does not support named structures of '
                        'model weights. Received: {old_value}')
      if len(old_value) != len(new_value):
        raise TypeError('Model weights have different lengths: '
                        f'(old) {len(old_value)} != (new) {len(new_value)})\n'
                        f'Old values: {old_value}\nNew values: {new_value}')
      for old, new in zip(old_value, new_value):
        assert_weight_lists_match(old, new)
    else:
      raise TypeError('Model weights structures contains types that cannot be '
                      'handled.\nOld weights structure: {old}\n'
                      'New weights structure: {new}\n'
                      'Must be one of (int, float, np.ndarray, tf.Tensor, '
                      'collections.Sequence)'.format(
                          old=tf.nest.map_structure(type, old_value),
                          new=tf.nest.map_structure(type, new_value)))

  assert_weight_lists_match(server_state.model.trainable, trainable_weights)
  assert_weight_lists_match(server_state.model.non_trainable,
                            non_trainable_weights)
  # TODO(b/123092620): We can't use tff.utils.update_state because this
  # is an AnonymousTuple, not a ServerState. We should do something
  # that doesn't mention every entry in the state.
  return ServerState(
      model=model_utils.ModelWeights(
          trainable=trainable_weights, non_trainable=non_trainable_weights),
      optimizer_state=server_state.optimizer_state,
      delta_aggregate_state=server_state.delta_aggregate_state,
      model_broadcast_state=server_state.model_broadcast_state)


def _apply_delta(
    *,
    optimizer: tf.keras.optimizers.Optimizer,
    model: model_lib.Model,
    delta,
) -> None:
  """Applies `delta` to `model` using `optimizer`."""
  model_variables = model_utils.ModelWeights.from_model(model)
  tf.nest.assert_same_structure(delta, model_variables.trainable)
  grads_and_vars = tf.nest.map_structure(
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta),
      tf.nest.flatten(model_variables.trainable))
  # Note: this may create variables inside `optimizer`, for example if this is
  # the first usage of Adam or momentum optmizers.
  optimizer.apply_gradients(grads_and_vars)


def _eagerly_create_optimizer_variables(
    *, model: model_lib.Model,
    optimizer: tf.keras.optimizers.Optimizer) -> List[tf.Variable]:
  """Forces eager construction of the optimizer variables.

  This code is needed both in `server_init` and `server_update` (to introduce
  variables so we can read their initial values for the initial state).

  Args:
    model: a `tff.learning.Model`.
    optimizer: a `tf.keras.optimizers.Optimizer`.

  Returns:
    A list of optimizer variables.
  """
  delta_tensor_spec = tf.nest.map_structure(
      lambda v: tf.TensorSpec.from_tensor(v.read_value()),
      model_utils.ModelWeights.from_model(model).trainable)
  # Trace the function, which forces eager variable creation.
  tf.function(_apply_delta).get_concrete_function(
      optimizer=optimizer, model=model, delta=delta_tensor_spec)
  return optimizer.variables()


# ==============================================================================
# Federated Computations
#
# These constructors setup the system level orchestration logic.
# ==============================================================================


def _build_initialize_computaiton(
    *,
    model_fn: _ModelConstructor,
    server_optimizer_fn: _OptimizerConstructor,
    delta_aggregate_fn: tff.utils.StatefulAggregateFn,
    model_broadcast_fn: tff.utils.StatefulBroadcastFn,
) -> tff.Computation:
  """Builds the `initialize` computation for a model delta averaging process.

  Args:
    model_fn: a no-argument callable that constructs and returns a
      `tff.learning.Model`. *Must* construct and return a new model when called.
      Returning captured models from other scopes will raise errors.
    server_optimizer_fn: a no-argument callable that constructs and returns a
      `tf.keras.optimizers.Optimizer`. *Must* construct and return a new
      optimizer when called. Returning captured optimizers from other scopes
      will raise errors.
    delta_aggregate_fn: a `tff.utils.StatefulAggregateFn` to aggregate client
      model deltas.
    model_broadcast_fn: a `tff.utils.StatefulBroadcastFn` to broadcast the
      global model to the clients.

  Returns:
    A `tff.Computation` that initializes the process. The computation takes no
    arguments and returns `ServerState` value with `tff.SERVER` placement.
  """

  @tff.tf_computation
  def server_init() -> ServerState:
    """Returns initial `tff.learning.framework.ServerState`.

    Returns:
      A `tff.learning.framework.ServerState` namedtuple.
    """
    model = model_fn()
    optimizer = server_optimizer_fn()
    # We must force variable creation for momentum and adaptive optimizers.
    optimizer_vars = _eagerly_create_optimizer_variables(
        model=model, optimizer=optimizer)
    return ServerState(
        model=model_utils.ModelWeights.from_model(model),
        optimizer_state=optimizer_vars,
        delta_aggregate_state=delta_aggregate_fn.initialize(),
        model_broadcast_state=model_broadcast_fn.initialize())

  @tff.federated_computation()
  def initialize_computation():
    """Orchestration logic for server model initialization."""
    return tff.federated_eval(server_init, tff.SERVER)

  return initialize_computation


def _build_one_round_computation(
    *,
    model_fn: _ModelConstructor,
    server_optimizer_fn: _OptimizerConstructor,
    model_to_client_delta_fn: Callable[[Callable[[], model_lib.Model]],
                                       ClientDeltaFn],
    delta_aggregate_fn: tff.utils.StatefulAggregateFn,
    model_broadcast_fn: tff.utils.StatefulBroadcastFn,
    delta_aggregate_state_type: tff.Type,
    model_broadcast_state_type: tff.Type,
) -> tff.Computation:
  """Builds the `next` computation for a model delta averaging process.

  Args:
    model_fn: a no-argument callable that constructs and returns a
      `tff.learning.Model`. *Must* construct and return a new model when called.
      Returning captured models from other scopes will raise errors.
    server_optimizer_fn: a no-argument callable that constructs and returns a
      `tf.keras.optimizers.Optimizer`. *Must* construct and return a new
      optimizer when called. Returning captured optimizers from other scopes
      will raise errors.
    model_to_client_delta_fn: a callable that takes a single no-arg callable
      that returns `tff.learning.Model` as an argument and returns a
      `ClientDeltaFn` which performs the local training loop and model delta
      computation.
    delta_aggregate_fn: a `tff.utils.StatefulAggregateFn` to aggregate client
      model deltas.
    model_broadcast_fn: a `tff.utils.StatefulBroadcastFn` to broadcast the
      global model to the clients.
    delta_aggregate_state_type: a `tff.Type` specifying the type structure of
      the state of `delta_aggregate_fn`.
    model_broadcast_state_type: a `tff.Type` specifying the type structure of
      the state of `model_broadcast_fn`.

  Returns:
    A `tff.Computation` that initializes the process. The computation takes
    a tuple of `(ServerState@SERVER, tf.data.Dataset@CLIENTS)` argument, and
    returns a tuple of `(ServerState@SERVER, metrics@SERVER)`.
  """
  # TODO(b/124477628): would be nice not to have the construct a throwaway model
  # here just to get the types. After fully moving to TF2.0 and eager-mode, we
  # should re-evaluate what happens here.
  # TODO(b/144382142): Keras name uniquification is probably the main reason we
  # still need this.
  with tf.Graph().as_default():
    dummy_model_for_metadata = model_fn()
    model_weights_type = tff.framework.type_from_tensors(
        model_utils.ModelWeights.from_model(dummy_model_for_metadata))

    dummy_optimizer = server_optimizer_fn()
    # We must force variable creation for momentum and adaptive optimizers.
    _eagerly_create_optimizer_variables(
        model=dummy_model_for_metadata, optimizer=dummy_optimizer)
    optimizer_variable_type = tff.framework.type_from_tensors(
        dummy_optimizer.variables())

  @tff.tf_computation(model_weights_type, model_weights_type.trainable,
                      optimizer_variable_type)
  def server_update(global_model, model_delta, optimizer_state):
    """Converts args to correct python types and calls server_update_model."""
    # Construct variables first.
    model = model_fn()
    optimizer = server_optimizer_fn()
    # We must force variable creation for momentum and adaptive optimizers.
    _eagerly_create_optimizer_variables(model=model, optimizer=optimizer)

    @tf.function
    def update_model_inner(weights_delta):
      """Applies the update to the global model."""
      model_variables = model_utils.ModelWeights.from_model(model)
      optimizer_variables = optimizer.variables()
      # We might have a NaN value e.g. if all of the clients processed
      # had no data, so the denominator in the federated_mean is zero.
      # If we see any NaNs, zero out the whole update.
      no_nan_weights_delta, _ = tensor_utils.zero_all_if_any_non_finite(
          weights_delta)

      # TODO(b/124538167): We should increment a server counter to
      # track the fact a non-finite weights_delta was encountered.

      # Set the variables to the current global model (before update).
      tf.nest.map_structure(lambda a, b: a.assign(b),
                            (model_variables, optimizer_variables),
                            (global_model, optimizer_state))
      # Update the variables with the delta, and return the new global model.
      _apply_delta(optimizer=optimizer, model=model, delta=no_nan_weights_delta)
      return model_variables, optimizer_variables

    return update_model_inner(model_delta)

  dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)

  @tff.tf_computation(dataset_type, model_weights_type)
  def _compute_local_training_and_client_delta(dataset, initial_model_weights):
    """Performs client local model optimization.

    Args:
      dataset: a `tf.data.Dataset` that provides training examples.
      initial_model_weights: a `model_utils.ModelWeights` containing the
        starting weights.

    Returns:
      A `ClientOutput` structure.
    """
    client_delta_fn = model_to_client_delta_fn(model_fn)
    client_output = client_delta_fn(dataset, initial_model_weights)
    return client_output

  server_state_type = ServerState(
      model=model_weights_type,
      optimizer_state=optimizer_variable_type,
      delta_aggregate_state=delta_aggregate_state_type,
      model_broadcast_state=model_broadcast_state_type)

  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(dataset_type, tff.CLIENTS))
  def one_round_computation(server_state, federated_dataset):
    """Orchestration logic for one round of optimization.

    Args:
      server_state: a `tff.learning.framework.ServerState` named tuple.
      federated_dataset: a federated `tf.Dataset` with placement tff.CLIENTS.

    Returns:
      A tuple of updated `tff.learning.framework.ServerState` and the result of
      `tff.learning.Model.federated_output_computation`, both having
      `tff.SERVER` placement.
    """
    new_broadcaster_state, client_model = model_broadcast_fn(
        server_state.model_broadcast_state, server_state.model)

    client_outputs = tff.federated_map(_compute_local_training_and_client_delta,
                                       (federated_dataset, client_model))

    new_delta_aggregate_state, round_model_delta = delta_aggregate_fn(
        server_state.delta_aggregate_state,
        client_outputs.weights_delta,
        weight=client_outputs.weights_delta_weight)

    new_global_model, new_optimizer_state = tff.federated_map(
        server_update,
        (server_state.model, round_model_delta, server_state.optimizer_state))

    new_server_state = tff.federated_zip(
        ServerState(new_global_model, new_optimizer_state,
                    new_delta_aggregate_state, new_broadcaster_state))

    aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
        client_outputs.model_output)

    if isinstance(aggregated_outputs.type_signature, tff.NamedTupleType):
      # Promote the FederatedType outside the NamedTupleType.
      aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return new_server_state, aggregated_outputs

  return one_round_computation


def build_stateless_mean() -> tff.utils.StatefulAggregateFn:
  """Just tff.federated_mean with empty state, to use as a default."""

  @tff.tf_computation
  def _cast_weight_to_float(x):
    return tf.cast(x, tf.float32)

  def cast_to_float_mean(state, value, weight):
    return state, tff.federated_mean(
        value, weight=tff.federated_map(_cast_weight_to_float, weight))

  return tff.utils.StatefulAggregateFn(
      initialize_fn=lambda: (), next_fn=cast_to_float_mean)


def build_stateless_broadcaster() -> tff.utils.StatefulBroadcastFn:
  """Just tff.federated_broadcast with empty state, to use as a default."""
  return tff.utils.StatefulBroadcastFn(
      initialize_fn=lambda: (),
      next_fn=lambda state, value: (  # pylint: disable=g-long-lambda
          state, tff.federated_broadcast(value)))


def build_model_delta_optimizer_process(
    model_fn: _ModelConstructor,
    model_to_client_delta_fn: Callable[[model_lib.Model], ClientDeltaFn],
    server_optimizer_fn: _OptimizerConstructor,
    stateful_delta_aggregate_fn: tff.utils
    .StatefulAggregateFn = build_stateless_mean(),
    stateful_model_broadcast_fn: tff.utils
    .StatefulBroadcastFn = build_stateless_broadcaster(),
) -> tff.templates.IterativeProcess:
  """Constructs `tff.templates.IterativeProcess` for Federated Averaging or SGD.

  This provides the TFF orchestration logic connecting the common server logic
  which applies aggregated model deltas to the server model with a
  `ClientDeltaFn` that specifies how `weight_deltas` are computed on device.

  Note: We pass in functions rather than constructed objects so we can ensure
  any variables or ops created in constructors are placed in the correct graph.

  Args:
    model_fn: a no-arg function that returns a `tff.learning.Model`.
    model_to_client_delta_fn: a function from a `model_fn` to a `ClientDeltaFn`.
    server_optimizer_fn: a no-arg function that returns a `tf.Optimizer`. The
      `apply_gradients` method of this optimizer is used to apply client updates
      to the server model.
    stateful_delta_aggregate_fn: a `tff.utils.StatefulAggregateFn` where the
      `next_fn` performs a federated aggregation and upates state. That is, it
      has TFF type `(state@SERVER, value@CLIENTS, weights@CLIENTS) ->
      (state@SERVER, aggregate@SERVER)`, where the `value` type is
      `tff.learning.framework.ModelWeights.trainable` corresponding to the
      object returned by `model_fn`.
    stateful_model_broadcast_fn: a `tff.utils.StatefulBroadcastFn` where the
      `next_fn` performs a federated broadcast and upates state. That is, it has
      TFF type `(state@SERVER, value@SERVER) -> (state@SERVER, value@CLIENTS)`,
      where the `value` type is `tff.learning.framework.ModelWeights`
      corresponding to the object returned by `model_fn`.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(model_to_client_delta_fn)
  py_typecheck.check_callable(server_optimizer_fn)
  py_typecheck.check_type(stateful_delta_aggregate_fn,
                          tff.utils.StatefulAggregateFn)
  py_typecheck.check_type(stateful_model_broadcast_fn,
                          tff.utils.StatefulBroadcastFn)

  initialize_computation = _build_initialize_computaiton(
      model_fn=model_fn,
      server_optimizer_fn=server_optimizer_fn,
      delta_aggregate_fn=stateful_delta_aggregate_fn,
      model_broadcast_fn=stateful_model_broadcast_fn)

  delta_aggregate_state_type = initialize_computation.type_signature.result.member.delta_aggregate_state
  model_broadcast_state_type = initialize_computation.type_signature.result.member.model_broadcast_state
  run_one_round_computation = _build_one_round_computation(
      model_fn=model_fn,
      server_optimizer_fn=server_optimizer_fn,
      model_to_client_delta_fn=model_to_client_delta_fn,
      delta_aggregate_fn=stateful_delta_aggregate_fn,
      model_broadcast_fn=stateful_model_broadcast_fn,
      delta_aggregate_state_type=delta_aggregate_state_type,
      model_broadcast_state_type=model_broadcast_state_type)

  return tff.templates.IterativeProcess(
      initialize_fn=initialize_computation, next_fn=run_one_round_computation)
