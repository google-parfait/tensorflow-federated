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
from typing import Callable, List, Optional, Tuple, Union

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.utils import computation_utils
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils

# Type aliases.
_ModelConstructor = Callable[[], model_lib.Model]
_OptimizerConstructor = Callable[[], tf.keras.optimizers.Optimizer]


class ProcessTypeError(Exception):
  """Error raised when a `MeasuredProcess` does not have the correct type signature."""
  pass


class DisjointArgumentError(Exception):
  """Error raised when two disjoint arguments are specified (only one allowed)."""
  pass


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
  py_typecheck.check_type(server_state, ServerState)
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
          isinstance(old_value, collections.Sequence)):
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
  new_server_state = computation_utils.update_state(
      server_state,
      model=model_utils.ModelWeights(
          trainable=trainable_weights, non_trainable=non_trainable_weights))
  return new_server_state


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
    model: A `tff.learning.Model`.
    optimizer: A `tf.keras.optimizers.Optimizer`.

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


def _build_initialize_computation(
    *,
    model_fn: _ModelConstructor,
    server_optimizer_fn: _OptimizerConstructor,
    broadcast_process: measured_process.MeasuredProcess,
    aggregation_process: measured_process.MeasuredProcess,
) -> computation_base.Computation:
  """Builds the `initialize` computation for a model delta averaging process.

  Args:
    model_fn: a no-argument callable that constructs and returns a
      `tff.learning.Model`. *Must* construct and return a new model when called.
      Returning captured models from other scopes will raise errors.
    server_optimizer_fn: a no-argument callable that constructs and returns a
      `tf.keras.optimizers.Optimizer`. *Must* construct and return a new
      optimizer when called. Returning captured optimizers from other scopes
      will raise errors.
    broadcast_process: a `tff.templates.MeasuredProcess` to broadcast the global
      model to the clients.
    aggregation_process: a `tff.templates.MeasuredProcess` to aggregate client
      model deltas.

  Returns:
    A `tff.Computation` that initializes the process. The computation takes no
    arguments and returns a `tuple` of global model weights and server state
    with `tff.SERVER` placement.
  """

  @computations.tf_computation
  def server_init() -> Tuple[model_utils.ModelWeights, List[tf.Variable]]:
    """Returns initial `tff.learning.framework.ServerState`.

    Returns:
      A `tuple` of `tff.learning.framework.ModelWeights` and a `list` of
      `tf.Variable`s for the global optimizer state.
    """
    model = model_fn()
    optimizer = server_optimizer_fn()
    # We must force variable creation for momentum and adaptive optimizers.
    optimizer_vars = _eagerly_create_optimizer_variables(
        model=model, optimizer=optimizer)
    return model_utils.ModelWeights.from_model(model), optimizer_vars,

  @computations.federated_computation()
  def initialize_computation():
    """Orchestration logic for server model initialization."""
    initial_global_model, initial_global_optimizer_state = intrinsics.federated_eval(
        server_init, placements.SERVER)
    return intrinsics.federated_zip(
        ServerState(
            model=initial_global_model,
            optimizer_state=initial_global_optimizer_state,
            delta_aggregate_state=aggregation_process.initialize(),
            model_broadcast_state=broadcast_process.initialize()))

  return initialize_computation


def _build_one_round_computation(
    *,
    model_fn: _ModelConstructor,
    server_optimizer_fn: _OptimizerConstructor,
    model_to_client_delta_fn: Callable[[Callable[[], model_lib.Model]],
                                       ClientDeltaFn],
    broadcast_process: measured_process.MeasuredProcess,
    aggregation_process: measured_process.MeasuredProcess,
) -> computation_base.Computation:
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
    broadcast_process: a `tff.templates.MeasuredProcess` to broadcast the global
      model to the clients.
    aggregation_process: a `tff.templates.MeasuredProcess` to aggregate client
      model deltas.

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
    model_weights_type = model_utils.weights_type_from_model(
        dummy_model_for_metadata)

    dummy_optimizer = server_optimizer_fn()
    # We must force variable creation for momentum and adaptive optimizers.
    _eagerly_create_optimizer_variables(
        model=dummy_model_for_metadata, optimizer=dummy_optimizer)
    optimizer_variable_type = type_conversions.type_from_tensors(
        dummy_optimizer.variables())

  @computations.tf_computation(model_weights_type, model_weights_type.trainable,
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

  dataset_type = computation_types.SequenceType(
      dummy_model_for_metadata.input_spec)

  @computations.tf_computation(dataset_type, model_weights_type)
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

  broadcast_state = broadcast_process.initialize.type_signature.result.member
  aggregation_state = aggregation_process.initialize.type_signature.result.member

  server_state_type = ServerState(
      model=model_weights_type,
      optimizer_state=optimizer_variable_type,
      delta_aggregate_state=aggregation_state,
      model_broadcast_state=broadcast_state)

  @computations.federated_computation(
      computation_types.FederatedType(server_state_type, placements.SERVER),
      computation_types.FederatedType(dataset_type, placements.CLIENTS))
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
    broadcast_output = broadcast_process.next(
        server_state.model_broadcast_state, server_state.model)
    client_outputs = intrinsics.federated_map(
        _compute_local_training_and_client_delta,
        (federated_dataset, broadcast_output.result))
    aggregation_output = aggregation_process.next(
        server_state.delta_aggregate_state, client_outputs.weights_delta,
        client_outputs.weights_delta_weight)
    new_global_model, new_optimizer_state = intrinsics.federated_map(
        server_update, (server_state.model, aggregation_output.result,
                        server_state.optimizer_state))
    new_server_state = intrinsics.federated_zip(
        ServerState(new_global_model, new_optimizer_state,
                    aggregation_output.state, broadcast_output.state))
    aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
        client_outputs.model_output)
    measurements = intrinsics.federated_zip(
        collections.OrderedDict(
            broadcast=broadcast_output.measurements,
            aggregation=aggregation_output.measurements,
            train=aggregated_outputs))
    return new_server_state, measurements

  return one_round_computation


def _is_valid_stateful_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates whether a `MeasuredProcess` is valid for model delta processes.

  Valid processes must have `state` and `measurements` placed on the server.
  This method is intended to be used with additional validation on the non-state
  parameters, inputs and result.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff `process` is a valid stateful process, `False` otherwise.
  """
  init_type = process.initialize.type_signature
  next_type = process.next.type_signature
  return (init_type.result.placement is placements.SERVER and
          next_type.parameter[0].placement is placements.SERVER and
          next_type.result.state.placement is placements.SERVER and
          next_type.result.measurements.placement is placements.SERVER)


def _is_valid_broadcast_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates a `MeasuredProcess` adheres to the broadcast signature.

  A valid broadcast process is one whose argument is placed at `SERVER` and
  whose output is placed at `CLIENTS`.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff the process is a validate broadcast process, otherwise `False`.
  """
  next_type = process.next.type_signature
  return (isinstance(process, measured_process.MeasuredProcess) and
          _is_valid_stateful_process(process) and
          next_type.parameter[1].placement is placements.SERVER and
          next_type.result.result.placement is placements.CLIENTS)


def _is_valid_aggregation_process(
    process: measured_process.MeasuredProcess) -> bool:
  """Validates a `MeasuredProcess` adheres to the aggregation signature.

  A valid aggregation process is one whose argument is placed at `SERVER` and
  whose output is placed at `CLIENTS`.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff the process is a validate aggregation process, otherwise `False`.
  """
  next_type = process.next.type_signature
  return (isinstance(process, measured_process.MeasuredProcess) and
          _is_valid_stateful_process(process) and
          next_type.parameter[1].placement is placements.CLIENTS and
          next_type.result.result.placement is placements.SERVER)


# ============================================================================

NONE_SERVER_TYPE = computation_types.FederatedType((), placements.SERVER)


def _wrap_in_measured_process(
    stateful_fn: Union[computation_utils.StatefulBroadcastFn,
                       computation_utils.StatefulAggregateFn],
    input_type: computation_types.Type) -> measured_process.MeasuredProcess:
  """Converts a `computation_utils.StatefulFn` to a `tff.templates.MeasuredProcess`."""
  py_typecheck.check_type(stateful_fn, (computation_utils.StatefulBroadcastFn,
                                        computation_utils.StatefulAggregateFn))

  @computations.federated_computation()
  def initialize_comp():
    if not isinstance(stateful_fn.initialize, computation_base.Computation):
      initialize = computations.tf_computation(stateful_fn.initialize)
    else:
      initialize = stateful_fn.initialize
    return intrinsics.federated_eval(initialize, placements.SERVER)

  state_type = initialize_comp.type_signature.result

  if isinstance(stateful_fn, computation_utils.StatefulBroadcastFn):

    @computations.federated_computation(
        state_type,
        computation_types.FederatedType(input_type, placements.SERVER),
    )
    def next_comp(state, value):
      empty_metrics = intrinsics.federated_value((), placements.SERVER)
      state, result = stateful_fn(state, value)
      return collections.OrderedDict(
          state=state, result=result, measurements=empty_metrics)

  elif isinstance(stateful_fn, computation_utils.StatefulAggregateFn):

    @computations.federated_computation(
        state_type,
        computation_types.FederatedType(input_type, placements.CLIENTS),
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def next_comp(state, value, weight):
      empty_metrics = intrinsics.federated_value((), placements.SERVER)
      state, result = stateful_fn(state, value, weight)
      return collections.OrderedDict(
          state=state, result=result, measurements=empty_metrics)

  else:
    raise TypeError(
        'Received a {t}, expected either a computation_utils.StatefulAggregateFn or a '
        'computation_utils.StatefulBroadcastFn.'.format(t=type(stateful_fn)))

  return measured_process.MeasuredProcess(
      initialize_fn=initialize_comp, next_fn=next_comp)


@computations.federated_computation()
def _empty_server_initialization():
  return intrinsics.federated_value((), placements.SERVER)


def build_stateless_mean(
    *, model_delta_type: Union[computation_types.StructType,
                               computation_types.TensorType]
) -> measured_process.MeasuredProcess:
  """Builds a `MeasuredProcess` that wraps` tff.federated_mean`."""

  @computations.federated_computation(
      NONE_SERVER_TYPE,
      computation_types.FederatedType(model_delta_type, placements.CLIENTS),
      computation_types.FederatedType(tf.float32, placements.CLIENTS))
  def stateless_mean(state, value, weight):
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    return collections.OrderedDict(
        state=state,
        result=intrinsics.federated_mean(value, weight=weight),
        measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=_empty_server_initialization, next_fn=stateless_mean)


def build_stateless_broadcaster(
    *, model_weights_type: Union[computation_types.StructType,
                                 computation_types.TensorType]
) -> measured_process.MeasuredProcess:
  """Builds a `MeasuredProcess` that wraps `tff.federated_broadcast`."""

  @computations.federated_computation(
      NONE_SERVER_TYPE,
      computation_types.FederatedType(model_weights_type, placements.SERVER),
  )
  def stateless_broadcast(state, value):
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    return collections.OrderedDict(
        state=state,
        result=intrinsics.federated_broadcast(value),
        measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=_empty_server_initialization, next_fn=stateless_broadcast)


def build_model_delta_optimizer_process(
    model_fn: _ModelConstructor,
    model_to_client_delta_fn: Callable[[model_lib.Model], ClientDeltaFn],
    server_optimizer_fn: _OptimizerConstructor,
    stateful_delta_aggregate_fn: Optional[
        computation_utils.StatefulAggregateFn] = None,
    stateful_model_broadcast_fn: Optional[
        computation_utils.StatefulBroadcastFn] = None,
    *,
    broadcast_process: Optional[measured_process.MeasuredProcess] = None,
    aggregation_process: Optional[measured_process.MeasuredProcess] = None,
) -> iterative_process.IterativeProcess:
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
      `next_fn` performs a federated aggregation and updates state. That is, it
      has TFF type `(state@SERVER, value@CLIENTS, weights@CLIENTS) ->
      (state@SERVER, aggregate@SERVER)`, where the `value` type is
      `tff.learning.framework.ModelWeights.trainable` corresponding to the
      object returned by `model_fn`.
    stateful_model_broadcast_fn: a `tff.utils.StatefulBroadcastFn` where the
      `next_fn` performs a federated broadcast and updates state. That is, it
      has TFF type `(state@SERVER, value@SERVER) -> (state@SERVER,
      value@CLIENTS)`, where the `value` type is
      `tff.learning.framework.ModelWeights` corresponding to the object returned
      by `model_fn`.
    broadcast_process: a `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`.
    aggregation_process: a `tff.templates.MeasuredProcess` that aggregates the
      model updates on the clients back to the server. It must support the
      signature `({input_values}@CLIENTS-> output_values@SERVER)`.

  Returns:
    A `tff.templates.IterativeProcess`.

  Raises:
    ProcessTypeError: if `broadcast_process` or `aggregation_process` do not
    conform to the signature of broadcast (SERVER->CLIENTS) or aggregation
    (CLIENTS->SERVER).
  """
  py_typecheck.check_callable(model_fn)
  py_typecheck.check_callable(model_to_client_delta_fn)
  py_typecheck.check_callable(server_optimizer_fn)

  model_weights_type = model_utils.weights_type_from_model(model_fn)

  # TODO(b/159138779): remove the StatefulFn arguments and these validation
  # functions once all callers are migrated.
  def validate_disjoint_optional_arguments(
      stateful_fn: Optional[Union[computation_utils.StatefulBroadcastFn,
                                  computation_utils.StatefulAggregateFn]],
      process: Optional[measured_process.MeasuredProcess],
      process_input_type: Union[computation_types.StructType,
                                computation_types.TensorType],
  ) -> Optional[measured_process.MeasuredProcess]:
    """Validate that only one of two arguments is specified.

    This validates that only the `tff.templates.MeasuredProcess` or the
    `tff.utils.StatefulFn` is specified, and converts the `tff.utils.StatefulFn`
    to a `tff.templates.MeasuredProcess` if possible.  This a bridge while
    transition to `tff.templates.MeasuredProcess`.

    Args:
      stateful_fn: an optional `tff.utils.StatefulFn` that will be wrapped if
        specified.
      process: an optional `tff.templates.MeasuredProcess` that will be returned
        as-is.
      process_input_type: the input type used when wrapping `stateful_fn`.

    Returns:
      `None` if neither argument is specified, otherwise a
      `tff.templates.MeasuredProcess`.

    Raises:
      DisjointArgumentError: if both `stateful_fn` and `process` are not `None`.
    """
    if stateful_fn is not None:
      py_typecheck.check_type(stateful_fn,
                              (computation_utils.StatefulBroadcastFn,
                               computation_utils.StatefulAggregateFn))
      if process is not None:
        raise DisjointArgumentError(
            'Specifying both arguments is an error. Only one may be used')
      return _wrap_in_measured_process(
          stateful_fn, input_type=process_input_type)
    return process

  try:
    aggregation_process = validate_disjoint_optional_arguments(
        stateful_delta_aggregate_fn, aggregation_process,
        model_weights_type.trainable)
  except DisjointArgumentError as e:
    raise DisjointArgumentError(
        'Specifying both `stateful_delta_aggregate_fn` and '
        '`aggregation_process` is an error. Only one may be used') from e

  try:
    broadcast_process = validate_disjoint_optional_arguments(
        stateful_model_broadcast_fn, broadcast_process, model_weights_type)
  except DisjointArgumentError as e:
    raise DisjointArgumentError(
        'Specifying both `stateful_model_broadcast_fn` and '
        '`broadcast_process` is an error. Only one may be used') from e

  if broadcast_process is None:
    broadcast_process = build_stateless_broadcaster(
        model_weights_type=model_weights_type)
  if not _is_valid_broadcast_process(broadcast_process):
    raise ProcessTypeError(
        'broadcast_process type signature does not conform to expected '
        'signature (<state@S, input@S> -> <state@S, result@C, measurements@S>).'
        ' Got: {t}'.format(t=broadcast_process.next.type_signature))

  if aggregation_process is None:
    aggregation_process = build_stateless_mean(
        model_delta_type=model_weights_type.trainable)
  if not _is_valid_aggregation_process(aggregation_process):
    raise ProcessTypeError(
        'aggregation_process type signature does not conform to expected '
        'signature (<state@S, input@C> -> <state@S, result@S, measurements@S>).'
        ' Got: {t}'.format(t=aggregation_process.next.type_signature))

  initialize_computation = _build_initialize_computation(
      model_fn=model_fn,
      server_optimizer_fn=server_optimizer_fn,
      broadcast_process=broadcast_process,
      aggregation_process=aggregation_process)

  run_one_round_computation = _build_one_round_computation(
      model_fn=model_fn,
      server_optimizer_fn=server_optimizer_fn,
      model_to_client_delta_fn=model_to_client_delta_fn,
      broadcast_process=broadcast_process,
      aggregation_process=aggregation_process)

  return iterative_process.IterativeProcess(
      initialize_fn=initialize_computation, next_fn=run_one_round_computation)
