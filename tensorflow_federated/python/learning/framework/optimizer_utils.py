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

from collections.abc import Sequence
from typing import Union

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib


@attr.s(eq=False, frozen=True)
class ClientOutput:
  """Structure for outputs returned from clients during federated optimization.

  Attributes:
    weights_delta: A dictionary of updates to the model's trainable variables.
    weights_delta_weight: Weight to use in a weighted mean when aggregating
      `weights_delta`.
    model_output: A structure matching
      `tff.learning.Model.report_local_unfinalized_metrics`, reflecting the
      results of training on the input dataset.
    optimizer_output: Additional metrics or other outputs defined by the
      optimizer.
  """

  weights_delta = attr.ib()
  weights_delta_weight = attr.ib()
  model_output = attr.ib()
  optimizer_output = attr.ib(default=None)


@attr.s(eq=False, frozen=True)
class ServerState:
  """Represents the state of the server carried between rounds.

  Attributes:
    model: A `ModelWeights` structure, containing Tensors or Variables.
    optimizer_state: A list of Tensors or Variables, in the order returned by
      `optimizer.variables()`
    delta_aggregate_state: State (possibly empty) of the delta_aggregate_fn.
    model_broadcast_state: State (possibly empty) of the model_broadcast_fn.
  """

  model = attr.ib()
  optimizer_state = attr.ib()
  delta_aggregate_state = attr.ib()
  model_broadcast_state = attr.ib()


def state_with_new_model_weights(
    server_state: ServerState,
    trainable_weights: list[np.ndarray],
    non_trainable_weights: list[np.ndarray],
) -> ServerState:
  """Returns a `ServerState` with updated model weights.

  This function cannot currently be used for `tff.learning.algorithms`, which
  can have a different `ServerState` structure. For those methods, use the
  `tff.learning.algorithms.templates.LearningProcess.set_model_weights` method
  on the created process.

  Args:
    server_state: A `tff.learning.framework.ServerState`.
    trainable_weights: A list of `numpy` values in the order of the original
      model's `trainable_variables`.
    non_trainable_weights: A list of `numpy` values in the order of the original
      model's `non_trainable_variables`.

  Returns:
    A `tff.learning.framework.ServerState`.
  """
  py_typecheck.check_type(server_state, ServerState)
  tensor_leaf_types = (np.ndarray, tf.Tensor, np.number)
  python_leaf_types = (int, float, bytes)

  def assert_weight_lists_match(old_value, new_value):
    """Assert two flat lists of ndarrays or tensors match."""
    # First try to normalize python scalar leaves to numpy. If one of the
    # values is not a Python value we can try casting the other value to it.
    if isinstance(old_value, python_leaf_types) and isinstance(
        new_value, tensor_leaf_types
    ):
      old_value = np.add(
          old_value, 0, dtype=new_value.dtype  # pytype: disable=attribute-error
      )
    elif isinstance(old_value, tensor_leaf_types) and isinstance(
        new_value, python_leaf_types
    ):
      new_value = np.add(new_value, 0, dtype=old_value.dtype)

    if isinstance(new_value, python_leaf_types) and isinstance(
        old_value, python_leaf_types
    ):
      # If both values are python types, just compare the types.
      if type(new_value) is not type(old_value):
        raise TypeError(
            'Element is not the same type. old '
            f'({type(old_value)} != new ({type(new_value)}).'
        )
    elif isinstance(new_value, tensor_leaf_types) and isinstance(
        old_value, tensor_leaf_types
    ):
      # pytype: disable=attribute-error
      if (
          old_value.dtype != new_value.dtype
          or old_value.shape != new_value.shape
      ):
        raise TypeError(
            'Element is not the same tensor type. old '
            f'({old_value.dtype}, {old_value.shape}) != '
            f'new ({new_value.dtype}, {new_value.shape})'
        )
      # pytype: enable=attribute-error
    elif isinstance(new_value, Sequence) and isinstance(old_value, Sequence):
      if len(old_value) != len(new_value):
        raise TypeError(
            'Model weights have different lengths: '
            f'(old) {len(old_value)} != (new) {len(new_value)})\n'
            f'Old values: {old_value}\nNew values: {new_value}'
        )
      for old, new in zip(old_value, new_value):
        assert_weight_lists_match(old, new)
    else:
      raise TypeError(
          'Model weights structures contains types that cannot be '
          'handled.\nOld weights structure: {old}\n'
          'New weights structure: {new}\n'
          'Must be one of (int, float, np.ndarray, tf.Tensor, '
          'collections.abc.Sequence)'.format(
              old=tf.nest.map_structure(type, old_value),
              new=tf.nest.map_structure(type, new_value),
          )
      )

  assert_weight_lists_match(server_state.model.trainable, trainable_weights)
  assert_weight_lists_match(
      server_state.model.non_trainable, non_trainable_weights
  )
  new_server_state = ServerState(
      model=model_weights_lib.ModelWeights(
          trainable=trainable_weights, non_trainable=non_trainable_weights
      ),
      optimizer_state=server_state.optimizer_state,
      delta_aggregate_state=server_state.delta_aggregate_state,
      model_broadcast_state=server_state.model_broadcast_state,
  )
  return new_server_state


def _is_valid_stateful_process(
    process: measured_process.MeasuredProcess,
) -> bool:
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
  return (
      init_type.result.placement is placements.SERVER
      and next_type.parameter[0].placement is placements.SERVER
      and next_type.result.state.placement is placements.SERVER
      and next_type.result.measurements.placement is placements.SERVER
  )


def is_valid_broadcast_process(
    process: measured_process.MeasuredProcess,
) -> bool:
  """Validates a `MeasuredProcess` adheres to the broadcast signature.

  A valid broadcast process is one whose argument is placed at `SERVER` and
  whose output is placed at `CLIENTS`.

  Args:
    process: A measured process to validate.

  Returns:
    `True` iff the process is a validate broadcast process, otherwise `False`.
  """
  next_type = process.next.type_signature
  return (
      isinstance(process, measured_process.MeasuredProcess)
      and _is_valid_stateful_process(process)
      and next_type.parameter[1].placement is placements.SERVER
      and next_type.result.result.placement is placements.CLIENTS
  )


def build_stateless_broadcaster(
    *,
    model_weights_type: Union[
        computation_types.StructType, computation_types.TensorType
    ],
) -> measured_process.MeasuredProcess:
  """Builds a `MeasuredProcess` that wraps `tff.federated_broadcast`."""

  @federated_computation.federated_computation()
  def _empty_server_initialization():
    return intrinsics.federated_value((), placements.SERVER)

  @federated_computation.federated_computation(
      computation_types.at_server(()),
      computation_types.at_server(model_weights_type),
  )
  def stateless_broadcast(state, value):
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(
        state=state,
        result=intrinsics.federated_broadcast(value),
        measurements=empty_metrics,
    )

  return measured_process.MeasuredProcess(
      initialize_fn=_empty_server_initialization, next_fn=stateless_broadcast
  )
