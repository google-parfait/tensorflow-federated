# Copyright 2022, The TensorFlow Federated Authors.
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
"""A federated platform implemented using native TFF components."""

import asyncio
from typing import Optional, TypeVar, Union

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


_T = TypeVar('_T')
# This type defines values of type `_T` nested in a structure of
# `tff.structure.Struct`'s.
# TODO: b/232433269 - Update `tff.structure.Struct` to be able to define nested
# homogeneous structures of `tff.structure.Struct`s.
_StructStructure = Union[
    _T,
    structure.Struct,
]


class NativeValueReference(value_reference.MaterializableValueReference):
  """A `tff.program.MaterializableValueReference` backed by a task."""

  def __init__(
      self,
      task: asyncio.Task,
      type_signature: value_reference.MaterializableTypeSignature,
  ):
    """Returns an initialized `tff.program.NativeValueReference`.

    Args:
      task: An `asyncio.Task` to run.
      type_signature: The `tff.Type` of this object.
    """
    self._task = task
    self._type_signature = type_signature

  @property
  def type_signature(self) -> value_reference.MaterializableTypeSignature:
    """The `tff.TensorType` of this object."""
    return self._type_signature

  async def get_value(self) -> value_reference.MaterializedValue:
    """Returns the referenced value as a numpy scalar or array."""
    return await self._task

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, NativeValueReference):
      return NotImplemented
    return (
        self._type_signature,
        self._task,
    ) == (
        other._type_signature,
        other._task,
    )


def _create_structure_of_references(
    task: asyncio.Task,
    type_signature: computation_types.Type,
) -> _StructStructure[NativeValueReference]:
  """Returns a structure of `tff.program.NativeValueReference`s.

  Args:
    task: A task used to create the structure of
      `tff.program.NativeValueReference`s.
    type_signature: The `tff.Type` of the value returned by `task`; must contain
      only structures, server-placed values, or tensors.

  Raises:
    NotImplementedError: If `type_signature` contains an unexpected type.
  """

  if isinstance(type_signature, computation_types.StructType):

    async def _to_structure(task: asyncio.Task) -> structure.Struct:
      value = await task
      return structure.from_container(value)

    coro = _to_structure(task)
    task = asyncio.create_task(coro)

    async def _get_item(
        task: asyncio.Task, index: int
    ) -> value_reference.MaterializedValue:
      value = await task
      return value[index]

    elements = []
    element_types = structure.iter_elements(type_signature)
    for index, (name, element_type) in enumerate(element_types):
      element = _get_item(task, index)
      element_task = asyncio.create_task(element)
      element = _create_structure_of_references(element_task, element_type)
      elements.append((name, element))
    return structure.Struct(elements)
  elif (
      isinstance(type_signature, computation_types.FederatedType)
      and type_signature.placement == placements.SERVER
  ):
    return _create_structure_of_references(task, type_signature.member)
  elif isinstance(type_signature, computation_types.SequenceType):
    return NativeValueReference(task, type_signature)
  elif isinstance(type_signature, computation_types.TensorType):
    return NativeValueReference(task, type_signature)
  else:
    raise NotImplementedError(f'Unexpected type found: {type_signature}.')


async def _materialize_structure_of_references(
    value: value_reference.MaterializableStructure,
    type_signature: computation_types.Type,
) -> _StructStructure[value_reference.MaterializedValue]:
  """Returns a structure of materialized values."""

  async def _materialize(
      value: value_reference.MaterializableValue,
  ) -> value_reference.MaterializedValue:
    if isinstance(value, value_reference.MaterializableValueReference):
      return await value.get_value()
    else:
      return value

  if isinstance(type_signature, computation_types.StructType):
    value = structure.from_container(value)
    element_types = list(structure.iter_elements(type_signature))
    element_awaitables = [
        _materialize_structure_of_references(v, t)
        for v, (_, t) in zip(value, element_types)
    ]
    elements = await asyncio.gather(*element_awaitables)
    elements = [(n, v) for v, (n, _) in zip(elements, element_types)]
    return structure.Struct(elements)
  elif isinstance(type_signature, computation_types.FederatedType):
    return await _materialize_structure_of_references(
        value, type_signature.member
    )
  elif isinstance(type_signature, computation_types.SequenceType):
    return await _materialize(value)
  elif isinstance(type_signature, computation_types.TensorType):
    return await _materialize(value)
  else:
    raise NotImplementedError(f'Unexpected type found: {type_signature}.')


class NativeFederatedContext(federated_context.FederatedContext):
  """A `tff.program.FederatedContext` backed by an execution context."""

  def __init__(self, context: async_execution_context.AsyncExecutionContext):
    """Returns an initialized `tff.program.NativeFederatedContext`.

    Args:
      context: An `tff.framework.AsyncExecutionContext`.
    """
    self._context = context

  def invoke(
      self,
      comp: computation_base.Computation,
      arg: Optional[federated_context.ComputationArgValue],
  ) -> structure_utils.Structure[NativeValueReference]:
    """Invokes the `comp` with the argument `arg`.

    Args:
      comp: The `tff.Computation` being invoked.
      arg: The optional argument of `comp`; server-placed values must be
        represented by `tff.program.MaterializableStructure`, and client-placed
        values must be represented by structures of values returned by a
        `tff.program.FederatedDataSourceIterator`.

    Returns:
      The result of invocation; a structure of
      `tff.program.MaterializableValueReference`.

    Raises:
      ValueError: If the result type of the invoked computation does not contain
      only structures, server-placed values, or tensors.
    Raises:
      ValueError: If the result type of `comp` does not contain only structures,
      server-placed values, or tensors.
    """
    result_type = comp.type_signature.result
    if not federated_context.contains_only_server_placed_data(result_type):
      raise ValueError(
          'Expected the result type of `comp` to contain only structures, '
          f'server-placed values, or tensors, found {result_type}.'
      )

    async def _invoke(
        context: async_execution_context.AsyncExecutionContext,
        comp: computation_base.Computation,
        arg: value_reference.MaterializableStructure,
    ) -> value_reference.MaterializedStructure:
      if comp.type_signature.parameter is not None:
        arg = await _materialize_structure_of_references(
            arg, comp.type_signature.parameter
        )
      return await context.invoke(comp, arg)

    coro = _invoke(self._context, comp, arg)
    task = asyncio.create_task(coro)
    result = _create_structure_of_references(task, result_type)
    result = type_conversions.type_to_py_container(result, result_type)
    return result
