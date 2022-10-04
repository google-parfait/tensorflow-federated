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
import inspect
import typing
from typing import Any, Awaitable

from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import value_reference


class AwaitableValueReference(value_reference.MaterializableValueReference):
  """A `tff.program.MaterializableValueReference` backed by an `Awaitable`."""

  def __init__(self,
               awaitable: Awaitable[value_reference.MaterializablePythonType],
               type_signature: value_reference.MaterializableTffType):
    """Returns an initialized `tff.program.AwaitableValueReference`.

    Args:
      awaitable: An `Awaitable` that returns the referenced value.
      type_signature: The `tff.Type` of this object.
    """
    if not inspect.isawaitable(awaitable):
      raise TypeError(f'Expected a `Awaitable`, found {type(awaitable)}')
    py_typecheck.check_type(
        type_signature, typing.get_args(value_reference.MaterializableTffType))

    self._awaitable = awaitable
    self._type_signature = type_signature
    self._value = None

  @property
  def type_signature(self) -> value_reference.MaterializableTffType:
    """The `tff.TensorType` of this object."""
    return self._type_signature

  async def get_value(self) -> value_reference.MaterializablePythonType:
    """Returns the referenced value as a numpy scalar or array."""
    if self._value is None:
      self._value = await self._awaitable
    return self._value

  def __eq__(self, other: Any) -> bool:
    if self is other:
      return True
    elif not isinstance(other, AwaitableValueReference):
      return NotImplemented
    return (self._type_signature == other._type_signature and
            self._awaitable == other._awaitable)


def _create_structure_of_awaitable_references(
    awaitable: Awaitable[Any], type_signature: computation_types.Type) -> Any:
  """Returns a structure of `tff.program.AwaitableValueReference`s."""
  if not inspect.isawaitable(awaitable):
    raise TypeError(f'Expected an `Awaitable`, found {type(awaitable)}')
  py_typecheck.check_type(type_signature, computation_types.Type)

  if type_signature.is_struct():

    async def _to_structure(
        awaitable: Awaitable[structure.Struct]) -> structure.Struct:
      return structure.from_container(await awaitable)

    awaitable = _to_structure(awaitable)
    # A `async_utils.SharedAwaitable` is required to materialize structures of
    # values sequentially or concurrently. This happens when `get_item` is
    # invoked for eaceh element.
    shared_awaitable = async_utils.SharedAwaitable(awaitable)

    async def _get_item(awaitable: Awaitable[structure.Struct],
                        index: int) -> Any:
      value = await awaitable
      return value[index]

    elements = []
    element_types = structure.iter_elements(type_signature)
    for index, (name, element_type) in enumerate(element_types):
      element_awaitable = _get_item(shared_awaitable, index)
      # A `async_utils.SharedAwaitable` is required to materialize structures of
      # values multiple times. This happens when a value is released using
      # multiple `tff.program.ReleaseManager`s.
      element_shared_awaitable = async_utils.SharedAwaitable(element_awaitable)
      element = _create_structure_of_awaitable_references(
          element_shared_awaitable, element_type)
      elements.append((name, element))
    return structure.Struct(elements)
  elif (type_signature.is_federated() and
        type_signature.placement == placements.SERVER):
    return _create_structure_of_awaitable_references(awaitable,
                                                     type_signature.member)
  elif type_signature.is_sequence():
    return AwaitableValueReference(awaitable, type_signature)
  elif type_signature.is_tensor():
    return AwaitableValueReference(awaitable, type_signature)
  else:
    raise NotImplementedError(f'Unexpected type found: {type_signature}.')


async def _materialize_structure_of_value_references(
    value: Any, type_signature: computation_types.Type) -> Any:
  """Returns a structure of materialized values."""
  py_typecheck.check_type(type_signature, computation_types.Type)

  async def _materialize(value: Any) -> Any:
    if isinstance(value, value_reference.MaterializableValueReference):
      return await value.get_value()
    else:
      return value

  if type_signature.is_struct():
    value = structure.from_container(value)
    element_types = list(structure.iter_elements(type_signature))
    element_awaitables = [
        _materialize_structure_of_value_references(v, t)
        for v, (_, t) in zip(value, element_types)
    ]
    elements = await asyncio.gather(*element_awaitables)
    elements = [(n, v) for v, (n, _) in zip(elements, element_types)]
    return structure.Struct(elements)
  elif (type_signature.is_federated() and
        type_signature.placement == placements.SERVER):
    return await _materialize_structure_of_value_references(
        value, type_signature.member)
  elif type_signature.is_sequence():
    return await _materialize(value)
  elif type_signature.is_tensor():
    return await _materialize(value)
  else:
    return value


class NativeFederatedContext(federated_context.FederatedContext):
  """A `tff.program.FederatedContext` backed by a `tff.framework.Context`."""

  def __init__(self, context: context_base.Context):
    """Returns an initialized `tff.program.NativeFederatedContext`.

    Args:
      context: A `tff.framework.Context` with an async `invoke`.

    Raises:
      ValueError: If `context` does not have an async `invoke`.
    """
    py_typecheck.check_type(context, context_base.Context)
    if not asyncio.iscoroutinefunction(context.invoke):
      raise ValueError('Expected a `context` with an async `invoke`, received '
                       f'{context}.')

    self._context = context

  def invoke(self, comp: computation_base.Computation, arg: Any) -> Any:
    """Invokes the `comp` with the argument `arg`.

    Args:
      comp: The `tff.Computation` being invoked.
      arg: The optional argument of `comp`.

    Returns:
      The result of invocation, must contain only structures, server-placed
      values, or tensors.

    Raises:
      ValueError: If the result type of the invoked comptuation does not contain
      only structures, server-placed values, or tensors.
    """
    py_typecheck.check_type(comp, computation_base.Computation)

    result_type = comp.type_signature.result
    if not federated_context.contains_only_server_placed_data(result_type):
      raise ValueError(
          'Expected the result type of the invoked computation to contain only '
          'structures, server-placed values, or tensors, found '
          f'\'{result_type}\'.')

    async def _invoke(context: context_base.Context,
                      comp: computation_base.Computation, arg: Any) -> Any:
      if comp.type_signature.parameter is not None:
        arg = await _materialize_structure_of_value_references(
            arg, comp.type_signature.parameter)
      return await context.invoke(comp, arg)

    result_coro = _invoke(self._context, comp, arg)
    result = _create_structure_of_awaitable_references(result_coro, result_type)
    result = type_conversions.type_to_py_container(result, result_type)
    return result
