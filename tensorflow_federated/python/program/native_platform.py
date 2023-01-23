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
from collections.abc import Awaitable, Callable
import functools
import typing
from typing import Any, Optional, TypeVar, Union

from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


_MaterializedValueFn = Callable[
    [], Awaitable[value_reference.MaterializedValue]
]
_T = TypeVar('_T')
# This type defines values of type `_T` nested in a structure of
# `tff.structure.Struct`'s.
# TODO(b/232433269) Update `tff.structure.Struct` to be able to define nested
# homogeneous structures of `tff.structure.Struct`s.
_StructStructure = Union[
    _T,
    structure.Struct,
]


class AwaitableValueReference(value_reference.MaterializableValueReference):
  """A `tff.program.MaterializableValueReference` backed by a coroutine function."""

  def __init__(
      self,
      fn: _MaterializedValueFn,
      type_signature: value_reference.MaterializableTypeSignature,
  ):
    """Returns an initialized `tff.program.AwaitableValueReference`.

    Args:
      fn: A function that returns an `Awaitable` representing the referenced
        value.
      type_signature: The `tff.Type` of this object.
    """
    if not callable(fn):
      raise TypeError(
          f'Expected a function that returns an `Awaitable`, found {type(fn)}'
      )
    py_typecheck.check_type(
        type_signature,
        typing.get_args(value_reference.MaterializableTypeSignature),
    )

    self._fn = fn
    self._type_signature = type_signature
    self._value = None

  @property
  def type_signature(self) -> value_reference.MaterializableTypeSignature:
    """The `tff.TensorType` of this object."""
    return self._type_signature

  async def get_value(self) -> value_reference.MaterializedValue:
    """Returns the referenced value as a numpy scalar or array."""
    if self._value is None:
      self._value = await self._fn()
    return self._value

  def __eq__(self, other: Any) -> bool:
    if self is other:
      return True
    elif not isinstance(other, AwaitableValueReference):
      return NotImplemented
    return (
        self._type_signature == other._type_signature and self._fn == other._fn
    )


def _wrap_in_shared_awaitable(
    fn: Callable[..., Awaitable[Any]]
) -> Callable[..., async_utils.SharedAwaitable]:
  """Wraps the returned awaitable in a `tff.async_utils.SharedAwaitable`.

  Args:
    fn: A function that returns an `Awaitable`.

  Returns:
    A function that returns a `tff.async_utils.SharedAwaitable`
  """
  if not callable(fn):
    raise TypeError(
        f'Expected a function that returns an `Awaitable`, found {type(fn)}'
    )

  @functools.cache
  def wrapper(*args: Any, **kwargs: Any) -> async_utils.SharedAwaitable:
    awaitable = fn(*args, **kwargs)
    return async_utils.SharedAwaitable(awaitable)

  return wrapper


def _create_structure_of_awaitable_references(
    fn: _MaterializedValueFn, type_signature: computation_types.Type
) -> _StructStructure[AwaitableValueReference]:
  """Returns a structure of `tff.program.AwaitableValueReference`s.

  Args:
    fn: A function that returns an `Awaitable` used to create the structure of
      `tff.program.AwaitableValueReference`s.
    type_signature: The `tff.Type` of the value returned by `coro_fn`; must
      contain only structures, server-placed values, or tensors.

  Raises:
    NotImplementedError: If `type_signature` contains an unexpected type.
  """
  if not callable(fn):
    raise TypeError(
        f'Expected a function that returns an `Awaitable`, found {type(fn)}'
    )
  py_typecheck.check_type(type_signature, computation_types.Type)

  # A `async_utils.SharedAwaitable` is required to materialize structures of
  # values multiple times. This happens when a value is released using multiple
  # `tff.program.ReleaseManager`s.
  fn = _wrap_in_shared_awaitable(fn)

  if type_signature.is_struct():

    async def _to_structure(fn: _MaterializedValueFn) -> structure.Struct:
      value = await fn()
      return structure.from_container(value)

    fn = functools.partial(_to_structure, fn)

    # A `tff.async_utils.SharedAwaitable` is required to materialize structures
    # of values concurrently. This happens when the structure is flattened and
    # the `tff.program.AwaitableValueReference`s are materialized concurrently,
    # see `tff.program.materialize_value` for an example.
    fn = _wrap_in_shared_awaitable(fn)

    async def _get_item(
        fn: _MaterializedValueFn, index: int
    ) -> value_reference.MaterializedValue:
      value = await fn()
      return value[index]

    elements = []
    element_types = structure.iter_elements(type_signature)
    for index, (name, element_type) in enumerate(element_types):
      element_fn = functools.partial(_get_item, fn, index)
      element = _create_structure_of_awaitable_references(
          element_fn, element_type
      )
      elements.append((name, element))
    return structure.Struct(elements)
  elif (
      type_signature.is_federated()
      and type_signature.placement == placements.SERVER
  ):
    return _create_structure_of_awaitable_references(fn, type_signature.member)
  elif type_signature.is_sequence():
    return AwaitableValueReference(fn, type_signature)
  elif type_signature.is_tensor():
    return AwaitableValueReference(fn, type_signature)
  else:
    raise NotImplementedError(f'Unexpected type found: {type_signature}.')


async def _materialize_structure_of_value_references(
    value: value_reference.MaterializableStructure,
    type_signature: computation_types.Type,
) -> _StructStructure[value_reference.MaterializedValue]:
  """Returns a structure of materialized values."""
  py_typecheck.check_type(type_signature, computation_types.Type)

  async def _materialize(
      value: value_reference.MaterializableValue,
  ) -> value_reference.MaterializedValue:
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
  elif (
      type_signature.is_federated()
      and type_signature.placement == placements.SERVER
  ):
    return await _materialize_structure_of_value_references(
        value, type_signature.member
    )
  elif type_signature.is_sequence():
    return await _materialize(value)
  elif type_signature.is_tensor():
    return await _materialize(value)
  else:
    return value


class NativeFederatedContext(federated_context.FederatedContext):
  """A `tff.program.FederatedContext` backed by an execution context."""

  def __init__(self, context: async_execution_context.AsyncExecutionContext):
    """Returns an initialized `tff.program.NativeFederatedContext`.

    Args:
      context: An `tff.framework.AsyncExecutionContext`.
    """
    py_typecheck.check_type(
        context, async_execution_context.AsyncExecutionContext
    )

    self._context = context

  def invoke(
      self,
      comp: computation_base.Computation,
      arg: Optional[
          Union[
              value_reference.MaterializableStructure,
              Any,
              computation_base.Computation,
          ]
      ],
  ) -> structure_utils.Structure[AwaitableValueReference]:
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
    py_typecheck.check_type(comp, computation_base.Computation)
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
        arg = await _materialize_structure_of_value_references(
            arg, comp.type_signature.parameter
        )
      return await context.invoke(comp, arg)

    coro_fn = functools.partial(_invoke, self._context, comp, arg)
    result = _create_structure_of_awaitable_references(coro_fn, result_type)
    result = type_conversions.type_to_py_container(result, result_type)
    return result
