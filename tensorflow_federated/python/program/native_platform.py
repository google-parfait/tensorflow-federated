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
"""A federated platform implemented using native components."""

import asyncio
import collections
import random
import typing
from typing import Any, Awaitable, Coroutine, List, Optional, Sequence

import tensorflow as tf

from tensorflow_federated.python.common_libs import async_utils
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import federated_context
from tensorflow_federated.python.program import value_reference


class CoroValueReference(value_reference.MaterializableValueReference):
  """A `tff.program.MaterializableValueReference` backed by a coroutine."""

  def __init__(self, coro: Coroutine[Any, Any,
                                     value_reference.MaterializablePythonType],
               type_signature: value_reference.MaterializableTffType):
    """Returns an initialized `tff.program.CoroValueReference`.

    Args:
      coro: An `asyncio.Coroutine` that returns the referenced value.
      type_signature: The `tff.Type` of this object.
    """
    if not asyncio.iscoroutine(coro):
      raise TypeError(f'Expected a `Coroutine`, found {type(coro)}')
    py_typecheck.check_type(
        type_signature, typing.get_args(value_reference.MaterializableTffType))

    self._coro = coro
    self._type_signature = type_signature
    self._value = None

  @property
  def type_signature(self) -> value_reference.MaterializableTffType:
    """The `tff.TensorType` of this object."""
    return self._type_signature

  async def get_value(self) -> value_reference.MaterializablePythonType:
    """Returns the referenced value as a numpy scalar or array."""
    if self._value is None:
      self._value = await self._coro
    return self._value

  def __eq__(self, other: Any) -> bool:
    if self is other:
      return True
    elif not isinstance(other, CoroValueReference):
      return NotImplemented
    return (self._type_signature == other._type_signature and
            self._coro == other._coro)


def _create_structure_of_coro_references(
    coro: Coroutine[Any, Any,
                    Any], type_signature: computation_types.Type) -> Any:
  """Returns a structure of `tff.program.CoroValueReference`s."""
  if not asyncio.iscoroutine(coro):
    raise TypeError(f'Expected a `Coroutine`, found {type(coro)}')
  py_typecheck.check_type(type_signature, computation_types.Type)

  if type_signature.is_struct():

    async def _to_structure(coro: Coroutine[Any, Any, Any]) -> structure.Struct:
      return structure.from_container(await coro)

    coro = _to_structure(coro)
    shared_awaitable = async_utils.SharedAwaitable(coro)

    async def _get_item(awaitable: Awaitable[structure.Struct],
                        index: int) -> Any:
      value = await awaitable
      return value[index]

    elements = []
    element_types = structure.iter_elements(type_signature)
    for index, (name, element_type) in enumerate(element_types):
      element_coro = _get_item(shared_awaitable, index)
      element = _create_structure_of_coro_references(element_coro, element_type)
      elements.append((name, element))
    return structure.Struct(elements)
  elif (type_signature.is_federated() and
        type_signature.placement == placements.SERVER):
    return _create_structure_of_coro_references(coro, type_signature.member)
  elif type_signature.is_sequence():
    return CoroValueReference(coro, type_signature)
  elif type_signature.is_tensor():
    return CoroValueReference(coro, type_signature)
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
    element_coros = [
        _materialize_structure_of_value_references(v, t)
        for v, (_, t) in zip(value, element_types)
    ]
    elements = await asyncio.gather(*element_coros)
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
    result = _create_structure_of_coro_references(result_coro, result_type)
    result = type_conversions.type_to_py_container(result, result_type)
    return result


class DatasetDataSourceIterator(data_source.FederatedDataSourceIterator):
  """A `tff.program.FederatedDataSourceIterator` backed by `tf.data.Dataset`s.

  A `tff.program.FederatedDataSourceIterator` backed by a sequence of
  `tf.data.Dataset's, one `tf.data.Dataset' per client, and selects data
  uniformly random with replacement.
  """

  def __init__(self, datasets: Sequence[tf.data.Dataset],
               federated_type: computation_types.FederatedType):
    """Returns an initialized `tff.program.DatasetDataSourceIterator`.

    Args:
      datasets: A sequence of `tf.data.Dataset's to use to yield the data from
        this data source.
      federated_type: The type of the data returned by calling `select` on an
        iterator.

    Raises:
      ValueError: If `datasets` is empty or if each `tf.data.Dataset` in
        `datasets` does not have the same type specification.
    """
    py_typecheck.check_type(datasets, collections.abc.Sequence)
    if not datasets:
      raise ValueError('Expected `datasets` to not be empty.')
    for dataset in datasets:
      py_typecheck.check_type(dataset, tf.data.Dataset)
      element_spec = datasets[0].element_spec
      if dataset.element_spec != element_spec:
        raise ValueError('Expected each `tf.data.Dataset` in `datasets` to '
                         'have the same type specification, found '
                         f'\'{element_spec}\' and \'{dataset.element_spec}\'.')
    py_typecheck.check_type(federated_type, computation_types.FederatedType)

    self._datasets = datasets
    self._federated_type = federated_type

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select`."""
    return self._federated_type

  def select(self, number_of_clients: Optional[int] = None) -> Any:
    """Returns a new selection of data from this iterator.

    Args:
      number_of_clients: A number of clients to use when selecting data, must be
        a positive integer and less than the number of `datasets`.

    Raises:
      ValueError: If `number_of_clients` is not a positive integer or if
        `number_of_clients` is not less than the number of `datasets`.
    """
    if number_of_clients is not None:
      py_typecheck.check_type(number_of_clients, int)
    if (number_of_clients is None or number_of_clients < 0 or
        number_of_clients > len(self._datasets)):
      raise ValueError('Expected `number_of_clients` to be a positive integer '
                       'and less than the number of `datasets`.')
    return random.choices(population=self._datasets, k=number_of_clients)


class DatasetDataSource(data_source.FederatedDataSource):
  """A `tff.program.FederatedDataSource` backed by `tf.data.Dataset`s.

  A `tff.program.FederatedDataSource` backed by a sequence of
  `tf.data.Dataset's, one `tf.data.Dataset' per client, and selects data
  uniformly random with replacement.
  """

  def __init__(self, datasets: Sequence[tf.data.Dataset]):
    """Returns an initialized `tff.program.DatasetDataSource`.

    Args:
      datasets: A sequence of `tf.data.Dataset's to use to yield the data from
        this data source.

    Raises:
      ValueError: If `datasets` is empty or if each `tf.data.Dataset` in
        `datasets` does not have the same type specification.
    """
    py_typecheck.check_type(datasets, collections.abc.Sequence)
    if not datasets:
      raise ValueError('Expected `datasets` to not be empty.')
    for dataset in datasets:
      py_typecheck.check_type(dataset, tf.data.Dataset)
      element_spec = datasets[0].element_spec
      if dataset.element_spec != element_spec:
        raise ValueError('Expected each `tf.data.Dataset` in `datasets` to '
                         'have the same type specification, found '
                         f'\'{element_spec}\' and \'{dataset.element_spec}\'.')

    self._datasets = datasets
    self._federated_type = computation_types.FederatedType(
        computation_types.SequenceType(element_spec), placements.CLIENTS)

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._federated_type

  @property
  def capabilities(self) -> List[data_source.Capability]:
    """The list of capabilities supported by this data source."""
    return [data_source.Capability.RANDOM_UNIFORM]

  def iterator(self) -> DatasetDataSourceIterator:
    """Returns a new iterator for retrieving data from this data source."""
    return DatasetDataSourceIterator(self._datasets, self._federated_type)
