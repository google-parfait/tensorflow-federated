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
"""Utilities for prefetching federated data."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import struct
import threading
from typing import Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.executors import cardinality_carrying_base
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import ingestable_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source as data_source_lib
from tensorflow_federated.python.program import serialization_utils


class FetchedValue(
    ingestable_base.Ingestable, cardinality_carrying_base.CardinalityCarrying
):
  """Represents a value constructed by the prefetching data source."""

  def __init__(
      self,
      executor: executor_base.Executor,
      executor_value: executor_value_base.ExecutorValue,
      cardinality: Mapping[placements.PlacementLiteral, int],
      defining_coro_fn: Callable[
          [executor_base.Executor], Awaitable[executor_value_base.ExecutorValue]
      ],
  ):
    """Initializes a FetchedValue, intended to amortize cost across rounds.

    Args:
      executor: Instance of `tff.framework.Executor` in which `executor_value`
        is embedded.
      executor_value: A representation of a value embedded in a TFF executor.
        Must be embedded in the `executor` argument of the initializer.
      cardinality: The cardinality of the executor value. A mapping from TFF
        placement literals (e.g. tff.CLIENTS, tff.SERVER) to ints.
      defining_coro_fn: A coroutine function accepting an executor argument, and
        returning the result of embedding the concrete value which backs this
        `FetchedValue` in that executor.
    """
    self._executor = executor
    self._executor_value = executor_value
    self._cardinality = cardinality
    self._defining_coro_fn = defining_coro_fn

  @property
  def type_signature(self) -> computation_types.Type:
    return self._executor_value.type_signature

  @property
  def cardinality(self) -> Mapping[placements.PlacementLiteral, int]:
    return self._cardinality

  async def ingest(
      self, executor: executor_base.Executor
  ) -> executor_value_base.ExecutorValue:
    if self._executor is executor:
      # We are addressing the same executor we already embedded this value in;
      # we can shortcut, since this executor has already ingested this value.
      return self._executor_value
    else:
      # Executor has been swapped underneath us, perhaps e.g. due to worker
      # failure--we need to re-embed.
      self._executor_value = await self._defining_coro_fn(executor)
      self._executor = executor
      return self._executor_value


class PrefetchingDataSourceIterator(
    data_source_lib.FederatedDataSourceIterator
):
  """A `tff.program.FederatedDataSourceIterator` that prefetches data.

  Note: Instances of `tff.program.FederatedDataSourceIterator` constructed in
  different `tff.framework.AsyncExecutionContext` will behave differently.
  """

  def __init__(
      self,
      iterator: data_source_lib.FederatedDataSourceIterator,
      total_rounds: int,
      num_rounds_to_prefetch: int,
      num_clients_to_prefetch: int,
      prefetch_threshold: int = 0,
  ):
    """Returns an initialized `tff.program.FederatedDataSourceIterator`.

    Args:
      iterator: A `tff.program.FederatedDataSourceIterator` used to prefetch
        data from.
      total_rounds: The total number of rounds.
      num_rounds_to_prefetch: The number of rounds to prefetch.
      num_clients_to_prefetch: The number of clients to prefetch per round. Must
        be greater than 1 and must be identical across all rounds; attempts to
        select any other number of clients will fail.
      prefetch_threshold: The threshold below which the data source starts
        prefetching.

    Raises:
      ValueError: If `num_clients_to_prefetch` is not greater than 1.
      RuntimeError: If the iterator is not constructed in an
        `tff.framework.AsyncExecutionContext`.
    """
    py_typecheck.check_type(
        iterator, data_source_lib.FederatedDataSourceIterator
    )
    py_typecheck.check_type(total_rounds, int)
    py_typecheck.check_type(num_rounds_to_prefetch, int)
    py_typecheck.check_type(num_clients_to_prefetch, int)
    py_typecheck.check_type(prefetch_threshold, int)
    if num_clients_to_prefetch < 1:
      raise ValueError(
          'Expected `num_clients_to_prefetch` to be greater than 1, found '
          f'{num_clients_to_prefetch}.'
      )
    context = get_context_stack.get_context_stack().current
    if not isinstance(context, async_execution_context.AsyncExecutionContext):
      raise RuntimeError(
          'Expected the `tff.program.PrefetchingDataSourceIterator` to be '
          'constructed in a `tff.framework.AsyncExecutionContext`, found '
          f'{context}.'
      )

    self._iterator = iterator
    self._total_rounds = total_rounds
    self._num_rounds_to_prefetch = num_rounds_to_prefetch
    self._num_clients_to_prefetch = num_clients_to_prefetch
    self._prefetch_threshold = prefetch_threshold
    self._executor_factory = context.executor_factory
    self._cardinality = {placements.CLIENTS: num_clients_to_prefetch}
    self._num_rounds_prefetched = 0
    self._prefetched_data = []
    self._lock = threading.Lock()
    self._active_threads = []

    self._start_prefetching()

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'PrefetchingDataSourceIterator':
    """Deserializes the object from bytes."""
    offset = 0
    iterator, iterator_size = serialization_utils.unpack_serializable_from(
        buffer, offset=offset
    )
    if not isinstance(iterator, data_source_lib.FederatedDataSourceIterator):
      raise TypeError(
          'Expected `iterator` to be a '
          '`tff.program.FederatedDataSourceIterator`, found '
          f'`{type(iterator)}`.'
      )
    offset += iterator_size
    (
        total_rounds,
        num_rounds_to_prefetch,
        num_clients_to_prefetch,
        prefetch_threshold,
    ) = struct.unpack_from('!QQQQ', buffer, offset=offset)
    return PrefetchingDataSourceIterator(
        iterator=iterator,
        total_rounds=total_rounds,
        num_rounds_to_prefetch=num_rounds_to_prefetch,
        num_clients_to_prefetch=num_clients_to_prefetch,
        prefetch_threshold=prefetch_threshold,
    )

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    iterator_bytes = serialization_utils.pack_serializable(self._iterator)
    data_bytes = struct.pack(
        '!QQQQ',
        self._total_rounds,
        self._num_rounds_to_prefetch,
        self._num_clients_to_prefetch,
        self._prefetch_threshold,
    )
    return iterator_bytes + data_bytes

  def _single_round_fn(self) -> None:
    data = self._iterator.select(
        self._num_clients_to_prefetch
    )  # gen-stub-imports

    # We assume the executor factory uses a cache, so most calls to this
    # function should result in a hit.
    executor_at_invocation = self._executor_factory.create_executor(
        self._cardinality
    )

    # Force ingestion of the round data in the configured executor. If a worker
    # goes down, the stack will be rebuilt and this state lost--but this is a
    # quick-and-dirty version of 'persistent values' that should help amortize
    # the cost of this work.
    if isinstance(data, ingestable_base.Ingestable):
      executor_value_coro = data.ingest(executor_at_invocation)

      async def defining_coro_fn(
          executor: executor_base.Executor,
      ) -> executor_value_base.ExecutorValue:
        return await data.ingest(executor)

    else:
      executor_value_coro = executor_at_invocation.create_value(
          data, self._iterator.federated_type
      )

      async def defining_coro_fn(
          executor: executor_base.Executor,
      ) -> executor_value_base.ExecutorValue:
        return await executor.create_value(data, self._iterator.federated_type)

    event_loop = asyncio.new_event_loop()
    executor_value = event_loop.run_until_complete(executor_value_coro)
    fetched_data = FetchedValue(
        executor_at_invocation,
        executor_value,
        self._cardinality,
        defining_coro_fn,  # pytype: disable=wrong-arg-types  # b/150782658
    )

    with self._lock:
      self._prefetched_data.append(fetched_data)

  def _start_prefetching(self) -> None:
    with self._lock:
      if self._active_threads:
        # Already prefetching
        return
      if len(self._prefetched_data) > self._prefetch_threshold:
        # Only fetch data when _prefetched_data has a low volume of data
        # to avoid new threads created in each round.
        return
      num_to_prefetch = min(
          self._num_rounds_to_prefetch - len(self._prefetched_data),
          self._total_rounds - self._num_rounds_prefetched,
      )
      if num_to_prefetch < 1:
        # Already have enough
        return
      self._num_rounds_prefetched = (
          self._num_rounds_prefetched + num_to_prefetch
      )
      for _ in range(num_to_prefetch):
        thread = threading.Thread(target=self._single_round_fn)
        thread.start()
        self._active_threads.append(thread)

  def _finish_prefetching(self) -> None:
    threads = []
    with self._lock:
      threads = self._active_threads
      self._active_threads = []
    for thread in threads:
      thread.join()
    with self._lock:
      if len(self._prefetched_data) < 1:
        raise RuntimeError('Failed to prefetch at least one item.')

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select`."""
    return self._iterator.federated_type

  def select(self, num_clients: Optional[int] = None) -> object:
    """Returns a new selection of data from this iterator.

    Args:
      num_clients: A number of clients to use when selecting data. Must be a
        positive integer and equal to `num_clients_to_prefetch`.

    Raises:
      ValueError: If `num_clients` is not a positive integer or if `num_clients`
        is not equal to `num_clients_to_prefetch`.
    """
    if num_clients is not None:
      py_typecheck.check_type(num_clients, int)
    if (
        num_clients is None
        or num_clients < 0
        or num_clients != self._num_clients_to_prefetch
    ):
      raise ValueError(
          'Expected `num_clients` to be a positive integer and equal to '
          f'`num_clients_to_prefetch`, found `num_clients`: {num_clients}, '
          f'`num_clients_to_prefetch`: {self._num_clients_to_prefetch}'
      )

    self._finish_prefetching()
    with self._lock:
      data = self._prefetched_data[0]
      self._prefetched_data = self._prefetched_data[1:]
    self._start_prefetching()
    return data

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, PrefetchingDataSourceIterator):
      return NotImplemented
    # The prefetched data should not be considered to determine equality.
    return (
        self._iterator,
        self._total_rounds,
        self._num_rounds_to_prefetch,
        self._num_clients_to_prefetch,
        self._prefetch_threshold,
        self._executor_factory,
    ) == (
        other._iterator,
        other._total_rounds,
        other._num_rounds_to_prefetch,
        other._num_clients_to_prefetch,
        other._prefetch_threshold,
        other._executor_factory,
    )


class PrefetchingDataSource(data_source_lib.FederatedDataSource):
  """A `tff.program.FederatedDataSource` that prefetches data."""

  def __init__(
      self,
      data_source: data_source_lib.FederatedDataSource,
      total_rounds: int,
      num_rounds_to_prefetch: int,
      num_clients_to_prefetch: int,
      prefetch_threshold: int = 0,
  ):
    """Returns an initialized `tff.program.PrefetchingDataSource`.

    Args:
      data_source: A `tff.program.FederatedDataSource` used to prefetch data
        from.
      total_rounds: The total number of rounds.
      num_rounds_to_prefetch: The number of rounds to prefetch.
      num_clients_to_prefetch: The number of clients to prefetch per round. Must
        be greater than 1 and must be the same across all rounds; attempts to
        select any other number of clients will fail.
      prefetch_threshold: The threshold below which the data source starts
        prefetching.

    Raises:
      ValueError: If `num_clients_to_prefetch` is not greater than 1.
    """
    py_typecheck.check_type(data_source, data_source_lib.FederatedDataSource)
    py_typecheck.check_type(total_rounds, int)
    py_typecheck.check_type(num_rounds_to_prefetch, int)
    py_typecheck.check_type(num_clients_to_prefetch, int)
    py_typecheck.check_type(prefetch_threshold, int)
    if num_clients_to_prefetch < 1:
      raise ValueError(
          'Expected `num_clients_to_prefetch` to be greater than 1, found '
          f'{num_clients_to_prefetch}.'
      )

    self._data_source = data_source
    self._total_rounds = total_rounds
    self._num_rounds_to_prefetch = num_rounds_to_prefetch
    self._num_clients_to_prefetch = num_clients_to_prefetch
    self._prefetch_threshold = prefetch_threshold

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._data_source.federated_type

  @property
  def capabilities(self) -> list[data_source_lib.Capability]:
    """The list of capabilities supported by this data source."""
    return self._data_source.capabilities

  def iterator(self) -> PrefetchingDataSourceIterator:
    """Returns a new iterator for retrieving data from this data source."""
    iterator = self._data_source.iterator()
    return PrefetchingDataSourceIterator(
        iterator=iterator,
        total_rounds=self._total_rounds,
        num_rounds_to_prefetch=self._num_rounds_to_prefetch,
        num_clients_to_prefetch=self._num_clients_to_prefetch,
        prefetch_threshold=self._prefetch_threshold,
    )
