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
"""Data source that performs prefetching."""

import asyncio
import threading
from typing import Any, Awaitable, Callable, List, Mapping, Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.executors import cardinality_carrying_base
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import ingestable_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source as tff_data_source


class FetchedValue(ingestable_base.Ingestable,
                   cardinality_carrying_base.CardinalityCarrying):
  """Represents a value constructed by the prefetching data source."""

  def __init__(
      self, executor: executor_base.Executor,
      executor_value: executor_value_base.ExecutorValue,
      cardinality: Mapping[Any, int],
      defining_coro_fn: Callable[[executor_base.Executor],
                                 Awaitable[executor_value_base.ExecutorValue]]):
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
  def cardinality(self) -> Mapping[Any, int]:
    return self._cardinality

  async def ingest(self, executor):
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


class PrefetchingDataSourceIterator(tff_data_source.FederatedDataSourceIterator
                                   ):
  """An instance of `DataSourceIterator` with built-in prefetching."""

  def __init__(
      self,
      data_iterator: tff_data_source.FederatedDataSourceIterator,
      total_rounds: int,
      num_rounds_to_prefetch: int,
      num_clients_to_prefetch: Optional[int],
      # TODO(b/193543632): C++ execution is not fully supported in OSS.
      context: async_execution_context.AsyncExecutionContext,
      buffer_size: int = 0,
  ):
    """Constructs this data source iterator.

    Args:
      data_iterator: An iterstor over the original data source.
      total_rounds: The total (maximum) number of rounds to fetch.
      num_rounds_to_prefetch: The number of rounds to fetch in advance.
      num_clients_to_prefetch: The number of clients to prefetch per round,
        which must be identical across all rounds; attempts to sample any other
        number of clients from this data source will fail.
      context: The current context stack. Now only syncronous execution context
        is supported.
      buffer_size: The number below which the data source starts to prefetching.

    Raises:
      RuntimeError: if the iterator is not being created in a valid execution
        context, same as the context for all the subsequent selections.
    """
    self._cardinality = {placements.CLIENTS: num_clients_to_prefetch}
    self._executor_factory = context.executor_factory
    self._data_iterator = data_iterator
    self._total_rounds = total_rounds
    self._num_rounds_to_prefetch = num_rounds_to_prefetch
    self._buffer_size = buffer_size
    self._num_clients_to_prefetch = num_clients_to_prefetch
    self._lock = threading.Lock()
    self._num_rounds_prefetched = 0
    self._prefetched_round_data = []
    self._active_threads = []
    self._start_prefetching()

  def _single_round_fn(self):
    round_data = self._data_iterator.select(
        number_of_clients=self._num_clients_to_prefetch)  # gen-stub-imports

    # We assume the executor factory uses a cache, so most calls to this
    # function should result in a hit.
    executor_at_invocation = self._executor_factory.create_executor(
        self._cardinality)

    # Force ingestion of the round data in the configured executor. If a worker
    # goes down, the stack will be rebuilt and this state lost--but this is a
    # quick-and-dirty version of 'persistent values' that should help amortize
    # the cost of this work.
    if isinstance(round_data, ingestable_base.Ingestable):
      executor_value_coro = round_data.ingest(executor_at_invocation)

      async def defining_coro_fn(executor):
        return await round_data.ingest(executor)

    else:
      executor_value_coro = executor_at_invocation.create_value(
          round_data, self._data_iterator.federated_type)

      async def defining_coro_fn(executor):
        return await executor.create_value(round_data,
                                           self._data_iterator.federated_type)

    event_loop = asyncio.new_event_loop()
    executor_value = event_loop.run_until_complete(executor_value_coro)
    fetched_round_data = FetchedValue(executor_at_invocation, executor_value,
                                      self._cardinality, defining_coro_fn)

    with self._lock:
      self._prefetched_round_data.append(fetched_round_data)

  def _start_prefetching(self) -> None:
    with self._lock:
      if self._active_threads:
        # Already prefetching
        return
      if len(self._prefetched_round_data) > self._buffer_size:
        # Only fetch data when _prefetched_round_data has a low volume of data
        # to avoid new threads created in each round.
        return
      num_to_prefetch = min(
          self._num_rounds_to_prefetch - len(self._prefetched_round_data),
          self._total_rounds - self._num_rounds_prefetched)
      if num_to_prefetch < 1:
        # Already have enough
        return
      self._num_rounds_prefetched = self._num_rounds_prefetched + num_to_prefetch
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
      if len(self._prefetched_round_data) < 1:
        raise RuntimeError('Failed to prefetch at least one item.')

  @property
  def federated_type(self) -> computation_types.FederatedType:
    return self._data_iterator.federated_type

  def select(self, number_of_clients: Optional[int] = None) -> Any:
    if number_of_clients != self._num_clients_to_prefetch:
      raise ValueError(
          'The requested number of clients is {}, but this prefetching '
          'data source is already hard-wired to fetch {} clients/round '
          'and cannot accept dynamically changing different numbers of '
          'clients on a per-round basis.'.format(number_of_clients,
                                                 self._num_clients_to_prefetch))
    self._finish_prefetching()
    with self._lock:
      round_data = self._prefetched_round_data[0]
      self._prefetched_round_data = self._prefetched_round_data[1:]
    self._start_prefetching()
    return round_data


class PrefetchingDataSource(tff_data_source.FederatedDataSource):
  """An instance of `DataSource` that performs prefetching."""

  def __init__(self,
               data_source: tff_data_source.FederatedDataSource,
               total_rounds: int,
               num_rounds_to_prefetch: int,
               num_clients_to_prefetch: int,
               context: async_execution_context.AsyncExecutionContext,
               buffer_size: int = 0):
    """Constructs this data source.

    Args:
      data_source: The original source of data.
      total_rounds: The total (maximum) number of rounds to fetch.
      num_rounds_to_prefetch: The number of rounds to fetch in advance.
      num_clients_to_prefetch: The number of clients per round, which must be
        identical across all rounds; attempts to sample any other number of
        clients from this data source will fail.
      context: The current context stack. Now only syncronous execution context
        is supported.
      buffer_size: The number below which the data source starts to prefetching.

    Raises:
      ValueError: If the argument values are outside the supported range.
    """
    py_typecheck.check_type(context,
                            async_execution_context.AsyncExecutionContext)
    py_typecheck.check_type(data_source, tff_data_source.FederatedDataSource)
    if num_clients_to_prefetch < 1:
      raise ValueError(
          f'The number of clients per round {num_clients_to_prefetch} is smaller '
          'than 1.')
    self._data_source = data_source
    self._total_rounds = total_rounds
    self._num_rounds_to_prefetch = num_rounds_to_prefetch
    self._num_clients_to_prefetch = num_clients_to_prefetch
    self._buffer_size = buffer_size
    self._context = context

  @property
  def federated_type(self) -> computation_types.FederatedType:
    return self._data_source.federated_type

  @property
  def capabilities(self) -> List[tff_data_source.Capability]:
    return self._data_source.capabilities

  def iterator(self) -> PrefetchingDataSourceIterator:
    return PrefetchingDataSourceIterator(self._data_source.iterator(),
                                         self._total_rounds,
                                         self._num_rounds_to_prefetch,
                                         self._num_clients_to_prefetch,
                                         self._context, self._buffer_size)
