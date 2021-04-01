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
"""A collection of constructors for basic types of executor stacks."""

import asyncio
from concurrent import futures
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from absl import logging
import attr
import cachetools
import grpc
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import caching_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import federated_composing_strategy
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.executors import sequence_executor
from tensorflow_federated.python.core.impl.executors import sizing_executor
from tensorflow_federated.python.core.impl.executors import thread_delegating_executor
from tensorflow_federated.python.core.impl.types import placements


# Place a limit on the maximum size of the executor caches managed by the
# ExecutorFactories, to prevent unbounded thread and memory growth in the case
# of rapidly-changing cross-round cardinalities.
_EXECUTOR_CACHE_SIZE = 10


def _get_hashable_key(cardinalities: executor_factory.CardinalitiesType):
  return tuple(sorted((str(k), v) for k, v in cardinalities.items()))


class ResourceManagingExecutorFactory(executor_factory.ExecutorFactory):
  """Implementation of executor factory holding an executor per cardinality."""

  def __init__(
      self,
      executor_stack_fn: Callable[[executor_factory.CardinalitiesType],
                                  executor_base.Executor],
      ensure_closed: Optional[Sequence[executor_base.Executor]] = None):
    """Initializes `ResourceManagingExecutorFactory`.

    `ResourceManagingExecutorFactory` manages a mapping from `cardinalities`
    to `executor_base.Executors`, closing and destroying the executors in this
    mapping when asked.

    Args:
      executor_stack_fn: Callable taking a mapping from
        `placements.PlacementLiteral` to integers, and returning an
        `executor_base.Executor`. The returned executor will be configured to
        handle these cardinalities.
      ensure_closed: Optional sequence of `executor_base.Excutors` which should
        always be closed on a `clean_up_executors` call. Defaults to empty.
    """

    py_typecheck.check_callable(executor_stack_fn)
    self._executor_stack_fn = executor_stack_fn
    self._executors = cachetools.LRUCache(_EXECUTOR_CACHE_SIZE)
    if ensure_closed is None:
      ensure_closed = ()
    self._ensure_closed = ensure_closed

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Constructs or gets existing executor.

    Returns a previously-constructed executor if this method has already been
    invoked with `cardinalities`. If not, invokes `self._executor_stack_fn`
    with `cardinalities` and returns the result.

    Args:
      cardinalities: `dict` with `placements.PlacementLiteral` keys and
        integer values, specifying the population size at each placement. The
        executor stacks returned from this method are not themselves
        polymorphic; a concrete stack must have fixed sizes at each placement.

    Returns:
      Instance of `executor_base.Executor` as described above.
    """
    py_typecheck.check_type(cardinalities, dict)
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is not None:
      return ex
    ex = self._executor_stack_fn(cardinalities)
    py_typecheck.check_type(ex, executor_base.Executor)
    self._executors[key] = ex
    return ex

  def clean_up_executors(self):
    """Calls `close` on all constructed executors, resetting internal cache.

    If a caller holds a name bound to any of the executors returned from
    `create_executor`, this executor should be assumed to be in an invalid
    state, and should not be used after this method is called. Instead, callers
    should again invoke `create_executor`.
    """
    for _, ex in self._executors.items():
      ex.close()
    for ex in self._ensure_closed:
      ex.close()
    self._executors = {}


@attr.s(auto_attribs=True, eq=False, order=False, frozen=True)
class SizeInfo(object):
  """Structure for size information from SizingExecutorFactory.get_size_info().

  Attribues:
    `broadcast_history`: 2D ragged list of 2-tuples which represents the
      broadcast history.
    `aggregate_history`: 2D ragged list of 2-tuples which represents the
      aggregate history.
    `broadcast_bits`: A list of shape [number_of_execs] representing the
      number of broadcasted bits passed through each executor.
    `aggregate_bits`: A list of shape [number_of_execs] representing the
      number of aggregated bits passed through each executor.
  """
  broadcast_history: Dict[Any, sizing_executor.SizeAndDTypes]
  aggregate_history: Dict[Any, sizing_executor.SizeAndDTypes]
  broadcast_bits: List[int]
  aggregate_bits: List[int]


class SizingExecutorFactory(ResourceManagingExecutorFactory):
  """A executor factory holding an executor per cardinality."""

  def __init__(
      self,
      executor_stack_fn: Callable[[executor_factory.CardinalitiesType],
                                  Tuple[executor_base.Executor,
                                        List[sizing_executor.SizingExecutor]]]):
    """Initializes `SizingExecutorFactory`.

    Args:
      executor_stack_fn: Similar to base class but the second return value of
        the callable is used to expose the SizingExecutors.
    """

    super().__init__(executor_stack_fn)
    # Sizing executors are intended to record the entire history of execution,
    # and therefore we don't want to be silently clearing them. So we leave
    # sizing_executors as a proper dict.
    self._sizing_executors = {}

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """See base class."""

    py_typecheck.check_type(cardinalities, dict)
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is not None:
      return ex
    ex, sizing_executors = self._executor_stack_fn(cardinalities)
    self._sizing_executors[key] = []
    for executor in sizing_executors:
      if not isinstance(executor, sizing_executor.SizingExecutor):
        raise ValueError('Expected all input executors to be sizing executors')
      self._sizing_executors[key].append(executor)
    py_typecheck.check_type(ex, executor_base.Executor)
    self._executors[key] = ex
    return ex

  def get_size_info(self) -> SizeInfo:
    """Returns information about the transferred data of each SizingExecutor.

    Returns the history of broadcast and aggregation for each executor as well
    as the number of aggregated bits that has been passed through.

    Returns:
      An instance of `SizeInfo`.
    """
    size_ex_dict = self._sizing_executors

    def _extract_history(sizing_exs: List[sizing_executor.SizingExecutor]):
      broadcast_history, aggregate_history = [], []
      for ex in sizing_exs:
        broadcast_history.extend(ex.broadcast_history)
        aggregate_history.extend(ex.aggregate_history)
      return broadcast_history, aggregate_history

    broadcast_history, aggregate_history = {}, {}
    for key, size_exs in size_ex_dict.items():
      current_broadcast_history, current_aggregate_history = _extract_history(
          size_exs)
      broadcast_history[key] = current_broadcast_history
      aggregate_history[key] = current_aggregate_history

    broadcast_bits = [
        self._calculate_bit_size(hist) for hist in broadcast_history.values()
    ]
    aggregate_bits = [
        self._calculate_bit_size(hist) for hist in aggregate_history.values()
    ]
    return SizeInfo(
        broadcast_history=broadcast_history,
        aggregate_history=aggregate_history,
        broadcast_bits=broadcast_bits,
        aggregate_bits=aggregate_bits)

  def _bits_per_element(self, dtype: tf.DType) -> int:
    """Returns the number of bits that a tensorflow DType uses per element."""
    if dtype == tf.string:
      return 8
    elif dtype == tf.bool:
      return 1
    return dtype.size * 8

  def _calculate_bit_size(self, history: sizing_executor.SizeAndDTypes) -> int:
    """Takes a list of 2 element lists and calculates the number of bits represented.

    The input list should follow the format of self.broadcast_history or
    self.aggregate_history. That is, each 2 element list should be
    [num_elements, dtype].

    Args:
      history: The history of values passed through the executor.

    Returns:
      The number of bits represented in the history.
    """
    bit_size = 0
    for num_elements, dtype in history:
      bit_size += num_elements * self._bits_per_element(dtype)
    return bit_size


# pylint:disable=missing-function-docstring
def _wrap_executor_in_threading_stack(ex: executor_base.Executor,
                                      use_caching: Optional[bool] = False,
                                      support_sequence_ops: bool = False,
                                      can_resolve_references=True):
  threaded_ex = thread_delegating_executor.ThreadDelegatingExecutor(ex)
  if use_caching:
    threaded_ex = caching_executor.CachingExecutor(threaded_ex)
  if support_sequence_ops:
    if not can_resolve_references:
      raise ValueError(
          'Support for sequence ops requires ability to resolve references.')
    threaded_ex = sequence_executor.SequenceExecutor(
        reference_resolving_executor.ReferenceResolvingExecutor(threaded_ex))
  if can_resolve_references:
    threaded_ex = reference_resolving_executor.ReferenceResolvingExecutor(
        threaded_ex)
  return threaded_ex


class UnplacedExecutorFactory(executor_factory.ExecutorFactory):
  """ExecutorFactory to construct executors which cannot understand placement.

  This factory constructs executors which represent "local execution": work
  that happens at the clients, at the server, or without placements. As such,
  this executor manages the placement of work on local executors.
  """

  def __init__(self,
               *,
               use_caching: bool,
               support_sequence_ops: bool = False,
               can_resolve_references: bool = True,
               server_device: Optional[tf.config.LogicalDevice] = None,
               client_devices: Optional[Sequence[tf.config.LogicalDevice]] = (),
               leaf_executor_fn=eager_tf_executor.EagerTFExecutor):
    self._use_caching = use_caching
    self._support_sequence_ops = support_sequence_ops
    self._can_resolve_references = can_resolve_references
    self._server_device = server_device
    self._client_devices = client_devices
    self._client_device_index = 0
    self._leaf_executor_fn = leaf_executor_fn

  def _get_next_client_device(self) -> Optional[tf.config.LogicalDevice]:
    if not self._client_devices:
      return None
    device = self._client_devices[self._client_device_index]
    self._client_device_index = (self._client_device_index + 1) % len(
        self._client_devices)
    return device

  def create_executor(
      self,
      *,
      cardinalities: Optional[executor_factory.CardinalitiesType] = None,
      placement: Optional[placements.PlacementLiteral] = None
  ) -> executor_base.Executor:
    if cardinalities:
      raise ValueError(
          'Unplaced executors cannot accept nonempty cardinalities as '
          'arguments. Received cardinalities: {}.'.format(cardinalities))
    if placement == placements.CLIENTS:
      device = self._get_next_client_device()
    elif placement == placements.SERVER:
      device = self._server_device
    else:
      device = None
    leaf_ex = self._leaf_executor_fn(device=device)
    return _wrap_executor_in_threading_stack(
        leaf_ex,
        use_caching=self._use_caching,
        support_sequence_ops=self._support_sequence_ops,
        can_resolve_references=self._can_resolve_references)

  def clean_up_executors(self):
    # Does not hold any executors internally, so nothing to clean up.
    pass


class FederatingExecutorFactory(executor_factory.ExecutorFactory):
  """Executor factory for stacks which delegate placed computations.

  `FederatingExecutorFactory` validates cardinality requests and manages
  the relationship between the clients and the client executors. Additionally,
  `FederatingExecutorFactory` allows for the measurement of tensors crossing
  the federated communication boundary via the
  `sizing_executor.SizingExecutor`.

  This factory is initialized with:
    * An integer number of client executors, indicating the number of client
      executors that should be run in parallel. Setting this parameter to a low
      number can aid with OOMs on accelerators, or speed up the computation in
      the case of extremely lightweight client work.
    * An `UnplacedExecutorFactory` to use to construct the executors for
      computations after they have had their placement ingested and stripped by
      the `FederatingExecutor`. That is, this factory produces the executors
      used to run client, server and unplaced computations.
    * An optional number of clients. In the case that this parameter is
      unspecified, this factory will be polymorphic to number of clients
      requested, relying on inference of this value from a higher level to
      populate the `cardinalities` parameter to its `create_executor` method.
      In the case that this parameter is specified, the `create_executor`
      method will check that the requested cardinalities is consistent with
      this hardcoded parameter.
    * A boolean `use_sizing` to indicate whether to wire instances of
      `sizing_executors.SizingExecutor` on top of the client stacks.
    * An optional instance of `LocalComputationFactory` to use to construct
      local computations used as parameters in certain federated operators
      (such as `tff.federated_sum`, etc.). Defaults to a TensorFlow factory.

  """

  def __init__(self,
               *,
               clients_per_thread: int,
               unplaced_ex_factory: UnplacedExecutorFactory,
               num_clients: Optional[int] = None,
               use_sizing: bool = False,
               local_computation_factory: local_computation_factory_base
               .LocalComputationFactory = tensorflow_computation_factory
               .TensorFlowComputationFactory(),
               federated_strategy_factory=federated_resolving_strategy
               .FederatedResolvingStrategy.factory):
    py_typecheck.check_type(clients_per_thread, int)
    py_typecheck.check_type(unplaced_ex_factory, UnplacedExecutorFactory)
    py_typecheck.check_type(
        local_computation_factory,
        local_computation_factory_base.LocalComputationFactory)
    self._clients_per_thread = clients_per_thread
    self._unplaced_executor_factory = unplaced_ex_factory
    if num_clients is not None:
      py_typecheck.check_type(num_clients, int)
      if num_clients <= 0:
        raise ValueError('Number of clients must be positive.')
    self._num_clients = num_clients
    self._use_sizing = use_sizing
    if self._use_sizing:
      self._sizing_executors = []
    else:
      self._sizing_executors = None
    self._federated_strategy_factory = federated_strategy_factory
    self._local_computation_factory = local_computation_factory

  @property
  def sizing_executors(self) -> List[sizing_executor.SizingExecutor]:
    if not self._use_sizing:
      raise ValueError('This federated factory is not configured to produce '
                       'size information. Construct a new federated factory, '
                       'passing argument `use_sizing=True`.')
    else:
      return self._sizing_executors

  def _validate_requested_clients(
      self, cardinalities: executor_factory.CardinalitiesType) -> int:
    num_requested_clients = cardinalities.get(placements.CLIENTS)
    if num_requested_clients is None:
      if self._num_clients is not None:
        return self._num_clients
      else:
        return 0
    if (self._num_clients is not None and
        self._num_clients != num_requested_clients):
      raise ValueError(
          'FederatingStackFactory configured to return {} '
          'clients, but encountered a request for {} clients.'
          'If your computation accepts CLIENTS-placed arguments, it is '
          'recommended to avoid setting the num_clients parameter in the TFF '
          'runtime.'.format(self._num_clients, num_requested_clients))
    return num_requested_clients

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Constructs a federated executor with requested cardinalities."""
    num_clients = self._validate_requested_clients(cardinalities)
    num_client_executors = math.ceil(num_clients / self._clients_per_thread)
    client_stacks = [
        self._unplaced_executor_factory.create_executor(
            cardinalities={}, placement=placements.CLIENTS)
        for _ in range(num_client_executors)
    ]
    if self._use_sizing:
      client_stacks = [
          sizing_executor.SizingExecutor(ex) for ex in client_stacks
      ]
      self._sizing_executors.extend(client_stacks)

    federating_strategy_factory = self._federated_strategy_factory(
        {
            placements.CLIENTS: [
                client_stacks[k % len(client_stacks)]
                for k in range(num_clients)
            ],
            placements.SERVER:
                self._unplaced_executor_factory.create_executor(
                    placement=placements.SERVER),
        },
        local_computation_factory=self._local_computation_factory)
    unplaced_executor = self._unplaced_executor_factory.create_executor()
    executor = federating_executor.FederatingExecutor(
        federating_strategy_factory, unplaced_executor)
    return _wrap_executor_in_threading_stack(executor)

  def clean_up_executors(self):
    # Does not hold any executors internally, so nothing to clean up.
    pass


def create_minimal_length_flat_stack_fn(
    max_clients_per_stack: int,
    federated_stack_factory: executor_factory.ExecutorFactory
) -> Callable[[executor_factory.CardinalitiesType],
              List[executor_base.Executor]]:
  """Creates a function returning a list of executors to run `cardinalities`.

  This list is of minimal length among all lists subject to the constraint that
  each executor can be responsible for executing no more than
  `max_clients_per_stack`. For example, given a `cardinalities` argument of 10
  and `max_clients_per_stack` of 3, this function may return a list of
  executors 3 of which run 3 clients and 1 of which runs 1, or 2 of which run 3
  and 2 of which run 2.

  Args:
    max_clients_per_stack: Integer determining the maximum number of clients a
      single executor in the list returned by the function may execute.
    federated_stack_factory: The `executor_factory.ExecutorFactory` for use in
      actually constructing these executors.

  Returns:
    A callable taking a parameter of type `executor_factory.CardinalitiesType`,
    and returning a list of `executor_base.Executors` which in total can execute
    the cardinalities specified by the argument. This callable will raise if it
    is passed a cardinalitites dict with a negative number of clients.
  """

  def create_executor_list(
      cardinalities: executor_factory.CardinalitiesType
  ) -> List[executor_base.Executor]:
    num_clients = cardinalities.get(placements.CLIENTS, 0)
    if num_clients < 0:
      raise ValueError('Number of clients cannot be negative.')
    elif num_clients < 1:
      return [
          federated_stack_factory.create_executor(cardinalities=cardinalities)
      ]
    executors = []
    while num_clients > 0:
      n = min(num_clients, max_clients_per_stack)
      sub_executor_cardinalities = {**cardinalities}
      sub_executor_cardinalities[placements.CLIENTS] = n
      executors.append(
          federated_stack_factory.create_executor(sub_executor_cardinalities))
      num_clients -= n
    return executors

  return create_executor_list


class ComposingExecutorFactory(executor_factory.ExecutorFactory):
  """Factory class encapsulating executor compositional logic.

  This class is responsible for aggregating lists of executors into a
  compositional hierarchy based on the `max_fanout` parameter.
  """

  def __init__(self,
               *,
               max_fanout: int,
               unplaced_ex_factory: UnplacedExecutorFactory,
               flat_stack_fn: Callable[[executor_factory.CardinalitiesType],
                                       Sequence[executor_base.Executor]],
               local_computation_factory: local_computation_factory_base
               .LocalComputationFactory = tensorflow_computation_factory
               .TensorFlowComputationFactory()):
    if max_fanout < 2:
      raise ValueError('Max fanout must be greater than 1.')
    self._flat_stack_fn = flat_stack_fn
    self._max_fanout = max_fanout
    self._unplaced_ex_factory = unplaced_ex_factory
    self._local_computation_factory = local_computation_factory

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Creates an executor hierarchy of maximum width `self._max_fanout`.

    First creates a flat list of executors to aggregate, using the
    `flat_stack_fn` passed in at initialization. Then composes this list
    into a hierarchy of width at most `self._max_fanout`.

    Args:
      cardinalities: A mapping from placements to integers specifying the
        cardinalities at each placement

    Returns:
      An `executor_base.Executor` satisfying the conditions above.
    """
    executors = self._flat_stack_fn(cardinalities)
    return self._aggregate_stacks(executors)

  def clean_up_executors(self):
    """Holds no executors internally, so passes on cleanup."""
    pass

  def _create_composing_stack(
      self, *, target_executors: Sequence[executor_base.Executor]
  ) -> executor_base.Executor:
    server_executor = self._unplaced_ex_factory.create_executor(
        placement=placements.SERVER)
    composing_strategy_factory = federated_composing_strategy.FederatedComposingStrategy.factory(
        server_executor,
        target_executors,
        local_computation_factory=self._local_computation_factory)
    unplaced_executor = self._unplaced_ex_factory.create_executor()
    composing_executor = federating_executor.FederatingExecutor(
        composing_strategy_factory, unplaced_executor)
    threaded_composing_executor = _wrap_executor_in_threading_stack(
        composing_executor, can_resolve_references=False)
    return threaded_composing_executor

  def _aggregate_stacks(
      self,
      executors: Sequence[executor_base.Executor],
  ) -> executor_base.Executor:
    """Hierarchically aggregates a sequence of executors via composing strategy.

    Constructs as many levels as it takes to support all executors, reducing
    by the factor of `self._max_fanout` in each iteration, for up to
    `log(len(address_list)) / log(max_fanout)` iterations.

    Args:
      executors: Sequence of `executor_base.Executors` to aggregate into a
        composing hierarchy.

    Returns:
      A single `executor_base.Executor` representing the aggregated hierarchy.
      The particular architecture of this hierarchy depends on the interplay
      between `self._max_fanout` and the length of the sequence of executors.

    Raises:
      RuntimeError: If hierarchy construction fails.
    """
    if len(executors) <= 1:
      return reference_resolving_executor.ReferenceResolvingExecutor(
          self._create_composing_stack(target_executors=executors))
    while len(executors) > 1:
      new_executors = []
      offset = 0
      while offset < len(executors):
        new_offset = offset + self._max_fanout
        target_executors = executors[offset:new_offset]
        composing_executor = self._create_composing_stack(
            target_executors=target_executors)
        new_executors.append(composing_executor)
        offset = new_offset
      executors = new_executors
    if len(executors) != 1:
      raise RuntimeError('Expected 1 executor, got {}.'.format(len(executors)))
    return reference_resolving_executor.ReferenceResolvingExecutor(executors[0])


def local_executor_factory(
    num_clients=None,
    max_fanout=100,
    clients_per_thread=1,
    server_tf_device=None,
    client_tf_devices=tuple(),
    reference_resolving_clients=True,
    support_sequence_ops=False,
    leaf_executor_fn=eager_tf_executor.EagerTFExecutor,
    local_computation_factory=tensorflow_computation_factory
    .TensorFlowComputationFactory()
) -> executor_factory.ExecutorFactory:
  """Constructs an executor factory to execute computations locally.

  Note: The `tff.federated_secure_sum()` intrinsic is not implemented by this
  executor.

  Args:
    num_clients: The number of clients. If specified, the executor factory
      function returned by `local_executor_factory` will be configured to have
      exactly `num_clients` clients. If unspecified (`None`), then the function
      returned will attempt to infer cardinalities of all placements for which
      it is passed values.
    max_fanout: The maximum fanout at any point in the aggregation hierarchy. If
      `num_clients > max_fanout`, the constructed executor stack will consist of
      multiple levels of aggregators. The height of the stack will be on the
      order of `log(num_clients) / log(max_fanout)`.
    clients_per_thread: Integer number of clients for each of TFF's threads to
      run in sequence. Increasing `clients_per_thread` therefore reduces the
      concurrency of the TFF runtime, which can be useful if client work is very
      lightweight or models are very large and multiple copies cannot fit in
      memory.
    server_tf_device: A `tf.config.LogicalDevice` to place server and other
      computation without explicit TFF placement.
    client_tf_devices: List/tuple of `tf.config.LogicalDevice` to place clients
      for simulation. Possibly accelerators returned by
      `tf.config.list_logical_devices()`.
    reference_resolving_clients: Boolean indicating whether executors
      representing clients must be able to handle unplaced TFF lambdas.
    support_sequence_ops: Boolean indicating whether this executor supports
      sequence ops (currently False by default).
    leaf_executor_fn: A function that constructs leaf-level executors. Default
      is the eager TF executor (other possible options: XLA, IREE). Should
      accept the `device` keyword argument if the executor is to be configured
      with explicitly chosen devices.
    local_computation_factory: An instance of `LocalComputationFactory` to
      use to construct local computations used as parameters in certain
      federated operators (such as `tff.federated_sum`, etc.). Defaults to
      a TensorFlow computation factory that generates TensorFlow code.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.

  Raises:
    ValueError: If the number of clients is specified and not one or larger.
  """
  if server_tf_device is not None:
    py_typecheck.check_type(server_tf_device, tf.config.LogicalDevice)
  py_typecheck.check_type(client_tf_devices, (tuple, list))
  py_typecheck.check_type(max_fanout, int)
  py_typecheck.check_type(clients_per_thread, int)
  if num_clients is not None:
    py_typecheck.check_type(num_clients, int)
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  unplaced_ex_factory = UnplacedExecutorFactory(
      use_caching=False,
      support_sequence_ops=support_sequence_ops,
      can_resolve_references=reference_resolving_clients,
      server_device=server_tf_device,
      client_devices=client_tf_devices,
      leaf_executor_fn=leaf_executor_fn)
  federating_executor_factory = FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=False,
      local_computation_factory=local_computation_factory)
  flat_stack_fn = create_minimal_length_flat_stack_fn(
      max_fanout, federating_executor_factory)
  full_stack_factory = ComposingExecutorFactory(
      max_fanout=max_fanout,
      unplaced_ex_factory=unplaced_ex_factory,
      flat_stack_fn=flat_stack_fn,
      local_computation_factory=local_computation_factory)

  def _factory_fn(cardinalities):
    if cardinalities.get(placements.CLIENTS, 0) < max_fanout:
      return federating_executor_factory.create_executor(cardinalities)
    return full_stack_factory.create_executor(cardinalities)

  return ResourceManagingExecutorFactory(_factory_fn)


def thread_debugging_executor_factory(
    num_clients=None,
    clients_per_thread=1,
    leaf_executor_fn=eager_tf_executor.EagerTFExecutor
) -> executor_factory.ExecutorFactory:
  r"""Constructs a simplified execution stack to execute local computations.

  The constructed executors support a limited set of TFF's computations. In
  particular, the debug executor can only resolve references at the top level
  of the stack, and therefore assumes that all local computation is expressed
  in pure TensorFlow.

  The debug executor makes particular guarantees about the structure of the
  stacks it constructs which are intended to make them maximally easy to reason
  about. That is, the debug executor will essentially execute exactly the
  computation it is passed (in particular, this implies that there are no
  caching layers), and uses its declared inability to execute arbitrary TFF
  lambdas to reduce the complexity of the constructed stack. Every debugging
  executor will have a similar structure, the simplest structure that can
  execute the full expressivity of TFF while running each client in a dedicated
  thread:


                        ReferenceResolvingExecutor
                                  |
                            FederatingExecutor
                          /       ...         \
    ThreadDelegatingExecutor                  ThreadDelegatingExecutor
                |                                         |
        EagerTFExecutor                            EagerTFExecutor


  This structure can be useful in understanding the concurrency pattern of TFF
  execution, and where the TFF runtime infers data dependencies.

  Args:
    num_clients: The number of clients. If specified, the executor factory
      function returned by `local_executor_factory` will be configured to have
      exactly `num_clients` clients. If unspecified (`None`), then the function
      returned will attempt to infer cardinalities of all placements for which
      it is passed values.
    clients_per_thread: Integer number of clients for each of TFF's threads to
      run in sequence. Increasing `clients_per_thread` therefore reduces the
      concurrency of the TFF runtime, which can be useful if client work is very
      lightweight or models are very large and multiple copies cannot fit in
      memory.
    leaf_executor_fn: A function that constructs leaf-level executors. Default
      is the eager TF executor (other possible options: XLA, IREE). Should
      accept the `device` keyword argument if the executor is to be configured
      with explicitly chosen devices.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.

  Raises:
    ValueError: If the number of clients is specified and not one or larger.
  """
  py_typecheck.check_type(clients_per_thread, int)
  if num_clients is not None:
    py_typecheck.check_type(num_clients, int)
  unplaced_ex_factory = UnplacedExecutorFactory(
      use_caching=False,
      can_resolve_references=False,
      leaf_executor_fn=leaf_executor_fn)
  federating_executor_factory = FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=False)

  return ResourceManagingExecutorFactory(
      federating_executor_factory.create_executor)


def sizing_executor_factory(
    num_clients: int = None,
    max_fanout: int = 100,
    clients_per_thread: int = 1,
    leaf_executor_fn=eager_tf_executor.EagerTFExecutor
) -> executor_factory.ExecutorFactory:
  """Constructs an executor factory to execute computations locally with sizing.

  Args:
    num_clients: The number of clients. If specified, the executor factory
      function returned by `sizing_executor_factory` will be configured to have
      exactly `num_clients` clients. If unspecified (`None`), then the function
      returned will attempt to infer cardinalities of all placements for which
      it is passed values.
    max_fanout: The maximum fanout at any point in the aggregation hierarchy. If
      `num_clients > max_fanout`, the constructed executor stack will consist of
      multiple levels of aggregators. The height of the stack will be on the
      order of `log(num_clients) / log(max_fanout)`.
    clients_per_thread: Integer number of clients for each of TFF's threads to
      run in sequence. Increasing `clients_per_thread` therefore reduces the
      concurrency of the TFF runtime, which can be useful if client work is very
      lightweight or models are very large and multiple copies cannot fit in
      memory.
    leaf_executor_fn: A function that constructs leaf-level executors. Default
      is the eager TF executor (other possible options: XLA, IREE). Should
      accept the `device` keyword argument if the executor is to be configured
      with explicitly chosen devices.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.

  Raises:
    ValueError: If the number of clients is specified and not one or larger.
  """
  py_typecheck.check_type(max_fanout, int)
  py_typecheck.check_type(clients_per_thread, int)
  if num_clients is not None:
    py_typecheck.check_type(num_clients, int)
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  unplaced_ex_factory = UnplacedExecutorFactory(
      use_caching=False, leaf_executor_fn=leaf_executor_fn)
  federating_executor_factory = FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=True)
  flat_stack_fn = create_minimal_length_flat_stack_fn(
      max_fanout, federating_executor_factory)
  full_stack_factory = ComposingExecutorFactory(
      max_fanout=max_fanout,
      unplaced_ex_factory=unplaced_ex_factory,
      flat_stack_fn=flat_stack_fn)

  def _factory_fn(
      cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    if cardinalities.get(placements.CLIENTS, 0) < max_fanout:
      executor = federating_executor_factory.create_executor(cardinalities)
    else:
      executor = full_stack_factory.create_executor(cardinalities)
    sizing_executor_list = federating_executor_factory.sizing_executors
    return executor, sizing_executor_list

  return SizingExecutorFactory(_factory_fn)


class ReconstructOnChangeExecutorFactory(executor_factory.ExecutorFactory):
  """ExecutorFactory exposing hook to construct executors on environment change.

  When the initialization parameter `change_query` returns `True`,
  ReconstructOnChangeExecutorFactory` constructs a new executor, bypassing
  any previously constructed executors.
  """

  def __init__(self,
               underlying_stack: executor_factory.ExecutorFactory,
               ensure_closed: Optional[Sequence[executor_base.Executor]] = None,
               change_query: Callable[[executor_factory.CardinalitiesType],
                                      bool] = lambda _: True):
    self._change_query = change_query
    self._underlying_stack = underlying_stack
    self._executors = cachetools.LRUCache(_EXECUTOR_CACHE_SIZE)
    if ensure_closed is None:
      ensure_closed = ()
    self._ensure_closed = ensure_closed

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Returns a new or existing executor, depending on `change_query`.

    `create_executor` constructs a new executor whenever `change_query` returns
    `True`  when called with argument `cardinalities`. If `change_query` returns
    `False`, `create_executor` is free to inspect its internal executor cache
    and return a previously constructed executor if one is available.

    Args:
      cardinalities: A mapping from placement literals to ints.

    Returns:
      An `executor_base.Executor` obeying the semantics above.
    """
    py_typecheck.check_type(cardinalities, dict)
    key = _get_hashable_key(cardinalities)
    if self._change_query(cardinalities):
      reconstructed = self._underlying_stack.create_executor(cardinalities)
      self._executors[key] = reconstructed
      return reconstructed
    elif self._executors.get(key):
      return self._executors[key]
    else:
      constructed = self._underlying_stack.create_executor(cardinalities)
      self._executors[key] = constructed
      return constructed

  def clean_up_executors(self):
    for _, ex in self._executors.items():
      ex.close()
    self._executors = {}
    for ex in self._ensure_closed:
      ex.close()
    self._underlying_stack.clean_up_executors()


def _configure_remote_executor(ex, cardinalities, loop):
  """Configures `ex` to run the appropriate number of clients."""
  if loop.is_running():
    asyncio.run_coroutine_threadsafe(ex.set_cardinalities(cardinalities), loop)
  else:
    loop.run_until_complete(ex.set_cardinalities(cardinalities))
  return


def _get_event_loop() -> Tuple[asyncio.AbstractEventLoop, bool]:
  """Returns an event loop and whether the loop should be closed once done."""
  try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
      loop = asyncio.new_event_loop()
      should_close_loop = True
    else:
      should_close_loop = False
  except RuntimeError:
    loop = asyncio.new_event_loop()
    should_close_loop = True
  return loop, should_close_loop


class _CardinalitiesOrReadyListChanged():
  """Callable checking for changes to either the argument or a ready list.

  Note: the contents of the provided list are expected to change over time.
  Elements of the list must offer an `is_ready` property which will be checked
  each time.
  """

  def __init__(self, maybe_ready_list):
    self._previous_cardinalities = None
    self._previous_ready_list = ()
    self._maybe_ready_list = maybe_ready_list

  def __call__(self, cardinalities: executor_factory.CardinalitiesType) -> bool:
    cardinalities_changed = self._previous_cardinalities != cardinalities
    ready_list = tuple(x for x in self._maybe_ready_list if x.is_ready)
    ready_list_changed = ready_list != self._previous_ready_list
    self._previous_cardinalities = cardinalities
    self._previous_ready_list = ready_list
    return cardinalities_changed or ready_list_changed


def _configure_remote_workers(num_clients, remote_executors):
  """"Configures `num_clients` across `remote_executors`."""
  loop, must_close_loop = _get_event_loop()
  available_executors = [ex for ex in remote_executors if ex.is_ready]
  logging.info('%s TFF workers available out of a total of %s.',
               len(available_executors), len(remote_executors))
  if not available_executors:
    raise execution_context.RetryableError(
        'No workers are ready; try again to reconnect.')
  try:
    remaining_clients = num_clients
    live_workers = []
    for ex_idx, ex in enumerate(available_executors):
      remaining_executors = len(available_executors) - ex_idx
      num_clients_to_host = remaining_clients // remaining_executors
      remaining_clients -= num_clients_to_host
      if num_clients_to_host > 0:
        _configure_remote_executor(ex,
                                   {placements.CLIENTS: num_clients_to_host},
                                   loop)
        live_workers.append(ex)
  finally:
    if must_close_loop:
      loop.stop()
      loop.close()
  return [
      _wrap_executor_in_threading_stack(e, can_resolve_references=False)
      for e in live_workers
  ]


def remote_executor_factory(
    channels: List[grpc.Channel],
    rpc_mode: str = 'REQUEST_REPLY',
    thread_pool_executor: Optional[futures.Executor] = None,
    dispose_batch_size: int = 20,
    max_fanout: int = 100,
    default_num_clients: int = 0,
) -> executor_factory.ExecutorFactory:
  """Create an executor backed by remote workers.

  Args:
    channels: A list of `grpc.Channels` hosting services which can execute TFF
      work.
    rpc_mode: A string specifying the connection mode between the local host and
      `channels`.
    thread_pool_executor: Optional concurrent.futures.Executor used to wait for
      the reply to a streaming RPC message. Uses the default Executor if not
      specified.
    dispose_batch_size: The batch size for requests to dispose of remote worker
      values. Lower values will result in more requests to the remote worker,
      but will result in values being cleaned up sooner and therefore may result
      in lower memory usage on the remote worker.
    max_fanout: The maximum fanout at any point in the aggregation hierarchy. If
      `num_clients > max_fanout`, the constructed executor stack will consist of
      multiple levels of aggregators. The height of the stack will be on the
      order of `log(num_clients) / log(max_fanout)`.
    default_num_clients: The number of clients to use for simulations where the
      number of clients cannot be inferred. Usually the number of clients will
      be inferred from the number of values passed to computations which accept
      client-placed values. However, when this inference isn't possible (such as
      in the case of a no-argument or non-federated computation) this default
      will be used instead.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.
  """
  py_typecheck.check_type(channels, list)
  if not channels:
    raise ValueError('The list of channels cannot be empty.')
  py_typecheck.check_type(rpc_mode, str)
  if thread_pool_executor is not None:
    py_typecheck.check_type(thread_pool_executor, futures.Executor)
  py_typecheck.check_type(dispose_batch_size, int)
  py_typecheck.check_type(max_fanout, int)
  py_typecheck.check_type(default_num_clients, int)

  remote_executors = []
  for channel in channels:
    remote_executors.append(
        remote_executor.RemoteExecutor(
            channel=channel,
            rpc_mode=rpc_mode,
            thread_pool_executor=thread_pool_executor,
            dispose_batch_size=dispose_batch_size))

  def _flat_stack_fn(cardinalities):
    num_clients = cardinalities.get(placements.CLIENTS, default_num_clients)
    return _configure_remote_workers(num_clients, remote_executors)

  unplaced_ex_factory = UnplacedExecutorFactory(use_caching=False)
  composing_executor_factory = ComposingExecutorFactory(
      max_fanout=max_fanout,
      unplaced_ex_factory=unplaced_ex_factory,
      flat_stack_fn=_flat_stack_fn,
  )

  return ReconstructOnChangeExecutorFactory(
      underlying_stack=composing_executor_factory,
      ensure_closed=remote_executors,
      change_query=_CardinalitiesOrReadyListChanged(
          maybe_ready_list=remote_executors))
