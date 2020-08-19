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

import math
from typing import List, Callable, Optional, Sequence

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import caching_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import federated_composing_strategy
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import sizing_executor
from tensorflow_federated.python.core.impl.executors import thread_delegating_executor
from tensorflow_federated.python.core.impl.types import placement_literals


def _wrap_executor_in_threading_stack(ex: executor_base.Executor,
                                      use_caching: Optional[bool] = True):
  threaded_ex = thread_delegating_executor.ThreadDelegatingExecutor(ex)
  if use_caching:
    threaded_ex = caching_executor.CachingExecutor(threaded_ex)
  rre_wrapped_ex = reference_resolving_executor.ReferenceResolvingExecutor(
      threaded_ex)
  return rre_wrapped_ex


class UnplacedExecutorFactory(executor_factory.ExecutorFactory):
  """ExecutorFactory to construct executors which cannot understand placement.

  This factory constructs executors which represent "local execution": work
  that happens at the clients, at the server, or without placements. As such,
  this executor manages the placement of work on local executors.
  """

  def __init__(
      self,
      *,
      use_caching: bool,
      server_device: Optional[tf.config.LogicalDevice] = None,
      client_devices: Optional[Sequence[tf.config.LogicalDevice]] = ()):
    self._use_caching = use_caching
    self._server_device = server_device
    self._client_devices = client_devices
    self._client_device_index = 0

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
      placement: Optional[placement_literals.PlacementLiteral] = None
  ) -> executor_base.Executor:
    if cardinalities:
      raise ValueError(
          'Unplaced executors cannot accept nonempty cardinalities as '
          'arguments. Received cardinalities: {}.'.format(cardinalities))
    if placement == placement_literals.CLIENTS:
      device = self._get_next_client_device()
    elif placement == placement_literals.SERVER:
      device = self._server_device
    else:
      device = None
    eager_ex = eager_tf_executor.EagerTFExecutor(device=device)
    return _wrap_executor_in_threading_stack(eager_ex)

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
  """

  def __init__(self,
               *,
               clients_per_thread: int,
               unplaced_ex_factory: UnplacedExecutorFactory,
               num_clients: Optional[int] = None,
               use_sizing: bool = False):
    py_typecheck.check_type(clients_per_thread, int)
    py_typecheck.check_type(unplaced_ex_factory, UnplacedExecutorFactory)
    self._clients_per_thread = clients_per_thread
    self._unplaced_executor_factory = unplaced_ex_factory
    if num_clients is not None:
      py_typecheck.check_type(num_clients, int)
      if num_clients < 0:
        raise ValueError('Number of clients cannot be negative.')
    self._num_clients = num_clients
    self._use_sizing = use_sizing
    if self._use_sizing:
      self._sizing_executors = []
    else:
      self._sizing_executors = None

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
    num_requested_clients = cardinalities.get(placement_literals.CLIENTS)
    if num_requested_clients is None:
      if self._num_clients is not None:
        num_clients = self._num_clients
      else:
        num_clients = 0
    elif (self._num_clients is not None and
          self._num_clients != num_requested_clients):
      raise ValueError(
          'FederatingStackFactory configured to return {} '
          'clients, but encountered a request for {} clients.'.format(
              self._num_clients, num_requested_clients))
    else:
      num_clients = num_requested_clients
    return num_clients

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Constructs a federated executor with requested cardinalities."""
    num_clients = self._validate_requested_clients(cardinalities)
    num_client_executors = math.ceil(num_clients / self._clients_per_thread)
    client_stacks = [
        self._unplaced_executor_factory.create_executor(
            cardinalities={}, placement=placement_literals.CLIENTS)
        for _ in range(num_client_executors)
    ]
    if self._use_sizing:
      client_stacks = [
          sizing_executor.SizingExecutor(ex) for ex in client_stacks
      ]
      self._sizing_executors.extend(client_stacks)

    federating_strategy_factory = federated_resolving_strategy.FederatedResolvingStrategy.factory(
        {
            placement_literals.CLIENTS: [
                client_stacks[k % len(client_stacks)]
                for k in range(num_clients)
            ],
            placement_literals.SERVER:
                self._unplaced_executor_factory.create_executor(
                    cardinalities={}, placement=placement_literals.SERVER),
        })
    unplaced_executor = self._unplaced_executor_factory.create_executor(
        cardinalities={})
    executor = federating_executor.FederatingExecutor(
        federating_strategy_factory, unplaced_executor)
    return _wrap_executor_in_threading_stack(executor)

  def clean_up_executors(self):
    # Does not hold any executors internally, so nothing to clean up.
    pass


def _create_composite_stack(
    target_executors,
    unplaced_ex_factory: UnplacedExecutorFactory) -> executor_base.Executor:
  """Creates a single composite stack."""
  server_executor = unplaced_ex_factory.create_executor(
      placement=placement_literals.SERVER)
  federating_strategy_factory = federated_composing_strategy.FederatedComposingStrategy.factory(
      server_executor, target_executors)
  unplaced_executor = unplaced_ex_factory.create_executor(
      placement=placement_literals.SERVER)
  executor = federating_executor.FederatingExecutor(federating_strategy_factory,
                                                    unplaced_executor)
  return _wrap_executor_in_threading_stack(executor)


def _aggregate_stacks(
    executors: Sequence[executor_base.Executor], max_fanout: int,
    unplaced_ex_factory: UnplacedExecutorFactory) -> executor_base.Executor:
  """Aggregates multiple stacks into a single composite executor.

  Args:
    executors: Executors to aggregate as a `list`.
    max_fanout: The max fanout (see below).
    unplaced_ex_factory: The unplaced executor factory to use in constructing
      executors to execute unplaced computations in the hierarchy.

  Returns:
    An executor stack, potentially multi-level, that spans all `executors`.

  Raises:
    RuntimeError: If it can't create composite executors.
  """
  py_typecheck.check_type(executors, list)
  py_typecheck.check_type(max_fanout, int)
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  for ex in executors:
    py_typecheck.check_type(ex, executor_base.Executor)
  # Recursively construct as many levels as it takes to support all clients,
  # reducing by the factor of `max_fanout` in each iteration, for up to
  # `log(len(address_list)) / log(max_fanout)` iterations.
  while len(executors) > 1:
    new_executors = []
    offset = 0
    while offset < len(executors):
      new_offset = offset + max_fanout
      new_executors.append(
          _create_composite_stack(
              executors[offset:new_offset],
              unplaced_ex_factory=unplaced_ex_factory))
      offset = new_offset
    executors = new_executors
  if len(executors) != 1:
    raise RuntimeError('Expected 1 executor, got {}.'.format(len(executors)))
  return executors[0]


def _create_full_stack(
    cardinalities: executor_factory.CardinalitiesType,
    max_fanout: int,
    stack_func: Callable[[executor_factory.CardinalitiesType],
                         executor_base.Executor],
    unplaced_ex_factory: UnplacedExecutorFactory,
) -> executor_base.Executor:
  """Creates a full executor stack.

  Args:
    cardinalities: The cardinalities to create at each placement.
    max_fanout: The maximum fanout at any point in the hierarchy. Must be 2 or
      larger.
    stack_func: A function taking a dict of cardinalities and returning an
      `executor_base.Executor`.
    unplaced_ex_factory: The unplaced executor factory to use in constructing
      executors to execute unplaced computations in the hierarchy.

  Returns:
    An executor stack, potentially multi-level, that spans all clients.

  Raises:
    ValueError: If the number of clients or fanout are not as specified.
    RuntimeError: If the stack construction fails.
  """
  num_clients = cardinalities.get(placement_literals.CLIENTS, 0)
  py_typecheck.check_type(max_fanout, int)
  if num_clients < 0:
    raise ValueError('Number of clients cannot be negative.')
  if num_clients < 1:
    return stack_func(cardinalities=cardinalities)  # pytype: disable=wrong-keyword-args
  else:
    executors = []
    while num_clients > 0:
      n = min(num_clients, max_fanout)
      executors.append(
          stack_func(cardinalities={placement_literals.CLIENTS: n}))  # pytype: disable=wrong-keyword-args
      num_clients -= n
    return _aggregate_stacks(executors, max_fanout, unplaced_ex_factory)


def local_executor_factory(
    num_clients=None,
    max_fanout=100,
    clients_per_thread=1,
    server_tf_device=None,
    client_tf_devices=tuple()
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
      concurrency of the TFF runtime, which can be useful if client work is
      very lightweight or models are very large and multiple copies cannot fit
      in memory.
    server_tf_device: A `tf.config.LogicalDevice` to place server and other
      computation without explicit TFF placement.
    client_tf_devices: List/tuple of `tf.config.LogicalDevice` to place clients
      for simulation. Possibly accelerators returned by
      `tf.config.list_logical_devices()`.

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
      use_caching=True,
      server_device=server_tf_device,
      client_devices=client_tf_devices)
  federating_executor_factory = FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=False)

  def _factory_fn(
      cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    return _create_full_stack(
        cardinalities,
        max_fanout,
        stack_func=federating_executor_factory.create_executor,
        unplaced_ex_factory=unplaced_ex_factory)

  return executor_factory.ExecutorFactoryImpl(_factory_fn)


def sizing_executor_factory(
    num_clients: int = None,
    max_fanout: int = 100,
    clients_per_thread: int = 1) -> executor_factory.ExecutorFactory:
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
  unplaced_ex_factory = UnplacedExecutorFactory(use_caching=True)
  federating_executor_factory = FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=True)

  def _factory_fn(
      cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    executor = _create_full_stack(
        cardinalities,
        max_fanout,
        stack_func=federating_executor_factory.create_executor,
        unplaced_ex_factory=unplaced_ex_factory)
    sizing_executor_list = federating_executor_factory.sizing_executors
    return executor, sizing_executor_list

  return executor_factory.SizingExecutorFactory(_factory_fn)


def worker_pool_executor_factory(executors,
                                 max_fanout=100
                                ) -> executor_factory.ExecutorFactory:
  """Create an executor backed by a worker pool.

  Args:
    executors: A list of `tff.framework.Executor` instances that forward work to
      workers in the worker pool. These can be any type of executors, but in
      most scenarios, they will be instances of `tff.framework.RemoteExecutor`.
    max_fanout: The maximum fanout at any point in the aggregation hierarchy. If
      `num_clients > max_fanout`, the constructed executor stack will consist of
      multiple levels of aggregators. The height of the stack will be on the
      order of `log(num_clients) / log(max_fanout)`.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.
  """
  py_typecheck.check_type(executors, list)
  py_typecheck.check_type(max_fanout, int)
  if not executors:
    raise ValueError('The list executors cannot be empty.')
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  executors = [_wrap_executor_in_threading_stack(e) for e in executors]
  unplaced_ex_factory = UnplacedExecutorFactory(use_caching=True)

  def _stack_fn(cardinalities):
    del cardinalities  # Unused
    return _aggregate_stacks(
        executors, max_fanout, unplaced_ex_factory=unplaced_ex_factory)

  return executor_factory.ExecutorFactoryImpl(executor_stack_fn=_stack_fn)
