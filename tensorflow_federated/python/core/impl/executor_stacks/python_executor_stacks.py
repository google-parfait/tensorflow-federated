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

from collections.abc import Callable, Sequence
from concurrent import futures
import math
from typing import Optional, Union
import warnings

from absl import logging
import cachetools
import grpc
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import local_computation_factory_base
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.executors import federated_composing_strategy
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.executors import remote_executor_grpc_stub
from tensorflow_federated.python.core.impl.executors import remote_executor_stub
from tensorflow_federated.python.core.impl.executors import sequence_executor
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
      executor_stack_fn: Callable[
          [executor_factory.CardinalitiesType], executor_base.Executor
      ],
  ):
    """Initializes `ResourceManagingExecutorFactory`.

    `ResourceManagingExecutorFactory` manages a mapping from `cardinalities`
    to `executor_base.Executors`, closing and destroying the executors in this
    mapping when asked.

    Args:
      executor_stack_fn: Callable taking a mapping from
        `placements.PlacementLiteral` to integers, and returning an
        `executor_base.Executor`. The returned executor will be configured to
        handle these cardinalities.
    """

    py_typecheck.check_callable(executor_stack_fn)
    self._executor_stack_fn = executor_stack_fn
    self._executors = cachetools.LRUCache(_EXECUTOR_CACHE_SIZE)

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Constructs or gets existing executor.

    Returns a previously-constructed executor if this method has already been
    invoked with `cardinalities`. If not, invokes `self._executor_stack_fn`
    with `cardinalities` and returns the result.

    Args:
      cardinalities: `dict` with `placements.PlacementLiteral` keys and integer
        values, specifying the population size at each placement. The executor
        stacks returned from this method are not themselves polymorphic; a
        concrete stack must have fixed sizes at each placement.

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

  def clean_up_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ):
    """Calls `close` on constructed executors, resetting internal cache.

    If a caller holds a name bound to any of the executors returned from
    `create_executor` with the specified cardinalities, this executor
    should be assumed to be in an invalid state, and should not be used after
    this method is called. Instead, callers should again invoke
    `create_executor`.

    Args:
      cardinalities: The cardinalities of the executor to be cleaned up.
    """
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is None:
      return
    ex.close()
    del self._executors[key]


# pylint: disable=missing-function-docstring
def _wrap_executor_in_threading_stack(
    ex: executor_base.Executor,
    support_sequence_ops: bool = False,
    can_resolve_references=True,
):
  threaded_ex = thread_delegating_executor.ThreadDelegatingExecutor(ex)
  if support_sequence_ops:
    if not can_resolve_references:
      raise ValueError(
          'Support for sequence ops requires ability to resolve references.'
      )
    threaded_ex = sequence_executor.SequenceExecutor(
        reference_resolving_executor.ReferenceResolvingExecutor(threaded_ex)
    )
  if can_resolve_references:
    threaded_ex = reference_resolving_executor.ReferenceResolvingExecutor(
        threaded_ex
    )
  return threaded_ex


# pylint: enable=missing-function-docstring


class UnplacedExecutorFactory:
  """ExecutorFactory to construct executors which cannot understand placement.

  This factory constructs executors which represent "local execution": work
  that happens at the clients, at the server, or without placements. As such,
  this executor manages the placement of work on local executors.
  """

  def __init__(
      self,
      *,
      support_sequence_ops: bool = False,
      can_resolve_references: bool = True,
      server_device: Optional[tf.config.LogicalDevice] = None,
      client_devices: Optional[Sequence[tf.config.LogicalDevice]] = (),
      leaf_executor_fn=eager_tf_executor.EagerTFExecutor,
  ):
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
        self._client_devices
    )
    return device

  def create_executor(
      self,
      *,
      cardinalities: Optional[executor_factory.CardinalitiesType] = None,
      placement: Optional[placements.PlacementLiteral] = None,
  ) -> executor_base.Executor:
    """Constructs instance of `executor_base.Executor`."""
    if cardinalities:
      raise ValueError(
          'Unplaced executors cannot accept nonempty cardinalities as '
          'arguments. Received cardinalities: {}.'.format(cardinalities)
      )
    if placement == placements.CLIENTS:
      device = self._get_next_client_device()
    elif placement == placements.SERVER:
      device = self._server_device
    else:
      device = None
    leaf_ex = self._leaf_executor_fn(device=device)
    return _wrap_executor_in_threading_stack(
        leaf_ex,
        support_sequence_ops=self._support_sequence_ops,
        can_resolve_references=self._can_resolve_references,
    )

  def clean_up_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ):
    # Does not hold any executors internally, so nothing to clean up.
    pass


class FederatingExecutorFactory(executor_factory.ExecutorFactory):
  """Executor factory for stacks which delegate placed computations.

  `FederatingExecutorFactory` validates cardinality requests and manages
  the relationship between the clients and the client executors.

  This factory is initialized with:
    * An integer number of clients per thread, indicating the number of client
      executors that should be run in a single thread. Setting this parameter to
      a high number can aid with OOMs on accelerators, or speed up the
      computation in the case of extremely lightweight client work.
    * An `UnplacedExecutorFactory` to use to construct the executors for
      computations after they have had their placement ingested and stripped by
      the `FederatingExecutor`. That is, this factory produces the executors
      used to run client, server and unplaced computations.
    * A number of default clients. If client cardinalities cannot be inferred
      from data, this value will be used as the default number of clients to
      be supported by the returned executors.
    * An optional instance of `LocalComputationFactory` to use to construct
      local computations used as parameters in certain federated operators
      (such as `tff.federated_sum`, etc.). Defaults to a TensorFlow factory.
  """

  def __init__(
      self,
      *,
      clients_per_thread: int,
      unplaced_ex_factory: UnplacedExecutorFactory,
      default_num_clients: int = 0,
      local_computation_factory: local_computation_factory_base.LocalComputationFactory = tensorflow_computation_factory.TensorFlowComputationFactory(),
      federated_strategy_factory=federated_resolving_strategy.FederatedResolvingStrategy.factory,
  ):
    py_typecheck.check_type(clients_per_thread, int)
    py_typecheck.check_type(unplaced_ex_factory, UnplacedExecutorFactory)
    py_typecheck.check_type(
        local_computation_factory,
        local_computation_factory_base.LocalComputationFactory,
    )
    self._clients_per_thread = clients_per_thread
    self._unplaced_executor_factory = unplaced_ex_factory
    py_typecheck.check_type(default_num_clients, int)
    if default_num_clients < 0:
      raise ValueError('Number of clients must be nonnegative.')
    self._default_num_clients = default_num_clients
    self._federated_strategy_factory = federated_strategy_factory
    self._local_computation_factory = local_computation_factory

  def _validate_requested_clients(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> int:
    num_requested_clients = cardinalities.get(placements.CLIENTS)
    if num_requested_clients is None:
      return self._default_num_clients
    return num_requested_clients

  def create_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ) -> executor_base.Executor:
    """Constructs a federated executor with requested cardinalities."""
    num_clients = self._validate_requested_clients(cardinalities)
    num_client_executors = math.ceil(num_clients / self._clients_per_thread)
    client_stacks = [
        self._unplaced_executor_factory.create_executor(
            cardinalities={}, placement=placements.CLIENTS
        )
        for _ in range(num_client_executors)
    ]

    federating_strategy_factory = self._federated_strategy_factory(
        {
            placements.CLIENTS: [
                client_stacks[k % len(client_stacks)]
                for k in range(num_clients)
            ],
            placements.SERVER: self._unplaced_executor_factory.create_executor(
                placement=placements.SERVER
            ),
        },
        local_computation_factory=self._local_computation_factory,
    )
    unplaced_executor = self._unplaced_executor_factory.create_executor()
    executor = federating_executor.FederatingExecutor(
        federating_strategy_factory, unplaced_executor
    )
    return _wrap_executor_in_threading_stack(executor)

  def clean_up_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ):
    # Does not hold any executors internally, so nothing to clean up.
    pass


def create_minimal_length_flat_stack_fn(
    max_clients_per_stack: int,
    federated_stack_factory: executor_factory.ExecutorFactory,
) -> Callable[
    [executor_factory.CardinalitiesType], list[executor_base.Executor]
]:
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
      cardinalities: executor_factory.CardinalitiesType,
  ) -> list[executor_base.Executor]:
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
          federated_stack_factory.create_executor(sub_executor_cardinalities)
      )
      num_clients -= n
    return executors

  return create_executor_list


class ComposingExecutorFactory(executor_factory.ExecutorFactory):
  """Factory class encapsulating executor compositional logic.

  This class is responsible for aggregating lists of executors into a
  compositional hierarchy based on the `max_fanout` parameter.
  """

  def __init__(
      self,
      *,
      max_fanout: int,
      unplaced_ex_factory: UnplacedExecutorFactory,
      flat_stack_fn: Callable[
          [executor_factory.CardinalitiesType], Sequence[executor_base.Executor]
      ],
      local_computation_factory: local_computation_factory_base.LocalComputationFactory = tensorflow_computation_factory.TensorFlowComputationFactory(),
  ):
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

  def clean_up_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ):
    """Holds no executors internally, so passes on cleanup."""
    pass

  def _create_composing_stack(
      self, *, target_executors: Sequence[executor_base.Executor]
  ) -> executor_base.Executor:
    server_executor = self._unplaced_ex_factory.create_executor(
        placement=placements.SERVER
    )
    composing_strategy_factory = (
        federated_composing_strategy.FederatedComposingStrategy.factory(
            server_executor,
            target_executors,
            local_computation_factory=self._local_computation_factory,
        )
    )
    unplaced_executor = self._unplaced_ex_factory.create_executor()
    composing_executor = federating_executor.FederatingExecutor(
        composing_strategy_factory, unplaced_executor
    )
    threaded_composing_executor = _wrap_executor_in_threading_stack(
        composing_executor, can_resolve_references=False
    )
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
          self._create_composing_stack(target_executors=executors)
      )
    while len(executors) > 1:
      new_executors = []
      offset = 0
      while offset < len(executors):
        new_offset = offset + self._max_fanout
        target_executors = executors[offset:new_offset]
        composing_executor = self._create_composing_stack(
            target_executors=target_executors
        )
        new_executors.append(composing_executor)
        offset = new_offset
      executors = new_executors
    if len(executors) != 1:
      raise RuntimeError('Expected 1 executor, got {}.'.format(len(executors)))
    return reference_resolving_executor.ReferenceResolvingExecutor(executors[0])


def normalize_num_clients_and_default_num_clients(
    num_clients: Optional[int], default_num_clients: int
) -> int:
  if num_clients is not None:
    warnings.warn(
        'num_clients is deprecated; please use default_num_clients instead.'
    )
    py_typecheck.check_type(num_clients, int)
    return num_clients
  return default_num_clients


def local_executor_factory(
    default_num_clients: int = 0,
    max_fanout=100,
    clients_per_thread=1,
    server_tf_device=None,
    client_tf_devices=tuple(),
    reference_resolving_clients=True,
    support_sequence_ops=False,
    leaf_executor_fn=eager_tf_executor.EagerTFExecutor,
    local_computation_factory=tensorflow_computation_factory.TensorFlowComputationFactory(),
) -> executor_factory.ExecutorFactory:
  """Constructs an executor factory to execute computations locally.

  Note: The `tff.federated_secure_sum_bitwidth()` intrinsic is not implemented
  by this executor.

  Args:
    default_num_clients: The number of clients to run by default if cardinality
      cannot be inferred from arguments.
    max_fanout: The maximum fanout at any point in the aggregation hierarchy. If
      `num_cients > max_fanout`, the constructed executor stack will consist of
      multiple levels of aggregators. The height of the stack will be on the
      order of `log(num_cients) / log(max_fanout)`.
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
      is the eager TF executor (other possible options: XLA). Should accept the
      `device` keyword argument if the executor is to be configured with
      explicitly chosen devices.
    local_computation_factory: An instance of `LocalComputationFactory` to use
      to construct local computations used as parameters in certain federated
      operators (such as `tff.federated_sum`, etc.). Defaults to a TensorFlow
      computation factory that generates TensorFlow code.

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
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  unplaced_ex_factory = UnplacedExecutorFactory(
      support_sequence_ops=support_sequence_ops,
      can_resolve_references=reference_resolving_clients,
      server_device=server_tf_device,
      client_devices=client_tf_devices,
      leaf_executor_fn=leaf_executor_fn,
  )
  federating_executor_factory = FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      default_num_clients=default_num_clients,
      local_computation_factory=local_computation_factory,
  )
  flat_stack_fn = create_minimal_length_flat_stack_fn(
      max_fanout, federating_executor_factory
  )
  full_stack_factory = ComposingExecutorFactory(
      max_fanout=max_fanout,
      unplaced_ex_factory=unplaced_ex_factory,
      flat_stack_fn=flat_stack_fn,
      local_computation_factory=local_computation_factory,
  )

  def _factory_fn(cardinalities):
    if cardinalities.get(placements.CLIENTS, 0) < max_fanout:
      return federating_executor_factory.create_executor(cardinalities)
    return full_stack_factory.create_executor(cardinalities)

  return ResourceManagingExecutorFactory(_factory_fn)


class ReconstructOnChangeExecutorFactory(executor_factory.ExecutorFactory):
  """ExecutorFactory exposing hook to construct executors on environment change.

  When the initialization parameter `change_query` returns `True`,
  ReconstructOnChangeExecutorFactory` constructs a new executor, bypassing
  any previously constructed executors.
  """

  def __init__(
      self,
      underlying_stack: executor_factory.ExecutorFactory,
      change_query: Callable[
          [executor_factory.CardinalitiesType], bool
      ] = lambda _: True,
  ):
    self._change_query = change_query
    self._underlying_stack = underlying_stack
    self._executors = cachetools.LRUCache(_EXECUTOR_CACHE_SIZE)

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

  def clean_up_executor(
      self, cardinalities: executor_factory.CardinalitiesType
  ):
    key = _get_hashable_key(cardinalities)
    ex = self._executors.get(key)
    if ex is None:
      return
    ex.close()
    del self._executors[key]
    self._underlying_stack.clean_up_executor(cardinalities)


class _CardinalitiesOrReadyListChanged:
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


def _configure_remote_workers(
    default_num_clients,
    stubs,
    thread_pool_executor,
    dispose_batch_size,
    stream_structs: bool = False,
):
  """Configures `default_num_clients` across `remote_executors`."""
  available_stubs = [stub for stub in stubs if stub.is_ready]
  logging.info(
      '%s TFF workers available out of a total of %s.',
      len(available_stubs),
      len(stubs),
  )
  if not available_stubs:
    raise executors_errors.RetryableError(
        'No workers are ready; try again to reconnect.'
    )
  remaining_clients = default_num_clients
  live_workers = []
  for stub_idx, stub in enumerate(available_stubs):
    remaining_stubs = len(available_stubs) - stub_idx
    default_num_clients_to_host = remaining_clients // remaining_stubs
    remaining_clients -= default_num_clients_to_host
    if default_num_clients_to_host > 0:
      ex = remote_executor.RemoteExecutor(
          stub, thread_pool_executor, dispose_batch_size, stream_structs
      )
      ex.set_cardinalities({placements.CLIENTS: default_num_clients_to_host})
      live_workers.append(ex)
  return [
      _wrap_executor_in_threading_stack(e, can_resolve_references=False)
      for e in live_workers
  ]


def remote_executor_factory(
    channels: list[grpc.Channel],
    thread_pool_executor: Optional[futures.Executor] = None,
    dispose_batch_size: int = 20,
    max_fanout: int = 100,
    default_num_clients: int = 0,
    stream_structs: bool = False,
) -> executor_factory.ExecutorFactory:
  """Create an executor backed by remote workers.

  Args:
    channels: A list of `grpc.Channels` hosting services which can execute TFF
      work.
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
      order of `log(default_num_clients) / log(max_fanout)`.
    default_num_clients: The number of clients to use for simulations where the
      number of clients cannot be inferred. Usually the number of clients will
      be inferred from the number of values passed to computations which accept
      client-placed values. However, when this inference isn't possible (such as
      in the case of a no-argument or non-federated computation) this default
      will be used instead.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.
  """
  py_typecheck.check_type(channels, list)
  if not channels:
    raise ValueError('The list of channels cannot be empty.')
  if thread_pool_executor is not None:
    py_typecheck.check_type(thread_pool_executor, futures.Executor)
  py_typecheck.check_type(dispose_batch_size, int)
  py_typecheck.check_type(max_fanout, int)
  py_typecheck.check_type(default_num_clients, int)

  stubs = [
      remote_executor_grpc_stub.RemoteExecutorGrpcStub(channel)
      for channel in channels
  ]
  return remote_executor_factory_from_stubs(
      stubs,
      thread_pool_executor,
      dispose_batch_size,
      max_fanout,
      default_num_clients,
      stream_structs,
  )


def remote_executor_factory_from_stubs(
    stubs: list[
        Union[
            remote_executor_grpc_stub.RemoteExecutorGrpcStub,
            remote_executor_stub.RemoteExecutorStub,
        ]
    ],
    thread_pool_executor: Optional[futures.Executor] = None,
    dispose_batch_size: int = 20,
    max_fanout: int = 100,
    default_num_clients: int = 0,
    stream_structs: bool = False,
) -> executor_factory.ExecutorFactory:
  """Create an executor backed by remote workers.

  Args:
    stubs: A list stubs to the TFF executor service, running on remote machines.
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
      order of `log(default_num_clients) / log(max_fanout)`.
    default_num_clients: The number of clients to use for simulations where the
      number of clients cannot be inferred. Usually the number of clients will
      be inferred from the number of values passed to computations which accept
      client-placed values. However, when this inference isn't possible (such as
      in the case of a no-argument or non-federated computation) this default
      will be used instead.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.
  """
  py_typecheck.check_type(stubs, list)
  if not stubs:
    raise ValueError('The list of stubs cannot be empty.')
  if thread_pool_executor is not None:
    py_typecheck.check_type(thread_pool_executor, futures.Executor)
  py_typecheck.check_type(dispose_batch_size, int)
  py_typecheck.check_type(max_fanout, int)
  py_typecheck.check_type(default_num_clients, int)

  def _flat_stack_fn(cardinalities):
    num_clients = cardinalities.get(placements.CLIENTS, default_num_clients)
    return _configure_remote_workers(
        num_clients,
        stubs,
        thread_pool_executor,
        dispose_batch_size,
        stream_structs,
    )

  unplaced_ex_factory = UnplacedExecutorFactory()
  composing_executor_factory = ComposingExecutorFactory(
      max_fanout=max_fanout,
      unplaced_ex_factory=unplaced_ex_factory,
      flat_stack_fn=_flat_stack_fn,
  )

  return ReconstructOnChangeExecutorFactory(
      underlying_stack=composing_executor_factory,
      change_query=_CardinalitiesOrReadyListChanged(maybe_ready_list=stubs),
  )
