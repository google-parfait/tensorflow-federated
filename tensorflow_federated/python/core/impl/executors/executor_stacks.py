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

import functools
from typing import List, Tuple

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import caching_executor
from tensorflow_federated.python.core.impl.executors import composing_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.executors import sizing_executor
from tensorflow_federated.python.core.impl.executors import thread_delegating_executor
from tensorflow_federated.python.core.impl.types import placement_literals


def _complete_stack(ex):
  return reference_resolving_executor.ReferenceResolvingExecutor(
      caching_executor.CachingExecutor(
          thread_delegating_executor.ThreadDelegatingExecutor(ex)))


def _create_bottom_stack(device=None):
  return _complete_stack(eager_tf_executor.EagerTFExecutor(device=device))


def _create_federated_stack(num_clients, num_client_executors,
                            device_scheduler):
  """Constructs local federated stack."""
  client_bottom_stacks = [
      _create_bottom_stack(device=device_scheduler.next_client_device())
      for _ in range(num_client_executors)
  ]
  executor_dict = {
      placement_literals.CLIENTS: [
          client_bottom_stacks[k % len(client_bottom_stacks)]
          for k in range(num_clients)
      ],
      placement_literals.SERVER:
          _create_bottom_stack(device=device_scheduler.server_device()),
      None:
          _create_bottom_stack(device=device_scheduler.server_device())
  }
  return _complete_stack(federating_executor.FederatingExecutor(executor_dict))


def _create_sizing_stack(num_clients, num_client_executors):
  """Constructs local stack with sizing wired in at each client."""
  sizing_stacks = [
      sizing_executor.SizingExecutor(_create_bottom_stack())
      for _ in range(num_client_executors)
  ]
  executor_dict = {
      placement_literals.CLIENTS: [
          sizing_stacks[k % len(sizing_stacks)] for k in range(num_clients)
      ],
      placement_literals.SERVER: _create_bottom_stack(),
      None: _create_bottom_stack()
  }
  return _complete_stack(
      federating_executor.FederatingExecutor(executor_dict)), sizing_stacks


def _create_composite_stack(children):
  return _complete_stack(
      composing_executor.ComposingExecutor(_create_bottom_stack(), children))


def _aggregate_stacks(executors, max_fanout):
  """Aggregates multiple stacks into a single composite executor.

  Args:
    executors: Executors to aggregate as a `list`.
    max_fanout: The max fanout (see below).

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
          _create_composite_stack(executors[offset:new_offset]))
      offset = new_offset
    executors = new_executors
  if len(executors) != 1:
    raise RuntimeError('Expected 1 executor, got {}.'.format(len(executors)))
  return executors[0]


def _create_full_stack(num_clients, max_fanout, stack_func,
                       num_client_executors):
  """Creates a full executor stack.

  Args:
    num_clients: The number of clients to support. Must be 0 or larger.
    max_fanout: The maximum fanout at any point in the hierarchy. Must be 2 or
      larger.
    stack_func: A function taking one argument which is the number of clients
      and returns an executor_base.Executor.
    num_client_executors: The maximum number of threads spun up locally to
      represent clients.

  Returns:
    An executor stack, potentially multi-level, that spans all clients.

  Raises:
    ValueError: If the number of clients or fanout are not as specified.
    RuntimeError: If the stack construction fails.
  """
  py_typecheck.check_type(num_clients, int)
  py_typecheck.check_type(max_fanout, int)
  if num_clients < 0:
    raise ValueError('Number of clients cannot be negative.')
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  if num_clients < 1:
    return stack_func(0, 1)
  else:
    executors = []
    while num_clients > 0:
      n = min(num_clients, max_fanout)
      executors.append(stack_func(n, num_client_executors))
      num_clients -= n
    return _aggregate_stacks(executors, max_fanout)


def _create_explicit_cardinality_factory(
    num_clients, max_fanout, stack_func,
    num_client_executors) -> executor_factory.ExecutorFactory:
  """Creates executor function with fixed cardinality."""

  def _return_executor(cardinalities):
    n_requested_clients = cardinalities.get(placement_literals.CLIENTS)
    if n_requested_clients is not None and n_requested_clients != num_clients:
      raise ValueError('Expected to construct an executor with {} clients, '
                       'but executor is hardcoded for {}'.format(
                           n_requested_clients, num_clients))
    return _create_full_stack(num_clients, max_fanout, stack_func,
                              num_client_executors)

  return executor_factory.ExecutorFactoryImpl(
      executor_stack_fn=_return_executor)


def _create_inferred_cardinality_factory(
    max_fanout, stack_func,
    num_client_executors) -> executor_factory.ExecutorFactory:
  """Creates executor function with variable cardinality."""

  def _create_variable_clients_executors(cardinalities):
    """Constructs executor stacks from `dict` argument."""
    py_typecheck.check_type(cardinalities, dict)
    for k, v in cardinalities.items():
      py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      if k not in [placement_literals.CLIENTS, placement_literals.SERVER]:
        raise ValueError('Unsupported placement: {}.'.format(k))
      if v <= 0:
        raise ValueError(
            'Cardinality must be at '
            'least one; you have passed {} for placement {}.'.format(v, k))

    return _create_full_stack(
        cardinalities.get(placement_literals.CLIENTS, 0), max_fanout,
        stack_func, num_client_executors)

  return executor_factory.ExecutorFactoryImpl(
      executor_stack_fn=_create_variable_clients_executors)


class _DeviceScheduler():
  """Assign server and clients to devices. Useful in multi-GPU environment."""

  def __init__(self, server_tf_device, client_tf_devices):
    """Initialize with server and client TF device placement.

    Args:
      server_tf_device: A `tf.config.LogicalDevice` to place server and other
        computation without explicit TFF placement.
      client_tf_devices: List/tuple of `tf.config.LogicalDevice` to place
        clients for simulation. Possibly accelerators returned by
        `tf.config.list_logical_devices()`.
    """
    py_typecheck.check_type(client_tf_devices, (tuple, list))
    for device in client_tf_devices:
      py_typecheck.check_type(device, tf.config.LogicalDevice)
    self._client_devices = [d.name for d in client_tf_devices]
    self._idx = 0
    if server_tf_device is None:
      self._server_device = None
    else:
      py_typecheck.check_type(server_tf_device, tf.config.LogicalDevice)
      self._server_device = server_tf_device.name

  def next_client_device(self):
    """Gets a device to place the next client in cyclic order."""
    if len(self._client_devices) < 1:
      return None
    self._idx = (self._idx + 1) % len(self._client_devices)
    return self._client_devices[self._idx]

  def server_device(self):
    return self._server_device


def local_executor_factory(
    num_clients=None,
    max_fanout=100,
    num_client_executors=32,
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
    num_client_executors: The number of distinct client executors to run
      concurrently; executing more clients than this number results in
      multiple clients having their work pinned on a single executor in a
      synchronous fashion.
    server_tf_device: A `tf.config.LogicalDevice` to place server and
      other computation without explicit TFF placement.
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
  device_scheduler = _DeviceScheduler(server_tf_device, client_tf_devices)
  stack_func = functools.partial(
      _create_federated_stack, device_scheduler=device_scheduler)
  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')
  if num_clients is not None:
    py_typecheck.check_type(num_clients, int)
    if num_clients <= 0:
      raise ValueError('If specifying `num_clients`, cardinality must be at '
                       'least one; you have passed {}.'.format(num_clients))
    return _create_explicit_cardinality_factory(num_clients, max_fanout,
                                                stack_func,
                                                num_client_executors)
  else:
    return _create_inferred_cardinality_factory(max_fanout, stack_func,
                                                num_client_executors)


def sizing_executor_factory(
    num_clients: int = None,
    max_fanout: int = 100,
    num_client_executors: int = 1) -> executor_factory.ExecutorFactory:
  """Constructs an executor to execute computations on the local machine with sizing.

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
    num_client_executors: The number of distinct client executors to run
      concurrently; executing more clients than this number results in
      multiple clients having their work pinned on a single executor in a
      synchronous fashion.

  Returns:
    An instance of `executor_factory.ExecutorFactory` encapsulating the
    executor construction logic specified above.

  Raises:
    ValueError: If the number of clients is specified and not one or larger.
  """

  def _executor_stack_fn(
      cardinalities: executor_factory.CardinalitiesType
  ) -> Tuple[executor_base.Executor, List[sizing_executor.SizingExecutor]]:
    """The function passed to SizingExecutorFactoryImpl to convert cardinalities into executor stack.

    Unlike the function that is passed into ExecutorFactoryImpl, this one
    outputs the sizing executors as well.

    Args:
      cardinalities: Cardinality representation used to determine how many
        clients there are.

    Returns:
      A Tuple of the top level executor created from the cardinalities, and the
      list of sizing executors underneath the top level executors.
    """
    sizing_exs = []

    def _standalone_stack_func(num_clients, num_client_executors):
      # pylint: disable= unused-variable
      nonlocal sizing_exs
      stack, current_sizing_exs = _create_sizing_stack(num_clients,
                                                       num_client_executors)
      sizing_exs.extend(current_sizing_exs)

      # pylint: enable= unused-variable
      return stack

    # Explicit case.
    if num_clients is not None:
      py_typecheck.check_type(num_clients, int)
      if num_clients <= 0:
        raise ValueError('If specifying `num_clients`, cardinality must be at '
                         'least one; you have passed {}.'.format(num_clients))
      n_requested_clients = cardinalities.get(placement_literals.CLIENTS)
      if n_requested_clients is not None and n_requested_clients != num_clients:
        raise ValueError('Expected to construct an executor with {} clients, '
                         'but executor is hardcoded for {}'.format(
                             n_requested_clients, num_clients))
      return _create_full_stack(num_clients, max_fanout, _standalone_stack_func,
                                num_client_executors), sizing_exs
    # Inferred case.
    else:
      n_requested_clients = cardinalities.get(placement_literals.CLIENTS, 0)
      return _create_full_stack(n_requested_clients, max_fanout,
                                _standalone_stack_func,
                                num_client_executors), sizing_exs

  if max_fanout < 2:
    raise ValueError('Max fanout must be greater than 1.')

  return executor_factory.SizingExecutorFactoryImpl(_executor_stack_fn)


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
  executors = [_complete_stack(e) for e in executors]

  def _stack_fn(cardinalities):
    del cardinalities  # Unused
    return _aggregate_stacks(executors, max_fanout)

  return executor_factory.ExecutorFactoryImpl(executor_stack_fn=_stack_fn)
