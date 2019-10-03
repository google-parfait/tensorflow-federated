# Lint as: python3
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

import six

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import caching_executor
from tensorflow_federated.python.core.impl import composite_executor
from tensorflow_federated.python.core.impl import concurrent_executor
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import federated_executor
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl.compiler import placement_literals


def _complete_stack(ex):
  return lambda_executor.LambdaExecutor(
      caching_executor.CachingExecutor(
          concurrent_executor.ConcurrentExecutor(ex)))


def _create_bottom_stack():
  return _complete_stack(eager_executor.EagerExecutor())


def _create_federated_stack(num_clients):
  executor_dict = {
      placement_literals.CLIENTS: [
          _create_bottom_stack() for _ in range(num_clients)
      ],
      placement_literals.SERVER: _create_bottom_stack(),
      None: _create_bottom_stack()
  }
  return _complete_stack(federated_executor.FederatedExecutor(executor_dict))


def _create_composite_stack(children):
  return _complete_stack(
      composite_executor.CompositeExecutor(_create_bottom_stack(), children))


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


def _create_full_stack(num_clients, max_fanout):
  """Creates a full executor stack.

  Args:
    num_clients: The number of clients to support. Must be 0 or larger.
    max_fanout: The maximum fanout at any point in the hierarchy. Must be 1 or
      larger.

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
  if max_fanout < 1:
    raise ValueError('Max fanout must be positive.')
  if num_clients < 1:
    return _create_federated_stack(0)
  else:
    executors = []
    while num_clients > 0:
      n = min(num_clients, max_fanout)
      executors.append(_create_federated_stack(n))
      num_clients -= n
    return _aggregate_stacks(executors, max_fanout)


def _create_explicit_cardinality_executor_fn(num_clients, max_fanout):
  """Creates executor function with fixed cardinality."""

  def _return_executor(_):
    return _create_full_stack(num_clients, max_fanout)

  return _return_executor


def _create_inferred_cardinality_executor_fn(max_fanout):
  """Creates executor function with variable cardinality."""

  def _create_variable_clients_executors(cardinalities):
    """Constructs executor stacks from `dict` argument."""
    py_typecheck.check_type(cardinalities, dict)
    for k, v in six.iteritems(cardinalities):
      py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      if k not in [placement_literals.CLIENTS, placement_literals.SERVER]:
        raise ValueError('Unsupported placement: {}.'.format(k))
      if v <= 0:
        raise ValueError(
            'Cardinality must be at '
            'least one; you have passed {} for placement {}.'.format(v, k))

    return _create_full_stack(
        cardinalities.get(placement_literals.CLIENTS, 0), max_fanout)

  return _create_variable_clients_executors


def create_local_executor(num_clients=None, max_fanout=100):
  """Constructs an executor to execute computations on the local machine.

  NOTE: This function is only available in Python 3.

  Args:
    num_clients: The number of clients. If specified, the executor factory
      function returned by `create_local_executor` will be configured to have
      exactly `num_clients` clients. If unspecified (`None`), then the function
      returned will attempt to infer cardinalities of all placements for which
      it is passed values.
    max_fanout: The maximum fanout at any point in the aggregation hierarchy.
      If `num_clients > max_fanout`, the constructed executor stack will consist
      of multiple levels of aggregators. The height of the stack will be on the
      order of `log(num_clients) / log(max_fanout)`.

  Returns:
    An executor factory function which returns a
    `tff.framework.Executor` upon invocation with a dict mapping placements
    to positive integers.

  Raises:
    ValueError: If the number of clients is specified and not one or larger.
  """
  # TODO(b/140112504): Follow up with an ExecutorFactory abstract class.

  if num_clients is not None:
    py_typecheck.check_type(num_clients, int)
    if num_clients <= 0:
      raise ValueError('If specifying `num_clients`, cardinality must be at '
                       'least one; you have passed {}.'.format(num_clients))
    return _create_explicit_cardinality_executor_fn(num_clients, max_fanout)
  else:
    return _create_inferred_cardinality_executor_fn(max_fanout)


def create_worker_pool_executor(executors, max_fanout=100):
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
    An instance of `tff.framework.Executor`.
  """
  py_typecheck.check_type(executors, list)
  py_typecheck.check_type(max_fanout, int)
  if not executors:
    raise ValueError('The list executors cannot be empty.')
  if max_fanout < 1:
    raise ValueError('Max fanout must be positive.')
  executors = [_complete_stack(e) for e in executors]
  return _aggregate_stacks(executors, max_fanout)
