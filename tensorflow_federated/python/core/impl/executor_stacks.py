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
from tensorflow_federated.python.core.impl import concurrent_executor
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import federated_executor
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl.compiler import placement_literals


def _create_single_worker_stack():
  ex = eager_executor.EagerExecutor()
  ex = concurrent_executor.ConcurrentExecutor(ex)
  ex = caching_executor.CachingExecutor(ex)
  return lambda_executor.LambdaExecutor(ex)


def _create_multiple_worker_stacks(num_workers):
  return [_create_single_worker_stack() for _ in range(num_workers)]


def _create_explicit_cardinality_executor_fn(num_clients):
  """Creates executor function with fixed cardinality."""
  executor_dict = {
      None: _create_multiple_worker_stacks(1),
      placement_literals.SERVER: _create_multiple_worker_stacks(1),
      placement_literals.CLIENTS: _create_multiple_worker_stacks(num_clients)
  }

  def _return_executor(x):
    del x  # Unused
    return lambda_executor.LambdaExecutor(
        caching_executor.CachingExecutor(
            federated_executor.FederatedExecutor(executor_dict)))

  return _return_executor


def _create_inferred_cardinality_executor_fn():
  """Creates executor function with variable cardinality."""

  def _create_variable_clients_executors(x):
    """Constructs executor stacks from `dict` argument."""
    py_typecheck.check_type(x, dict)
    for k, v in six.iteritems(x):
      py_typecheck.check_type(k, placement_literals.PlacementLiteral)
      if v <= 0:
        raise ValueError(
            'Cardinality must be at '
            'least one; you have passed {} for placement {}.'.format(v, k))
    executor_dict = dict([(placement, _create_multiple_worker_stacks(n))
                          for placement, n in six.iteritems(x)])
    executor_dict.update({None: _create_multiple_worker_stacks(1)})
    executor_dict.update(
        {placement_literals.SERVER: _create_multiple_worker_stacks(1)})
    return lambda_executor.LambdaExecutor(
        caching_executor.CachingExecutor(
            federated_executor.FederatedExecutor(executor_dict)))

  return _create_variable_clients_executors


def create_local_executor(num_clients=None):
  """Constructs an executor to execute computations on the local machine.

  NOTE: This function is only available in Python 3.

  Args:
    num_clients: The number of clients. If specified, the executor factory
      function returned by `create_local_executor` will be configured to have
      exactly `num_clients` clients. If unspecified (`None`), then the function
      returned will attempt to infer cardinalities of all placements for which
      it is passed values.

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
    return _create_explicit_cardinality_executor_fn(num_clients)
  else:
    return _create_inferred_cardinality_executor_fn()
