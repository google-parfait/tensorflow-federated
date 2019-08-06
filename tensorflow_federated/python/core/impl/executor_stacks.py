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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import caching_executor
from tensorflow_federated.python.core.impl import concurrent_executor
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import federated_executor
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl import placement_literals


def create_local_executor(num_clients=None):
  """Constructs an executor to execute computations on the local machine.

  The initial temporary implementation requires that the number of clients be
  specified in advance. This limitation will be removed in the near future.

  NOTE: This function is only available in Python 3.

  Args:
    num_clients: The number of clients. If not specified (`None`), then this
      executor is not federated (can only execute unplaced computations).

  Returns:
    An instance of `tff.framework.Executor` for single-machine use only.

  Raises:
    ValueError: If the number of clients is not one or larger.
  """

  def _create_single_worker_stack():
    ex = eager_executor.EagerExecutor()
    ex = concurrent_executor.ConcurrentExecutor(ex)
    ex = caching_executor.CachingExecutor(ex)
    return lambda_executor.LambdaExecutor(ex)

  if num_clients is None:
    return _create_single_worker_stack()
  else:
    # TODO(b/134543154): We shouldn't have to specif the number of clients; this
    # needs to go away once we flesh out all the remaining bits ad pieces.
    py_typecheck.check_type(num_clients, int)
    if num_clients < 1:
      raise ValueError('If the number of clients is present, it must be >= 1.')

    def _create_multiple_worker_stacks(num_workers):
      return [_create_single_worker_stack() for _ in range(num_workers)]

    return lambda_executor.LambdaExecutor(
        caching_executor.CachingExecutor(
            federated_executor.FederatedExecutor({
                None:
                    _create_multiple_worker_stacks(1),
                placement_literals.SERVER:
                    _create_multiple_worker_stacks(1),
                placement_literals.CLIENTS:
                    (_create_multiple_worker_stacks(num_clients))
            })))
