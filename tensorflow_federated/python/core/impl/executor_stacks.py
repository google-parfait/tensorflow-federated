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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import concurrent_executor
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import federated_executor
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl import placement_literals


def create_local_executor(num_clients):
  """Constructs an executor to execute computations on the local machine.

  The initial temporary implementation requires that the number of clients be
  specified in advance. This limitation will be removed in the near future.

  Args:
    num_clients: The number of clients.

  Returns:
    An instance of `tff.framework.Executor` for single-machine use only.
  """
  # TODO(b/134543154): We should not have to specif the number of clients; this
  # needs to go away once we flesh out all the remaining bits ad pieces.

  py_typecheck.check_type(num_clients, int)
  bottom_ex = lambda_executor.LambdaExecutor(eager_executor.EagerExecutor())

  def _make(n):
    return [concurrent_executor.ConcurrentExecutor(bottom_ex) for _ in range(n)]

  return lambda_executor.LambdaExecutor(
      federated_executor.FederatedExecutor({
          None: _make(1),
          placement_literals.SERVER: _make(1),
          placement_literals.CLIENTS: _make(num_clients)
      }))
