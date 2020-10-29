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

from typing import Optional

from tensorflow_federated.python.core.backends.test import federated_strategy
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_stacks


def test_executor_factory(
    num_clients: Optional[int] = None,
    clients_per_thread: int = 1) -> executor_factory.ExecutorFactory:
  """Constructs a test execution stack to execute local computations.

  This factory is similar to `tff.framework.thread_debugging_executor_factory`
  except that it is configured to delegate the implementation of federated
  intrinsics to a `federated_strategy.TestFederatedStrategy`.

  This execution stack can be useful when testing federated algorithms that
  require unique implementations for the intrinsics provided by TFF.

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

  Returns:
    An `executor_factory.ExecutorFactory`.
  """
  unplaced_ex_factory = executor_stacks.UnplacedExecutorFactory(
      use_caching=False,
      can_resolve_references=True,
  )
  federating_executor_factory = executor_stacks.FederatingExecutorFactory(
      clients_per_thread=clients_per_thread,
      unplaced_ex_factory=unplaced_ex_factory,
      num_clients=num_clients,
      use_sizing=False,
      federated_strategy_factory=federated_strategy.TestFederatedStrategy
      .factory)

  return executor_stacks.ResourceManagingExecutorFactory(
      federating_executor_factory.create_executor)
