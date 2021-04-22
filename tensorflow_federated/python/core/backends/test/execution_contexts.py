# Copyright 2018, The TensorFlow Federated Authors.
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
"""Execution contexts for the test backend."""

from tensorflow_federated.python.core.backends.test import executor_stacks
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context


def create_test_execution_context(num_clients=None,
                                  clients_per_thread=1,
                                  *,
                                  default_num_clients: int = 0):
  """Creates an execution context that executes computations locally."""
  factory = executor_stacks.test_executor_factory(
      num_clients=num_clients,
      clients_per_thread=clients_per_thread,
      default_num_clients=default_num_clients)
  return execution_context.ExecutionContext(executor_fn=factory)


def set_test_execution_context(num_clients=None,
                               clients_per_thread=1,
                               *,
                               default_num_clients: int = 0):
  """Sets an execution context that executes computations locally."""
  context = create_test_execution_context(
      num_clients=num_clients,
      clients_per_thread=clients_per_thread,
      default_num_clients=default_num_clients)
  context_stack_impl.context_stack.set_default_context(context)
