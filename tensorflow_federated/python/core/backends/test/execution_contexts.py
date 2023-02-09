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

from tensorflow_federated.python.common_libs import deprecation
from tensorflow_federated.python.core.backends.test import compiler
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import python_executor_stacks


# TODO(b/240972950): Remove deprecated API.
@deprecation.deprecated(
    '`tff.backends.test.create_test_python_execution_context` is deprecated, '
    'currently there is no alternative.'
)
def create_test_python_execution_context(
    default_num_clients=0, clients_per_thread=1
):
  """Creates an execution context that executes computations locally."""
  factory = python_executor_stacks.local_executor_factory(
      default_num_clients=default_num_clients,
      clients_per_thread=clients_per_thread,
  )

  return sync_execution_context.SyncExecutionContext(
      executor_fn=factory,
      compiler_fn=compiler.replace_secure_intrinsics_with_bodies,
  )


# TODO(b/240972950): Remove deprecated API.
@deprecation.deprecated(
    '`tff.backends.test.set_test_python_execution_context` is deprecated, '
    'currently there is no alternative.'
)
def set_test_python_execution_context(
    default_num_clients=0, clients_per_thread=1
):
  """Sets an execution context that executes computations locally."""
  context = create_test_python_execution_context(
      default_num_clients=default_num_clients,
      clients_per_thread=clients_per_thread,
  )
  context_stack_impl.context_stack.set_default_context(context)
