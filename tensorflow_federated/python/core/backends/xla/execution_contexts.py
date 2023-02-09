# Copyright 2020, The TensorFlow Federated Authors.
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
"""Execution contexts for the XLA backend."""

from tensorflow_federated.python.common_libs import deprecation
from tensorflow_federated.python.core.backends.xla import compiler
from tensorflow_federated.python.core.backends.xla import executor
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import python_executor_stacks


# TODO(b/240972950): Remove deprecated API.
@deprecation.deprecated(
    '`tff.backends.xla.create_local_python_execution_context` is deprecated, '
    'currently there is no alternative.'
)
def create_local_python_execution_context():
  """Creates an XLA-based local execution context.

  NOTE: This context is only directly backed by an XLA executor. It does not
  support any intrinsics, lambda expressions, etc.

  Returns:
    An instance of `tff.framework.SyncExecutionContext` backed by XLA executor.
  """
  # TODO(b/175888145): Extend this into a complete local executor stack.

  factory = python_executor_stacks.local_executor_factory(
      support_sequence_ops=True,
      leaf_executor_fn=executor.XlaExecutor,
      local_computation_factory=compiler.XlaComputationFactory(),
  )
  return sync_execution_context.SyncExecutionContext(executor_fn=factory)


# TODO(b/240972950): Remove deprecated API.
@deprecation.deprecated(
    '`tff.backends.xla.set_local_python_execution_context` is deprecated, '
    'currently there is no alternative.'
)
def set_local_python_execution_context(*args, **kwargs):
  """Sets an XLA-based local execution context.

  Invokes `create_local_execution_context` to construct an execution context,
  and sets it as the default. Accepts the same parameters as
  `create_local_execution_context`.

  Args:
    *args: Positional args for `create_local_execution_context`.
    **kwargs: Keyword args for `create_local_execution_context`.
  """
  context = create_local_python_execution_context(*args, **kwargs)
  context_stack_impl.context_stack.set_default_context(context)
