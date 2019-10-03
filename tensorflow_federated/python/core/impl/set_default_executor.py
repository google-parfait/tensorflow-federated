# Lint as: python3
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
"""A utility to change the default executor."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import execution_context
from tensorflow_federated.python.core.impl import executor_base


def set_default_executor(executor=None):
  """Places an `executor`-backed execution context at the top of the stack.

  NOTE: This function is only available in Python 3.

  Args:
    executor: Either an instance of `executor_base.Executor`, a factory
      function returning such executors, or `None`. If `executor` is a factory
      function, the constructed context will infer the number of clients from
      the data it is passed, if possible. If `None`, causes the default
      reference executor to be installed (as is the default).
  """
  # TODO(b/140112504): Follow up here when we implement the ExecutorFactory
  # interface.
  if isinstance(executor, executor_base.Executor):
    context = execution_context.ExecutionContext(lambda x: executor)
  elif callable(executor):
    context = execution_context.ExecutionContext(executor)
  else:
    py_typecheck.check_none(executor)
    context = None
  context_stack_impl.context_stack.set_default_context(context)
