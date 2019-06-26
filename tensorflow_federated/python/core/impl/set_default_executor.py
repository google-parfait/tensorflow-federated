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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import execution_context
from tensorflow_federated.python.core.impl import executor_base


def set_default_executor(executor=None):
  """Places an `executor`-backed execution context at the top of the stack.

  Args:
    executor: Either an instance of `executor_base.Executor`, or `None` which
      causes the default reference executor to be installed (as is the default).
  """
  if executor is not None:
    py_typecheck.check_type(executor, executor_base.Executor)
    context = execution_context.ExecutionContext(executor)
  else:
    context = None
  context_stack_impl.context_stack.set_default_context(context)
