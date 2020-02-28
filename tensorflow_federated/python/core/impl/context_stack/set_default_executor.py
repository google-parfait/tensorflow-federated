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

from tensorflow_federated.python.core.impl import reference_executor
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_factory


def set_default_executor(executor_factory_instance):
  """Places an `executor`-backed execution context at the top of the stack.

  Args:
    executor_factory_instance: An instance of
      `executor_factory.ExecutorFactory`.
  """
  if isinstance(executor_factory_instance, executor_factory.ExecutorFactory):
    context = execution_context.ExecutionContext(executor_factory_instance)
  elif isinstance(executor_factory_instance,
                  reference_executor.ReferenceExecutor):
    # TODO(b/148233458): ReferenceExecutor inherits from ExectionContext and is
    # used as-is here. The plan is to migrate it to the new Executor base class
    # and stand it up inside a factory like all other executors.
    context = executor_factory_instance
  else:
    raise TypeError('Expected `executor_factory_instance` to be of type '
                    '`executor_factory.ExecutorFactory`, found {}.'.format(
                        type(executor_factory_instance)))
  context_stack_impl.context_stack.set_default_context(context)
