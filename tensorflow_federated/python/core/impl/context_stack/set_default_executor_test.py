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

from absl.testing import absltest

from tensorflow_federated.python.core.impl import reference_executor
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.context_stack import set_default_executor
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_factory


class TestSetDefaultExecutor(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # In these tests we are setting the default context of the
    # `context_stack_impl.context_stack`, so here we reset that context back to
    # some known state.
    self.context = context_stack_test_utils.TestContext()
    context_stack_impl.context_stack.set_default_context(self.context)

  def test_with_executor_factory(self):
    context_stack = context_stack_impl.context_stack
    executor_factory_impl = executor_factory.ExecutorFactoryImpl(lambda _: None)
    self.assertNotIsInstance(context_stack.current,
                             execution_context.ExecutionContext)

    set_default_executor.set_default_executor(executor_factory_impl)

    self.assertIsInstance(context_stack.current,
                          execution_context.ExecutionContext)
    self.assertIs(context_stack.current._executor_factory,
                  executor_factory_impl)

  # TODO(b/148233458): ReferenceExecutor is special cased by the implementation
  # of `set_default_executor.set_default_executor`. This test exists to ensure
  # that this case is handled well, but can be removed when that special casing
  # is removed.
  def test_with_reference_executor(self):
    context_stack = context_stack_impl.context_stack
    executor = reference_executor.ReferenceExecutor()
    self.assertIsNot(context_stack.current, executor)

    set_default_executor.set_default_executor(executor)

    self.assertIs(context_stack.current, executor)

  def test_raises_type_error_with_none(self):
    with self.assertRaises(TypeError):
      set_default_executor.set_default_executor(None)


if __name__ == '__main__':
  absltest.main()
