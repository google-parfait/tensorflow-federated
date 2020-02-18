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

from tensorflow_federated.python.core.impl import context_stack_test_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context


class ContextStackTest(absltest.TestCase):

  def test_basic_functionality(self):
    ctx_stack = context_stack_impl.context_stack
    self.assertIsInstance(ctx_stack, context_stack_impl.ContextStackImpl)
    self.assertIsInstance(ctx_stack.current, execution_context.ExecutionContext)

    with ctx_stack.install(context_stack_test_utils.TestContext('foo')):
      self.assertIsInstance(ctx_stack.current,
                            context_stack_test_utils.TestContext)
      self.assertEqual(ctx_stack.current.name, 'foo')

      with ctx_stack.install(context_stack_test_utils.TestContext('bar')):
        self.assertIsInstance(ctx_stack.current,
                              context_stack_test_utils.TestContext)
        self.assertEqual(ctx_stack.current.name, 'bar')

      self.assertEqual(ctx_stack.current.name, 'foo')

    self.assertIsInstance(ctx_stack.current, execution_context.ExecutionContext)

  def test_set_default_context(self):

    ctx_stack = context_stack_impl.context_stack
    self.assertIsInstance(ctx_stack.current, execution_context.ExecutionContext)
    foo = context_stack_test_utils.TestContext('foo')
    ctx_stack.set_default_context(foo)
    self.assertIs(ctx_stack.current, foo)
    ctx_stack.set_default_context()
    self.assertIsInstance(ctx_stack.current, execution_context.ExecutionContext)


if __name__ == '__main__':
  absltest.main()
