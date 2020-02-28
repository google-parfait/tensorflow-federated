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

from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils


class ContextStackTest(absltest.TestCase):

  def test_set_default_context_with_context(self):
    default_context = context_stack_test_utils.TestContext()
    context_stack = context_stack_impl.ContextStackImpl(default_context)
    context = context_stack_test_utils.TestContext()
    self.assertIsNot(context_stack.current, context)

    context_stack.set_default_context(context)

    self.assertIs(context_stack.current, context)

  def test_set_default_context_raises_type_error_with_none(self):
    default_context = context_stack_test_utils.TestContext()
    context_stack = context_stack_impl.ContextStackImpl(default_context)

    with self.assertRaises(TypeError):
      context_stack.set_default_context(None)

  def test_install_pushes_context_on_stack(self):
    default_context = context_stack_test_utils.TestContext()
    context_stack = context_stack_impl.ContextStackImpl(default_context)
    self.assertIs(context_stack.current, default_context)

    context_two = context_stack_test_utils.TestContext()
    with context_stack.install(context_two):
      self.assertIs(context_stack.current, context_two)

      context_three = context_stack_test_utils.TestContext()
      with context_stack.install(context_three):
        self.assertIs(context_stack.current, context_three)

      self.assertIs(context_stack.current, context_two)

    self.assertIs(context_stack.current, default_context)


if __name__ == '__main__':
  absltest.main()
