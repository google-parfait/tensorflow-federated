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

from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_test_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import set_default_context


class SetDefaultContextTest(absltest.TestCase):

  def test_with_none(self):
    context = context_stack_test_utils.TestContext('test')
    context_stack = context_stack_impl.context_stack
    context_stack.set_default_context(context)
    self.assertIs(context_stack.current, context)

    set_default_context.set_default_context(None)

    self.assertIsNot(context_stack.current, context)
    self.assertIsInstance(context_stack.current, context_base.Context)

  def test_with_context(self):
    context = context_stack_test_utils.TestContext('test')
    context_stack = context_stack_impl.context_stack
    self.assertIsNot(context_stack.current, context)

    set_default_context.set_default_context(context)

    self.assertIs(context_stack.current, context)


if __name__ == '__main__':
  absltest.main()
