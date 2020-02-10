# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for ExecutorFactory."""

from unittest import mock

from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory


class ExecutorFactoryImplTest(common_test.TestCase):

  def test_subclass_base_fails_no_create_method(self):

    class NotCallable(executor_factory.ExecutorFactory):

      def clean_up_executors(self):
        pass

    with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
      NotCallable()

  def test_subclass_base_fails_no_cleanup(self):

    class NoCleanup(executor_factory.ExecutorFactory):

      def create_executor(self, x):
        pass

    with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
      NoCleanup()

  def test_instantiation_succeeds_both_methods_specified(self):

    class Fine(executor_factory.ExecutorFactory):

      def create_executor(self, x):
        pass

      def clean_up_executors(self):
        pass

    Fine()

  def test_concrete_class_instantiates_stack_fn(self):

    def _stack_fn(x):
      del x  # Unused
      return eager_executor.EagerExecutor()

    factory = executor_factory.ExecutorFactoryImpl(_stack_fn)
    self.assertIsInstance(factory, executor_factory.ExecutorFactoryImpl)

  def test_call_constructs_executor(self):

    def _stack_fn(x):
      del x  # Unused
      return eager_executor.EagerExecutor()

    factory = executor_factory.ExecutorFactoryImpl(_stack_fn)
    ex = factory.create_executor({})
    self.assertIsInstance(ex, executor_base.Executor)

  def test_cleanup_succeeds_without_init(self):

    def _stack_fn(x):
      del x  # Unused
      return eager_executor.EagerExecutor()

    factory = executor_factory.ExecutorFactoryImpl(_stack_fn)
    factory.clean_up_executors()

  def test_cleanup_calls_close(self):
    ex = eager_executor.EagerExecutor()
    ex.close = mock.MagicMock()

    def _stack_fn(x):
      del x  # Unused
      return ex

    factory = executor_factory.ExecutorFactoryImpl(_stack_fn)
    factory.create_executor({})
    factory.clean_up_executors()
    ex.close.assert_called_once()

  def test_construction_with_multiple_cardinalities_reuses_existing_stacks(
      self):
    ex = eager_executor.EagerExecutor()
    ex.close = mock.MagicMock()
    num_times_invoked = 0

    def _stack_fn(x):
      del x  # Unused
      nonlocal num_times_invoked
      num_times_invoked += 1
      return ex

    factory = executor_factory.ExecutorFactoryImpl(_stack_fn)
    for _ in range(2):
      factory.create_executor({})
      factory.create_executor({placement_literals.SERVER: 1})
    self.assertEqual(num_times_invoked, 2)


if __name__ == '__main__':
  common_test.main()
