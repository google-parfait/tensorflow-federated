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

from unittest import mock
from absl.testing import parameterized

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.types import placement_literals


class ExecutorFactoryImplTest(parameterized.TestCase):

  def _maybe_wrap_stack_fn(self, stack_fn, ex_factory):
    """The stack_fn for SizingExecutorFactory requires two outputs.

    If required, we will wrap the stack_fn and provide a dummy value as the
    second return value.

    Args:
      stack_fn: The original stack_fn
      ex_factory: A class which inherits from ExecutorFactory.

    Returns:
      A stack_fn that might additionally return a list as the second value.
    """
    if ex_factory == executor_factory.SizingExecutorFactory:
      return lambda x: (stack_fn(x), [])
    else:
      return stack_fn

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

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_factory.SizingExecutorFactory),
      ('ExecutorFactoryImpl', executor_factory.ExecutorFactoryImpl))
  def test_concrete_class_instantiates_stack_fn(self, ex_factory):

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    self.assertIsInstance(factory, ex_factory)

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_factory.SizingExecutorFactory),
      ('ExecutorFactoryImpl', executor_factory.ExecutorFactoryImpl))
  def test_call_constructs_executor(self, ex_factory):

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    ex = factory.create_executor({})
    self.assertIsInstance(ex, executor_base.Executor)

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_factory.SizingExecutorFactory),
      ('ExecutorFactoryImpl', executor_factory.ExecutorFactoryImpl))
  def test_cleanup_succeeds_without_init(self, ex_factory):

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    factory.clean_up_executors()

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_factory.SizingExecutorFactory),
      ('ExecutorFactoryImpl', executor_factory.ExecutorFactoryImpl))
  def test_cleanup_calls_close(self, ex_factory):
    ex = eager_tf_executor.EagerTFExecutor()
    ex.close = mock.MagicMock()

    def _stack_fn(x):
      del x  # Unused
      return ex

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    factory.create_executor({})
    factory.clean_up_executors()
    ex.close.assert_called_once()

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_factory.SizingExecutorFactory),
      ('ExecutorFactoryImpl', executor_factory.ExecutorFactoryImpl))
  def test_construction_with_multiple_cardinalities_reuses_existing_stacks(
      self, ex_factory):
    ex = eager_tf_executor.EagerTFExecutor()
    ex.close = mock.MagicMock()
    num_times_invoked = 0

    def _stack_fn(x):
      del x  # Unused
      nonlocal num_times_invoked
      num_times_invoked += 1
      return ex

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    for _ in range(2):
      factory.create_executor({})
      factory.create_executor({placement_literals.SERVER: 1})
    self.assertEqual(num_times_invoked, 2)


if __name__ == '__main__':
  test.main()
