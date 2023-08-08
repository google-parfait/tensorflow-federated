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

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.impl.executor_stacks import python_executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.types import placements


class ExecutorMock(mock.MagicMock, executor_base.Executor):

  async def create_value(self, *args):
    pass

  async def create_call(self, *args):
    pass

  async def create_selection(self, *args):
    pass

  async def create_struct(self, *args):
    pass

  async def close(self, *args):
    pass


class ConcreteExecutorFactoryTest(parameterized.TestCase):

  def test_subclass_base_fails_no_create_method(self):
    class NotCallable(executor_factory.ExecutorFactory):

      def clean_up_executor(self, x):
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

      def clean_up_executor(self, x):
        pass

    Fine()

  @parameterized.named_parameters((
      'ResourceManagingExecutorFactory',
      python_executor_stacks.ResourceManagingExecutorFactory,
  ))
  def test_concrete_class_instantiates_stack_fn(self, ex_factory):
    def _stack_fn(x):
      del x  # Unused
      return ExecutorMock()

    factory = ex_factory(_stack_fn)
    self.assertIsInstance(factory, ex_factory)

  @parameterized.named_parameters((
      'ResourceManagingExecutorFactory',
      python_executor_stacks.ResourceManagingExecutorFactory,
  ))
  def test_call_constructs_executor(self, ex_factory):
    def _stack_fn(x):
      del x  # Unused
      return ExecutorMock()

    factory = ex_factory(_stack_fn)
    ex = factory.create_executor({})
    self.assertIsInstance(ex, executor_base.Executor)

  @parameterized.named_parameters((
      'ResourceManagingExecutorFactory',
      python_executor_stacks.ResourceManagingExecutorFactory,
  ))
  def test_cleanup_succeeds_without_init(self, ex_factory):
    def _stack_fn(x):
      del x  # Unused
      return ExecutorMock()

    factory = ex_factory(_stack_fn)
    factory.clean_up_executor({placements.CLIENTS: 1})

  @parameterized.named_parameters((
      'ResourceManagingExecutorFactory',
      python_executor_stacks.ResourceManagingExecutorFactory,
  ))
  def test_cleanup_calls_close(self, ex_factory):
    ex = ExecutorMock()
    ex.close = mock.MagicMock()

    def _stack_fn(x):
      del x  # Unused
      return ex

    factory = ex_factory(_stack_fn)
    factory.create_executor({})
    factory.clean_up_executor({})
    ex.close.assert_called_once()

  @parameterized.named_parameters((
      'ResourceManagingExecutorFactory',
      python_executor_stacks.ResourceManagingExecutorFactory,
  ))
  def test_construction_with_multiple_cardinalities_reuses_existing_stacks(
      self, ex_factory
  ):
    ex = ExecutorMock()
    ex.close = mock.MagicMock()
    num_times_invoked = 0

    def _stack_fn(x):
      del x  # Unused
      nonlocal num_times_invoked
      num_times_invoked += 1
      return ex

    factory = ex_factory(_stack_fn)
    for _ in range(2):
      factory.create_executor({})
      factory.create_executor({placements.SERVER: 1})
    self.assertEqual(num_times_invoked, 2)

  def test_executors_persisted_is_capped(self):
    ex = ExecutorMock()

    factory = python_executor_stacks.ResourceManagingExecutorFactory(
        lambda _: ex
    )
    for num_clients in range(100):
      factory.create_executor({placements.CLIENTS: num_clients})
    self.assertLess(len(factory._executors), 20)


if __name__ == '__main__':
  absltest.main()
