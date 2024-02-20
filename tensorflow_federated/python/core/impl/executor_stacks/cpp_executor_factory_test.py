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

from absl.testing import absltest

from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_test_utils_bindings
from tensorflow_federated.python.core.impl.types import placements


def _create_mock_execution_stack(
    max_concurrent_computation_calls: int,
) -> executor_bindings.Executor:
  """Constructs the default leaf executor stack."""
  del max_concurrent_computation_calls  # Unused.

  mock_executor = executor_test_utils_bindings.create_mock_executor()
  reference_resolving_executor = (
      executor_bindings.create_reference_resolving_executor(mock_executor)
  )
  return executor_bindings.create_sequence_executor(
      reference_resolving_executor
  )


class CPPExecutorFactoryTest(absltest.TestCase):

  def test_create_local_cpp_factory_constructs(self):
    local_cpp_factory = cpp_executor_factory.local_cpp_executor_factory(
        default_num_clients=0, leaf_executor_fn=_create_mock_execution_stack
    )
    self.assertIsInstance(local_cpp_factory, executor_factory.ExecutorFactory)

  def test_clean_up_executors_clears_state(self):
    local_cpp_factory = cpp_executor_factory.local_cpp_executor_factory(
        default_num_clients=0, leaf_executor_fn=_create_mock_execution_stack
    )
    cardinalities = {placements.CLIENTS: 1}
    local_cpp_factory.create_executor(cardinalities)
    for executor in local_cpp_factory._executors.values():
      self.assertIsInstance(executor, executor_base.Executor)
    local_cpp_factory.clean_up_executor(cardinalities)
    self.assertEmpty(local_cpp_factory._executors)

  def test_create_local_cpp_factory_constructs_executor_implementation(self):
    local_cpp_factory = cpp_executor_factory.local_cpp_executor_factory(
        default_num_clients=0, leaf_executor_fn=_create_mock_execution_stack
    )
    self.assertIsInstance(local_cpp_factory, executor_factory.ExecutorFactory)
    executor = local_cpp_factory.create_executor({placements.CLIENTS: 1})
    self.assertIsInstance(executor, executor_base.Executor)

  def test_create_remote_cpp_factory_constructs(self):
    targets = ['localhost:8000', 'localhost:8001']
    channels = [
        executor_bindings.create_insecure_grpc_channel(t) for t in targets
    ]
    remote_cpp_factory = cpp_executor_factory.remote_cpp_executor_factory(
        channels=channels, default_num_clients=0
    )
    self.assertIsInstance(remote_cpp_factory, executor_factory.ExecutorFactory)

  def test_create_remote_cpp_factory_raises_with_no_available_workers(self):
    targets = ['localhost:8000', 'localhost:8001']
    channels = [
        executor_bindings.create_insecure_grpc_channel(t) for t in targets
    ]
    remote_cpp_factory = cpp_executor_factory.remote_cpp_executor_factory(
        channels=channels, default_num_clients=0
    )
    self.assertIsInstance(remote_cpp_factory, executor_factory.ExecutorFactory)
    with self.assertRaises(Exception):
      remote_cpp_factory.create_executor({placements.CLIENTS: 1})

  def test_create_cpp_factory_raises_with_invalid_default_num_clients(self):
    with self.subTest('local_nonnegative'):
      with self.assertRaisesRegex(ValueError, 'nonnegative'):
        cpp_executor_factory.local_cpp_executor_factory(
            default_num_clients=-1,
            leaf_executor_fn=_create_mock_execution_stack,
        )

    with self.subTest('remote_nonnegative'):
      with self.assertRaisesRegex(ValueError, 'nonnegative'):
        cpp_executor_factory.remote_cpp_executor_factory(
            channels=[], default_num_clients=-1
        )

    with self.subTest('local_non_integer'):
      with self.assertRaisesRegex(TypeError, 'int'):
        cpp_executor_factory.local_cpp_executor_factory(
            default_num_clients=1.0,
            leaf_executor_fn=_create_mock_execution_stack,
        )

    with self.subTest('remote_non_integer'):
      with self.assertRaisesRegex(TypeError, 'int'):
        cpp_executor_factory.remote_cpp_executor_factory(
            channels=[], default_num_clients=1.0
        )


if __name__ == '__main__':
  absltest.main()
