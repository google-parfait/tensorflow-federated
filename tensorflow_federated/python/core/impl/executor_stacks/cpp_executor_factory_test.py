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
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.types import placements


class CPPExecutorFactoryTest(absltest.TestCase):

  def _assert_cpp_executor_interface(self, executor):
    self.assertTrue(hasattr(executor, 'create_value'))
    self.assertTrue(hasattr(executor, 'create_struct'))
    self.assertTrue(hasattr(executor, 'create_selection'))
    self.assertTrue(hasattr(executor, 'create_call'))
    self.assertTrue(hasattr(executor, 'materialize'))

  def test_create_local_cpp_factory_constructs(self):
    local_cpp_factory = cpp_executor_factory.local_cpp_executor_factory(
        default_num_clients=0)
    self.assertIsInstance(local_cpp_factory, executor_factory.ExecutorFactory)

  def test_clean_up_executors_clears_state(self):
    local_cpp_factory = cpp_executor_factory.local_cpp_executor_factory(
        default_num_clients=0)
    local_cpp_factory.create_executor({placements.CLIENTS: 1})
    for executor in local_cpp_factory._executors.values():
      self._assert_cpp_executor_interface(executor)
    local_cpp_factory.clean_up_executors()
    self.assertEmpty(local_cpp_factory._executors)

  def test_create_local_cpp_factory_constructs_executor_implementation(self):
    local_cpp_factory = cpp_executor_factory.local_cpp_executor_factory(
        default_num_clients=0)
    self.assertIsInstance(local_cpp_factory, executor_factory.ExecutorFactory)
    executor = local_cpp_factory.create_executor({placements.CLIENTS: 1})
    self._assert_cpp_executor_interface(executor)

  def test_create_remote_cpp_factory_constructs(self):
    targets = ['localhost:8000', 'localhost:8001']
    channels = [
        executor_bindings.create_insecure_grpc_channel(t) for t in targets
    ]
    remote_cpp_factory = cpp_executor_factory.remote_cpp_executor_factory(
        channels=channels, default_num_clients=0)
    self.assertIsInstance(remote_cpp_factory, executor_factory.ExecutorFactory)

  def test_create_remote_cpp_factory_raises_with_no_available_workers(self):
    targets = ['localhost:8000', 'localhost:8001']
    channels = [
        executor_bindings.create_insecure_grpc_channel(t) for t in targets
    ]
    remote_cpp_factory = cpp_executor_factory.remote_cpp_executor_factory(
        channels=channels, default_num_clients=0)
    self.assertIsInstance(remote_cpp_factory, executor_factory.ExecutorFactory)
    with self.assertRaises(Exception):
      remote_cpp_factory.create_executor({placements.CLIENTS: 1})


if __name__ == '__main__':
  absltest.main()
