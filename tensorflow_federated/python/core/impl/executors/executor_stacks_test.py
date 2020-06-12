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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_factory


def _temperature_sensor_example_next_fn():

  @computations.tf_computation(
      computation_types.SequenceType(tf.float32), tf.float32)
  def count_over(ds, t):
    return ds.reduce(
        np.float32(0), lambda n, x: n + tf.cast(tf.greater(x, t), tf.float32))

  @computations.tf_computation(computation_types.SequenceType(tf.float32))
  def count_total(ds):
    return ds.reduce(np.float32(0.0), lambda n, _: n + 1.0)

  @computations.federated_computation(
      type_factory.at_clients(computation_types.SequenceType(tf.float32)),
      type_factory.at_server(tf.float32))
  def comp(temperatures, threshold):
    return intrinsics.federated_mean(
        intrinsics.federated_map(
            count_over,
            intrinsics.federated_zip(
                [temperatures,
                 intrinsics.federated_broadcast(threshold)])),
        intrinsics.federated_map(count_total, temperatures))

  return comp


def _create_concurrent_maxthread_tuples():
  tuples = []
  for concurrency in range(1, 5):
    local_ex_string = 'local_executor_{}_client_thread'.format(concurrency)
    ex_factory = executor_stacks.local_executor_factory(
        num_client_executors=concurrency)
    tuples.append((local_ex_string, ex_factory, concurrency))
    sizing_ex_string = 'sizing_executor_{}_client_thread'.format(concurrency)
    ex_factory = executor_stacks.sizing_executor_factory(
        num_client_executors=concurrency)
    tuples.append((sizing_ex_string, ex_factory, concurrency))
  return tuples


class ExecutorMock(mock.MagicMock, executor_base.Executor):

  def create_value(self, *args):
    pass

  def create_call(self, *args):
    pass

  def create_selection(self, *args):
    pass

  def create_tuple(self, *args):
    pass

  def close(self, *args):
    pass


class ExecutorStacksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_construction_with_no_args(self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn()
    self.assertIsInstance(executor_factory_impl,
                          executor_factory.ExecutorFactoryImpl)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_construction_raises_with_max_fanout_one(self, executor_factory_fn):
    with self.assertRaises(ValueError):
      executor_factory_fn(max_fanout=1)

  @parameterized.named_parameters(
      ('local_executor_none_clients', executor_stacks.local_executor_factory()),
      ('sizing_executor_none_clients',
       executor_stacks.sizing_executor_factory()),
      ('local_executor_three_clients',
       executor_stacks.local_executor_factory(num_clients=3)),
      ('sizing_executor_three_clients',
       executor_stacks.sizing_executor_factory(num_clients=3)),
  )
  def test_execution_of_temperature_sensor_example(self, executor):
    comp = _temperature_sensor_example_next_fn()
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float),
    ]
    threshold = 15.0

    with executor_test_utils.install_executor(executor):
      result = comp(temperatures, threshold)

    self.assertAlmostEqual(result, 8.333, places=3)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_execution_with_inferred_clients_larger_than_fanout(
      self, executor_factory_fn):

    @computations.federated_computation(type_factory.at_clients(tf.int32))
    def foo(x):
      return intrinsics.federated_sum(x)

    executor = executor_factory_fn(max_fanout=3)
    with executor_test_utils.install_executor(executor):
      result = foo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    self.assertEqual(result, 55)

  @parameterized.named_parameters(
      ('local_executor_none_clients', executor_stacks.local_executor_factory()),
      ('sizing_executor_none_clients',
       executor_stacks.sizing_executor_factory()),
      ('local_executor_one_client',
       executor_stacks.local_executor_factory(num_clients=1)),
      ('sizing_executor_one_client',
       executor_stacks.sizing_executor_factory(num_clients=1)),
  )
  def test_execution_of_tensorflow(self, executor):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    with executor_test_utils.install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)

  @parameterized.named_parameters(*_create_concurrent_maxthread_tuples())
  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.eager_tf_executor.EagerTFExecutor',
      return_value=ExecutorMock())
  def test_limiting_concurrency_constructs_one_eager_executor(
      self, ex_factory, concurrency_level, tf_executor_mock):
    ex_factory.create_executor({placement_literals.CLIENTS: 10})
    args_list = tf_executor_mock.call_args_list
    # One for server, one for `None`-placed, concurrency_level for clients.
    self.assertLen(args_list, concurrency_level + 2)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_create_executor_raises_with_wrong_cardinalities(
      self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn(num_clients=5)
    cardinalities = {
        placement_literals.SERVER: 1,
        None: 1,
        placement_literals.CLIENTS: 1,
    }
    with self.assertRaises(ValueError,):
      executor_factory_impl.create_executor(cardinalities)


class UnplacedExecutorFactoryTest(parameterized.TestCase):

  def test_constructs_executor_factory(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    self.assertIsInstance(unplaced_factory, executor_factory.ExecutorFactory)

  def test_constructs_executor_factory_without_caching(self):
    unplaced_factory_no_caching = executor_stacks.UnplacedExecutorFactory(
        use_caching=False)
    self.assertIsInstance(unplaced_factory_no_caching,
                          executor_factory.ExecutorFactory)

  def test_create_executor_returns_executor(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    unplaced_executor = unplaced_factory.create_executor(cardinalities={})
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  def test_create_executor_raises_with_nonempty_cardinalitites(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    with self.assertRaises(ValueError):
      unplaced_factory.create_executor(
          cardinalities={placement_literals.SERVER: 1})

  @parameterized.named_parameters(
      ('server_on_cpu', 'CPU'),
      ('server_on_gpu', 'GPU'),
  )
  def test_create_executor_with_server_device(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=True, server_device=server_tf_device)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  @parameterized.named_parameters(
      ('clients_on_cpu', 'CPU'),
      ('clients_on_gpu', 'GPU'),
  )
  def test_create_executor_with_client_devices(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=True, client_devices=tf_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  @parameterized.named_parameters(
      ('server_clients_on_cpu', 'CPU'),
      ('server_clients_on_gpu', 'GPU'),
  )
  def test_create_executor_with_server_client_devices(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=True,
        server_device=server_tf_device,
        client_devices=tf_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  def test_create_executor_with_server_cpu_client_gpu(self):
    cpu_devices = tf.config.list_logical_devices('CPU')
    gpu_devices = tf.config.list_logical_devices('GPU')
    server_tf_device = None if not cpu_devices else cpu_devices[0]
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=True,
        server_device=server_tf_device,
        client_devices=gpu_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)


if __name__ == '__main__':
  absltest.main()
