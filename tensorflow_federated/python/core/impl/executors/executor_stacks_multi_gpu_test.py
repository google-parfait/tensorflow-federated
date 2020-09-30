# Copyright 2020, The TensorFlow Federated Authors.
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
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils


def _create_tff_parallel_clients_with_dataset_reduce():

  @tf.function
  def reduce_fn(x, y):
    return x + y

  @tf.function
  def dataset_reduce_fn(ds, initial_val):
    return ds.reduce(initial_val, reduce_fn)

  @computations.tf_computation(computation_types.SequenceType(tf.int64))
  def dataset_reduce_fn_wrapper(ds):
    initial_val = tf.Variable(np.int64(1.0))
    return dataset_reduce_fn(ds, initial_val)

  @computations.federated_computation(
      computation_types.at_clients(computation_types.SequenceType(tf.int64)))
  def parallel_client_run(client_datasets):
    return intrinsics.federated_map(dataset_reduce_fn_wrapper, client_datasets)

  return parallel_client_run


def _create_tff_parallel_clients_with_iter_dataset():

  @tf.function
  def reduce_fn(x, y):
    return x + y

  @tf.function
  def dataset_reduce_fn(ds, initial_val):
    for batch in iter(ds):
      initial_val = reduce_fn(initial_val, batch)
    return initial_val

  @computations.tf_computation(computation_types.SequenceType(tf.int64))
  def dataset_reduce_fn_wrapper(ds):
    initial_val = tf.Variable(np.int64(1.0))
    return dataset_reduce_fn(ds, initial_val)

  @computations.federated_computation(
      computation_types.at_clients(computation_types.SequenceType(tf.int64)))
  def parallel_client_run(client_datasets):
    return intrinsics.federated_map(dataset_reduce_fn_wrapper, client_datasets)

  return parallel_client_run


class MultiGPUTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    test_utils.create_logical_multi_gpus()

  def _create_logical_multi_gpus(self):
    # Multiple logical GPU devices will be created for tests in this module.
    # Only call this function once as logical deviceds have to be created before
    # listed in each indivisual test.
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
      # TODO(b/168138878): switch to raise and move out of MultiGPUTest
      self.skipTest('Skip GPU tests when no GPU is provided')
    if len(gpu_devices) == 1:
      tf.config.set_logical_device_configuration(gpu_devices[0], [
          tf.config.LogicalDeviceConfiguration(memory_limit=128),
          tf.config.LogicalDeviceConfiguration(memory_limit=128)
      ])

  @parameterized.named_parameters(
      ('server_on_cpu', 'CPU'),
      ('server_on_gpu', 'GPU'),
  )
  def test_create_executor_with_client_mgpu(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    gpu_devices = tf.config.list_logical_devices('GPU')
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=True,
        server_device=server_tf_device,
        client_devices=gpu_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  @parameterized.named_parameters(
      ('server_on_cpu', 'CPU'),
      ('server_on_gpu', 'GPU'),
  )
  def test_local_executor_multi_gpus_iter_dataset(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    gpu_devices = tf.config.list_logical_devices('GPU')
    local_executor = executor_stacks.local_executor_factory(
        server_tf_device=server_tf_device, client_tf_devices=gpu_devices)
    with executor_test_utils.install_executor(local_executor):
      parallel_client_run = _create_tff_parallel_clients_with_iter_dataset()
      client_data = [
          tf.data.Dataset.range(10),
          tf.data.Dataset.range(10).map(lambda x: x + 1)
      ]
      client_results = parallel_client_run(client_data)
      self.assertEqual(client_results, [np.int64(46), np.int64(56)])

  @parameterized.named_parameters(
      ('server_on_cpu', 'CPU'),
      ('server_on_gpu', 'GPU'),
  )
  def test_local_executor_multi_gpus_dataset_reduce(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    gpu_devices = tf.config.list_logical_devices('GPU')
    local_executor = executor_stacks.local_executor_factory(
        server_tf_device=server_tf_device, client_tf_devices=gpu_devices)
    with executor_test_utils.install_executor(local_executor):
      parallel_client_run = _create_tff_parallel_clients_with_dataset_reduce()
      client_data = [
          tf.data.Dataset.range(10),
          tf.data.Dataset.range(10).map(lambda x: x + 1)
      ]
      # TODO(b/159180073): merge this one into iter dataset test when the
      # dataset reduce function can be correctly used for GPU device.
      with self.assertRaisesRegex(
          ValueError,
          'Detected dataset reduce op in multi-GPU TFF simulation.*'):
        parallel_client_run(client_data)


if __name__ == '__main__':
  absltest.main()
