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
import tensorflow_federated as tff


def _create_tff_parallel_clients_with_dataset_reduce():

  @tf.function
  def reduce_fn(x, y):
    return x + y

  @tf.function
  def dataset_reduce_fn(ds, initial_val):
    return ds.reduce(initial_val, reduce_fn)

  @tff.tf_computation(tff.SequenceType(tf.int64))
  def dataset_reduce_fn_wrapper(ds):
    initial_val = tf.Variable(np.int64(1.0))
    return dataset_reduce_fn(ds, initial_val)

  @tff.federated_computation(tff.at_clients(tff.SequenceType(tf.int64)))
  def parallel_client_run(client_datasets):
    return tff.federated_map(dataset_reduce_fn_wrapper, client_datasets)

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

  @tff.tf_computation(tff.SequenceType(tf.int64))
  def dataset_reduce_fn_wrapper(ds):
    initial_val = tf.Variable(np.int64(1.0))
    return dataset_reduce_fn(ds, initial_val)

  @tff.federated_computation(tff.at_clients(tff.SequenceType(tf.int64)))
  def parallel_client_run(client_datasets):
    return tff.federated_map(dataset_reduce_fn_wrapper, client_datasets)

  return parallel_client_run


class LocalExecutorMultiTPUTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tpu_devices = tf.config.list_logical_devices('TPU')
    if len(tpu_devices) < 2:
      self.skipTest('Skip multi-tpu tests when {} tpus are provided'.format(
          len(tpu_devices)))

  @parameterized.named_parameters(
      ('iter_server_on_cpu', 'CPU',
       _create_tff_parallel_clients_with_iter_dataset),
      ('iter_server_on_tpu', 'TPU',
       _create_tff_parallel_clients_with_iter_dataset),
      ('reduce_server_on_cpu', 'CPU',
       _create_tff_parallel_clients_with_dataset_reduce),
      ('reduce_server_on_tpu', 'TPU',
       _create_tff_parallel_clients_with_dataset_reduce),
  )
  def test_local_executor_multi_tpus(self, tf_device,
                                     create_tff_parallel_clients_fn):
    self.skipTest('b/157625321')
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    client_devices = tf.config.list_logical_devices('TPU')
    tff.backends.native.set_local_python_execution_context(
        server_tf_device=server_tf_device, client_tf_devices=client_devices)
    parallel_client_run = create_tff_parallel_clients_fn()
    client_data = [
        tf.data.Dataset.range(10),
        tf.data.Dataset.range(10).map(lambda x: x + 1)
    ]
    client_results = parallel_client_run(client_data)
    self.assertEqual(client_results, [np.int64(46), np.int64(56)])


if __name__ == '__main__':
  absltest.main()
