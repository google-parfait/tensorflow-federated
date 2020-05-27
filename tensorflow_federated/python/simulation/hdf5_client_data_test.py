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

import os
import tempfile

from absl.testing import absltest
import h5py
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.simulation import hdf5_client_data

tf.compat.v1.enable_v2_behavior()

TEST_DATA = {
    'CLIENT A': {
        'w': np.asarray([100, 200, 300], dtype='i8'),
        'x': np.asarray([[1, 2], [3, 4], [5, 6]], dtype='i4'),
        'y': np.asarray([4.0, 5.0, 6.0], dtype='f4'),
        'z': np.asarray(['a', 'b', 'c'], dtype='S'),
    },
    'CLIENT B': {
        'w': np.asarray([1000], dtype='i8'),
        'x': np.asarray([[10, 11]], dtype='i4'),
        'y': np.asarray([7.0], dtype='f4'),
        'z': np.asarray(['d'], dtype='S'),
    },
    'CLIENT C': {
        'w': np.asarray([10000, 20000], dtype='i8'),
        'x': np.asarray([[100, 101], [200, 201]], dtype='i4'),
        'y': np.asarray([8.0, 9.0], dtype='f4'),
        'z': np.asarray(['e', 'f'], dtype='S'),
    },
}


def create_fake_hdf5():
  fd, filepath = tempfile.mkstemp()
  # close the pre-opened file descriptor immediately to avoid leaking.
  os.close(fd)
  with h5py.File(filepath, 'w') as f:
    examples_group = f.create_group('examples')
    for user_id, data in TEST_DATA.items():
      user_group = examples_group.create_group(user_id)
      for name, values in sorted(data.items()):
        user_group.create_dataset(name, data=values, dtype=values.dtype)
  return filepath


class HDF5ClientDataTest(tf.test.TestCase, absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.test_data_filepath = create_fake_hdf5()

  @classmethod
  def tearDownClass(cls):
    os.remove(cls.test_data_filepath)
    super().tearDownClass()

  def test_client_ids_property(self):
    client_data = hdf5_client_data.HDF5ClientData(
        HDF5ClientDataTest.test_data_filepath)
    self.assertEqual(client_data.client_ids, sorted(TEST_DATA.keys()))

  def test_element_type_structure(self):
    expected_structure = {
        'w': tf.TensorSpec(shape=[], dtype=tf.int64),
        'x': tf.TensorSpec(shape=[2], dtype=tf.int32),
        'y': tf.TensorSpec(shape=[], dtype=tf.float32),
        'z': tf.TensorSpec(shape=[], dtype=tf.string),
    }
    client_data = hdf5_client_data.HDF5ClientData(
        HDF5ClientDataTest.test_data_filepath)
    self.assertDictEqual(client_data.element_type_structure, expected_structure)

  def test_create_tf_dataset_for_client(self):
    client_data = hdf5_client_data.HDF5ClientData(
        HDF5ClientDataTest.test_data_filepath)

    with self.assertRaisesRegex(ValueError,
                                'is not a client in this ClientData'):
      client_data.create_tf_dataset_for_client('non_existent_id')

    # Iterate over each client, ensuring we received a tf.data.Dataset with the
    # correct data.
    for client_id, expected_data in TEST_DATA.items():
      tf_dataset = client_data.create_tf_dataset_for_client(client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)

      expected_examples = []
      for i in range(len(expected_data['x'])):
        expected_examples.append({k: v[i] for k, v in expected_data.items()})
      for actual in tf_dataset:
        expected = expected_examples.pop(0)
        actual = self.evaluate(actual)
        self.assertCountEqual(actual, expected)
      self.assertEmpty(expected_examples)

  def test_create_tf_dataset_from_all_clients(self):
    client_data = hdf5_client_data.HDF5ClientData(
        HDF5ClientDataTest.test_data_filepath)
    tf_dataset = client_data.create_tf_dataset_from_all_clients()
    self.assertIsInstance(tf_dataset, tf.data.Dataset)

    expected_examples = []
    for expected_data in TEST_DATA.values():
      for i in range(len(expected_data['x'])):
        expected_examples.append({k: v[i] for k, v in expected_data.items()})

    for actual in tf_dataset:
      expected = expected_examples.pop(0)
      actual = self.evaluate(actual)
      self.assertCountEqual(actual, expected)
    self.assertEmpty(expected_examples)


if __name__ == '__main__':
  tf.test.main()
