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
import re
import tempfile

from absl.testing import absltest
import h5py
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.simulation import hdf5_client_data
from tensorflow_federated.python.simulation import transforming_client_data

tf.compat.v1.enable_v2_behavior()

TEST_DATA = {
    'CLIENT A': {
        'x': np.asarray([[1, 2], [3, 4], [5, 6]], dtype='i4'),
        'y': np.asarray([4.0, 5.0, 6.0], dtype='f4'),
        'z': np.asarray(['a', 'b', 'c'], dtype='S'),
    },
    'CLIENT B': {
        'x': np.asarray([[10, 11]], dtype='i4'),
        'y': np.asarray([7.0], dtype='f4'),
        'z': np.asarray(['d'], dtype='S'),
    },
    'CLIENT C': {
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
      for name, values in data.items():
        user_group.create_dataset(name, data=values)
  return filepath


def _test_transform_cons(raw_client_id, index):
  del raw_client_id

  def fn(data):
    data['x'] = data['x'] + 10 * index
    return data

  return fn


class TransformingClientDataTest(tf.test.TestCase, absltest.TestCase):

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
        TransformingClientDataTest.test_data_filepath)
    num_transformed_clients = 7
    transformed_client_data = transforming_client_data.TransformingClientData(
        client_data, _test_transform_cons, num_transformed_clients)
    client_ids = transformed_client_data.client_ids

    # Check length of client_ids.
    self.assertLen(client_ids, 7)

    # Check that they are all strings.
    for client_id in client_ids:
      self.assertIsInstance(client_id, str)

    # Check ids are sorted.
    self.assertListEqual(client_ids, sorted(client_ids))

  def test_fail_on_bad_client_id(self):
    client_data = hdf5_client_data.HDF5ClientData(
        TransformingClientDataTest.test_data_filepath)
    transformed_client_data = transforming_client_data.TransformingClientData(
        client_data, _test_transform_cons, 7)

    # The following three should be valid.
    transformed_client_data.create_tf_dataset_for_client('CLIENT A_1')
    transformed_client_data.create_tf_dataset_for_client('CLIENT B_1')
    transformed_client_data.create_tf_dataset_for_client('CLIENT A_2')

    # This should not be valid: no corresponding client.
    with self.assertRaisesRegex(
        ValueError, 'client_id must be a valid string from client_ids.'):
      transformed_client_data.create_tf_dataset_for_client('CLIENT D_0')

    # This should not be valid: index out of range.
    with self.assertRaisesRegex(
        ValueError, 'client_id must be a valid string from client_ids.'):
      transformed_client_data.create_tf_dataset_for_client('CLIENT B_2')

  def test_create_tf_dataset_for_client(self):
    client_data = hdf5_client_data.HDF5ClientData(
        TransformingClientDataTest.test_data_filepath)

    transformed_client_data = transforming_client_data.TransformingClientData(
        client_data, _test_transform_cons, 9)

    for client_id in transformed_client_data.client_ids:
      tf_dataset = transformed_client_data.create_tf_dataset_for_client(
          client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)

      pattern = r'^(.*)_(\d*)$'
      match = re.search(pattern, client_id)
      client = match.group(1)
      index = int(match.group(2))
      for i, actual in enumerate(tf_dataset):
        actual = self.evaluate(actual)
        expected = {k: v[i].copy() for k, v in TEST_DATA[client].items()}
        expected['x'] += 10 * index
        self.assertCountEqual(actual, expected)
        for k, v in actual.items():
          self.assertAllEqual(v, expected[k])

  def test_create_tf_dataset_from_all_clients(self):
    client_data = hdf5_client_data.HDF5ClientData(
        TransformingClientDataTest.test_data_filepath)

    num_transformed_clients = 9
    transformed_client_data = transforming_client_data.TransformingClientData(
        client_data, _test_transform_cons, num_transformed_clients)
    expansion_factor = num_transformed_clients // len(TEST_DATA)

    tf_dataset = transformed_client_data.create_tf_dataset_from_all_clients()
    self.assertIsInstance(tf_dataset, tf.data.Dataset)

    expected_examples = []
    for expected_data in TEST_DATA.values():
      for index in range(expansion_factor):
        for i in range(len(expected_data['x'])):
          example = {k: v[i].copy() for k, v in expected_data.items()}
          example['x'] += 10 * index
          expected_examples.append(example)

    for actual in tf_dataset:
      actual = self.evaluate(actual)
      expected = expected_examples.pop(0)
      self.assertCountEqual(actual, expected)
    self.assertEmpty(expected_examples)


if __name__ == '__main__':
  tf.test.main()
