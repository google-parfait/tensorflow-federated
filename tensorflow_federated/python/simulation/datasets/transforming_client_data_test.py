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

import re

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data
from tensorflow_federated.python.simulation.datasets import transforming_client_data

TEST_DATA = {
    'CLIENT A': {
        'x': [[1, 2], [3, 4], [5, 6]],
        'y': [4.0, 5.0, 6.0],
        'z': ['a', 'b', 'c'],
    },
    'CLIENT B': {
        'x': [[10, 11]],
        'y': [7.0],
        'z': ['d'],
    },
    'CLIENT C': {
        'x': [[100, 101], [200, 201]],
        'y': [8.0, 9.0],
        'z': ['e', 'f'],
    },
}

TEST_CLIENT_DATA = from_tensor_slices_client_data.TestClientData(TEST_DATA)

# Client data class with only integers, in order to test exact correctness for
# transformations.
INTEGER_TEST_DATA = {
    'CLIENT A': {
        'x': [[1, 2], [3, 4], [5, 6]],
        'y': [4, 5, 6],
        'z': [7, 8, 9],
    },
    'CLIENT B': {
        'x': [[10, 11]],
        'y': [7],
        'z': [8],
    },
    'CLIENT C': {
        'x': [[100, 101], [200, 201]],
        'y': [8, 9],
        'z': [10, 11],
    },
}


def _test_transform_cons(raw_client_id, index):
  del raw_client_id

  def fn(data):
    return {'x': data['x'] + 10 * index, 'y': data['y'], 'z': data['z']}

  return fn


class TransformingClientDataTest(tf.test.TestCase, absltest.TestCase):

  def test_client_ids_property(self):
    num_transformed_clients = 7
    transformed_client_data = transforming_client_data.TransformingClientData(
        TEST_CLIENT_DATA, _test_transform_cons, num_transformed_clients)
    client_ids = transformed_client_data.client_ids
    self.assertLen(client_ids, num_transformed_clients)
    for client_id in client_ids:
      self.assertIsInstance(client_id, str)
    self.assertListEqual(client_ids, sorted(client_ids))
    self.assertTrue(transformed_client_data._has_pseudo_clients)

  def test_default_num_transformed_clients(self):
    transformed_client_data = transforming_client_data.TransformingClientData(
        TEST_CLIENT_DATA, _test_transform_cons)
    client_ids = transformed_client_data.client_ids
    self.assertLen(client_ids, len(TEST_DATA))
    self.assertFalse(transformed_client_data._has_pseudo_clients)

  def test_fail_on_bad_client_id(self):
    transformed_client_data = transforming_client_data.TransformingClientData(
        TEST_CLIENT_DATA, _test_transform_cons, 7)
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
    tensor_slices_dataset = from_tensor_slices_client_data.TestClientData(
        INTEGER_TEST_DATA)
    transformed_client_data = transforming_client_data.TransformingClientData(
        tensor_slices_dataset, _test_transform_cons, 9)
    for client_id in transformed_client_data.client_ids:
      actual_dataset = transformed_client_data.create_tf_dataset_for_client(
          client_id)
      self.assertIsInstance(actual_dataset, tf.data.Dataset)
      pattern = r'^(.*)_(\d*)$'
      match = re.search(pattern, client_id)
      client = match.group(1)
      index = int(match.group(2))
      base_client_dataset = tensor_slices_dataset.create_tf_dataset_for_client(
          client)
      expected_dataset = base_client_dataset.map(
          _test_transform_cons(client, index))
      self.assertAllClose(
          list(actual_dataset.as_numpy_iterator()),
          list(expected_dataset.as_numpy_iterator()))

  def test_create_tf_dataset_from_all_clients(self):
    flat_data = {'CLIENT A': [1], 'CLIENT B': [2], 'CLIENT C': [3]}

    def get_transform_fn(client_id, index):
      del client_id
      return lambda batch: batch * index

    num_transformed_clients = 9
    flat_client_data = from_tensor_slices_client_data.TestClientData(flat_data)
    transformed_client_data = transforming_client_data.TransformingClientData(
        flat_client_data, get_transform_fn, num_transformed_clients)
    expansion_factor = num_transformed_clients // len(TEST_DATA)
    actual_dataset = transformed_client_data.create_tf_dataset_from_all_clients(
    )
    self.assertIsInstance(actual_dataset, tf.data.Dataset)
    actual_examples = [batch.numpy() for batch in actual_dataset]

    expected_examples = []
    for client in flat_client_data.client_ids:
      for index in range(expansion_factor):
        base_client_dataset = flat_client_data.create_tf_dataset_for_client(
            client)
        transformed_dataset = base_client_dataset.map(
            get_transform_fn(client, index))
        transformed_examples = [batch.numpy() for batch in transformed_dataset]
        expected_examples += transformed_examples

    self.assertCountEqual(actual_examples, expected_examples)


if __name__ == '__main__':
  tf.test.main()
