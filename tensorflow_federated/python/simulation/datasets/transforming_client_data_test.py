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

import collections
import re

import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data
from tensorflow_federated.python.simulation.datasets import transforming_client_data

TEST_DATA = {
    'CLIENT A':
        collections.OrderedDict(
            x=[[1, 2], [3, 4], [5, 6]],
            y=[4.0, 5.0, 6.0],
            z=['a', 'b', 'c'],
        ),
    'CLIENT B':
        collections.OrderedDict(
            x=[[10, 11]],
            y=[7.0],
            z=['d'],
        ),
    'CLIENT C':
        collections.OrderedDict(
            x=[[100, 101], [200, 201]],
            y=[8.0, 9.0],
            z=['e', 'f'],
        ),
}

TEST_CLIENT_DATA = from_tensor_slices_client_data.TestClientData(TEST_DATA)


def _make_transform_expanded(client_id):
  index_str = tf.strings.split(client_id, sep='_', maxsplit=1)[0]
  index = tf.cast(tf.strings.to_number(index_str), tf.int32)

  def fn(data):
    return collections.OrderedDict([('x', data['x'] + 10 * index),
                                    ('y', data['y']), ('z', data['z'])])

  return fn


def _make_transform_raw(client_id):
  del client_id

  def fn(data):
    data['x'] = data['x'] + 10
    return data

  return fn


NUM_EXPANDED_CLIENTS = 3


def test_expand_client_id(client_id):
  return [str(i) + '_' + client_id for i in range(NUM_EXPANDED_CLIENTS)]


def test_reduce_client_id(client_id):
  return tf.strings.split(client_id, sep='_')[1]


TRANSFORMED_CLIENT_DATA = transforming_client_data.TransformingClientData(
    TEST_CLIENT_DATA, _make_transform_expanded, test_expand_client_id,
    test_reduce_client_id)


class TransformingClientDataTest(tf.test.TestCase):

  def test_client_ids_property(self):
    num_transformed_clients = len(TEST_DATA) * NUM_EXPANDED_CLIENTS
    client_ids = TRANSFORMED_CLIENT_DATA.client_ids
    self.assertLen(client_ids, num_transformed_clients)
    for client_id in client_ids:
      self.assertIsInstance(client_id, str)
    self.assertListEqual(client_ids, sorted(client_ids))

  def test_default_num_transformed_clients(self):
    transformed_client_data = transforming_client_data.TransformingClientData(
        TEST_CLIENT_DATA, _make_transform_raw)
    client_ids = transformed_client_data.client_ids
    self.assertCountEqual(client_ids, TEST_DATA.keys())

  def test_fail_on_bad_client_id(self):
    # The following three should be valid.
    TRANSFORMED_CLIENT_DATA.create_tf_dataset_for_client('0_CLIENT A')
    TRANSFORMED_CLIENT_DATA.create_tf_dataset_for_client('1_CLIENT B')
    TRANSFORMED_CLIENT_DATA.create_tf_dataset_for_client('0_CLIENT C')

    # This should not be valid: no prefix.
    with self.assertRaisesRegex(ValueError,
                                'is not a client in this ClientData'):
      TRANSFORMED_CLIENT_DATA.create_tf_dataset_for_client('CLIENT A')

    # This should not be valid: no corresponding client.
    with self.assertRaisesRegex(ValueError,
                                'is not a client in this ClientData'):
      TRANSFORMED_CLIENT_DATA.create_tf_dataset_for_client('0_CLIENT D')

    # This should not be valid: index out of range.
    with self.assertRaisesRegex(ValueError,
                                'is not a client in this ClientData'):
      TRANSFORMED_CLIENT_DATA.create_tf_dataset_for_client('3_CLIENT B')

  def test_dataset_computation(self):
    for client_id in TRANSFORMED_CLIENT_DATA.client_ids:
      actual_dataset = TRANSFORMED_CLIENT_DATA.dataset_computation(client_id)
      self.assertIsInstance(actual_dataset, tf.data.Dataset)
      pattern = r'^(\d*)_(.*)$'
      match = re.search(pattern, client_id)
      client = match.group(2)
      base_client_dataset = TEST_CLIENT_DATA.create_tf_dataset_for_client(
          client)
      expected_dataset = base_client_dataset.map(
          _make_transform_expanded(client_id))
      for actual_client_data, expected_client_data in zip(
          actual_dataset.as_numpy_iterator(),
          expected_dataset.as_numpy_iterator()):
        for actual_datum, expected_datum in zip(actual_client_data,
                                                expected_client_data):
          self.assertEqual(actual_datum, expected_datum)

  def test_create_tf_dataset_from_all_clients(self):

    # Expands `CLIENT {N}` into N clients which add range(N) to the feature.
    def expand_client_id(client_id):
      return [client_id + '-' + str(i) for i in range(int(client_id[-1]))]

    def make_transform_fn(client_id):
      split_client_id = tf.strings.split(client_id, '-')
      index = tf.cast(tf.strings.to_number(split_client_id[1]), tf.int32)
      return lambda x: x + index

    reduce_client_id = lambda client_id: tf.strings.split(client_id, sep='-')[0]

    # pyformat: disable
    raw_data = {
        'CLIENT 1': [0],        # expanded to [0]
        'CLIENT 2': [1, 3, 5],  # expanded to [1, 3, 5], [2, 4, 6]
        'CLIENT 3': [7, 10]     # expanded to [7, 10], [8, 11], [9, 12]
    }
    # pyformat: enable
    client_data = from_tensor_slices_client_data.TestClientData(raw_data)
    transformed_client_data = transforming_client_data.TransformingClientData(
        client_data, make_transform_fn, expand_client_id, reduce_client_id)

    flat_data = transformed_client_data.create_tf_dataset_from_all_clients()
    self.assertIsInstance(flat_data, tf.data.Dataset)
    all_features = [batch.numpy() for batch in flat_data]
    self.assertCountEqual(all_features, range(13))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
