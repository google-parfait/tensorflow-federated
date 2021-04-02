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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data

TEST_DATA = {
    'CLIENT A': [[1, 2], [3, 4], [5, 6]],
    'CLIENT B': [[10, 11]],
    'CLIENT C': [[100, 101], [200, 201]],
    'CLIENT C2': [[100, 101], [202, 203]],
}
CLIENT_ID_NOT_IN_TEST_DATA = 'CLIENT D'
TEST_DATA_NOT_DICT = [[[1, 2]], [[1.0, 2.0]]]
ExampleNamedTuple = collections.namedtuple('ExampleNamedTuple', 'x')
TEST_DATA_WITH_NAMEDTUPLES = {
    'CLIENT A':
        ExampleNamedTuple(x=np.asarray([[1, 2], [3, 4], [5, 6]], dtype='i4'),),
}
TEST_DATA_WITH_INCONSISTENT_TYPE = {
    'CLIENT A': [[1, 2]],
    'CLIENT B': [[1.0, 2.0]],
}
TEST_DATA_WITH_TUPLES = {
    'CLIENT A': tuple([[1, 3, 5], [2, 4, 6]]),
    'CLIENT B': tuple([[10], [11]]),
    'CLIENT C': tuple([[100, 200], [101, 201]]),
    'CLIENT C2': tuple([[100, 202], [101, 203]]),
}
TEST_DATA_WITH_ORDEREDDICTS = {
    'CLIENT A':
        collections.OrderedDict(
            x=np.asarray([[1, 2], [3, 4], [5, 6]], dtype='i4'),
            y=np.asarray([4.0, 5.0, 6.0], dtype='f4'),
            z=np.asarray(['a', 'b', 'c'], dtype='S'),
        ),
    'CLIENT B':
        collections.OrderedDict(
            x=np.asarray([[10, 11]], dtype='i4'),
            y=np.asarray([7.0], dtype='f4'),
            z=np.asarray(['d'], dtype='S'),
        ),
    'CLIENT C':
        collections.OrderedDict(
            x=np.asarray([[100, 101], [200, 201]], dtype='i4'),
            y=np.asarray([8.0, 9.0], dtype='f4'),
            z=np.asarray(['e', 'f'], dtype='S'),
        ),
    'CLIENT C2':
        collections.OrderedDict(
            x=np.asarray([[100, 101], [202, 203]], dtype='i4'),
            y=np.asarray([100.0, 200.0], dtype='f4'),
            z=np.asarray(['abc', 'def'], dtype='S'),
        ),
}
TEST_DATA_WITH_PART_LIST_AND_PART_DICT = {
    'CLIENT A':
        collections.OrderedDict(
            x=np.asarray([[1, 2], [3, 4], [5, 6]], dtype='i4'),
            y=np.asarray([4.0, 5.0, 6.0], dtype='f4'),
            z=np.asarray(['a', 'b', 'c'], dtype='S'),
        ),
    'CLIENT B': [[1.0, 2.0]],
}


class TestClientDataTest(tf.test.TestCase):

  def assertSameDatasets(self, a_dataset, b_dataset):
    self.assertEqual(len(a_dataset), len(b_dataset))
    for a, b in zip(a_dataset, b_dataset):
      self.assertAllEqual(a, b)

  def assertSameDatasetsOfDicts(self, a_dataset, b_dataset):
    self.assertEqual(len(a_dataset), len(b_dataset))
    for a_dict, b_dict in zip(a_dataset, b_dataset):
      # Check that everything in a_dataset is an exact match for the contents
      # of b_dataset at the corresponding index.
      self.assertDictsWithEqualTensors(a_dict, b_dict)

  def assertDictsWithEqualTensors(self, a_dict, b_dict):
    self.assertEqual(len(a_dict), len(b_dict))
    self.assertAllEqual(a_dict.keys(), b_dict.keys())
    for key in a_dict:
      self.assertAllEqual(a_dict[key].numpy(), b_dict[key].numpy())

  def test_basic(self):
    tensor_slices_dict = {'a': [1, 2, 3], 'b': [4, 5]}
    client_data = from_tensor_slices_client_data.TestClientData(
        tensor_slices_dict)
    self.assertCountEqual(client_data.client_ids, ['a', 'b'])
    self.assertEqual(client_data.element_type_structure,
                     tf.TensorSpec(shape=(), dtype=tf.int32))

    def as_list(dataset):
      return [self.evaluate(x) for x in dataset]

    self.assertEqual(
        as_list(client_data.create_tf_dataset_for_client('a')), [1, 2, 3])
    self.assertEqual(
        as_list(client_data.create_tf_dataset_for_client('b')), [4, 5])

  def test_where_client_data_is_tensors(self):
    client_data = from_tensor_slices_client_data.TestClientData(TEST_DATA)
    self.assertCountEqual(TEST_DATA.keys(), client_data.client_ids)

    self.assertEqual(client_data.element_type_structure,
                     tf.TensorSpec(shape=(2,), dtype=tf.int32))

    for client_id in TEST_DATA:
      self.assertSameDatasets(
          tf.data.Dataset.from_tensor_slices(TEST_DATA[client_id]),
          client_data.create_tf_dataset_for_client(client_id))

  def test_where_client_data_is_tuples(self):
    client_data = from_tensor_slices_client_data.TestClientData(
        TEST_DATA_WITH_TUPLES)
    self.assertCountEqual(TEST_DATA_WITH_TUPLES.keys(), client_data.client_ids)

    self.assertEqual(client_data.element_type_structure, (tf.TensorSpec(
        shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32)))

    for client_id in TEST_DATA_WITH_TUPLES:
      self.assertSameDatasets(
          tf.data.Dataset.from_tensor_slices(TEST_DATA_WITH_TUPLES[client_id]),
          client_data.create_tf_dataset_for_client(client_id))

  def test_where_client_data_is_ordered_dicts(self):
    client_data = from_tensor_slices_client_data.TestClientData(
        TEST_DATA_WITH_ORDEREDDICTS)
    self.assertCountEqual(TEST_DATA_WITH_ORDEREDDICTS.keys(),
                          client_data.client_ids)
    self.assertEqual(
        collections.OrderedDict([
            ('x', tf.TensorSpec(shape=(2,), dtype=tf.int32)),
            ('y', tf.TensorSpec(shape=(), dtype=tf.float32)),
            ('z', tf.TensorSpec(shape=(), dtype=tf.string))
        ]), client_data.element_type_structure)

    for client_id in TEST_DATA_WITH_ORDEREDDICTS:
      self.assertSameDatasetsOfDicts(
          tf.data.Dataset.from_tensor_slices(
              TEST_DATA_WITH_ORDEREDDICTS[client_id]),
          client_data.create_tf_dataset_for_client(client_id))

  def test_raises_error_if_empty_client_found(self):
    with self.assertRaises(ValueError):
      from_tensor_slices_client_data.TestClientData({'a': []})

  def test_init_raises_error_if_slices_is_not_dict(self):
    with self.assertRaises(TypeError):
      from_tensor_slices_client_data.TestClientData(TEST_DATA_NOT_DICT)

  def test_init_raises_error_if_slices_are_namedtuples(self):
    with self.assertRaises(TypeError):
      from_tensor_slices_client_data.TestClientData(TEST_DATA_WITH_NAMEDTUPLES)

  def test_init_raises_error_if_slices_are_inconsistent_type(self):
    with self.assertRaises(TypeError):
      from_tensor_slices_client_data.TestClientData(
          TEST_DATA_WITH_INCONSISTENT_TYPE)

  def test_init_raises_error_if_slices_are_part_list_and_part_dict(self):
    with self.assertRaises(TypeError):
      from_tensor_slices_client_data.TestClientData(
          TEST_DATA_WITH_PART_LIST_AND_PART_DICT)

  def test_shuffle_client_ids(self):
    tensor_slices_dict = {'a': [1, 1], 'b': [2, 2, 2], 'c': [3], 'd': [4, 4]}
    all_examples = [1, 1, 2, 2, 2, 3, 4, 4]
    client_data = from_tensor_slices_client_data.TestClientData(
        tensor_slices_dict)

    def get_flat_dataset(seed):
      ds = client_data.create_tf_dataset_from_all_clients(seed=seed)
      return [x.numpy() for x in ds]

    d1 = get_flat_dataset(123)
    d2 = get_flat_dataset(456)
    self.assertNotEqual(d1, d2)  # Different random seeds, different order.
    self.assertCountEqual(d1, all_examples)
    self.assertCountEqual(d2, all_examples)

    # Test that the default behavior is to use a fresh random seed.
    # We could get unlucky, but we are very unlikely to get unlucky
    # 100 times in a row.
    found_not_equal = False
    for _ in range(100):
      if get_flat_dataset(seed=None) != get_flat_dataset(seed=None):
        found_not_equal = True
        break
    self.assertTrue(found_not_equal)

  def test_dataset_computation_where_client_data_is_tensors(self):
    client_data = from_tensor_slices_client_data.TestClientData(TEST_DATA)

    dataset_computation = client_data.dataset_computation
    self.assertIsInstance(dataset_computation, computation_base.Computation)

    expected_dataset_comp_type_signature = computation_types.FunctionType(
        computation_types.to_type(tf.string),
        computation_types.SequenceType(
            computation_types.TensorType(
                client_data.element_type_structure.dtype,
                tf.TensorShape(None))))

    self.assertTrue(
        dataset_computation.type_signature.is_equivalent_to(
            expected_dataset_comp_type_signature))

    # Iterate over each client, invoking the dataset_computation and ensuring
    # we received a tf.data.Dataset with the correct data.
    for client_id, expected_data in TEST_DATA.items():
      tf_dataset = dataset_computation(client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)
      self.assertLen(expected_data, tf_dataset.cardinality())
      # Check that everything in tf_dataset is an exact match for the contents
      # of expected_data at the corresponding index.
      for expected, actual in zip(expected_data, tf_dataset):
        self.assertAllEqual(np.asarray(expected), actual.numpy())

  def test_dataset_computation_where_client_data_is_tuples(self):
    client_data = from_tensor_slices_client_data.TestClientData(
        TEST_DATA_WITH_TUPLES)

    dataset_computation = client_data.dataset_computation
    self.assertIsInstance(dataset_computation, computation_base.Computation)

    expected_dataset_comp_type_signature = computation_types.FunctionType(
        computation_types.to_type(tf.string),
        computation_types.SequenceType(
            computation_types.TensorType(
                client_data.element_type_structure[0].dtype,
                tf.TensorShape(None))))

    self.assertTrue(
        dataset_computation.type_signature.is_equivalent_to(
            expected_dataset_comp_type_signature))

    # Iterate over each client, invoking the dataset_computation and ensuring
    # we received a tf.data.Dataset with the correct data.
    for client_id, expected_data in TEST_DATA_WITH_TUPLES.items():
      tf_dataset = dataset_computation(client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)
      self.assertLen(expected_data, tf_dataset.cardinality())
      # Check that everything in tf_dataset is an exact match for the contents
      # of expected_data at the corresponding index.
      for expected, actual in zip(expected_data, tf_dataset):
        self.assertAllEqual(np.asarray(expected), actual.numpy())

  def test_dataset_computation_where_client_data_is_ordered_dicts(self):
    client_data = from_tensor_slices_client_data.TestClientData(
        TEST_DATA_WITH_ORDEREDDICTS)

    dataset_computation = client_data.dataset_computation
    self.assertIsInstance(dataset_computation, computation_base.Computation)

    expected_dataset_comp_type_signature = computation_types.FunctionType(
        computation_types.to_type(tf.string),
        computation_types.SequenceType(
            collections.OrderedDict([
                ('x',
                 computation_types.TensorType(
                     client_data.element_type_structure['x'].dtype,
                     tf.TensorShape(2))),
                ('y',
                 computation_types.TensorType(
                     client_data.element_type_structure['y'].dtype, None)),
                ('z',
                 computation_types.TensorType(
                     client_data.element_type_structure['z'].dtype, None))
            ])))

    self.assertTrue(
        dataset_computation.type_signature.is_equivalent_to(
            expected_dataset_comp_type_signature))

    # Iterate over each client, invoking the computation and ensuring
    # we received a tf.data.Dataset with the correct data.
    for client_id in TEST_DATA_WITH_ORDEREDDICTS:
      dataset = dataset_computation(client_id)
      self.assertIsInstance(dataset, tf.data.Dataset)

      expected_dataset = tf.data.Dataset.from_tensor_slices(
          TEST_DATA_WITH_ORDEREDDICTS[client_id])
      self.assertSameDatasetsOfDicts(expected_dataset, dataset)

  def test_dataset_computation_raises_error_if_unknown_client_id(self):
    client_data = from_tensor_slices_client_data.TestClientData(TEST_DATA)

    dataset_computation = client_data.dataset_computation

    with self.assertRaises(tf.errors.InvalidArgumentError):
      dataset_computation(CLIENT_ID_NOT_IN_TEST_DATA)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
