# Copyright 2022, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import dataset_data_source


class DatasetDataSourceIteratorTest(parameterized.TestCase):

  def test_init_does_not_raise_type_error(self):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )

    try:
      dataset_data_source.DatasetDataSourceIterator(datasets, federated_type)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_datasets(self, datasets):
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )

    with self.assertRaises(TypeError):
      dataset_data_source.DatasetDataSourceIterator(datasets, federated_type)

  # pyformat: disable
  @parameterized.named_parameters(
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('placement', computation_types.PlacementType()),
      ('sequence', computation_types.SequenceType(np.int32)),
      ('struct', computation_types.StructWithPythonType(
          [np.bool_, np.int32, np.str_], list)),
      ('tensor', computation_types.TensorType(np.int32)),
  )
  # pyformat: enable
  def test_init_raises_type_error_with_federated_type(self, federated_type):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3

    with self.assertRaises(TypeError):
      dataset_data_source.DatasetDataSourceIterator(datasets, federated_type)

  def test_init_raises_value_error_with_datasets_empty(self):
    datasets = []
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )

    with self.assertRaises(ValueError):
      dataset_data_source.DatasetDataSourceIterator(datasets, federated_type)

  def test_init_raises_value_error_with_datasets_different_types(self):
    datasets = [
        tf.data.Dataset.from_tensor_slices([1, 2, 3]),
        tf.data.Dataset.from_tensor_slices(['a', 'b', 'c']),
    ]
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )

    with self.assertRaises(ValueError):
      dataset_data_source.DatasetDataSourceIterator(datasets, federated_type)

  @parameterized.named_parameters(
      ('zero', 0),
      ('one', 1),
      ('two', 2),
  )
  def test_select_returns_datasets_with_k(self, k):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )
    iterator = dataset_data_source.DatasetDataSourceIterator(
        datasets, federated_type
    )

    actual_datasets = iterator.select(k)

    self.assertLen(actual_datasets, k)
    for actual_dataset in actual_datasets:
      expected_dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
      self.assertSameElements(actual_dataset, expected_dataset)

  @parameterized.named_parameters(
      ('str', 'a'),
      ('list', []),
  )
  def test_select_raises_type_error_with_k(self, k):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )
    iterator = dataset_data_source.DatasetDataSourceIterator(
        datasets, federated_type
    )

    with self.assertRaises(TypeError):
      iterator.select(k)

  @parameterized.named_parameters(
      ('none', None),
      ('negative', -1),
      ('greater', 4),
  )
  def test_select_raises_value_error_with_k(self, k):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )
    iterator = dataset_data_source.DatasetDataSourceIterator(
        datasets, federated_type
    )

    with self.assertRaises(ValueError):
      iterator.select(k)

  def test_serializable(self):
    datasets = [tf.data.Dataset.from_tensor_slices([1, 2, 3])] * 3
    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(np.int32), placements.CLIENTS
    )
    iterator = dataset_data_source.DatasetDataSourceIterator(
        datasets, federated_type
    )
    iterator_bytes = iterator.to_bytes()

    actual_iterator = dataset_data_source.DatasetDataSourceIterator.from_bytes(
        iterator_bytes
    )

    self.assertIsNot(actual_iterator, iterator)
    self.assertEqual(actual_iterator, iterator)


class DatasetDataSourceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', [1, 2, 3], np.int32),
      ('str', ['a', 'b', 'c'], np.str_),
  )
  def test_init_sets_federated_type(self, tensors, dtype):
    datasets = [tf.data.Dataset.from_tensor_slices(tensors)] * 3

    data_source = dataset_data_source.DatasetDataSource(datasets)

    federated_type = computation_types.FederatedType(
        computation_types.SequenceType(dtype), placements.CLIENTS
    )
    self.assertEqual(data_source.federated_type, federated_type)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', [True, 1, 'a']),
  )
  def test_init_raises_type_error_with_datasets(self, datasets):
    with self.assertRaises(TypeError):
      dataset_data_source.DatasetDataSource(datasets)

  def test_init_raises_value_error_with_datasets_empty(self):
    datasets = []

    with self.assertRaises(ValueError):
      dataset_data_source.DatasetDataSource(datasets)

  def test_init_raises_value_error_with_datasets_different_types(self):
    datasets = [
        tf.data.Dataset.from_tensor_slices([1, 2, 3]),
        tf.data.Dataset.from_tensor_slices(['a', 'b', 'c']),
    ]

    with self.assertRaises(ValueError):
      dataset_data_source.DatasetDataSource(datasets)


if __name__ == '__main__':
  absltest.main()
