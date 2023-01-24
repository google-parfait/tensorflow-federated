# Copyright 2021, The TensorFlow Federated Authors.
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
from absl.testing import parameterized

import tensorflow as tf

from tensorflow_federated.python.analytics import data_processing
from tensorflow_federated.python.analytics import histogram_test_utils


class DataProcessingTest(
    parameterized.TestCase, histogram_test_utils.HistogramTest
):

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 2, []),
      (
          'batch_size_1',
          ['a', 'b', 'a', 'b', 'c'],
          1,
          [b'a', b'b', b'a', b'b', b'c'],
      ),
      (
          'batch_size_3',
          ['a', 'b', 'a', 'b', 'c'],
          3,
          [b'a', b'b', b'a', b'b', b'c'],
      ),
  )
  def test_all_elements_returns_expected_values(
      self, input_data, batch_size, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_all_elements(ds)
    self.assertAllEqual(all_elements, expected_result)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 3, []),
      (
          'string',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          1,
          3,
          ['abc', 'abc', 'bcd', 'bcd', 'def'],
      ),
      (
          'unicode',
          ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'],
          3,
          6,
          ['新年', '新年', '☺️', '☺️', '☺️'],
      ),
  )
  def test_all_elements_with_max_len_returns_expected_values(
      self, input_data, batch_size, string_max_bytes, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_all_elements(
        ds, string_max_bytes=string_max_bytes
    )
    all_elements = [
        elem.decode('utf-8', 'ignore') for elem in all_elements.numpy()
    ]
    self.assertEqual(all_elements, expected_result)

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_all_elements_raise_value_error(self, dataset_rank):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    batch_size = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaises(ValueError):
      data_processing.get_all_elements(ds)

  @parameterized.named_parameters(
      ('string_max_bytes_0', 0),
      ('string_max_bytes_neg', -1),
  )
  def test_all_elements_raise_params_value_error(self, string_max_bytes):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )

    with self.assertRaisesRegex(
        ValueError, '`string_max_bytes` must be at least 1.'
    ):
      data_processing.get_all_elements(ds, string_max_bytes=string_max_bytes)

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_all_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_all_elements(ds)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 2, 10, [], []),
      (
          'batch_size_1',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          4,
          [b'a', b'b', b'c'],
          [2, 1, 1],
      ),
      (
          'batch_size_3',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          3,
          4,
          [b'a', b'b'],
          [2, 1],
      ),
  )
  def test_capped_elements_with_counts_returns_expected_values(
      self,
      input_data,
      batch_size,
      max_user_contribution,
      expected_elements,
      expected_counts,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    capped_elements, counts = data_processing.get_capped_elements_with_counts(
        ds, max_user_contribution=max_user_contribution, batch_size=batch_size
    )
    self.assert_histograms_all_close(
        capped_elements, counts, expected_elements, expected_counts
    )

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 10, 3, [], []),
      (
          'string_1',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          1,
          4,
          3,
          ['abc', 'bcd'],
          [2, 2],
      ),
      (
          'string_2',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          2,
          2,
          2,
          ['ab'],
          [2],
      ),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 2, 3, 6, ['新年'], [2]),
  )
  def test_capped_elements_with_counts_max_len_returns_expected_values(
      self,
      input_data,
      batch_size,
      max_user_contribution,
      string_max_bytes,
      expected_elements,
      expected_counts,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    capped_elements, counts = data_processing.get_capped_elements_with_counts(
        ds,
        batch_size=batch_size,
        max_user_contribution=max_user_contribution,
        string_max_bytes=string_max_bytes,
    )
    capped_elements = [
        elem.decode('utf-8', 'ignore') for elem in capped_elements.numpy()
    ]
    self.assert_histograms_all_close(
        capped_elements, counts, expected_elements, expected_counts
    )

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_capped_elements_with_counts_raise_rank_value_error(
      self, dataset_rank
  ):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    max_user_contribution = 3
    batch_size = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaisesRegex(
        ValueError, 'The shape of elements in `dataset` must be of rank 1.*'
    ):
      data_processing.get_capped_elements_with_counts(
          ds, max_user_contribution=max_user_contribution, batch_size=batch_size
      )

  @parameterized.named_parameters(
      ('batch_size', 5, 0, 10, '`batch_size` must be at least 1.'),
      (
          'max_user_contribution',
          -10,
          10,
          10,
          '`max_user_contribution` must be at least 1.',
      ),
      ('string_max_bytes', 10, 5, 0, '`string_max_bytes` must be at least 1.'),
  )
  def test_capped_elements_with_counts_raise_params_value_error(
      self, max_user_contribution, batch_size, string_max_bytes, raises_regex
  ):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )
    with self.assertRaisesRegex(ValueError, raises_regex):
      data_processing.get_capped_elements_with_counts(
          ds,
          max_user_contribution=max_user_contribution,
          batch_size=batch_size,
          string_max_bytes=string_max_bytes,
      )

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_capped_elements_with_counts_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_capped_elements_with_counts(
          ds, max_user_contribution=10, batch_size=1
      )

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 2, 10, []),
      (
          'batch_size_1',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          4,
          [b'a', b'b', b'a', b'c'],
      ),
      (
          'batch_size_3',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          3,
          4,
          [b'a', b'b', b'a'],
      ),
  )
  def test_capped_elements_returns_expected_values(
      self, input_data, batch_size, max_user_contribution, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=max_user_contribution, batch_size=batch_size
    )
    self.assertAllEqual(capped_elements, expected_result)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 10, 3, []),
      (
          'string_1',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          1,
          4,
          3,
          ['abc', 'abc', 'bcd', 'bcd'],
      ),
      (
          'string_2',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          2,
          2,
          2,
          ['ab', 'ab'],
      ),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 2, 3, 6, ['新年', '新年']),
  )
  def test_capped_elements_with_max_len_returns_expected_values(
      self,
      input_data,
      batch_size,
      max_user_contribution,
      string_max_bytes,
      expected_result,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_capped_elements(
        ds,
        batch_size=batch_size,
        max_user_contribution=max_user_contribution,
        string_max_bytes=string_max_bytes,
    )
    all_elements = [
        elem.decode('utf-8', 'ignore') for elem in all_elements.numpy()
    ]
    self.assertEqual(all_elements, expected_result)

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_capped_elements_raise_rank_value_error(self, dataset_rank):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    max_user_contribution = 3
    batch_size = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaisesRegex(
        ValueError, 'The shape of elements in `dataset` must be of rank 1.*'
    ):
      data_processing.get_capped_elements(
          ds, max_user_contribution=max_user_contribution, batch_size=batch_size
      )

  @parameterized.named_parameters(
      ('batch_size', 5, 0, 10, '`batch_size` must be at least 1.'),
      (
          'max_user_contribution',
          -10,
          10,
          10,
          '`max_user_contribution` must be at least 1.',
      ),
      ('string_max_bytes', 10, 5, 0, '`string_max_bytes` must be at least 1.'),
  )
  def test_capped_elements_raise_params_value_error(
      self, max_user_contribution, batch_size, string_max_bytes, raises_regex
  ):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )
    with self.assertRaisesRegex(ValueError, raises_regex):
      data_processing.get_capped_elements(
          ds,
          max_user_contribution=max_user_contribution,
          batch_size=batch_size,
          string_max_bytes=string_max_bytes,
      )

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_capped_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_capped_elements(
          ds, max_user_contribution=10, batch_size=1
      )

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, []),
      (
          'string_dataset_batch_size_1',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          [b'a', b'b', b'c'],
      ),
      (
          'string_dataset_batch_size_3',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          3,
          [b'a', b'b', b'c'],
      ),
  )
  def test_unique_elements_returns_expected_values(
      self, input_data, batch_size, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    unique_elements = data_processing.get_unique_elements(ds)
    self.assertAllEqual(unique_elements, expected_result)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 3, []),
      (
          'string',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          1,
          3,
          ['abc', 'bcd', 'def'],
      ),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 3, 6, ['新年', '☺️']),
  )
  def test_get_unique_elements_with_max_len_returns_expected_values(
      self, input_data, batch_size, string_max_bytes, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    unique_elements = data_processing.get_unique_elements(
        ds, string_max_bytes=string_max_bytes
    )
    unique_elements = [
        elem.decode('utf-8', 'ignore') for elem in unique_elements.numpy()
    ]
    self.assertSetEqual(set(unique_elements), set(expected_result))

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_unique_elements_raise_value_error(self, dataset_rank):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    batch_size = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaises(ValueError):
      data_processing.get_unique_elements(ds)

  @parameterized.named_parameters(
      ('string_max_bytes_0', 0),
      ('string_max_bytes_neg', -1),
  )
  def test_unique_elements_raise_params_value_error(self, string_max_bytes):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )

    with self.assertRaisesRegex(
        ValueError, '`string_max_bytes` must be at least 1.'
    ):
      data_processing.get_unique_elements(ds, string_max_bytes=string_max_bytes)

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_get_unique_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_unique_elements(ds)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, [], []),
      (
          'string_dataset_batch_size_1',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          [b'a', b'b', b'c'],
          [2, 2, 3],
      ),
      (
          'string_dataset_batch_size_3',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          3,
          [b'a', b'b', b'c'],
          [2, 2, 3],
      ),
  )
  def test_unique_elements_with_counts_returns_expected_values(
      self, input_data, batch_size, expected_elements, expected_counts
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    (unique_elements, counts) = data_processing.get_unique_elements_with_counts(
        ds
    )
    self.assertAllEqual(unique_elements, expected_elements)
    self.assertAllEqual(counts, expected_counts)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 3, [], []),
      (
          'string',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'def'],
          1,
          3,
          ['abc', 'bcd', 'def'],
          [2, 2, 1],
      ),
      (
          'unicode',
          ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'],
          3,
          6,
          ['新年', '☺️'],
          [2, 3],
      ),
  )
  def test_get_unique_elements_with_counts_max_len_returns_expected_values(
      self,
      input_data,
      batch_size,
      string_max_bytes,
      expected_elements,
      expected_counts,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    unique_elements, counts = data_processing.get_unique_elements_with_counts(
        ds, string_max_bytes=string_max_bytes
    )
    unique_elements = [
        elem.decode('utf-8', 'ignore') for elem in unique_elements.numpy()
    ]
    self.assert_histograms_all_close(
        unique_elements, counts, expected_elements, expected_counts
    )

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_unique_elements_with_counts_raise_value_error(self, dataset_rank):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    batch_size = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaises(ValueError):
      data_processing.get_unique_elements_with_counts(ds)

  @parameterized.named_parameters(
      ('string_max_bytes_0', 0),
      ('string_max_bytes_neg', -1),
  )
  def test_unique_elements_with_counts_raise_params_value_error(
      self, string_max_bytes
  ):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )

    with self.assertRaisesRegex(
        ValueError, '`string_max_bytes` must be at least 1.'
    ):
      data_processing.get_unique_elements_with_counts(
          ds, string_max_bytes=string_max_bytes
      )

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_unique_elements_with_counts_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_unique_elements_with_counts(ds)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 1, []),
      (
          'string_batch_size_1_max_contrib_2',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          2,
          [b'a', b'c'],
      ),
      (
          'string_batch_size_3_max_contrib_2',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          3,
          2,
          [b'a', b'c'],
      ),
      (
          'string_batch_size_1_max_contrib_3',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          3,
          [b'a', b'b', b'c'],
      ),
      (
          'string_batch_size_2_max_contrib_4',
          ['a', 'b', 'c', 'd', 'e', 'a'],
          2,
          4,
          [b'a', b'b', b'c', b'd'],
      ),
  )
  def test_get_top_elements_returns_expected_values(
      self, input_data, batch_size, max_user_contribution, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_elements(ds, max_user_contribution)
    self.assertSetEqual(set(top_elements.numpy()), set(expected_result))

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 1, 3, []),
      (
          'max_contrib_max_string_len',
          ['abc', 'abcd', 'def', 'ghijk'],
          3,
          2,
          3,
          ['abc', 'def'],
      ),
      (
          'max_string_len',
          ['abcd', 'abcde', 'bcd', 'bcdef'],
          1,
          10,
          3,
          ['abc', 'bcd'],
      ),
      (
          'max_contrib',
          ['abcd', 'abcde', 'bcd', 'bcdef'],
          3,
          2,
          10,
          ['abcd', 'abcde'],
      ),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 1, 2, 6, ['新年', '☺️']),
  )
  def test_get_top_elements_with_max_len_returns_expected_values(
      self,
      input_data,
      batch_size,
      max_user_contribution,
      string_max_bytes,
      expected_result,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_elements(
        ds, max_user_contribution, string_max_bytes=string_max_bytes
    )
    top_elements = [
        elem.decode('utf-8', 'ignore') for elem in top_elements.numpy()
    ]
    self.assertSetEqual(set(top_elements), set(expected_result))

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_get_top_elements_raise_rank_value_error(self, dataset_rank):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    batch_size = 1
    max_user_contribution = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaisesRegex(
        ValueError, 'The shape of elements in `dataset` must be of rank 1.*'
    ):
      data_processing.get_top_elements(ds, max_user_contribution)

  @parameterized.named_parameters(
      (
          'max_user_contribution_0',
          0,
          1,
          '`max_user_contribution` must be at least 1.',
      ),
      (
          'max_user_contribution_neg',
          -10,
          10,
          '`max_user_contribution` must be at least 1.',
      ),
      ('string_max_bytes_0', 10, 0, '`string_max_bytes` must be at least 1.'),
      (
          'string_max_bytes_neg',
          20,
          -1,
          '`string_max_bytes` must be at least 1.',
      ),
  )
  def test_get_top_elements_raise_params_value_error(
      self, max_user_contribution, string_max_bytes, raises_regex
  ):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )

    with self.assertRaisesRegex(ValueError, raises_regex):
      data_processing.get_top_elements(
          ds,
          max_user_contribution=max_user_contribution,
          string_max_bytes=string_max_bytes,
      )

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_get_top_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_top_elements(ds, max_user_contribution=10)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 1, []),
      (
          'string_batch_size_1_max_contrib_2',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          2,
          [b'a', b'a', b'c', b'c', b'c'],
      ),
      (
          'string_batch_size_3_max_contrib_2',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          3,
          2,
          [b'a', b'a', b'c', b'c', b'c'],
      ),
      (
          'string_batch_size_1_max_contrib_3',
          ['a', 'b', 'a', 'c', 'b', 'c', 'c'],
          1,
          3,
          [b'a', b'a', b'b', b'b', b'c', b'c', b'c'],
      ),
      (
          'string_batch_size_2_max_contrib_4',
          ['a', 'b', 'c', 'd', 'e', 'a'],
          2,
          4,
          [b'a', b'a', b'b', b'c', b'd'],
      ),
  )
  def test_get_top_multi_elements_returns_expected_values(
      self, input_data, batch_size, max_user_contribution, expected_result
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_multi_elements(
        ds, max_user_contribution
    )
    self.assertSetEqual(set(top_elements.numpy()), set(expected_result))

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 1, 3, []),
      (
          'max_contrib_max_string_len',
          [
              'abc',
              'abcd',
              'def',
              'ghijk',
              'ghijk',
              'ghijk',
              'abc',
              'abc',
              'def',
          ],
          3,
          2,
          3,
          ['abc', 'abc', 'abc', 'ghi', 'ghi'],
      ),
      (
          'max_string_len',
          ['abcd', 'abcde', 'bcd', 'bcdef'],
          1,
          10,
          3,
          ['abc', 'bcd'],
      ),
      (
          'max_contrib',
          ['abcd', 'abcde', 'bcd', 'bcdef', 'bcd', 'bcdef'],
          3,
          2,
          10,
          ['bcd', 'bcd', 'bcdef', 'bcdef'],
      ),
      (
          'unicode',
          ['新年快乐', '新年', '☺️😇', '☺️😇', '안녕하세요', '☺️', '안녕하세요'],
          1,
          2,
          6,
          ['新年', '☺️'],
      ),
  )
  def test_get_top_multi_elements_with_max_len_returns_expected_values(
      self,
      input_data,
      batch_size,
      max_user_contribution,
      string_max_bytes,
      expected_result,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_multi_elements(
        ds, max_user_contribution, string_max_bytes=string_max_bytes
    )
    top_elements = [
        elem.decode('utf-8', 'ignore') for elem in top_elements.numpy()
    ]
    self.assertSetEqual(set(top_elements), set(expected_result))

  @parameterized.named_parameters(
      ('rank_0', None),
      ('rank_2', 2),
      ('rank_3', 3),
  )
  def test_get_top_multi_elements_raise_rank_value_error(self, dataset_rank):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c'])
    batch_size = 1
    max_user_contribution = 1
    while dataset_rank:
      ds = ds.batch(batch_size=batch_size)
      dataset_rank -= 1

    with self.assertRaisesRegex(
        ValueError, 'The shape of elements in `dataset` must be of rank 1.*'
    ):
      data_processing.get_top_multi_elements(ds, max_user_contribution)

  @parameterized.named_parameters(
      (
          'max_user_contribution_0',
          0,
          1,
          '`max_user_contribution` must be at least 1.',
      ),
      (
          'max_user_contribution_neg',
          -10,
          10,
          '`max_user_contribution` must be at least 1.',
      ),
      ('string_max_bytes_0', 10, 0, '`string_max_bytes` must be at least 1.'),
      (
          'string_max_bytes_neg',
          20,
          -1,
          '`string_max_bytes` must be at least 1.',
      ),
  )
  def test_get_top_multi_elements_raise_params_value_error(
      self, max_user_contribution, string_max_bytes, raises_regex
  ):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b', 'c']).batch(
        batch_size=1
    )

    with self.assertRaisesRegex(ValueError, raises_regex):
      data_processing.get_top_multi_elements(
          ds,
          max_user_contribution=max_user_contribution,
          string_max_bytes=string_max_bytes,
      )

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_get_top_multi_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'
    ):
      data_processing.get_top_multi_elements(ds, max_user_contribution=10)


class ToStackedTensorTest(tf.test.TestCase):

  def test_basic_encoding(self):
    ds = tf.data.Dataset.range(5)

    encoded = data_processing.to_stacked_tensor(ds)

    self.assertIsInstance(encoded, tf.Tensor)
    self.assertEqual(encoded.shape, [5])
    self.assertAllEqual(encoded, list(range(5)))

  def test_nested_structure(self):
    ds = tf.data.Dataset.from_tensors(collections.OrderedDict(x=42))

    encoded = data_processing.to_stacked_tensor(ds)

    self.assertAllEqual(encoded, collections.OrderedDict(x=[42]))

  def test_single_element(self):
    ds = tf.data.Dataset.from_tensors([42])

    encoded = data_processing.to_stacked_tensor(ds)

    self.assertAllEqual(encoded, [[42]])

  def test_non_scalar_tensor(self):
    ds = tf.data.Dataset.from_tensors([[1, 2], [3, 4]])
    assert len(ds.element_spec.shape) > 1, ds.element_spec.shape

    encoded = data_processing.to_stacked_tensor(ds)

    self.assertAllEqual(encoded, [[[1, 2], [3, 4]]])

  def test_empty_dataset(self):
    ds = tf.data.Dataset.range(-1)
    assert ds.cardinality() == 0

    encoded = data_processing.to_stacked_tensor(ds)

    self.assertAllEqual(encoded, list())

  def test_batched_drop_remainder(self):
    ds = tf.data.Dataset.range(6).batch(2, drop_remainder=True)

    encoded = data_processing.to_stacked_tensor(ds)

    self.assertAllEqual(encoded, [[0, 1], [2, 3], [4, 5]])

  def test_batched_with_remainder_unsupported(self):
    ds = tf.data.Dataset.range(6).batch(2, drop_remainder=False)

    with self.assertRaisesRegex(
        ValueError, 'Dataset elements must have fully-defined shapes'
    ):
      data_processing.to_stacked_tensor(ds)

  def test_roundtrip(self):
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[1, 2, 3],
            y=[['a'], ['b'], ['c']],
        )
    )

    encoded = data_processing.to_stacked_tensor(ds)
    roundtripped = tf.data.Dataset.from_tensor_slices(encoded)

    self.assertAllEqual(list(ds), list(roundtripped))

  def test_validates_input(self):
    not_a_dataset = [tf.constant(42)]

    with self.assertRaisesRegex(TypeError, 'ds'):
      data_processing.to_stacked_tensor(not_a_dataset)


if __name__ == '__main__':
  tf.test.main()
