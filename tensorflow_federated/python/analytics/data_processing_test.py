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

from absl.testing import parameterized

import tensorflow as tf

from tensorflow_federated.python.analytics import data_processing


class DataProcessingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 2, []),
      ('batch_size_1', ['a', 'b', 'a', 'b', 'c'
                       ], 1, [b'a', b'b', b'a', b'b', b'c']),
      ('batch_size_3', ['a', 'b', 'a', 'b', 'c'
                       ], 3, [b'a', b'b', b'a', b'b', b'c']),
  )
  def test_all_elements_returns_expected_values(self, input_data, batch_size,
                                                expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_all_elements(ds)
    self.assertAllEqual(all_elements, expected_result)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 3, []),
      ('string', ['abcd', 'abcde', 'bcd', 'bcdef', 'def'
                 ], 1, 3, ['abc', 'abc', 'bcd', 'bcd', 'def']),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'
                  ], 3, 6, ['新年', '新年', '☺️', '☺️', '☺️']),
  )
  def test_all_elements_with_max_len_returns_expected_values(
      self, input_data, batch_size, max_string_length, expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_all_elements(
        ds, max_string_length=max_string_length)
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
      ('max_string_length_0', 0),
      ('max_string_length_neg', -1),
  )
  def test_all_elements_raise_params_value_error(self, max_string_length):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size=1)

    with self.assertRaisesRegex(ValueError,
                                '`max_string_length` must be at least 1.'):
      data_processing.get_all_elements(ds, max_string_length=max_string_length)

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_all_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'):
      data_processing.get_all_elements(ds)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 2, 10, []),
      ('batch_size_1', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                       ], 1, 4, [b'a', b'b', b'a', b'c']),
      ('batch_size_3', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                       ], 3, 4, [b'a', b'b', b'a']),
  )
  def test_capped_elements_returns_expected_values(self, input_data, batch_size,
                                                   max_user_contribution,
                                                   expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=max_user_contribution, batch_size=batch_size)
    self.assertAllEqual(capped_elements, expected_result)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 10, 3, []),
      ('string_1', ['abcd', 'abcde', 'bcd', 'bcdef', 'def'
                   ], 1, 4, 3, ['abc', 'abc', 'bcd', 'bcd']),
      ('string_2', ['abcd', 'abcde', 'bcd', 'bcdef', 'def'
                   ], 2, 2, 2, ['ab', 'ab']),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 2, 3, 6, ['新年', '新年']),
  )
  def test_capped_elements_with_max_len_returns_expected_values(
      self, input_data, batch_size, max_user_contribution, max_string_length,
      expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_capped_elements(
        ds,
        batch_size=batch_size,
        max_user_contribution=max_user_contribution,
        max_string_length=max_string_length)
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
        ValueError, 'The shape of elements in `dataset` must be of rank 1.*'):
      data_processing.get_capped_elements(
          ds,
          max_user_contribution=max_user_contribution,
          batch_size=batch_size)

  @parameterized.named_parameters(
      ('batch_size', 5, 0, 10, '`batch_size` must be at least 1.'),
      ('max_user_contribution', -10, 10, 10,
       '`max_user_contribution` must be at least 1.'),
      ('max_string_length', 10, 5, 0,
       '`max_string_length` must be at least 1.'),
  )
  def test_capped_elements_raise_params_value_error(self, max_user_contribution,
                                                    batch_size,
                                                    max_string_length,
                                                    raises_regex):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size=1)
    with self.assertRaisesRegex(ValueError, raises_regex):
      data_processing.get_capped_elements(
          ds,
          max_user_contribution=max_user_contribution,
          batch_size=batch_size,
          max_string_length=max_string_length)

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_capped_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'):
      data_processing.get_capped_elements(
          ds, max_user_contribution=10, batch_size=1)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, []),
      ('string_dataset_batch_size_1', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                      ], 1, [b'a', b'b', b'c']),
      ('string_dataset_batch_size_3', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                      ], 3, [b'a', b'b', b'c']),
  )
  def test_unique_elements_returns_expected_values(self, input_data, batch_size,
                                                   expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    unique_elements = data_processing.get_unique_elements(ds)
    self.assertAllEqual(unique_elements, expected_result)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 3, []),
      ('string', ['abcd', 'abcde', 'bcd', 'bcdef', 'def'
                 ], 1, 3, ['abc', 'bcd', 'def']),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 3, 6, ['新年', '☺️']),
  )
  def test_get_unique_elements_with_max_len_returns_expected_values(
      self, input_data, batch_size, max_string_length, expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    unique_elements = data_processing.get_unique_elements(
        ds, max_string_length=max_string_length)
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
      ('max_string_length_0', 0),
      ('max_string_length_neg', -1),
  )
  def test_unique_elements_raise_params_value_error(self, max_string_length):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size=1)

    with self.assertRaisesRegex(ValueError,
                                '`max_string_length` must be at least 1.'):
      data_processing.get_unique_elements(
          ds, max_string_length=max_string_length)

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_get_unique_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'):
      data_processing.get_unique_elements(ds)

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 1, []),
      ('string_batch_size_1_max_contrib_2', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                            ], 1, 2, [b'a', b'c']),
      ('string_batch_size_3_max_contrib_2', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                            ], 3, 2, [b'a', b'c']),
      ('string_batch_size_1_max_contrib_3', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                            ], 1, 3, [b'a', b'b', b'c']),
      ('string_batch_size_2_max_contrib_4', ['a', 'b', 'c', 'd', 'e', 'a'
                                            ], 2, 4, [b'a', b'b', b'c', b'd']),
  )
  def test_get_top_elements_returns_expected_values(self, input_data,
                                                    batch_size,
                                                    max_user_contribution,
                                                    expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_elements(ds, max_user_contribution)
    self.assertSetEqual(set(top_elements.numpy()), set(expected_result))

  @parameterized.named_parameters(
      ('empty_dataset', tf.constant([], dtype=tf.string), 3, 1, 3, []),
      ('max_contrib_max_string_len', ['abc', 'abcd', 'def', 'ghijk'
                                     ], 3, 2, 3, ['abc', 'def']),
      ('max_string_len', ['abcd', 'abcde', 'bcd', 'bcdef'
                         ], 1, 10, 3, ['abc', 'bcd']),
      ('max_contrib', ['abcd', 'abcde', 'bcd', 'bcdef'
                      ], 3, 2, 10, ['abcd', 'abcde']),
      ('unicode', ['新年快乐', '新年', '☺️😇', '☺️😇', '☺️'], 1, 2, 6, ['新年', '☺️']),
  )
  def test_get_top_elements_with_max_len_returns_expected_values(
      self, input_data, batch_size, max_user_contribution, max_string_length,
      expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_elements(
        ds, max_user_contribution, max_string_length=max_string_length)
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
        ValueError, 'The shape of elements in `dataset` must be of rank 1.*'):
      data_processing.get_top_elements(ds, max_user_contribution)

  @parameterized.named_parameters(
      ('max_user_contribution_0', 0, 1,
       '`max_user_contribution` must be at least 1.'),
      ('max_user_contribution_neg', -10, 10,
       '`max_user_contribution` must be at least 1.'),
      ('max_string_length_0', 10, 0, '`max_string_length` must be at least 1.'),
      ('max_string_length_neg', 20, -1,
       '`max_string_length` must be at least 1.'),
  )
  def test_get_top_elements_raise_params_value_error(self,
                                                     max_user_contribution,
                                                     max_string_length,
                                                     raises_regex):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size=1)

    with self.assertRaisesRegex(ValueError, raises_regex):
      data_processing.get_top_elements(
          ds,
          max_user_contribution=max_user_contribution,
          max_string_length=max_string_length)

  @parameterized.named_parameters(
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0]),
      ('bool_dataset', [True, True, False]),
  )
  def test_get_top_elements_raise_type_error(self, input_data):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size=1)

    with self.assertRaisesRegex(
        TypeError, '`dataset.element_spec.dtype` must be `tf.string`.'):
      data_processing.get_top_elements(ds, max_user_contribution=10)


if __name__ == '__main__':
  tf.test.main()
