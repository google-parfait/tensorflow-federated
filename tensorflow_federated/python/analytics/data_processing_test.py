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
      ('empty_dataset', [], 2, []),
      ('string_dataset_batch_size_1', ['a', 'b', 'a', 'b', 'c'
                                      ], 1, [b'a', b'b', b'a', b'b', b'c']),
      ('string_dataset_batch_size_3', ['a', 'b', 'a', 'b', 'c'
                                      ], 3, [b'a', b'b', b'a', b'b', b'c']),
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3], 2, [1, 3, 2, 2, 4, 6, 3]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0], 2, [1.0, 4.0, 4.0, 6.0]),
  )
  def test_all_elements_returns_expected_values(self, input_data, batch_size,
                                                expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    all_elements = data_processing.get_all_elements(ds)
    self.assertAllEqual(all_elements, expected_result)

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
      _ = data_processing.get_all_elements(ds)

  @parameterized.named_parameters(
      ('empty_dataset', [], 2, 10, []),
      ('string_dataset_batch_size_1', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                      ], 1, 4, [b'a', b'b', b'a', b'c']),
      ('string_dataset_batch_size_3', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                      ], 3, 4, [b'a', b'b', b'a']),
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3], 2, 4, [1, 3, 2, 2]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0], 2, 2, [1.0, 4.0]),
  )
  def test_capped_elements_returns_expected_values(self, input_data, batch_size,
                                                   max_user_contribution,
                                                   expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    capped_elements = data_processing.get_capped_elements(
        ds, max_user_contribution=max_user_contribution, batch_size=batch_size)
    self.assertAllEqual(capped_elements, expected_result)

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
      _ = data_processing.get_capped_elements(
          ds,
          max_user_contribution=max_user_contribution,
          batch_size=batch_size)

  def test_capped_elements_raise_params_value_error(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size=1)

    with self.subTest('batch_size_value_error'):
      with self.assertRaisesRegex(ValueError,
                                  '`batch_size` must be at least 1.'):
        _ = data_processing.get_capped_elements(
            ds, max_user_contribution=3, batch_size=0)

    with self.subTest('max_user_contribution_value_error'):
      with self.assertRaisesRegex(
          ValueError, '`max_user_contribution` must be at least 1.'):
        _ = data_processing.get_capped_elements(
            ds, max_user_contribution=0, batch_size=1)

  @parameterized.named_parameters(
      ('empty_dataset', [], 3, []),
      ('string_dataset_batch_size_1', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                      ], 1, [b'a', b'b', b'c']),
      ('string_dataset_batch_size_3', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                      ], 3, [b'a', b'b', b'c']),
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3], 2, [1, 3, 2, 4, 6]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0], 2, [1.0, 4.0, 6.0]),
  )
  def test_unique_elements_returns_expected_values(self, input_data, batch_size,
                                                   expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    unique_elements = data_processing.get_unique_elements(ds)
    self.assertAllEqual(unique_elements, expected_result)

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
      _ = data_processing.get_unique_elements(ds)

  @parameterized.named_parameters(
      ('empty_dataset', [], 3, 1, []),
      ('string_batch_size_1_max_contrib_2', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                            ], 1, 2, [b'a', b'c']),
      ('string_batch_size_3_max_contrib_2', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                            ], 3, 2, [b'a', b'c']),
      ('string_batch_size_1_max_contrib_3', ['a', 'b', 'a', 'c', 'b', 'c', 'c'
                                            ], 1, 3, [b'a', b'b', b'c']),
      ('string_batch_size_2_max_contrib_4', ['a', 'b', 'c', 'd', 'e', 'a'
                                            ], 2, 4, [b'a', b'b', b'c', b'd']),
      ('int_dataset', [1, 3, 2, 2, 4, 6, 3], 2, 2, [3, 2]),
      ('float_dataset', [1.0, 4.0, 4.0, 6.0], 3, 1, [4.0]),
  )
  def test_get_top_elements_returns_expected_values(self, input_data,
                                                    batch_size,
                                                    max_user_contribution,
                                                    expected_result):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size)
    top_elements = data_processing.get_top_elements(ds, max_user_contribution)
    self.assertSetEqual(set(top_elements.numpy()), set(expected_result))

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
      _ = data_processing.get_top_elements(ds, max_user_contribution)

  def test_get_top_elements_raise_params_value_error(self):
    ds = tf.data.Dataset.from_tensor_slices(['a', 'b', 'a', 'b',
                                             'c']).batch(batch_size=1)

    with self.assertRaisesRegex(ValueError,
                                '`max_user_contribution` must be at least 1.'):
      _ = data_processing.get_top_elements(ds, max_user_contribution=0)


if __name__ == '__main__':
  tf.test.main()
