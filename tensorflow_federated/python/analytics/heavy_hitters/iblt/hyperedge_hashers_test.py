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

import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import hyperedge_hashers


# Table of critial values for chi square tests with 5% level of significance.
# The keys in the table are degrees of freedom.
_CHI_SQUARE_CRITICAL_VALUE = {
    4: 9.487,
    19: 30.143,
    99: 123.225,
    499: 552.074,
    581: 638.183,
    1508: 1599.454,
    1999: 2104.128,
    23923: 24283.924,
}


class HyperedgeHashersTest(tf.test.TestCase, parameterized.TestCase):

  def _assert_distribution_uniform(self, samples, degrees_of_freedom, *,
                                   is_uniform):
    """Checks if `samples` are uniformly distributed by chi square tests.

    Note: the critical values needed in this test file are hard coded in
    _CHI_SQUARE_CRITICAL_VALUE, which needs to be updated if the tests are
    updated.

    Args:
      samples: The input samples.
      degrees_of_freedom: Degrees of freedom in chi square tests. In this case,
        it is the number of possible unique values in `samples` minus 1.
      is_uniform: A boolean indicating if `samples` are expected to be uniformly
        distributed or not.
    """
    samples = np.array(samples, dtype=np.int)
    samples = samples.reshape(-1)
    _, counts = np.unique(samples, return_counts=True)
    expected_counts = np.sum(counts) / len(counts)
    chi_square = np.sum(np.square(counts - expected_counts) / expected_counts)

    if is_uniform:
      self.assertLess(chi_square,
                      _CHI_SQUARE_CRITICAL_VALUE[degrees_of_freedom])
    else:
      self.assertGreaterEqual(chi_square,
                              _CHI_SQUARE_CRITICAL_VALUE[degrees_of_freedom])

  @parameterized.named_parameters(
      ('table_size_0', 0, 3, 'table_size must be at least 1.'),
      ('table_size_neg', -1, 3, 'table_size must be at least 1.'),
      ('repetitions_0', 10, 0, 'repetitions must be at least 3.'),
      ('repetitions_2', 10, 2, 'repetitions must be at least 3.'),
      ('repetitions_neg', 10, -1, 'repetitions must be at least 3.'),
  )
  def test_random_hash_family_raise_params_value_error(self, table_size,
                                                       repetitions,
                                                       raises_regex):
    seed = 0
    with self.assertRaisesRegex(ValueError, raises_regex):
      hyperedge_hashers.RandomHyperEdgeHasher(seed, table_size, repetitions)

  @parameterized.named_parameters(
      ('seed_0_table_size_5', 0, 5, 3,
       ['hello', 'these', 'are', 'some', 'strings']),
      ('seed_1_table_size_100', 1, 100, 5,
       ['', 'some', 'unicodes', 'अ', 'क', 'æ', '☺️', '☺️']),
      ('seed_5_table_size_2000', 5, 2000, 3,
       ['has, some, comma', '新年快乐', '☺️😇']),
      ('empty_strings', 0, 5, 3, []),
  )
  def test_random_hash_family_as_expected(self, seed, table_size, repetitions,
                                          data_strings):
    hasher = hyperedge_hashers.RandomHyperEdgeHasher(seed, table_size,
                                                     repetitions)
    data_strings_tensor = tf.constant(data_strings, dtype=tf.string)
    hashes = np.array(hasher.get_hash_indices(data_strings))
    hashes_tensor = hasher.get_hash_indices_tf(data_strings_tensor)
    with self.cached_session() as sess:
      hashes_tensor = sess.run(hashes_tensor)

    # Both functions should return the same indices.
    self.assertAllEqual(list(hashes), list(np.squeeze(hashes_tensor[:, :, 2])))

    # Indices should be in the table.
    self.assertAllInRange(hashes, 0, table_size)

    # The return shape should be as expected.
    if not data_strings:  # when data_string is empty
      self.assertEqual(hashes.shape, (len(data_strings),))
    else:
      self.assertEqual(hashes.shape, (len(data_strings), repetitions))
    self.assertEqual(hashes_tensor.shape, (len(data_strings), repetitions, 3))

  @parameterized.named_parameters(
      ('table_size_5', 1, 5, 3, 10000),
      (
          'table_size_100_seed_100000',
          100000,
          100,
          5,
          100000,
      ),
      (
          'table_size_500_seed_1000',
          1000,
          500,
          4,
          20000,
      ),
      (
          'table_size_2000_seed_100',
          100,
          2000,
          3,
          20000,
      ),
  )
  def test_random_hash_family_distribution_as_expected(
      self, seed, table_size, repetitions, data_strings_upper_bound):
    hasher = hyperedge_hashers.RandomHyperEdgeHasher(seed, table_size,
                                                     repetitions)
    data_strings = [str(x) for x in range(data_strings_upper_bound)]
    data_strings_tensor = tf.constant(data_strings, dtype=tf.string)
    hashes = np.array(hasher.get_hash_indices(data_strings))
    hashes_tensor = hasher.get_hash_indices_tf(data_strings_tensor)
    with self.cached_session() as sess:
      hashes_tensor = sess.run(hashes_tensor)

    _, counts = np.unique(hashes, return_counts=True)
    min_conflicts = data_strings_upper_bound / table_size
    # No more than 10 * min_conflicts strings are hashed to the same int64 value
    self.assertLess(np.max(counts), 10 * min_conflicts)

    # All indices should be uniformly distributed.
    degrees_of_freedom = table_size - 1
    self._assert_distribution_uniform(
        hashes, degrees_of_freedom, is_uniform=True)
    self._assert_distribution_uniform(
        np.squeeze(hashes_tensor[:, :, 2]), degrees_of_freedom, is_uniform=True)

  @parameterized.named_parameters(
      ('table_size_0', 0, 3, 4, 'table_size must be at least 1.'),
      ('table_size_neg', -1, 3, 4, 'table_size must be at least 1.'),
      ('repetitions_0', 10, 0, 4, 'repetitions must be at least 3.'),
      ('repetitions_2', 10, 2, 4, 'repetitions must be at least 3.'),
      ('repetitions_neg', 10, -1, 4, 'repetitions must be at least 3.'),
      ('rescale_factor_0', 10, 3, 0, 'rescale_factor must be positive.*'),
      ('rescale_factor_neg', 10, 3, -1, 'rescale_factor must be positive.*'),
      ('rescale_factor_more_than_table_size', 10, 3, 15,
       'rescale_factor must be positive.*'),
  )
  def test_coupled_hash_family_raise_params_value_error(self, table_size,
                                                        repetitions,
                                                        rescale_factor,
                                                        raises_regex):
    seed = 0
    with self.assertRaisesRegex(ValueError, raises_regex):
      hyperedge_hashers.CoupledHyperEdgeHasher(seed, table_size, repetitions,
                                               rescale_factor)

  @parameterized.named_parameters(
      ('seed_0_table_size_5', 0, 5, 3, 4,
       ['hello', 'these', 'are', 'some', 'strings']),
      ('seed_1_table_size_100', 1, 100, 5, 1,
       ['', 'some', 'unicodes', 'अ', 'क', 'æ', '☺️', '☺️']),
      ('seed_0_table_size_2000', 0, 2000, 3, 4,
       ['has, some, comma', '新年快乐', '☺️😇']),
      ('empty_strings', 0, 5, 3, 4, []),
  )
  def test_coupled_hash_family_as_expected(self, seed, table_size, repetitions,
                                           rescale_factor, data_strings):
    hasher = hyperedge_hashers.CoupledHyperEdgeHasher(seed, table_size,
                                                      repetitions,
                                                      rescale_factor)
    data_strings_tensor = tf.constant(data_strings, dtype=tf.string)
    hashes = np.array(hasher.get_hash_indices(data_strings))
    hashes_tensor = hasher.get_hash_indices_tf(data_strings_tensor)
    with self.cached_session() as sess:
      hashes_tensor = sess.run(hashes_tensor)

    # Both functions should return the same indices.
    self.assertAllEqual(list(hashes), list(np.squeeze(hashes_tensor[:, :, 2])))

    # Indices should be in the table.
    self.assertAllInRange(hashes, 0, table_size)

    # The return shape should be as expected.
    if not data_strings:  # when data_string is empty
      self.assertEqual(hashes.shape, (len(data_strings),))
    else:
      self.assertEqual(hashes.shape, (len(data_strings), repetitions))
    self.assertEqual(hashes_tensor.shape, (len(data_strings), repetitions, 3))

    # In coupled hashing, all the indices for each string should be at most
    # `table_size/rescale_factor` distance from each other.
    if data_strings:
      max_per_element_index_difference = np.max(
          np.max(hashes, axis=1) - np.min(hashes, axis=1))
      self.assertLess(max_per_element_index_difference,
                      table_size / float(rescale_factor))

  @parameterized.named_parameters(
      ('table_size_500', 100, 500, 3, 4, 10000),
      (
          'table_size_100',
          0,
          100,
          5,
          5,
          20000,
      ),
      (
          'table_size_2000',
          10,
          2000,
          3,
          1,
          10000,
      ),
  )
  def test_coupled_hash_family_distribution_as_expected(
      self, seed, table_size, repetitions, rescale_factor,
      data_strings_upper_bound):
    hasher = hyperedge_hashers.CoupledHyperEdgeHasher(seed, table_size,
                                                      repetitions,
                                                      rescale_factor)
    data_strings = [str(x) for x in range(data_strings_upper_bound)]
    data_strings_tensor = tf.constant(data_strings, dtype=tf.string)
    hashes = np.array(hasher.get_hash_indices(data_strings))
    hashes_tensor = hasher.get_hash_indices_tf(data_strings_tensor)
    with self.cached_session() as sess:
      hashes_tensor = sess.run(hashes_tensor)

    _, counts = np.unique(hashes, return_counts=True)
    min_conflicts = data_strings_upper_bound / table_size
    # No more than 10 * min_conflicts strings are hashed to the same int64 value
    self.assertLess(np.max(counts), 10 * min_conflicts)

    # All indices should not be uniformly distributed.
    degrees_of_freedom = table_size - 1
    self._assert_distribution_uniform(
        hashes, degrees_of_freedom, is_uniform=False)
    self._assert_distribution_uniform(
        np.squeeze(hashes_tensor[:, :, 2]),
        degrees_of_freedom,
        is_uniform=False)

    # Indices in the middle should be uniformly distributed.
    lower_bound = max(1, math.ceil(table_size / 2) - 10)
    upper_bound = min(table_size - 1, math.ceil(table_size / 2) + 10)
    middle_indices = (lower_bound <= hashes) & (hashes < upper_bound)
    hashes_in_the_middle = hashes[middle_indices]
    hashes_tensor_in_the_middle = hashes_tensor[:, :, 2][middle_indices]
    degrees_of_freedom = len(hashes_in_the_middle) - 1
    self._assert_distribution_uniform(
        hashes_in_the_middle, degrees_of_freedom, is_uniform=True)
    self._assert_distribution_uniform(
        hashes_tensor_in_the_middle, degrees_of_freedom, is_uniform=True)


if __name__ == '__main__':
  tf.test.main()
