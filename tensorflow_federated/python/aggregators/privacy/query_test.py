# Copyright 2020, The TensorFlow Authors.
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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators.privacy import query as dp_query
from tensorflow_federated.python.aggregators.privacy import test_utils


class SumAggregationQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_initial_sample_state_works_on_tensorspecs(self):
    query = dp_query.NoPrivacySumQuery()
    template = tf.TensorSpec.from_tensor(tf.constant([1.0, 2.0]))
    sample_state = query.initial_sample_state(template)
    expected = [0.0, 0.0]
    self.assertAllClose(sample_state, expected)


class GaussianQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_gaussian_sum_no_clip_no_noise(self):
    record1 = tf.constant([2.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    query = dp_query.GaussianSumQuery(l2_norm_clip=10.0, stddev=0.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [1.0, 1.0]
    self.assertAllClose(query_result, expected)

  def test_gaussian_sum_with_clip_no_noise(self):
    record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
    record2 = tf.constant([4.0, -3.0])  # Not clipped.

    query = dp_query.GaussianSumQuery(l2_norm_clip=5.0, stddev=0.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [1.0, 1.0]
    self.assertAllClose(query_result, expected)

  def test_gaussian_sum_with_changing_clip_no_noise(self):
    record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
    record2 = tf.constant([4.0, -3.0])  # Not clipped.

    l2_norm_clip = tf.Variable(5.0)
    query = dp_query.GaussianSumQuery(l2_norm_clip=l2_norm_clip, stddev=0.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])

    expected = [1.0, 1.0]
    self.assertAllClose(query_result, expected)

    l2_norm_clip.assign(0.0)
    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [0.0, 0.0]
    self.assertAllClose(query_result, expected)

  def test_gaussian_sum_with_noise(self):
    record1, record2 = 2.71828, 3.14159
    stddev = 1.0

    query = dp_query.GaussianSumQuery(l2_norm_clip=5.0, stddev=stddev)

    noised_sums = []
    for _ in range(1000):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      noised_sums.append(query_result)

    result_stddev = np.std(noised_sums)
    self.assertNear(result_stddev, stddev, 0.1)

  def test_gaussian_sum_merge(self):
    records1 = [tf.constant([2.0, 0.0]), tf.constant([-1.0, 1.0])]
    records2 = [tf.constant([3.0, 5.0]), tf.constant([-1.0, 4.0])]

    def get_sample_state(records):
      query = dp_query.GaussianSumQuery(l2_norm_clip=10.0, stddev=1.0)
      global_state = query.initial_global_state()
      params = query.derive_sample_params(global_state)
      sample_state = query.initial_sample_state(records[0])
      for record in records:
        sample_state = query.accumulate_record(params, sample_state, record)
      return sample_state

    sample_state_1 = get_sample_state(records1)
    sample_state_2 = get_sample_state(records2)

    merged = dp_query.GaussianSumQuery(10.0, 1.0).merge_sample_states(
        sample_state_1, sample_state_2
    )

    expected = [3.0, 10.0]
    self.assertAllClose(merged, expected)

  @parameterized.named_parameters(
      ('type_mismatch', [1.0], (1.0,), TypeError),
      ('too_few_on_left', [1.0], [1.0, 1.0], ValueError),
      ('too_few_on_right', [1.0, 1.0], [1.0], ValueError),
  )
  def test_incompatible_records(self, record1, record2, error_type):
    query = dp_query.GaussianSumQuery(1.0, 0.0)
    with self.assertRaises(error_type):
      test_utils.run_query(query, [record1, record2])


class NoPrivacyQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_sum(self):
    record1 = tf.constant([2.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    query = dp_query.NoPrivacySumQuery()
    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [1.0, 1.0]
    self.assertAllClose(query_result, expected)

  def test_no_privacy_average(self):
    record1 = tf.constant([5.0, 0.0])
    record2 = tf.constant([-1.0, 2.0])

    query = dp_query.NoPrivacyAverageQuery()
    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [2.0, 1.0]
    self.assertAllClose(query_result, expected)

  def test_no_privacy_weighted_average(self):
    record1 = tf.constant([4.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    weights = [1, 3]

    query = dp_query.NoPrivacyAverageQuery()
    query_result, _ = test_utils.run_query(
        query, [record1, record2], weights=weights
    )
    expected = [0.25, 0.75]
    self.assertAllClose(query_result, expected)

  @parameterized.named_parameters(
      ('type_mismatch', [1.0], (1.0,), TypeError),
      ('too_few_on_left', [1.0], [1.0, 1.0], ValueError),
      ('too_few_on_right', [1.0, 1.0], [1.0], ValueError),
  )
  def test_incompatible_records(self, record1, record2, error_type):
    query = dp_query.NoPrivacySumQuery()
    with self.assertRaises(error_type):
      test_utils.run_query(query, [record1, record2])


class NormalizedQueryTest(tf.test.TestCase):

  def test_normalization(self):
    record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
    record2 = tf.constant([4.0, -3.0])  # Not clipped.

    sum_query = dp_query.GaussianSumQuery(l2_norm_clip=5.0, stddev=0.0)
    query = dp_query.NormalizedQuery(numerator_query=sum_query, denominator=2.0)

    query_result, _ = test_utils.run_query(query, [record1, record2])
    expected = [0.5, 0.5]
    self.assertAllClose(query_result, expected)


if __name__ == '__main__':
  tf.test.main()
