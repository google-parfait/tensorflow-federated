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

from tensorflow_federated.python.aggregators.privacy import quantile as quantile_query
from tensorflow_federated.python.aggregators.privacy import test_utils

tf.compat.v1.enable_eager_execution()


def _make_quantile_estimator_query(
    initial_estimate,
    target_quantile,
    learning_rate,
    below_estimate_stddev,
    expected_num_records,
    geometric_update,
    tree_aggregation=False,
):
  if expected_num_records is not None:
    if tree_aggregation:
      return quantile_query.TreeQuantileEstimatorQuery(
          initial_estimate,
          target_quantile,
          learning_rate,
          below_estimate_stddev,
          expected_num_records,
          geometric_update,
      )
    else:
      return quantile_query.QuantileEstimatorQuery(
          initial_estimate,
          target_quantile,
          learning_rate,
          below_estimate_stddev,
          expected_num_records,
          geometric_update,
      )
  else:
    if tree_aggregation:
      raise ValueError(
          'Cannot set expected_num_records to None for tree aggregation.'
      )
    return quantile_query.NoPrivacyQuantileEstimatorQuery(
        initial_estimate, target_quantile, learning_rate, geometric_update
    )


class QuantileEstimatorQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('exact', True, False), ('fixed', False, False), ('tree', False, True)
  )
  def test_target_zero(self, exact, tree):
    record1 = tf.constant(8.5)
    record2 = tf.constant(7.25)

    query = _make_quantile_estimator_query(
        initial_estimate=10.0,
        target_quantile=0.0,
        learning_rate=1.0,
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=False,
        tree_aggregation=tree,
    )

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 10.0)

    # On the first two iterations, both records are below, so the estimate goes
    # down by 1.0 (the learning rate). When the estimate reaches 8.0, only one
    # record is below, so the estimate goes down by only 0.5. After two more
    # iterations, both records are below, and the estimate stays there (at 7.0).

    expected_estimates = [9.0, 8.0, 7.5, 7.0, 7.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(
      ('exact', True, False), ('fixed', False, False), ('tree', False, True)
  )
  def test_target_zero_geometric(self, exact, tree):
    record1 = tf.constant(5.0)
    record2 = tf.constant(2.5)

    query = _make_quantile_estimator_query(
        initial_estimate=16.0,
        target_quantile=0.0,
        learning_rate=np.log(2.0),  # Geometric steps in powers of 2.
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=True,
        tree_aggregation=tree,
    )

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 16.0)

    # For two iterations, both records are below, so the estimate is halved.
    # Then only one record is below, so the estimate goes down by only sqrt(2.0)
    # to 4 / sqrt(2.0). Still only one record is below, so it reduces to 2.0.
    # Now no records are below, and the estimate norm stays there (at 2.0).

    four_div_root_two = 4 / np.sqrt(2.0)  # approx 2.828

    expected_estimates = [8.0, 4.0, four_div_root_two, 2.0, 2.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(
      ('exact', True, False), ('fixed', False, False), ('tree', False, True)
  )
  def test_target_one(self, exact, tree):
    record1 = tf.constant(1.5)
    record2 = tf.constant(2.75)

    query = _make_quantile_estimator_query(
        initial_estimate=0.0,
        target_quantile=1.0,
        learning_rate=1.0,
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=False,
        tree_aggregation=tree,
    )

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 0.0)

    # On the first two iterations, both are above, so the estimate goes up
    # by 1.0 (the learning rate). When it reaches 2.0, only one record is
    # above, so the estimate goes up by only 0.5. After two more iterations,
    # both records are below, and the estimate stays there (at 3.0).

    expected_estimates = [1.0, 2.0, 2.5, 3.0, 3.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(
      ('exact', True, False), ('fixed', False, False), ('tree', False, True)
  )
  def test_target_one_geometric(self, exact, tree):
    record1 = tf.constant(1.5)
    record2 = tf.constant(3.0)

    query = _make_quantile_estimator_query(
        initial_estimate=0.5,
        target_quantile=1.0,
        learning_rate=np.log(2.0),  # Geometric steps in powers of 2.
        below_estimate_stddev=0.0,
        expected_num_records=(None if exact else 2.0),
        geometric_update=True,
        tree_aggregation=tree,
    )

    global_state = query.initial_global_state()

    initial_estimate = global_state.current_estimate
    self.assertAllClose(initial_estimate, 0.5)

    # On the first two iterations, both are above, so the estimate is doubled.
    # When the estimate reaches 2.0, only one record is above, so the estimate
    # is multiplied by sqrt(2.0). Still only one is above so it increases to
    # 4.0. Now both records are above, and the estimate stays there (at 4.0).

    two_times_root_two = 2 * np.sqrt(2.0)  # approx 2.828

    expected_estimates = [1.0, 2.0, two_times_root_two, 4.0, 4.0]
    for expected_estimate in expected_estimates:
      actual_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      self.assertAllClose(actual_estimate.numpy(), expected_estimate)

  @parameterized.named_parameters(
      ('start_low_geometric_exact', True, True, True, False),
      ('start_low_arithmetic_exact', True, True, False, False),
      ('start_high_geometric_exact', True, False, True, False),
      ('start_high_arithmetic_exact', True, False, False, False),
      ('start_low_geometric_noised', False, True, True, False),
      ('start_low_arithmetic_noised', False, True, False, False),
      ('start_high_geometric_noised', False, False, True, False),
      ('start_high_arithmetic_noised', False, False, False, False),
      ('start_low_geometric_tree', False, True, True, True),
      ('start_low_arithmetic_tree', False, True, False, True),
      ('start_high_geometric_tree', False, False, True, True),
      ('start_high_arithmetic_tree', False, False, False, True),
  )
  def test_linspace(self, exact, start_low, geometric, tree):
    # 100 records equally spaced from 0 to 10 in 0.1 increments.
    # Test that we converge to the correct median value and bounce around it.
    num_records = 21
    records = [
        tf.constant(x)
        for x in np.linspace(0.0, 10.0, num=num_records, dtype=np.float32)
    ]

    query = _make_quantile_estimator_query(
        initial_estimate=(1.0 if start_low else 10.0),
        target_quantile=0.5,
        learning_rate=1.0,
        below_estimate_stddev=(0.0 if exact else 1e-2),
        expected_num_records=(None if exact else num_records),
        geometric_update=geometric,
        tree_aggregation=tree,
    )

    global_state = query.initial_global_state()

    for t in range(50):
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_estimate = global_state.current_estimate

      if t > 40:
        self.assertNear(actual_estimate, 5.0, 0.25)

  @parameterized.named_parameters(
      ('start_low_geometric_exact', True, True, True, False),
      ('start_low_arithmetic_exact', True, True, False, False),
      ('start_high_geometric_exact', True, False, True, False),
      ('start_high_arithmetic_exact', True, False, False, False),
      ('start_low_geometric_noised', False, True, True, False),
      ('start_low_arithmetic_noised', False, True, False, False),
      ('start_high_geometric_noised', False, False, True, False),
      ('start_high_arithmetic_noised', False, False, False, False),
      ('start_low_geometric_tree', False, True, True, True),
      ('start_low_arithmetic_tree', False, True, False, True),
      ('start_high_geometric_tree', False, False, True, True),
      ('start_high_arithmetic_tree', False, False, False, True),
  )
  def test_all_equal(self, exact, start_low, geometric, tree):
    # 20 equal records. Test that we converge to that record and bounce around
    # it. Unlike the linspace test, the quantile-matching objective is very
    # sharp at the optimum so a decaying learning rate is necessary.
    num_records = 20
    records = [tf.constant(5.0)] * num_records

    learning_rate = tf.Variable(1.0)

    query = _make_quantile_estimator_query(
        initial_estimate=(1.0 if start_low else 10.0),
        target_quantile=0.5,
        learning_rate=learning_rate,
        below_estimate_stddev=(0.0 if exact else 1e-2),
        expected_num_records=(None if exact else num_records),
        geometric_update=geometric,
        tree_aggregation=tree,
    )

    global_state = query.initial_global_state()

    for t in range(50):
      tf.compat.v1.assign(learning_rate, 1.0 / np.sqrt(t + 1))
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_estimate = global_state.current_estimate

      if t > 40:
        self.assertNear(actual_estimate, 5.0, 0.5)

  def test_raises_with_non_scalar_record(self):
    query = quantile_query.NoPrivacyQuantileEstimatorQuery(
        initial_estimate=1.0, target_quantile=0.5, learning_rate=1.0
    )

    with self.assertRaisesRegex(ValueError, 'scalar'):
      query.accumulate_record(None, None, [1.0, 2.0])

  def test_tree_noise_restart(self):
    sample_num, tolerance, stddev = 1000, 0.3, 0.1
    initial_estimate, expected_num_records = 5.0, 2.0
    record1 = tf.constant(1.0)
    record2 = tf.constant(10.0)

    query = _make_quantile_estimator_query(
        initial_estimate=initial_estimate,
        target_quantile=0.5,
        learning_rate=1.0,
        below_estimate_stddev=stddev,
        expected_num_records=expected_num_records,
        geometric_update=False,
        tree_aggregation=True,
    )

    global_state = query.initial_global_state()

    self.assertAllClose(global_state.current_estimate, initial_estimate)

    # As the target quantile is accurate, there is no signal and only noise.
    samples = []
    for _ in range(sample_num):
      noised_estimate, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )
      samples.append(noised_estimate.numpy())
      global_state = query.reset_state(noised_estimate, global_state)
      self.assertNotEqual(global_state.current_estimate, initial_estimate)
      global_state = global_state._replace(current_estimate=initial_estimate)

    self.assertAllClose(
        np.std(samples), stddev / expected_num_records, rtol=tolerance
    )


class QuantileAdaptiveClipSumQueryTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_sum_no_clip_no_noise(self):
    record1 = tf.constant([2.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
    )
    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected = [1.0, 1.0]
    self.assertAllClose(result, expected)

  def test_sum_with_clip_no_noise(self):
    record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
    record2 = tf.constant([4.0, -3.0])  # Not clipped.

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=5.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
    )

    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected = [1.0, 1.0]
    self.assertAllClose(result, expected)

  def test_sum_with_noise(self):
    record1, record2 = 2.71828, 3.14159
    stddev = 1.0
    clip = 5.0

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=clip,
        noise_multiplier=stddev / clip,
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
    )

    noised_sums = []
    for _ in range(1000):
      query_result, _ = test_utils.run_query(query, [record1, record2])
      noised_sums.append(query_result.numpy())

    result_stddev = np.std(noised_sums)
    self.assertNear(result_stddev, stddev, 0.1)

  def test_adaptation_target_zero(self):
    record1 = tf.constant([8.5])
    record2 = tf.constant([-7.25])

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=0.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=False,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.l2_norm_clip
    self.assertAllClose(initial_clip, 10.0)

    # On the first two iterations, nothing is clipped, so the clip goes down
    # by 1.0 (the learning rate). When the clip reaches 8.0, one record is
    # clipped, so the clip goes down by only 0.5. After two more iterations,
    # both records are clipped, and the clip norm stays there (at 7.0).

    expected_sums = [1.25, 1.25, 0.75, 0.25, 0.0]
    expected_clips = [9.0, 8.0, 7.5, 7.0, 7.0]
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      actual_sum, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      actual_clip = global_state.sum_state.l2_norm_clip

      self.assertAllClose(actual_clip.numpy(), expected_clip)
      self.assertAllClose(actual_sum.numpy(), (expected_sum,))

  def test_adaptation_target_zero_geometric(self):
    record1 = tf.constant([5.0])
    record2 = tf.constant([-2.5])

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=16.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=0.0,
        learning_rate=np.log(2.0),  # Geometric steps in powers of 2.
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=True,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.l2_norm_clip
    self.assertAllClose(initial_clip, 16.0)

    # For two iterations, nothing is clipped, so the clip is cut in half.
    # Then one record is clipped, so the clip goes down by only sqrt(2.0) to
    # 4 / sqrt(2.0). Still only one record is clipped, so it reduces to 2.0.
    # Now both records are clipped, and the clip norm stays there (at 2.0).

    four_div_root_two = 4 / np.sqrt(2.0)  # approx 2.828

    expected_sums = [2.5, 2.5, 1.5, four_div_root_two - 2.5, 0.0]
    expected_clips = [8.0, 4.0, four_div_root_two, 2.0, 2.0]
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      actual_sum, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      actual_clip = global_state.sum_state.l2_norm_clip

      self.assertAllClose(actual_clip.numpy(), expected_clip)
      self.assertAllClose(actual_sum.numpy(), (expected_sum,))

  def test_adaptation_target_one(self):
    record1 = tf.constant([-1.5])
    record2 = tf.constant([2.75])

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=0.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=False,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.l2_norm_clip
    self.assertAllClose(initial_clip, 0.0)

    # On the first two iterations, both are clipped, so the clip goes up
    # by 1.0 (the learning rate). When the clip reaches 2.0, only one record is
    # clipped, so the clip goes up by only 0.5. After two more iterations,
    # both records are clipped, and the clip norm stays there (at 3.0).

    expected_sums = [0.0, 0.0, 0.5, 1.0, 1.25]
    expected_clips = [1.0, 2.0, 2.5, 3.0, 3.0]
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      actual_sum, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      actual_clip = global_state.sum_state.l2_norm_clip

      self.assertAllClose(actual_clip.numpy(), expected_clip)
      self.assertAllClose(actual_sum.numpy(), (expected_sum,))

  def test_adaptation_target_one_geometric(self):
    record1 = tf.constant([-1.5])
    record2 = tf.constant([3.0])

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=0.5,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=np.log(2.0),  # Geometric steps in powers of 2.
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=True,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.l2_norm_clip
    self.assertAllClose(initial_clip, 0.5)

    # On the first two iterations, both are clipped, so the clip is doubled.
    # When the clip reaches 2.0, only one record is clipped, so the clip is
    # multiplied by sqrt(2.0). Still only one is clipped so it increases to 4.0.
    # Now both records are clipped, and the clip norm stays there (at 4.0).

    two_times_root_two = 2 * np.sqrt(2.0)  # approx 2.828

    expected_sums = [0.0, 0.0, 0.5, two_times_root_two - 1.5, 1.5]
    expected_clips = [1.0, 2.0, two_times_root_two, 4.0, 4.0]
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      actual_sum, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )

      actual_clip = global_state.sum_state.l2_norm_clip

      self.assertAllClose(actual_clip.numpy(), expected_clip)
      self.assertAllClose(actual_sum.numpy(), (expected_sum,))

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_linspace(self, start_low, geometric):
    # `num_records` records equally spaced from 0 to 10 in 0.1 increments.
    # Test that we converge to the correct median value and bounce around it.
    num_records = 21
    records = [
        tf.constant(x)
        for x in np.linspace(0.0, 10.0, num=num_records, dtype=np.float32)
    ]

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=0.0,
        target_unclipped_quantile=0.5,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=num_records,
        geometric_update=geometric,
    )

    global_state = query.initial_global_state()

    for t in range(50):
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_clip = global_state.sum_state.l2_norm_clip

      if t > 40:
        self.assertNear(actual_clip, 5.0, 0.25)

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_all_equal(self, start_low, geometric):
    # `num_records` equal records. Test that we converge to that record and
    # bounce around it. Unlike the linspace test, the quantile-matching
    # objective is very sharp at the optimum so a decaying learning rate is
    # necessary.
    num_records = 20
    records = [tf.constant(5.0)] * num_records

    learning_rate = tf.Variable(1.0)

    query = quantile_query.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=0.0,
        target_unclipped_quantile=0.5,
        learning_rate=learning_rate,
        clipped_count_stddev=0.0,
        expected_num_records=num_records,
        geometric_update=geometric,
    )

    global_state = query.initial_global_state()

    for t in range(50):
      tf.compat.v1.assign(learning_rate, 1.0 / np.sqrt(t + 1))
      _, global_state = test_utils.run_query(query, records, global_state)

      actual_clip = global_state.sum_state.l2_norm_clip

      if t > 40:
        self.assertNear(actual_clip, 5.0, 0.5)


class QAdaClipTreeResSumQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_sum_no_clip_no_noise(self):
    record1 = tf.constant([2.0, 0.0])
    record2 = tf.constant([-1.0, 1.0])

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([2]),
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
    )
    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected = [1.0, 1.0]
    self.assertAllClose(result, expected)

  def test_sum_with_clip_no_noise(self):
    record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
    record2 = tf.constant([4.0, -3.0])  # Not clipped.

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=5.0,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([2]),
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
    )

    query_result, _ = test_utils.run_query(query, [record1, record2])
    result = query_result.numpy()
    expected = [1.0, 1.0]
    self.assertAllClose(result, expected)

  def test_sum_with_noise(self):
    vector_size = 1000
    record1 = tf.constant(2.71828, shape=[vector_size])
    record2 = tf.constant(3.14159, shape=[vector_size])
    stddev = 1.0
    clip = 5.0

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=clip,
        noise_multiplier=stddev / clip,
        record_specs=tf.TensorSpec([vector_size]),
        target_unclipped_quantile=1.0,
        learning_rate=0.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        noise_seed=1,
    )

    query_result, _ = test_utils.run_query(query, [record1, record2])

    result_stddev = np.std(query_result.numpy())
    self.assertNear(result_stddev, stddev, 0.1)

  def _test_estimate_clip_expected_sum(
      self,
      query,
      global_state,
      records,
      expected_sums,
      expected_clips,
      reset=True,
  ):
    for expected_sum, expected_clip in zip(expected_sums, expected_clips):
      initial_clip = global_state.sum_state.clip_value
      actual_sum, global_state = test_utils.run_query(
          query, records, global_state
      )
      if reset:
        global_state = query.reset_state(actual_sum, global_state)
        actual_clip = global_state.sum_state.clip_value
        self.assertAllClose(actual_clip.numpy(), expected_clip)
        self.assertAllClose(actual_sum.numpy(), (expected_sum,))
      else:
        actual_clip = global_state.sum_state.clip_value
        estimate_clip = global_state.quantile_estimator_state.current_estimate
        self.assertAllClose(actual_clip.numpy(), initial_clip)
        self.assertAllClose(estimate_clip.numpy(), expected_clip)
        self.assertAllClose(actual_sum.numpy(), (expected_sums[0],))

  @parameterized.named_parameters(('adaptive', True), ('constant', False))
  def test_adaptation_target_zero(self, reset):
    record1 = tf.constant([8.5])
    record2 = tf.constant([-7.25])

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=10.0,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=0.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=False,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.clip_value
    self.assertAllClose(initial_clip, 10.0)

    # On the first two iterations, nothing is clipped, so the clip goes down
    # by 1.0 (the learning rate). When the clip reaches 8.0, one record is
    # clipped, so the clip goes down by only 0.5. After two more iterations,
    # both records are clipped, and the clip norm stays there (at 7.0).

    expected_sums = [1.25, 1.25, 0.75, 0.25, 0.0]
    expected_clips = [9.0, 8.0, 7.5, 7.0, 7.0]
    self._test_estimate_clip_expected_sum(
        query,
        global_state,
        [record1, record2],
        expected_sums,
        expected_clips,
        reset=reset,
    )

  @parameterized.named_parameters(('adaptive', True), ('constant', False))
  def test_adaptation_target_zero_geometric(self, reset):
    record1 = tf.constant([5.0])
    record2 = tf.constant([-2.5])

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=16.0,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=0.0,
        learning_rate=np.log(2.0),  # Geometric steps in powers of 2.
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=True,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.clip_value
    self.assertAllClose(initial_clip, 16.0)

    # For two iterations, nothing is clipped, so the clip is cut in half.
    # Then one record is clipped, so the clip goes down by only sqrt(2.0) to
    # 4 / sqrt(2.0). Still only one record is clipped, so it reduces to 2.0.
    # Now both records are clipped, and the clip norm stays there (at 2.0).

    four_div_root_two = 4 / np.sqrt(2.0)  # approx 2.828

    expected_sums = [2.5, 2.5, 1.5, four_div_root_two - 2.5, 0.0]
    expected_clips = [8.0, 4.0, four_div_root_two, 2.0, 2.0]
    self._test_estimate_clip_expected_sum(
        query,
        global_state,
        [record1, record2],
        expected_sums,
        expected_clips,
        reset=reset,
    )

  @parameterized.named_parameters(('adaptive', True), ('constant', False))
  def test_adaptation_target_one(self, reset):
    record1 = tf.constant([-1.5])
    record2 = tf.constant([2.75])

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=0.0,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=1.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=False,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.clip_value
    self.assertAllClose(initial_clip, 0.0)

    # On the first two iterations, both are clipped, so the clip goes up
    # by 1.0 (the learning rate). When the clip reaches 2.0, only one record is
    # clipped, so the clip goes up by only 0.5. After two more iterations,
    # both records are clipped, and the clip norm stays there (at 3.0).

    expected_sums = [0.0, 0.0, 0.5, 1.0, 1.25]
    expected_clips = [1.0, 2.0, 2.5, 3.0, 3.0]
    self._test_estimate_clip_expected_sum(
        query,
        global_state,
        [record1, record2],
        expected_sums,
        expected_clips,
        reset=reset,
    )

  @parameterized.named_parameters(('adaptive', True), ('constant', False))
  def test_adaptation_target_one_geometric(self, reset):
    record1 = tf.constant([-1.5])
    record2 = tf.constant([3.0])

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=0.5,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=1.0,
        learning_rate=np.log(2.0),  # Geometric steps in powers of 2.
        clipped_count_stddev=0.0,
        expected_num_records=2.0,
        geometric_update=True,
    )

    global_state = query.initial_global_state()

    initial_clip = global_state.sum_state.clip_value
    self.assertAllClose(initial_clip, 0.5)

    # On the first two iterations, both are clipped, so the clip is doubled.
    # When the clip reaches 2.0, only one record is clipped, so the clip is
    # multiplied by sqrt(2.0). Still only one is clipped so it increases to 4.0.
    # Now both records are clipped, and the clip norm stays there (at 4.0).

    two_times_root_two = 2 * np.sqrt(2.0)  # approx 2.828

    expected_sums = [0.0, 0.0, 0.5, two_times_root_two - 1.5, 1.5]
    expected_clips = [1.0, 2.0, two_times_root_two, 4.0, 4.0]
    self._test_estimate_clip_expected_sum(
        query,
        global_state,
        [record1, record2],
        expected_sums,
        expected_clips,
        reset=reset,
    )

  def _test_estimate_clip_converge(
      self,
      query,
      records,
      expected_clip,
      tolerance,
      learning_rate=None,
      total_steps=50,
      converge_steps=40,
  ):
    global_state = query.initial_global_state()
    for t in range(total_steps):
      if learning_rate is not None:
        learning_rate.assign(1.0 / np.sqrt(t + 1))
      actual_sum, global_state = test_utils.run_query(
          query, records, global_state
      )
      if t > converge_steps:
        global_state = query.reset_state(actual_sum, global_state)
        estimate_clip = global_state.sum_state.clip_value
        self.assertNear(estimate_clip, expected_clip, tolerance)

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_linspace(self, start_low, geometric):
    # `num_records` records equally spaced from 0 to 10 in 0.1 increments.
    # Test that we converge to the correct median value and bounce around it.
    num_records = 21
    records = [
        tf.constant(x)
        for x in np.linspace(0.0, 10.0, num=num_records, dtype=np.float32)
    ]

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=0.5,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=num_records,
        geometric_update=geometric,
    )

    self._test_estimate_clip_converge(
        query, records, expected_clip=5.0, tolerance=0.25
    )

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_all_equal(self, start_low, geometric):
    # `num_records` equal records. Test that we converge to that record and
    # bounce around it. Unlike the linspace test, the quantile-matching
    # objective is very sharp at the optimum so a decaying learning rate is
    # necessary.
    num_records = 20
    records = [tf.constant(5.0)] * num_records

    learning_rate = tf.Variable(1.0)

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=0.5,
        learning_rate=learning_rate,
        clipped_count_stddev=0.0,
        expected_num_records=num_records,
        geometric_update=geometric,
    )

    self._test_estimate_clip_converge(
        query,
        records,
        expected_clip=5.0,
        tolerance=0.5,
        learning_rate=learning_rate,
    )

  def _test_noise_multiplier(
      self,
      query,
      records,
      noise_multiplier,
      learning_rate=None,
      tolerance=0.15,
      total_steps=10,
  ):
    global_state = query.initial_global_state()
    for t in range(total_steps):
      if learning_rate is not None:
        learning_rate.assign((t + 1.0) ** (-0.5))
      params = query.derive_sample_params(global_state)
      sample_state = query.initial_sample_state(records[0])
      for record in records:
        sample_state = query.accumulate_record(params, sample_state, record)
      actual_sum, global_state, _ = query.get_noised_result(
          sample_state, global_state
      )
      expected_std = global_state.sum_state.clip_value * noise_multiplier
      self.assertAllClose(
          expected_std,
          global_state.sum_state.tree_state.value_generator_state.stddev,
      )
      global_state = query.reset_state(actual_sum, global_state)
      self.assertAllClose(
          expected_std, tf.math.reduce_std(actual_sum), rtol=tolerance
      )

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_linspace_noise(self, start_low, geometric):
    # `num_records` records equally spaced from 0 to 10 in 0.1 increments.
    # Test that we converge to the correct median value and bounce around it.
    num_records, vector_size, noise_multiplier = 11, 1000, 0.1
    records = [
        tf.constant(
            vector_size ** (-0.5) * x, shape=[vector_size], dtype=tf.float32
        )
        for x in np.linspace(0.0, 10.0, num=num_records)
    ]

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=noise_multiplier,
        record_specs=tf.TensorSpec([vector_size]),
        target_unclipped_quantile=0.5,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=num_records,
        geometric_update=geometric,
        noise_seed=1,
    )

    self._test_noise_multiplier(query, records, noise_multiplier)

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_all_equal_noise(self, start_low, geometric):
    # `num_records` equal records. Test that we converge to that record and
    # bounce around it. Unlike the linspace test, the quantile-matching
    # objective is very sharp at the optimum so a decaying learning rate is
    # necessary.
    num_records, vector_size, noise_multiplier = 10, 1000, 0.5
    records = [
        tf.constant(
            vector_size ** (-0.5) * 5.0, shape=[vector_size], dtype=tf.float32
        )
    ] * num_records

    learning_rate = tf.Variable(1.0)

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=noise_multiplier,
        record_specs=tf.TensorSpec([vector_size]),
        target_unclipped_quantile=0.5,
        learning_rate=learning_rate,
        clipped_count_stddev=0.0,
        expected_num_records=num_records,
        geometric_update=geometric,
        noise_seed=1,
    )

    self._test_noise_multiplier(
        query, records, noise_multiplier, learning_rate=learning_rate
    )

  def test_adaptation_clip_noise(self):
    sample_num, tolerance, stddev = 1000, 0.3, 0.1
    initial_clip, expected_num_records = 5.0, 2.0
    record1 = tf.constant(1.0)
    record2 = tf.constant(10.0)

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=initial_clip,
        noise_multiplier=0.0,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=0.5,
        learning_rate=1.0,
        clipped_count_stddev=stddev,
        expected_num_records=expected_num_records,
        geometric_update=False,
        noise_seed=1,
    )

    global_state = query.initial_global_state()
    samples = []
    for _ in range(sample_num):
      noised_results, global_state = test_utils.run_query(
          query, [record1, record2], global_state
      )
      samples.append(noised_results.numpy())
      global_state = query.reset_state(noised_results, global_state)
      self.assertNotEqual(
          global_state.quantile_estimator_state.current_estimate, initial_clip
      )
      # Force to use the same clip norm for noise estimation
      quantile_estimator_state = global_state.quantile_estimator_state._replace(
          current_estimate=initial_clip
      )
      global_state = global_state._replace(
          quantile_estimator_state=quantile_estimator_state
      )

    # The sum result is 1. (unclipped) + 5. (clipped) = 6.
    self.assertAllClose(np.mean(samples), 6.0, atol=4 * stddev)
    self.assertAllClose(
        np.std(samples), stddev / expected_num_records, rtol=tolerance
    )

  @parameterized.named_parameters(
      ('start_low_arithmetic', True, False),
      ('start_low_geometric', True, True),
      ('start_high_arithmetic', False, False),
      ('start_high_geometric', False, True),
  )
  def test_adaptation_linspace_noise_converge(self, start_low, geometric):
    # `num_records` records equally spaced from 0 to 10 in 0.1 increments.
    # Test that we converge to the correct median value and bounce around it.
    num_records = 21
    records = [
        tf.constant(x)
        for x in np.linspace(0.0, 10.0, num=num_records, dtype=np.float32)
    ]

    query = quantile_query.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip=(1.0 if start_low else 10.0),
        noise_multiplier=0.01,
        record_specs=tf.TensorSpec([]),
        target_unclipped_quantile=0.5,
        learning_rate=1.0,
        clipped_count_stddev=0.01,
        expected_num_records=num_records,
        geometric_update=geometric,
        noise_seed=1,
    )

    self._test_estimate_clip_converge(
        query, records, expected_clip=5.0, tolerance=0.25
    )


if __name__ == '__main__':
  tf.test.main()
