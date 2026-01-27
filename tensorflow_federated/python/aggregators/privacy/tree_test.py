# Copyright 2021, The TensorFlow Authors.
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

from tensorflow_federated.python.aggregators.privacy import test_utils
from tensorflow_federated.python.aggregators.privacy import tree as tree_query


class RoundRestartIndicatorTest(tf.test.TestCase, parameterized.TestCase):

  def assertRestartsOnPeriod(
      self,
      indicator: tree_query.RestartIndicator,
      state: tf.Tensor,
      total_steps: int,
      period: int,
      offset: int,
  ):
    """Asserts a restart occurs only every `period` steps."""
    for step in range(total_steps):
      flag, state = indicator.next(state)
      if step % period == offset - 1:
        self.assertTrue(flag)
      else:
        self.assertFalse(flag)

  @parameterized.named_parameters(('zero', 0), ('negative', -1))
  def test_round_raise(self, period):
    with self.assertRaisesRegex(
        ValueError, 'Restart period should be equal or larger than 1'
    ):
      tree_query.PeriodicRoundRestartIndicator(period)

  @parameterized.named_parameters(
      ('zero', 0), ('negative', -1), ('equal', 2), ('large', 3)
  )
  def test_round_raise_warmup(self, warmup):
    period = 2
    with self.assertRaisesRegex(
        ValueError, f'Warmup must be between 1 and `period`-1={period-1}'
    ):
      tree_query.PeriodicRoundRestartIndicator(period, warmup)

  @parameterized.named_parameters(
      ('period_1', 1), ('period_2', 2), ('period_4', 4), ('period_5', 5)
  )
  def test_round_indicator(self, period):
    total_steps = 20
    indicator = tree_query.PeriodicRoundRestartIndicator(period)
    state = indicator.initialize()

    self.assertRestartsOnPeriod(indicator, state, total_steps, period, period)

  @parameterized.named_parameters(
      ('period_2', 2, 1), ('period_4', 4, 3), ('period_5', 5, 2)
  )
  def test_round_indicator_warmup(self, period, warmup):
    total_steps = 20
    indicator = tree_query.PeriodicRoundRestartIndicator(period, warmup)
    state = indicator.initialize()

    self.assertRestartsOnPeriod(indicator, state, total_steps, period, warmup)


def _get_l2_clip_fn():

  def l2_clip_fn(record_as_list, clip_value):
    clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_value)
    return clipped_record

  return l2_clip_fn


class RestartQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('s0t1f1', 0.0, 1.0, 1),
      ('s0t1f2', 0.0, 1.0, 2),
      ('s0t1f5', 0.0, 1.0, 5),
      ('s1t1f5', 1.0, 1.0, 5),
      ('s1t2f2', 1.0, 2.0, 2),
      ('s1t5f6', 1.0, 5.0, 6),
  )
  def test_scalar_tree_aggregation_reset(
      self, scalar_value, tree_node_value, period
  ):
    total_steps = 20
    indicator = tree_query.PeriodicRoundRestartIndicator(period)
    query = tree_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.0,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False,
    )
    query = tree_query.RestartQuery(query, indicator)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state
      )
      # Expected value is the signal of the current round plus the residual of
      # two continous tree aggregation values. The tree aggregation value can
      # be inferred from the binary representation of the current step.
      expected = scalar_value + tree_node_value * (
          bin(i % period + 1)[2:].count('1') - bin(i % period)[2:].count('1')
      )
      self.assertEqual(query_result, expected)


STRUCT_RECORD = [
    tf.constant([[2.0, 0.0], [0.0, 1.0]]),
    tf.constant([-1.0, 0.0]),
]

SINGLE_VALUE_RECORDS = [tf.constant(1.0), tf.constant(3.0), tf.constant(5.0)]

STRUCTURE_SPECS = tf.nest.map_structure(
    lambda t: tf.TensorSpec(tf.shape(t)), STRUCT_RECORD
)
NOISE_STD = 5.0

STREAMING_SCALARS = np.array(range(7), dtype=np.single)


def _get_noise_generator(specs, stddev=NOISE_STD, seed=1):
  return tree_query.GaussianNoiseGenerator(
      noise_std=stddev, specs=specs, seed=seed
  )


def _get_noise_fn(specs, stddev=NOISE_STD, seed=1):
  random_generator = tf.random.Generator.from_seed(seed)

  def noise_fn():
    shape = tf.nest.map_structure(lambda spec: spec.shape, specs)
    return tf.nest.map_structure(
        lambda x: random_generator.normal(x, stddev=stddev), shape
    )

  return noise_fn


class TreeResidualQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('r5d10n0s1s16eff', 5, 10, 0.0, 1, 16, 0.1, True),
      ('r3d5n1s1s32eff', 3, 5, 1.0, 1, 32, 1.0, True),
      ('r10d3n1s2s16eff', 10, 3, 1.0, 2, 16, 10.0, True),
      ('r10d3n1s2s16', 10, 3, 1.0, 2, 16, 10.0, False),
  )
  def test_sum(
      self,
      records_num,
      record_dim,
      noise_multiplier,
      seed,
      total_steps,
      clip,
      use_efficient,
  ):
    record_specs = tf.TensorSpec(shape=[record_dim])
    query = tree_query.TreeResidualSumQuery.build_l2_gaussian_query(
        clip_norm=clip,
        noise_multiplier=noise_multiplier,
        record_specs=record_specs,
        noise_seed=seed,
        use_efficient=use_efficient,
    )
    sum_query = tree_query.TreeCumulativeSumQuery.build_l2_gaussian_query(
        clip_norm=clip,
        noise_multiplier=noise_multiplier,
        record_specs=record_specs,
        noise_seed=seed,
        use_efficient=use_efficient,
    )
    global_state = query.initial_global_state()
    sum_global_state = sum_query.initial_global_state()

    cumsum_result = tf.zeros(shape=[record_dim])
    for _ in range(total_steps):
      records = [
          tf.random.uniform(shape=[record_dim], maxval=records_num)
          for _ in range(records_num)
      ]
      query_result, global_state = test_utils.run_query(
          query, records, global_state
      )
      sum_query_result, sum_global_state = test_utils.run_query(
          sum_query, records, sum_global_state
      )
      cumsum_result += query_result
      self.assertAllClose(cumsum_result, sum_query_result, rtol=1e-6)

  @parameterized.named_parameters(
      ('efficient', True, tree_query.EfficientTreeAggregator),
      ('normal', False, tree_query.TreeAggregator),
  )
  def test_sum_tree_aggregator_instance(self, use_efficient, tree_class):
    specs = tf.TensorSpec([])
    query = tree_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=1.0,
        noise_generator=_get_noise_fn(specs, 1.0),
        record_specs=specs,
        use_efficient=use_efficient,
    )
    self.assertIsInstance(query._tree_aggregator, tree_class)

  def test_derive_metrics(self):
    specs = tf.TensorSpec([])
    l2_clip = 2
    query = tree_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=l2_clip,
        noise_generator=_get_noise_fn(specs, 1.0),
        record_specs=specs,
    )
    metrics = query.derive_metrics(query.initial_global_state())
    self.assertEqual(metrics['tree_agg_dpftrl_clip'], l2_clip)

  @parameterized.named_parameters(
      ('s0t1f1', 0.0, 1.0, 1),
      ('s0t1f2', 0.0, 1.0, 2),
      ('s0t1f5', 0.0, 1.0, 5),
      ('s1t1f5', 1.0, 1.0, 5),
      ('s1t2f2', 1.0, 2.0, 2),
      ('s1t5f6', 1.0, 5.0, 6),
  )
  def test_scalar_tree_aggregation_reset(
      self, scalar_value, tree_node_value, frequency
  ):
    total_steps = 20
    query = tree_query.TreeResidualSumQuery(
        clip_fn=_get_l2_clip_fn(),
        clip_value=scalar_value + 1.0,  # no clip
        noise_generator=lambda: tree_node_value,
        record_specs=tf.TensorSpec([]),
        use_efficient=False,
    )
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i in range(total_steps):
      sample_state = query.initial_sample_state(scalar_value)
      sample_state = query.accumulate_record(params, sample_state, scalar_value)
      query_result, global_state, _ = query.get_noised_result(
          sample_state, global_state
      )
      if i % frequency == frequency - 1:
        global_state = query.reset_state(query_result, global_state)
      # Expected value is the signal of the current round plus the residual of
      # two continous tree aggregation values. The tree aggregation value can
      # be inferred from the binary representation of the current step.
      expected = scalar_value + tree_node_value * (
          bin(i % frequency + 1)[2:].count('1')
          - bin(i % frequency)[2:].count('1')
      )
      self.assertEqual(query_result, expected)


class BuildTreeTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      leaf_nodes_size=[1, 2, 3, 4, 5],
      arity=[2, 3],
      dtype=[tf.int32, tf.float32],
  )
  def test_build_tree_from_leaf(self, leaf_nodes_size, arity, dtype):
    """Test whether `_build_tree_from_leaf` will output the correct tree."""

    leaf_nodes = tf.cast(tf.range(leaf_nodes_size), dtype)
    depth = math.ceil(math.log(leaf_nodes_size, arity)) + 1

    tree = tree_query._build_tree_from_leaf(leaf_nodes, arity)

    self.assertEqual(depth, tree.shape[0])

    for layer in range(depth):
      reverse_depth = tree.shape[0] - layer - 1
      span_size = arity**reverse_depth
      for idx in range(arity**layer):
        left = idx * span_size
        right = (idx + 1) * span_size
        expected_value = sum(leaf_nodes[left:right])
        self.assertEqual(tree[layer][idx], expected_value)


class TreeRangeSumQueryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.product(
      params=[(0.0, 1.0, 2), (1.0, -1.0, 2), (1.0, 1.0, 1)],
  )
  def test_raises_error(self, params):
    clip_norm, stddev, arity = params
    with self.assertRaises(ValueError):
      tree_query.TreeRangeSumQuery.build_central_gaussian_query(
          clip_norm, stddev, arity
      )

  @parameterized.product(
      clip_norm=[0.1, 1.0, 10.0],
      stddev=[0.1, 1.0, 10.0],
  )
  def test_initial_global_state_type(self, clip_norm, stddev):
    query = tree_query.TreeRangeSumQuery.build_central_gaussian_query(
        clip_norm, stddev
    )
    global_state = query.initial_global_state()
    self.assertIsInstance(
        global_state, tree_query.TreeRangeSumQuery.GlobalState
    )

  @parameterized.product(
      clip_norm=[0.1, 1.0, 10.0],
      stddev=[0.1, 1.0, 10.0],
      arity=[2, 3, 4],
  )
  def test_derive_sample_params(self, clip_norm, stddev, arity):
    query = tree_query.TreeRangeSumQuery.build_central_gaussian_query(
        clip_norm, stddev, arity
    )
    global_state = query.initial_global_state()
    derived_arity, inner_query_state = query.derive_sample_params(global_state)
    self.assertAllClose(derived_arity, arity)
    self.assertAllClose(inner_query_state, clip_norm)

  @parameterized.product(
      (
          dict(arity=2, expected_tree=[1, 1, 0, 1, 0, 0, 0]),
          dict(arity=3, expected_tree=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
      ),
  )
  def test_preprocess_record(self, arity, expected_tree):
    query = tree_query.TreeRangeSumQuery.build_central_gaussian_query(
        10.0, 0.0, arity
    )
    record = tf.constant([1, 0, 0, 0], dtype=tf.float32)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    self.assertAllClose(preprocessed_record, expected_tree)

  @parameterized.product(
      (
          dict(
              arity=2,
              expected_tree=tf.ragged.constant([[1], [1, 0], [1, 0, 0, 0]]),
          ),
          dict(
              arity=3,
              expected_tree=tf.ragged.constant(
                  [[1], [1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]
              ),
          ),
      ),
  )
  def test_get_noised_result(self, arity, expected_tree):
    query = tree_query.TreeRangeSumQuery.build_central_gaussian_query(
        10.0, 0.0, arity
    )
    record = tf.constant([1, 0, 0, 0], dtype=tf.float32)
    expected_tree = tf.cast(expected_tree, tf.float32)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(params, record)
    sample_state, global_state, _ = query.get_noised_result(
        preprocessed_record, global_state
    )

    self.assertAllClose(sample_state, expected_tree)

  @parameterized.product(stddev=[0.1, 1.0, 10.0])
  def test_central_get_noised_result_with_noise(self, stddev):
    query = tree_query.TreeRangeSumQuery.build_central_gaussian_query(
        10.0, stddev
    )
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    preprocessed_record = query.preprocess_record(
        params, tf.constant([1.0, 0.0])
    )
    sample_state, global_state, _ = query.get_noised_result(
        preprocessed_record, global_state
    )

    self.assertAllClose(
        sample_state, tf.ragged.constant([[1.0], [1.0, 0.0]]), atol=10 * stddev
    )


if __name__ == '__main__':
  tf.test.main()
