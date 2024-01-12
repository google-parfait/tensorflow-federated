# Copyright 2020, The TensorFlow Federated Authors.
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
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import aggregator_test_utils
from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

_test_dp_query = tfp.GaussianSumQuery(l2_norm_clip=1.0, stddev=0.0)

_test_struct_type = [(np.float32, (2,)), np.float32]
_test_inner_agg_factory = aggregator_test_utils.SumPlusOneFactory()


class DPFactoryComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_simple', np.float32, None),
      ('struct_simple', _test_struct_type, None),
      ('float_inner', np.float32, _test_inner_agg_factory),
      ('struct_inner', _test_struct_type, _test_inner_agg_factory),
  )
  def test_type_properties(self, value_type, inner_agg_factory):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(
        _test_dp_query, inner_agg_factory
    )
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = factory_.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    query_state_type = computation_types.StructType(
        [('l2_norm_clip', np.float32), ('stddev', np.float32)]
    )
    query_metrics_type = ()
    inner_state_type = np.int32 if inner_agg_factory else ()
    dp_event_type = computation_types.StructType([
        ('module_name', np.str_),
        ('class_name', np.str_),
        ('noise_multiplier', np.float32),
    ])
    server_state_type = computation_types.FederatedType(
        differential_privacy.DPAggregatorState(
            query_state_type, inner_state_type, dp_event_type, np.bool_
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )
    inner_measurements_type = np.int32 if inner_agg_factory else ()
    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            dp_query_metrics=query_metrics_type, dp=inner_measurements_type
        ),
        placements.SERVER,
    )

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  def test_init_non_dp_query_raises(self):
    with self.assertRaises(TypeError):
      differential_privacy.DifferentiallyPrivateFactory('not a dp_query')

  def test_init_non_agg_factory_raises(self):
    with self.assertRaises(TypeError):
      differential_privacy.DifferentiallyPrivateFactory(
          _test_dp_query, 'not an agg factory'
      )

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.float32, placements.SERVER),
      ),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(np.float32)),
  )
  def test_incorrect_value_type_raises(self, bad_value_type):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    with self.assertRaises(TypeError):
      factory_.create(bad_value_type)


class DPFactoryExecutionTest(tf.test.TestCase, parameterized.TestCase):

  def assertInnerSumPlusOnePerformed(
      self,
      output: measured_process.MeasuredProcessOutput,
      l2_clip: float,
      client_data: list[float],
  ):
    """Asserts one step of SumPlusOneFactory was performed in inner aggregation.

    This includes three tests:
      1: that the output sum is as desired, accounting for clipping each float
        and the additional +1 of the SumPlusOneFactory.
      2: that the aggregator_test_utils.SumPlusOneFactory was called exactly
        once.
      3: that the measurement constant was left untouched.

    Args:
      output: The output of the `AggregationProcess`.next
      l2_clip: The l2 clipping parameter passed to the DP factory of choice.
      client_data: The client data passed to the `AggregationProcess`.next
    """
    self.assertAllClose(
        sum([min(x, l2_clip) for x in client_data]) + 1.0, output.result
    )
    self.assertAllEqual(
        1, output.state.agg_state
    )  # incremented by one each time.
    self.assertAllEqual(
        aggregator_test_utils.MEASUREMENT_CONSTANT, output.measurements['dp']
    )

  def test_simple_sum(self):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    value_type = computation_types.TensorType(np.float32)
    process = factory_.create(value_type)

    # The test query has clip 1.0 and no noise, so this computes clipped sum.

    state = process.initialize()

    client_data = [0.5, 1.0, 1.5]
    output = process.next(state, client_data)
    self.assertAllClose(2.5, output.result)

  def test_structure_sum(self):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    value_type = computation_types.to_type([np.float32, np.float32])
    process = factory_.create(value_type)

    # The test query has clip 1.0 and no noise, so this computes clipped sum.

    state = process.initialize()

    # pyformat: disable
    client_data = [
        [0.1, 0.2],         # not clipped (norm < 1)
        [5 / 13, 12 / 13],  # not clipped (norm == 1)
        [3.0, 4.0]          # clipped to 3/5, 4/5
    ]
    output = process.next(state, client_data)

    expected_result = [0.1 +  5 / 13 + 3 / 5,
                       0.2 + 12 / 13 + 4 / 5]
    # pyformat: enable
    self.assertAllClose(expected_result, output.result)

  def test_inner_sum(self):
    value_type = computation_types.TensorType(np.float32)
    factory_ = differential_privacy.DifferentiallyPrivateFactory(
        _test_dp_query, _test_inner_agg_factory
    )
    process = factory_.create(value_type)

    state = process.initialize()
    self.assertAllEqual(0, state.agg_state)

    client_data = [0.5, 1.0, 1.5]
    output = process.next(state, client_data)
    self.assertInnerSumPlusOnePerformed(
        output, _test_dp_query._l2_norm_clip, client_data
    )

  def test_tree_aggregation_inner_sum(self):
    l2_clip = 1.0
    value_type = computation_types.TensorType(np.float32)
    tree_factory = (
        differential_privacy.DifferentiallyPrivateFactory.tree_aggregation(
            noise_multiplier=0.0,
            l2_norm_clip=l2_clip,
            record_specs=value_type,
            clients_per_round=1.0,
            record_aggregation_factory=_test_inner_agg_factory,
        )
    )
    process = tree_factory.create(value_type)

    state = process.initialize()
    self.assertAllEqual(0, state.agg_state)

    client_data = [0.5, 1.0, 1.5]
    output = process.next(state, client_data)
    self.assertInnerSumPlusOnePerformed(output, l2_clip, client_data)

  def test_adaptive_query(self):
    query = tfp.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=1.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=3.0,
        geometric_update=False,
    )
    factory_ = differential_privacy.DifferentiallyPrivateFactory(query)
    value_type = computation_types.TensorType(np.float32)
    process = factory_.create(value_type)

    state = process.initialize()

    client_data = [0.5, 1.5, 2.0]  # Two clipped on first round.
    expected_result = 0.5 + 1.0 + 1.0
    output = process.next(state, client_data)
    self.assertAllClose(expected_result, output.result)

    # Clip is increased by 2/3 to 5/3.
    expected_result = 0.5 + 1.5 + 5 / 3
    output = process.next(output.state, client_data)
    self.assertAllClose(expected_result, output.result)

  def test_extract_dp_event_from_state(self):
    value_type = computation_types.TensorType(np.float32)
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    process = factory_.create(value_type)
    state = process.initialize()
    client_data = [0.0]
    output = process.next(state, client_data)
    event = differential_privacy.extract_dp_event_from_state(output.state)
    initial_sample_state = _test_dp_query.initial_sample_state(
        type_conversions.type_to_tf_tensor_specs(value_type)
    )
    query_state = _test_dp_query.initial_global_state()
    expected_dp_event = _test_dp_query.get_noised_result(
        initial_sample_state, query_state
    )[2]
    self.assertEqual(event, expected_dp_event)

  def test_error_when_extracting_from_initial_state(self):
    value_type = computation_types.TensorType(np.float32)
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    process = factory_.create(value_type)
    state = process.initialize()
    with self.assertRaises(
        differential_privacy.ExtractingDpEventFromInitialStateError
    ):
      differential_privacy.extract_dp_event_from_state(state)

  def test_noise(self):
    noise = 3.14159
    factory_ = differential_privacy.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise, clients_per_round=1.0, clip=1.0
    )
    value_type = computation_types.TensorType(np.float32)
    process = factory_.create(value_type)

    state = process.initialize()
    client_data = [0.36788]
    outputs = []
    # num_iterations empirically chosen to flake < 1 in 10000 runs of this test.
    num_iterations = 500

    for _ in range(num_iterations):
      output = process.next(state, client_data)
      outputs.append(output.result)
      state = output.state

    stddev = np.std(outputs, ddof=1.0)
    self.assertAllClose(stddev, noise, rtol=0.15)

  def test_gaussian_adaptive_cls(self):
    process = (
        differential_privacy.DifferentiallyPrivateFactory.gaussian_adaptive(
            noise_multiplier=1e-2, clients_per_round=10
        )
    )
    self.assertIsInstance(
        process, differential_privacy.DifferentiallyPrivateFactory
    )

  def test_gaussian_fixed_cls(self):
    process = differential_privacy.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=1.0, clients_per_round=10, clip=1.0
    )
    self.assertIsInstance(
        process, differential_privacy.DifferentiallyPrivateFactory
    )

  @parameterized.named_parameters([
      ('0', None),
      ('1', 5.0),
  ])
  def test_adaptive_clip_noise_params(self, clip_count_stddev):
    noise_mult = 2.0
    num_clients = 100.0
    value_noise_mult, new_clip_count_stddev = (
        differential_privacy.adaptive_clip_noise_params(
            noise_mult, num_clients, clip_count_stddev
        )
    )

    # The effective noise for client values are larger as we're splitting the
    # privacy budget (intended by the input noise level) with adaptive clipping.
    self.assertGreater(value_noise_mult, noise_mult)

    if clip_count_stddev is None:
      # Check if the default value is assignend.
      self.assertEqual(new_clip_count_stddev, 0.05 * num_clients)
    else:
      # Check if the specified value is kept.
      self.assertEqual(new_clip_count_stddev, clip_count_stddev)

  @parameterized.named_parameters(
      ('total5_std2', 5, 8.0, 2.0, False),
      ('total6_std0d5', 6, 0.5, 0.5, False),
      ('total7_std1', 7, 3.0, 1.0, False),
      ('total8_std1', 8, 1.0, 1.0, False),
      ('total3_std1_eff', 3, 1.0 + 2.0 / 3.0, 1.0, True),
      ('total4_std1_eff', 4, 4.0 / 7.0, 1.0, True),
  )
  def test_tree_aggregation_factory(
      self, total_steps, expected_variance, noise_std, use_efficient
  ):
    variable_shape, tolerance = [10000], 0.05
    record = tf.zeros(variable_shape, tf.float32)
    record_shape = tf.nest.map_structure(lambda t: t.shape, record)
    record_type = computation_types.to_type((np.float32, variable_shape))
    specs = tf.nest.map_structure(tf.TensorSpec, record_shape)

    tree_factory = (
        differential_privacy.DifferentiallyPrivateFactory.tree_aggregation(
            noise_multiplier=noise_std,
            l2_norm_clip=1.0,
            record_specs=specs,
            clients_per_round=1.0,
            noise_seed=1,
            use_efficient=use_efficient,
        )
    )

    process = tree_factory.create(record_type)

    state = process.initialize()
    client_data = [record]
    cumsum_result = np.zeros(variable_shape, np.float32)
    for _ in range(total_steps):
      output = process.next(state, client_data)
      state = output.state
      cumsum_result += output.result
    self.assertAllClose(
        np.sqrt(expected_variance), np.std(cumsum_result), rtol=tolerance
    )

  @parameterized.named_parameters(
      ('negative_clip', -1.0, 0.0),
      ('zero_clip', 0.0, 0.0),
      ('negative_noise', 1.0, -1.0),
  )
  def test_tree_aggregation_factory_raise(self, clip_norm, noise_multiplier):
    with self.assertRaisesRegex(ValueError, 'must be'):
      differential_privacy.DifferentiallyPrivateFactory.tree_aggregation(
          noise_multiplier=noise_multiplier,
          l2_norm_clip=clip_norm,
          record_specs=tf.TensorSpec([]),
          clients_per_round=1.0,
          noise_seed=1,
      )

  def test_tree_adaptive_factory_estimate_clip(self):
    factory_ = differential_privacy.DifferentiallyPrivateFactory.tree_adaptive(
        noise_multiplier=0.0,
        clients_per_round=3,
        record_specs=tf.TensorSpec([]),
        initial_l2_norm_clip=1.0,
        restart_warmup=None,
        restart_frequency=1,
        target_unclipped_quantile=1.0,
        clip_learning_rate=1.0,
        clipped_count_stddev=0.0,
        noise_seed=1,
    )
    process = factory_.create(computation_types.TensorType(np.float32))

    state = process.initialize()

    client_data = [0.5, 1.5, 2.5]  # Two clipped on first round.
    expected_result = (0.5 + 1.0 + 1.0) / 3.0
    output = process.next(state, client_data)
    self.assertAllClose(expected_result, output.result)

    # Clip is increased to np.exp(2./3)~1.95.
    expected_result = (0.5 + 1.5 + np.exp(2.0 / 3)) / 3.0
    output = process.next(output.state, client_data)
    self.assertAllClose(expected_result, output.result)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
