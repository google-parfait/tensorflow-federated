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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning.metrics import aggregation_factory

# Convenience aliases.
TensorType = computation_types.TensorType
MetricRange = aggregation_factory._MetricRange


def _get_finalized_metrics_type(metric_finalizers, unfinalized_metrics):
  if callable(metric_finalizers):
    finalized_metrics = metric_finalizers(unfinalized_metrics)
  else:
    finalized_metrics = collections.OrderedDict(
        (metric, finalizer(unfinalized_metrics[metric]))
        for metric, finalizer in metric_finalizers.items()
    )
  return type_conversions.type_from_tensors(finalized_metrics)


@tf.function
def _tf_mean(x):
  return tf.math.divide_no_nan(x[0], x[1])


_DEFAULT_FLOAT_FACTORY_KEY = 'None/default_estimation_process/1'
_DEFAULT_INT_FACTORY_KEY = '0/1048575/3'

_DEFAULT_FIXED_FLOAT_RANGE = (
    float(aggregation_factory.DEFAULT_FIXED_SECURE_LOWER_BOUND),
    float(aggregation_factory.DEFAULT_FIXED_SECURE_UPPER_BOUND),
)
_DEFAULT_AUTO_TUNED_FLOAT_RANGE = (
    None,
    quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=50.0,
        target_quantile=0.95,
        learning_rate=1.0,
        multiplier=2.0,
        secure_estimation=True,
    ),
)
_DEFAULT_INT_RANGE = (
    int(aggregation_factory.DEFAULT_FIXED_SECURE_LOWER_BOUND),
    int(aggregation_factory.DEFAULT_FIXED_SECURE_UPPER_BOUND),
)


class SumThenFinalizeFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'scalar_metric',
          collections.OrderedDict(num_examples=tf.function(func=lambda x: x)),
          collections.OrderedDict(num_examples=1.0),
      ),
      (
          'non_scalar_metric',
          collections.OrderedDict(loss=_tf_mean),
          collections.OrderedDict(loss=[2.0, 1.0]),
      ),
      (
          'callable',
          tf.function(
              lambda x: collections.OrderedDict(mean_loss=_tf_mean(x['loss']))
          ),
          collections.OrderedDict(loss=[1.0, 2.0]),
      ),
  )
  def test_type_properties(self, metric_finalizers, unfinalized_metrics):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    self.assertIsInstance(
        aggregate_factory, factory.UnweightedAggregationFactory
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        unfinalized_metrics
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = computation_types.FederatedType(
        ((), local_unfinalized_metrics_type), placements.SERVER
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    finalized_metrics_type = _get_finalized_metrics_type(
        metric_finalizers, unfinalized_metrics
    )
    result_value_type = computation_types.FederatedType(
        (finalized_metrics_type, finalized_metrics_type), placements.SERVER
    )
    measurements_type = computation_types.FederatedType((), placements.SERVER)
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            unfinalized_metrics=computation_types.FederatedType(
                local_unfinalized_metrics_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, measurements_type
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('default_metric_value_ranges', None),
      (
          'custom_metric_value_ranges',
          collections.OrderedDict(
              num_examples=(0, 100),
              loss=[
                  None,
                  (0.0, 100.0),
              ],
          ),
      ),
  )
  def test_type_properties_with_inner_secure_sum_process(
      self, metric_value_ranges
  ):
    if metric_value_ranges is None:
      secure_summation_factory = aggregation_factory.SecureSumFactory()
    else:
      secure_summation_factory = aggregation_factory.SecureSumFactory(
          metric_value_ranges
      )
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
        custom_sum=tf.function(
            func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))
        ),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1,
        loss=[2.0, 1.0],
        custom_sum=[tf.constant(1.0), tf.constant([1.0, 1.0])],
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics
    )

    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers, inner_summation_factory=secure_summation_factory
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)

    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    secure_summation_process = secure_summation_factory.create(
        local_unfinalized_metrics_type
    )
    expected_state_type = computation_types.FederatedType(
        (
            secure_summation_process.initialize.type_signature.result.member,
            local_unfinalized_metrics_type,
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    finalized_metrics_type = _get_finalized_metrics_type(
        metric_finalizers, local_unfinalized_metrics
    )
    result_value_type = computation_types.FederatedType(
        (finalized_metrics_type, finalized_metrics_type), placements.SERVER
    )
    measurements_type = computation_types.FederatedType(
        secure_summation_process.next.type_signature.result.measurements.member,
        placements.SERVER,
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            unfinalized_metrics=computation_types.FederatedType(
                local_unfinalized_metrics_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, measurements_type
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Metric aggregation contains non-secure summation aggregation')

  @parameterized.named_parameters(
      ('float', 1.0),
      (
          'list',
          [tf.function(func=lambda x: x), tf.function(func=lambda x: x + 1)],
      ),
  )
  def test_incorrect_finalizers_type_raises(self, bad_finalizers):
    with self.assertRaises(TypeError):
      aggregation_factory.SumThenFinalizeFactory(bad_finalizers)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(tf.float32, placements.SERVER),
      ),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)),
  )
  def test_incorrect_unfinalized_metrics_type_raises(
      self, bad_unfinalized_metrics_type
  ):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    with self.assertRaisesRegex(
        TypeError, 'Expected .*`tff.types.StructWithPythonType`'
    ):
      aggregate_factory.create(bad_unfinalized_metrics_type)

  def test_finalizers_and_unfinalized_metrics_type_mismatch_raises(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        ),
        container_type=collections.OrderedDict,
    )
    with self.assertRaisesRegex(
        ValueError, 'The metric names in `metric_finalizers` do not match those'
    ):
      aggregate_factory.create(local_unfinalized_metrics_type)

  def test_unfinalized_metrics_type_and_initial_values_mismatch_raises(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        collections.OrderedDict(num_examples=1.0)
    )
    initial_unfinalized_metrics = collections.OrderedDict(num_examples=[1.0])
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers,
        initial_unfinalized_metrics=initial_unfinalized_metrics,
    )
    with self.assertRaisesRegex(TypeError, 'initial unfinalized metrics type'):
      aggregate_factory.create(local_unfinalized_metrics_type)


class SumThenFinalizeFactoryExecutionTest(tf.test.TestCase):

  def test_sum_then_finalize_metrics(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
        custom_sum=tf.function(
            func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))
        ),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1.0,
        loss=[2.0, 1.0],
        custom_sum=[tf.constant(1.0), tf.constant([1.0, 1.0])],
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics
    )
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)

    state = process.initialize()
    _, unfinalized_metrics_accumulators = state
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=0.0,
        loss=[0.0, 0.0],
        custom_sum=[tf.constant(0.0), tf.constant([0.0, 0.0])],
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_unfinalized_metrics_accumulators,
    )

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    _, unfinalized_metrics_accumulators = output.state
    current_round_metrics, total_rounds_metrics = output.result
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=2.0,
        loss=[4.0, 2.0],
        custom_sum=[tf.constant(2.0), tf.constant([2.0, 2.0])],
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_unfinalized_metrics_accumulators,
    )
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(6.0)
        ),
    )
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(6.0)
        ),
    )

  def test_sum_then_finalize_metrics_with_initial_values(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1.0, loss=[2.0, 1.0]
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics
    )
    initial_unfinalized_metrics = collections.OrderedDict(
        num_examples=2.0, loss=[3.0, 2.0]
    )
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers,
        initial_unfinalized_metrics=initial_unfinalized_metrics,
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)

    state = process.initialize()
    _, unfinalized_metrics_accumulators = state
    self.assertEqual(
        unfinalized_metrics_accumulators, initial_unfinalized_metrics
    )

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    _, unfinalized_metrics_accumulators = output.state
    current_round_metrics, total_rounds_metrics = output.result
    self.assertEqual(
        unfinalized_metrics_accumulators,
        collections.OrderedDict(num_examples=4.0, loss=[7.0, 4.0]),
    )
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(num_examples=2.0, loss=2.0),
    )
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(num_examples=4.0, loss=1.75),
    )

  def test_secure_sum_then_finalize_metrics(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
        custom_sum=tf.function(
            func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))
        ),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1,
        loss=[2.0, 1.0],
        custom_sum=[tf.constant(101.0), tf.constant([1.0, 1.0])],
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics
    )
    secure_sum_factory = aggregation_factory.SecureSumFactory()

    aggregate_factory = aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers, inner_summation_factory=secure_sum_factory
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)
    state = process.initialize()
    init_inner_summation_process_state, unfinalized_metrics_accumulators = state
    expected_init_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=0.0,
        loss=[0.0, 0.0],
        custom_sum=[tf.constant(0.0), tf.constant([0.0, 0.0])],
    )
    secure_sum_process = secure_sum_factory.create(
        local_unfinalized_metrics_type
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_init_unfinalized_metrics_accumulators,
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        init_inner_summation_process_state,
        secure_sum_process.initialize(),
    )

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Metric aggregation contains non-secure summation aggregation')

    _, unfinalized_metrics_accumulators = output.state
    # Inital clippling bounds for float values are [-100.0, 100.0], metric
    # values fall outside the ranges will be clipped to within the range (e.g.,
    # 101.0 in `custom_sum` will be clipped to 100.0).
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=2.0,
        loss=[4.0, 2.0],
        custom_sum=[tf.constant(200.0), tf.constant([2.0, 2.0])],
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_unfinalized_metrics_accumulators,
    )

    current_round_metrics, total_rounds_metrics = output.result
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(204.0)
        ),
    )
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(204.0)
        ),
    )

    self.assertEqual(
        output.measurements,
        collections.OrderedDict([
            (
                _DEFAULT_FLOAT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=2,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=100.0,
                    secure_lower_threshold=-100.0,
                ),
            ),
            (
                _DEFAULT_INT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=0,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=1048575,
                    secure_lower_threshold=0,
                ),
            ),
        ]),
    )


class SecureSumFactoryTest(tf.test.TestCase, parameterized.TestCase):

  def test_default_value_ranges_returns_correct_results(self):
    aggregate_factory = aggregation_factory.SecureSumFactory()
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1,
        loss=[2.0, 1.0],
        custom_sum=[tf.constant(1.0), tf.constant([1.0, 1.0])],
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Metric aggregation contains non-secure summation aggregation')

    state = process.initialize()

    expected_factory_keys = set(
        [_DEFAULT_FLOAT_FACTORY_KEY, _DEFAULT_INT_FACTORY_KEY]
    )
    self.assertEqual(state.keys(), expected_factory_keys)

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    self.assertEqual(output.state.keys(), expected_factory_keys)
    # Assert only default float bounds are updated.
    self.assertNotAllEqual(
        output.state[_DEFAULT_FLOAT_FACTORY_KEY],
        state[_DEFAULT_FLOAT_FACTORY_KEY],
    )
    self.assertAllEqual(
        output.state[_DEFAULT_INT_FACTORY_KEY], state[_DEFAULT_INT_FACTORY_KEY]
    )

    tf.nest.map_structure(
        self.assertAllEqual,
        output.result,
        collections.OrderedDict(
            num_examples=2,
            loss=[4.0, 2.0],
            custom_sum=[tf.constant(2.0), tf.constant([2.0, 2.0])],
        ),
    )

    self.assertEqual(
        output.measurements,
        collections.OrderedDict([
            (
                _DEFAULT_FLOAT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=0,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=100.0,
                    secure_lower_threshold=-100.0,
                ),
            ),
            (
                _DEFAULT_INT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=0,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=aggregation_factory.DEFAULT_FIXED_SECURE_UPPER_BOUND,
                    secure_lower_threshold=aggregation_factory.DEFAULT_FIXED_SECURE_LOWER_BOUND,
                ),
            ),
        ]),
    )

  def test_user_value_ranges_returns_correct_results(self):
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=150,
        loss=[2.0, 1.0],
        custom_sum=[tf.constant(101.0), tf.constant([1.0, 1.0])],
    )
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics
    )
    metric_value_ranges = collections.OrderedDict(
        num_examples=(0, 100),
        loss=[
            None,
            (0.0, 100.0),
        ],
    )
    aggregate_factory = aggregation_factory.SecureSumFactory(
        metric_value_ranges
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Metric aggregation contains non-secure summation aggregation')

    state = process.initialize()
    custom_float_factory_key = aggregation_factory.create_factory_key(
        0.0, 100.0, tf.float32
    )
    custom_int_factory_key = aggregation_factory.create_factory_key(
        0, 100, tf.int32
    )
    expected_factory_keys = set([
        _DEFAULT_FLOAT_FACTORY_KEY,
        custom_float_factory_key,
        custom_int_factory_key,
    ])
    self.assertEqual(state.keys(), expected_factory_keys)

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    self.assertEqual(output.state.keys(), expected_factory_keys)
    # Assert only default float bounds are updated.
    self.assertNotAllEqual(
        output.state[_DEFAULT_FLOAT_FACTORY_KEY],
        state[_DEFAULT_FLOAT_FACTORY_KEY],
    )
    self.assertAllEqual(
        output.state[custom_float_factory_key], state[custom_float_factory_key]
    )
    self.assertAllEqual(
        output.state[custom_int_factory_key], state[custom_int_factory_key]
    )

    tf.nest.map_structure(
        self.assertAllEqual,
        output.result,
        collections.OrderedDict(
            num_examples=200,
            loss=[4.0, 2.0],
            custom_sum=[tf.constant(200.0), tf.constant([2.0, 2.0])],
        ),
    )

    self.assertEqual(
        output.measurements,
        collections.OrderedDict([
            (
                _DEFAULT_FLOAT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=2,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=100.0,
                    secure_lower_threshold=-100.0,
                ),
            ),
            (
                custom_float_factory_key,
                collections.OrderedDict(
                    secure_upper_clipped_count=0,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=100.0,
                    secure_lower_threshold=0.0,
                ),
            ),
            (
                custom_int_factory_key,
                collections.OrderedDict(
                    secure_upper_clipped_count=2,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=100,
                    secure_lower_threshold=0,
                ),
            ),
        ]),
    )

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(tf.float32, placements.SERVER),
      ),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)),
  )
  def test_incorrect_unfinalized_metrics_type_raises(
      self, bad_unfinalized_metrics_type
  ):
    secure_sum_factory = aggregation_factory.SecureSumFactory()
    with self.assertRaisesRegex(
        TypeError, 'Expected .*`tff.types.StructWithPythonType`'
    ):
      secure_sum_factory.create(bad_unfinalized_metrics_type)

  def test_user_value_ranges_fails_invalid_dtype(self):
    local_unfinalized_metrics = collections.OrderedDict(
        custom_sum=[tf.constant('abc')]
    )
    secure_sum_factory = aggregation_factory.SecureSumFactory()
    with self.assertRaises(aggregation_factory.UnquantizableDTypeError):
      secure_sum_factory.create(
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics
          )
      )

  def test_user_value_ranges_fails_not_2_tuple(self):
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        collections.OrderedDict(accuracy=[tf.constant(1.0), tf.constant(2.0)])
    )
    metric_value_ranges = collections.OrderedDict(
        accuracy=[
            # Invalid specification
            (0.0, 1.0, 2.0),
            None,
        ]
    )
    secure_sum_factory = aggregation_factory.SecureSumFactory(
        metric_value_ranges
    )
    with self.assertRaisesRegex(ValueError, 'must be defined as a 2-tuple'):
      secure_sum_factory.create(local_unfinalized_metrics_type)


class CreateDefaultSecureSumQuantizationRangesTest(
    parameterized.TestCase, tf.test.TestCase
):

  # The auto-tuned bound of float values is a `tff.templates.EstimationProcess`,
  # simply check two bounds have the same type.
  def assertAutoTunedBoundEqual(self, a, b, msg=None):
    if isinstance(a, estimation_process.EstimationProcess):
      return self.assertIsInstance(b, estimation_process.EstimationProcess, msg)

  @parameterized.named_parameters(
      ('float32', TensorType(tf.float32, [3]), _DEFAULT_AUTO_TUNED_FLOAT_RANGE),
      ('float64', TensorType(tf.float64, [1]), _DEFAULT_AUTO_TUNED_FLOAT_RANGE),
      ('int32', TensorType(tf.int32, [1]), _DEFAULT_INT_RANGE),
      ('int64', TensorType(tf.int64, [3]), _DEFAULT_INT_RANGE),
      (
          '<int64,float32>',
          computation_types.to_type([tf.int64, tf.float32]),
          [_DEFAULT_INT_RANGE, _DEFAULT_AUTO_TUNED_FLOAT_RANGE],
      ),
      (
          '<a=int64,b=<c=float32,d=[int32,int32]>>',
          computation_types.to_type(
              collections.OrderedDict(
                  a=tf.int64,
                  b=collections.OrderedDict(
                      c=tf.float32, d=[tf.int32, tf.int32]
                  ),
              )
          ),
          collections.OrderedDict(
              a=_DEFAULT_INT_RANGE,
              b=collections.OrderedDict(
                  c=_DEFAULT_AUTO_TUNED_FLOAT_RANGE,
                  d=[_DEFAULT_INT_RANGE, _DEFAULT_INT_RANGE],
              ),
          ),
      ),
  )
  def test_default_auto_tuned_range_construction(
      self, type_spec, expected_range
  ):
    self.addTypeEqualityFunc(
        estimation_process.EstimationProcess, self.assertAutoTunedBoundEqual
    )
    tf.nest.map_structure(
        self.assertEqual,
        aggregation_factory.create_default_secure_sum_quantization_ranges(
            type_spec
        ),
        expected_range,
    )

  @parameterized.named_parameters(
      ('float32', TensorType(tf.float32, [3]), _DEFAULT_FIXED_FLOAT_RANGE),
      ('float64', TensorType(tf.float64, [1]), _DEFAULT_FIXED_FLOAT_RANGE),
      ('int32', TensorType(tf.int32, [1]), _DEFAULT_INT_RANGE),
      ('int64', TensorType(tf.int64, [3]), _DEFAULT_INT_RANGE),
      (
          '<int64,float32>',
          computation_types.to_type([tf.int64, tf.float32]),
          [_DEFAULT_INT_RANGE, _DEFAULT_FIXED_FLOAT_RANGE],
      ),
      (
          '<a=int64,b=<c=float32,d=[int32,int32]>>',
          computation_types.to_type(
              collections.OrderedDict(
                  a=tf.int64,
                  b=collections.OrderedDict(
                      c=tf.float32, d=[tf.int32, tf.int32]
                  ),
              )
          ),
          collections.OrderedDict(
              a=_DEFAULT_INT_RANGE,
              b=collections.OrderedDict(
                  c=_DEFAULT_FIXED_FLOAT_RANGE,
                  d=[_DEFAULT_INT_RANGE, _DEFAULT_INT_RANGE],
              ),
          ),
      ),
  )
  def test_default_fixed_range_construction(self, type_spec, expected_range):
    self.assertAllEqual(
        aggregation_factory.create_default_secure_sum_quantization_ranges(
            type_spec, use_auto_tuned_bounds_for_float_values=False
        ),
        expected_range,
    )

  @parameterized.named_parameters(
      (
          'float32_float_range',
          TensorType(tf.float32, [3]),
          0.1,
          0.5,
          _DEFAULT_AUTO_TUNED_FLOAT_RANGE,
      ),
      (
          'float32_int_range',
          TensorType(tf.float32, [3]),
          1,
          5,
          _DEFAULT_AUTO_TUNED_FLOAT_RANGE,
      ),
      ('int32_int_range', TensorType(tf.int32, [1]), 1, 5, (1, 5)),
      ('int32_float_range', TensorType(tf.int32, [1]), 1.0, 5.0, (1, 5)),
      (
          'int32_float_range_truncated',
          TensorType(tf.int32, [1]),
          1.5,
          5.5,
          (2, 5),
      ),
      (
          '<int64,float32>',
          computation_types.to_type([tf.int64, tf.float32]),
          1,
          5,
          [(1, 5), _DEFAULT_AUTO_TUNED_FLOAT_RANGE],
      ),
      (
          '<a=int64,b=<c=float32,d=[int32,int32]>>',
          computation_types.to_type(
              collections.OrderedDict(
                  a=tf.int64,
                  b=collections.OrderedDict(
                      c=tf.float32, d=[tf.int32, tf.int32]
                  ),
              )
          ),
          1,
          5,
          collections.OrderedDict(
              a=(1, 5),
              b=collections.OrderedDict(
                  c=_DEFAULT_AUTO_TUNED_FLOAT_RANGE, d=[(1, 5), (1, 5)]
              ),
          ),
      ),
  )
  def test_user_supplied_range_using_default_auto_tuned_range(
      self, type_spec, lower_bound, upper_bound, expected_range
  ):
    self.addTypeEqualityFunc(
        estimation_process.EstimationProcess, self.assertAutoTunedBoundEqual
    )
    tf.nest.map_structure(
        self.assertEqual,
        aggregation_factory.create_default_secure_sum_quantization_ranges(
            type_spec, lower_bound, upper_bound
        ),
        expected_range,
    )

  @parameterized.named_parameters(
      (
          'float32_float_range',
          TensorType(tf.float32, [3]),
          0.1,
          0.5,
          (0.1, 0.5),
      ),
      ('float32_int_range', TensorType(tf.float32, [3]), 1, 5, (1.0, 5.0)),
      ('int32_int_range', TensorType(tf.int32, [1]), 1, 5, (1, 5)),
      ('int32_float_range', TensorType(tf.int32, [1]), 1.0, 5.0, (1, 5)),
      (
          'int32_float_range_truncated',
          TensorType(tf.int32, [1]),
          1.5,
          5.5,
          (2, 5),
      ),
      (
          '<int64,float32>',
          computation_types.to_type([tf.int64, tf.float32]),
          1,
          5,
          [(1, 5), (1.0, 5.0)],
      ),
      (
          '<a=int64,b=<c=float32,d=[int32,int32]>>',
          computation_types.to_type(
              collections.OrderedDict(
                  a=tf.int64,
                  b=collections.OrderedDict(
                      c=tf.float32, d=[tf.int32, tf.int32]
                  ),
              )
          ),
          1,
          5,
          collections.OrderedDict(
              a=(1, 5),
              b=collections.OrderedDict(c=(1.0, 5.0), d=[(1, 5), (1, 5)]),
          ),
      ),
  )
  def test_user_supplied_range_using_default_fixed_range(
      self, type_spec, lower_bound, upper_bound, expected_range
  ):
    self.assertAllEqual(
        aggregation_factory.create_default_secure_sum_quantization_ranges(
            type_spec,
            lower_bound,
            upper_bound,
            use_auto_tuned_bounds_for_float_values=False,
        ),
        expected_range,
    )

  def test_invalid_dtype(self):
    with self.assertRaises(aggregation_factory.UnquantizableDTypeError):
      aggregation_factory.create_default_secure_sum_quantization_ranges(
          TensorType(tf.string)
      )

  def test_too_narrow_integer_range(self):
    with self.assertRaisesRegex(ValueError, 'not wide enough'):
      aggregation_factory.create_default_secure_sum_quantization_ranges(
          TensorType(tf.int32), lower_bound=0.7, upper_bound=1.3
      )

  def test_range_reversed(self):
    with self.assertRaisesRegex(ValueError, 'must be greater than'):
      aggregation_factory.create_default_secure_sum_quantization_ranges(
          TensorType(tf.int32), lower_bound=10, upper_bound=5
      )
    with self.assertRaisesRegex(ValueError, 'must be greater than'):
      aggregation_factory.create_default_secure_sum_quantization_ranges(
          TensorType(tf.int32), lower_bound=10.0, upper_bound=5.0
      )


class FillMissingMetricValueRangesTest(
    parameterized.TestCase, tf.test.TestCase
):

  @parameterized.named_parameters(
      (
          'none_user_ranges',
          None,
          collections.OrderedDict(
              num_example=MetricRange(0, 100),
              loss=[MetricRange(0.0, 100.0), MetricRange(0.0, 100.0)],
          ),
      ),
      (
          'partial_user_ranges',
          collections.OrderedDict(loss=[None, (0.0, 400.0)]),
          collections.OrderedDict(
              num_example=MetricRange(0, 100),
              loss=[MetricRange(0.0, 100.0), MetricRange(0.0, 400.0)],
          ),
      ),
      (
          'full_tuple_user_ranges',
          collections.OrderedDict(
              num_example=(10, 200), loss=[(-100.0, 300.0), (0.0, 400.0)]
          ),
          collections.OrderedDict(
              num_example=MetricRange(10, 200),
              loss=[MetricRange(-100.0, 300.0), MetricRange(0.0, 400.0)],
          ),
      ),
      (
          'full_metric_range_user_ranges',
          collections.OrderedDict(
              num_example=MetricRange(10, 200),
              loss=[MetricRange(-100.0, 300.0), MetricRange(0.0, 400.0)],
          ),
          collections.OrderedDict(
              num_example=MetricRange(10, 200),
              loss=[MetricRange(-100.0, 300.0), MetricRange(0.0, 400.0)],
          ),
      ),
  )
  def test_fill_user_ranges_returns_correct_results(
      self, user_ranges, expected_filled_ranges
  ):
    default_ranges = collections.OrderedDict(
        num_example=(0, 100), loss=[(0.0, 100.0), (0.0, 100.0)]
    )
    filled_ranges = aggregation_factory.fill_missing_values_with_defaults(
        default_ranges, user_ranges
    )
    tf.nest.map_structure(
        self.assertAllEqual, filled_ranges, expected_filled_ranges
    )

  @parameterized.named_parameters(
      (
          'range_as_list',
          collections.OrderedDict(
              num_example=[10, 200], loss=[None, [0.0, 400.0]]
          ),
          'range',
      ),
      (
          'invalid_bound_type',
          collections.OrderedDict(num_example=('lower', 'upper')),
          'lower bound',
      ),
      (
          'bounds_not_match',
          collections.OrderedDict(num_example=(1.0, 100)),
          'same type',
      ),
  )
  def test_invalid_user_ranges_type_raises(self, user_ranges, expected_regex):
    default_ranges = collections.OrderedDict(
        num_example=(0, 100), loss=[(0.0, 100.0), (0.0, 100.0)]
    )
    with self.assertRaisesRegex(TypeError, expected_regex):
      aggregation_factory.fill_missing_values_with_defaults(
          default_ranges, user_ranges
      )

  @parameterized.named_parameters(
      ('1_tuple', collections.OrderedDict(num_example=(10,))),
      ('3_tuple', collections.OrderedDict(num_example=(10, 50, 100))),
  )
  def test_invalid_user_ranges_value_raises(self, user_ranges):
    default_ranges = collections.OrderedDict(
        num_example=(0, 100), loss=[(0.0, 100.0), (0.0, 100.0)]
    )
    with self.assertRaisesRegex(ValueError, '2-tuple'):
      aggregation_factory.fill_missing_values_with_defaults(
          default_ranges, user_ranges
      )


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
