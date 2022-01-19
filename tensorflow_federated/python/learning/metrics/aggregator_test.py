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
"""Tests for aggregator."""

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import finalizer


# Convenience aliases.
TensorType = computation_types.TensorType

_UNUSED_METRICS_FINALIZERS = collections.OrderedDict(
    accuracy=tf.function(func=lambda x: x[0] / x[1]))
_UNUSED_UNFINALIZED_METRICS = collections.OrderedDict(
    accuracy=[tf.constant(1.0), tf.constant(2.0)])


class CustomSumMetric(tf.keras.metrics.Sum):
  """A custom metric whose result is total + extra scalar and vector values."""

  def __init__(self, name='custom_sum_metric'):
    super().__init__(name=name, dtype=tf.int32)
    self.scalar = self.add_weight(
        name='scalar', shape=(), initializer='zeros', dtype=tf.int32)
    self.vector = self.add_weight(
        name='vector', shape=(2,), initializer='zeros', dtype=tf.int32)

  # The method `update_state` is omitted here because only the `result` method
  # is useful in the tests below.
  def result(self):
    return self.total + self.scalar + tf.reduce_sum(self.vector)


_TEST_ARGUMENTS_KERAS_METRICS = {
    'testcase_name':
        'keras_metrics',
    # Besides standard and custom Keras metrics, this test also covers the cases
    # when the tensors in the unfinalized metrics have different numbers and
    # different shapes.
    'metric_finalizers':
        collections.OrderedDict(
            accuracy=finalizer.create_keras_metric_finalizer(
                tf.keras.metrics.SparseCategoricalAccuracy),
            custom_sum=finalizer.create_keras_metric_finalizer(CustomSumMetric)
        ),
    'local_unfinalized_metrics_at_clients': [
        collections.OrderedDict(
            # The unfinalized `accuracy` has two values: `total` and `count`.
            accuracy=[tf.constant(1.0), tf.constant(2.0)],
            # The unfinalized `custom_sum` has three values: `total`, `scalar`,
            # and `vector`.
            custom_sum=[tf.constant(1),
                        tf.constant(1),
                        tf.constant([1, 1])]),
        collections.OrderedDict(
            accuracy=[tf.constant(3.0), tf.constant(6.0)],
            custom_sum=[tf.constant(1),
                        tf.constant(1),
                        tf.constant([1, 1])])
    ],
    # The finalized metrics are computed by first summing the unfinalized values
    # from clients, and run the corresponding finalizers (a division for
    # `accuracy`, and a sum for `custom_sum`) at the server.
    'expected_aggregated_metrics':
        collections.OrderedDict(
            accuracy=(1.0 + 3.0) / (2.0 + 6.0), custom_sum=8)
}

_TEST_ARGUMENTS_NON_KERAS_METRICS = {
    'testcase_name':
        'non_keras_metrics',
    # Besides two type of finalizer functions (i.e., divide and sum), this test
    # also covers two ways of representing the unfinalized metrics: as a list or
    # as an OrderedDict.
    'metric_finalizers':
        collections.OrderedDict(
            divide=tf.function(func=lambda x: x[0] / x[1]),
            sum=tf.function(func=lambda x: x['count_1'] + x['count_2'])),
    'local_unfinalized_metrics_at_clients': [
        collections.OrderedDict(
            divide=[tf.constant(1.0), tf.constant(2.0)],
            sum=collections.OrderedDict(count_1=1, count_2=1)),
        collections.OrderedDict(
            divide=[tf.constant(3.0), tf.constant(6.0)],
            sum=collections.OrderedDict(count_1=1, count_2=1))
    ],
    'expected_aggregated_metrics':
        collections.OrderedDict(divide=(1.0 + 3.0) / (2.0 + 6.0), sum=4)
}

_TEST_ARGUMENTS_INVALID_INPUTS = [{
    'testcase_name': 'metric_finalizers_not_ordereddict',
    'metric_finalizers': {
        'accuracy': tf.function(func=lambda x: x[0] / x[1])
    },
    'local_unfinalized_metrics': _UNUSED_UNFINALIZED_METRICS,
    'error_type': TypeError,
    'error_message': 'Expected .*collections.OrderedDict'
}, {
    'testcase_name':
        'metric_finalizers_key_not_string',
    'metric_finalizers':
        collections.OrderedDict([(1, tf.function(func=lambda x: x))]),
    'local_unfinalized_metrics':
        _UNUSED_UNFINALIZED_METRICS,
    'error_type':
        TypeError,
    'error_message':
        'Expected .*str'
}, {
    'testcase_name': 'metric_finalizers_value_not_callable',
    'metric_finalizers': collections.OrderedDict(accuracy=tf.constant(1.0)),
    'local_unfinalized_metrics': _UNUSED_UNFINALIZED_METRICS,
    'error_type': TypeError,
    'error_message': 'Expected .*callable'
}, {
    'testcase_name': 'unfinalized_metrics_not_structure_type',
    'metric_finalizers': _UNUSED_METRICS_FINALIZERS,
    'local_unfinalized_metrics': tf.constant(1.0),
    'error_type': TypeError,
    'error_message': 'Expected .*`tff.types.StructWithPythonType`'
}, {
    'testcase_name': 'unfinalized_metrics_not_ordereddict',
    'metric_finalizers': _UNUSED_METRICS_FINALIZERS,
    'local_unfinalized_metrics': [tf.constant(1.0),
                                  tf.constant(2.0)],
    'error_type': TypeError,
    'error_message': 'with `collections.OrderedDict` as the Python container'
}, {
    'testcase_name':
        'unmatched_metric_names',
    'metric_finalizers':
        collections.OrderedDict(
            accuracy=tf.function(func=lambda x: x[0] / x[1])),
    'local_unfinalized_metrics':
        collections.OrderedDict(
            loss=[tf.constant(1.0), tf.constant(2.0)]),
    'error_type':
        ValueError,
    'error_message':
        'The metric names in `metric_finalizers` do not match'
}]


class SumThenFinalizeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(_TEST_ARGUMENTS_KERAS_METRICS,
                                  _TEST_ARGUMENTS_NON_KERAS_METRICS)
  def test_returns_correct_results(self, metric_finalizers,
                                   local_unfinalized_metrics_at_clients,
                                   expected_aggregated_metrics):
    aggregator_computation = aggregator.sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]))
    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients)
    self.assertAllEqual(aggregated_metrics, expected_aggregated_metrics)

  @parameterized.named_parameters(_TEST_ARGUMENTS_INVALID_INPUTS)
  def test_fails_with_invalid_inputs(self, metric_finalizers,
                                     local_unfinalized_metrics, error_type,
                                     error_message):
    with self.assertRaisesRegex(error_type, error_message):
      aggregator.sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics))


DEFAULT_FLOAT_RANGE = (float(aggregator.DEFAULT_SECURE_LOWER_BOUND),
                       float(aggregator.DEFAULT_SECURE_UPPER_BOUND))
DEFAULT_INT_RANGE = (float(aggregator.DEFAULT_SECURE_LOWER_BOUND),
                     float(aggregator.DEFAULT_SECURE_UPPER_BOUND))


class CreateDefaultSecureSumQuantizationRangesTest(parameterized.TestCase,
                                                   tf.test.TestCase):

  @parameterized.named_parameters(
      ('float32', TensorType(tf.float32, [3]), DEFAULT_FLOAT_RANGE),
      ('float64', TensorType(tf.float64, [1]), DEFAULT_FLOAT_RANGE),
      ('int32', TensorType(tf.int32, [1]), DEFAULT_INT_RANGE),
      ('int64', TensorType(tf.int64, [3]), DEFAULT_INT_RANGE),
      ('<int64,float32>', computation_types.to_type(
          [tf.int64, tf.float32]), [DEFAULT_INT_RANGE, DEFAULT_FLOAT_RANGE]),
      ('<a=int64,b=<c=float32,d=[int32,int32]>>',
       computation_types.to_type(
           collections.OrderedDict(
               a=tf.int64,
               b=collections.OrderedDict(c=tf.float32, d=[tf.int32, tf.int32
                                                         ]))),
       collections.OrderedDict(
           a=DEFAULT_INT_RANGE,
           b=collections.OrderedDict(
               c=DEFAULT_FLOAT_RANGE, d=[DEFAULT_INT_RANGE, DEFAULT_INT_RANGE
                                        ]))),
  )
  def test_default_construction(self, type_spec, expected_range):
    self.assertAllEqual(
        aggregator.create_default_secure_sum_quantization_ranges(type_spec),
        expected_range)

  @parameterized.named_parameters(
      ('float32_float_range', TensorType(tf.float32, [3]), 0.1, 0.5,
       (0.1, 0.5)),
      ('float32_int_range', TensorType(tf.float32, [3]), 1, 5, (1., 5.)),
      ('int32_int_range', TensorType(tf.int32, [1]), 1, 5, (1, 5)),
      ('int32_float_range', TensorType(tf.int32, [1]), 1., 5., (1, 5)),
      ('int32_float_range_truncated', TensorType(tf.int32, [1]), 1.5, 5.5,
       (2, 5)),
      ('<int64,float32>', computation_types.to_type(
          [tf.int64, tf.float32]), 1, 5, [(1, 5), (1., 5.)]),
      ('<a=int64,b=<c=float32,d=[int32,int32]>>',
       computation_types.to_type(
           collections.OrderedDict(
               a=tf.int64,
               b=collections.OrderedDict(c=tf.float32, d=[tf.int32, tf.int32
                                                         ]))), 1, 5,
       collections.OrderedDict(
           a=(1, 5), b=collections.OrderedDict(c=(1., 5.), d=[(1, 5),
                                                              (1, 5)]))),
  )
  def test_user_supplied_range(self, type_spec, lower_bound, upper_bound,
                               expected_range):
    self.assertAllEqual(
        aggregator.create_default_secure_sum_quantization_ranges(
            type_spec, lower_bound, upper_bound), expected_range)

  def test_invalid_dtype(self):
    with self.assertRaises(aggregator.UnquantizableDTypeError):
      aggregator.create_default_secure_sum_quantization_ranges(
          TensorType(tf.string))

  def test_too_narrow_integer_range(self):
    with self.assertRaisesRegex(ValueError, 'not wide enough'):
      aggregator.create_default_secure_sum_quantization_ranges(
          TensorType(tf.int32), lower_bound=0.7, upper_bound=1.3)

  def test_range_reversed(self):
    with self.assertRaisesRegex(ValueError, 'must be greater than'):
      aggregator.create_default_secure_sum_quantization_ranges(
          TensorType(tf.int32), lower_bound=10, upper_bound=5)
    with self.assertRaisesRegex(ValueError, 'must be greater than'):
      aggregator.create_default_secure_sum_quantization_ranges(
          TensorType(tf.int32), lower_bound=10., upper_bound=5.)


class SecureSumThenFinalizeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(_TEST_ARGUMENTS_KERAS_METRICS,
                                  _TEST_ARGUMENTS_NON_KERAS_METRICS)
  def test_default_value_ranges_returns_correct_results(
      self, metric_finalizers, local_unfinalized_metrics_at_clients,
      expected_aggregated_metrics):
    aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]))
    try:
      static_assert.assert_not_contains_unsecure_aggregation(
          aggregator_computation)
    except:  # pylint: disable=bare-except
      self.fail('Metric aggregation contains non-secure summation aggregation')

    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients)

    def no_clipped_values(*unused_args):
      return collections.OrderedDict(
          secure_upper_clipped_count=0,
          secure_lower_clipped_count=0,
          secure_upper_threshold=aggregator.DEFAULT_SECURE_UPPER_BOUND,
          secure_lower_threshold=aggregator.DEFAULT_SECURE_LOWER_BOUND)

    expected_measurements = tf.nest.map_structure(
        no_clipped_values, local_unfinalized_metrics_at_clients[0])
    secure_sum_measurements = aggregated_metrics.pop('secure_sum_measurements')
    self.assertAllClose(secure_sum_measurements, expected_measurements)
    self.assertAllClose(
        aggregated_metrics, expected_aggregated_metrics, rtol=1e-5, atol=1e-5)

  def test_user_value_ranges_returns_correct_results(self):
    metric_finalizers = collections.OrderedDict(
        accuracy=finalizer.create_keras_metric_finalizer(
            tf.keras.metrics.SparseCategoricalAccuracy),
        custom_sum=finalizer.create_keras_metric_finalizer(CustomSumMetric))
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(
            # The unfinalized `accuracy` has two values: `total` and `count`.
            accuracy=[tf.constant(1.0), tf.constant(2.0)],
            # The unfinalized `custom_sum` has three values: `total`, `scalar`,
            # and `vector`.
            custom_sum=[tf.constant(1),
                        tf.constant(1),
                        tf.constant([1, 1])]),
        collections.OrderedDict(
            accuracy=[tf.constant(3.0), tf.constant(6.0)],
            custom_sum=[tf.constant(1),
                        tf.constant(1),
                        tf.constant([1, 1])])
    ]
    aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]),
        # Note: Partial specification, only the `accuracy` metrics denominator
        # variable has a different range; all others get the default.
        metric_value_ranges=collections.OrderedDict(accuracy=[
            None,
            (0.0, 1.0),
        ]))
    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients)

    def clipped_values(clipped_count,
                       threshold=aggregator.DEFAULT_SECURE_UPPER_BOUND):
      return collections.OrderedDict(
          secure_upper_clipped_count=clipped_count,
          secure_lower_clipped_count=0,
          secure_upper_threshold=threshold,
          secure_lower_threshold=0.0)

    expected_secure_sum_measurements = collections.OrderedDict(
        accuracy=[
            clipped_values(0),
            # Denominator of accuracy is clipped.
            clipped_values(2, threshold=1.0)
        ],
        custom_sum=[clipped_values(0),
                    clipped_values(0),
                    clipped_values(0)])
    secure_sum_measurements = aggregated_metrics.pop('secure_sum_measurements')
    self.assertAllEqual(secure_sum_measurements,
                        expected_secure_sum_measurements)

    expected_aggregated_metrics = collections.OrderedDict(
        accuracy=(1.0 + 3.0) /
        # The accuracy denominator is clipped to the range [0.0, 1.0]
        (1.0 + 1.0),
        custom_sum=8.0)
    self.assertAllClose(
        aggregated_metrics, expected_aggregated_metrics, rtol=1e-5, atol=1e-5)

  def test_user_value_ranges_fails_invalid_dtype(self):

    class TestConcatMetric(tf.keras.metrics.Metric):
      """A custom metric that concatenates strings."""

      def __init__(self, name='custom_concat_metric'):
        super().__init__(name=name, dtype=tf.string)
        self._value = self.add_weight(
            name='value', shape=(), initializer='zeros', dtype=tf.string)

      def update_state(self, value):
        self._value.assign(tf.concat([self._value, value], axis=0))

      def result(self):
        return self._value.read_value()

    metric_finalizers = collections.OrderedDict(
        custom_sum=finalizer.create_keras_metric_finalizer(TestConcatMetric))
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(custom_sum=[tf.constant('abc')])
    ]
    with self.assertRaises(aggregator.UnquantizableDTypeError):
      aggregator.secure_sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics_at_clients[0]))

  def test_user_value_ranges_fails_not_2_tuple(self):
    metric_finalizers = collections.OrderedDict(
        accuracy=finalizer.create_keras_metric_finalizer(
            tf.keras.metrics.SparseCategoricalAccuracy))
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(
            accuracy=[tf.constant(1.0), tf.constant(2.0)])
    ]
    with self.assertRaisesRegex(ValueError, 'must be defined as a 2-tuple'):
      aggregator.secure_sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics_at_clients[0]),
          metric_value_ranges=collections.OrderedDict(accuracy=[
              # Invalid specification
              (0.0, 1.0, 2.0),
              None
          ]))


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  tf.test.main()
