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
from typing import Any

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning.metrics import aggregation_factory
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import keras_finalizer

_UNUSED_METRICS_FINALIZERS = collections.OrderedDict(
    accuracy=tf.function(func=lambda x: x[0] / x[1])
)
_UNUSED_UNFINALIZED_METRICS = collections.OrderedDict(
    accuracy=[tf.constant(1.0), tf.constant(2.0)]
)


class CustomSumMetric(tf.keras.metrics.Sum):
  """A custom metric whose result is total + extra scalar and vector values."""

  def __init__(self, name='custom_sum_metric'):
    super().__init__(name=name, dtype=tf.int32)
    self.scalar = self.add_weight(
        name='scalar', shape=(), initializer='zeros', dtype=tf.int32
    )
    self.vector = self.add_weight(
        name='vector', shape=(2,), initializer='zeros', dtype=tf.int32
    )

  # The method `update_state` is omitted here because only the `result` method
  # is useful in the tests below.
  def result(self):
    return self.total + self.scalar + tf.reduce_sum(self.vector)


_TEST_ARGUMENTS_KERAS_METRICS = {
    'testcase_name': 'keras_metrics',
    # Besides standard and custom Keras metrics, this test also covers the cases
    # when the tensors in the unfinalized metrics have different numbers and
    # different shapes.
    'metric_finalizers': collections.OrderedDict(
        accuracy=keras_finalizer.create_keras_metric_finalizer(
            tf.keras.metrics.SparseCategoricalAccuracy
        ),
        custom_sum=keras_finalizer.create_keras_metric_finalizer(
            CustomSumMetric
        ),
    ),
    'local_unfinalized_metrics_at_clients': [
        collections.OrderedDict(
            # The unfinalized `accuracy` has two values: `total` and `count`.
            accuracy=[tf.constant(1.0), tf.constant(2.0)],
            # The unfinalized `custom_sum` has three values: `total`, `scalar`,
            # and `vector`.
            custom_sum=[tf.constant(1), tf.constant(1), tf.constant([1, 1])],
        ),
        collections.OrderedDict(
            accuracy=[tf.constant(3.0), tf.constant(6.0)],
            custom_sum=[tf.constant(1), tf.constant(1), tf.constant([1, 1])],
        ),
    ],
    # The finalized metrics are computed by first summing the unfinalized values
    # from clients, and run the corresponding finalizers (a division for
    # `accuracy`, and a sum for `custom_sum`) at the server.
    'expected_aggregated_metrics': collections.OrderedDict(
        accuracy=(1.0 + 3.0) / (2.0 + 6.0), custom_sum=8
    ),
}


def _test_finalize_metrics(
    unfinalized_metrics: collections.OrderedDict[str, Any]
) -> collections.OrderedDict[str, Any]:
  return collections.OrderedDict(
      accuracy=keras_finalizer.create_keras_metric_finalizer(
          tf.keras.metrics.SparseCategoricalAccuracy
      )(unfinalized_metrics['accuracy']),
      custom_sum=keras_finalizer.create_keras_metric_finalizer(CustomSumMetric)(
          unfinalized_metrics['custom_sum']
      ),
  )


_TEST_CALLABLE_ARGUMENTS_KERAS_METRICS = {
    'testcase_name': 'keras_metrics_callable_finalizers',
    'metric_finalizers': _test_finalize_metrics,
    'local_unfinalized_metrics_at_clients': [
        collections.OrderedDict(
            # The unfinalized `accuracy` has two values: `total` and `count`.
            accuracy=[tf.constant(1.0), tf.constant(2.0)],
            # The unfinalized `custom_sum` has three values: `total`, `scalar`,
            # and `vector`.
            custom_sum=[tf.constant(1), tf.constant(1), tf.constant([1, 1])],
        ),
        collections.OrderedDict(
            accuracy=[tf.constant(3.0), tf.constant(6.0)],
            custom_sum=[tf.constant(1), tf.constant(1), tf.constant([1, 1])],
        ),
    ],
    # The finalized metrics are computed by first summing the unfinalized values
    # from clients, and run the corresponding finalizers (a division for
    # `accuracy`, and a sum for `custom_sum`) at the server.
    'expected_aggregated_metrics': collections.OrderedDict(
        accuracy=(1.0 + 3.0) / (2.0 + 6.0), custom_sum=8
    ),
}

_TEST_ARGUMENTS_NON_KERAS_METRICS = {
    'testcase_name': 'non_keras_metrics',
    # Besides two type of finalizer functions (i.e., divide and sum), this test
    # also covers two ways of representing the unfinalized metrics: as a list or
    # as an OrderedDict.
    'metric_finalizers': collections.OrderedDict(
        divide=tf.function(func=lambda x: x[0] / x[1]),
        sum=tf.function(func=lambda x: x['count_1'] + x['count_2']),
    ),
    'local_unfinalized_metrics_at_clients': [
        collections.OrderedDict(
            divide=[tf.constant(1.0), tf.constant(2.0)],
            sum=collections.OrderedDict(count_1=1, count_2=1),
        ),
        collections.OrderedDict(
            divide=[tf.constant(3.0), tf.constant(6.0)],
            sum=collections.OrderedDict(count_1=1, count_2=1),
        ),
    ],
    'expected_aggregated_metrics': collections.OrderedDict(
        divide=(1.0 + 3.0) / (2.0 + 6.0), sum=4
    ),
}

_TEST_METRICS_MIXED_DTYPES = {
    'testcase_name': 'metrics_mixed_dtypes',
    # This is similar to the `non_keras_metrics` above, except that each metric
    # contains values with two different dtyes (`tf.float32` and `tf.int32`).
    'metric_finalizers': collections.OrderedDict(
        divide=tf.function(func=lambda x: x[0] / tf.cast(x[1], tf.float32)),
        sum=tf.function(
            func=lambda x: tf.cast(x['count_1'], tf.float32) + x['count_2']  # pylint:disable=g-long-lambda
        ),
    ),
    'local_unfinalized_metrics_at_clients': [
        collections.OrderedDict(
            divide=[tf.constant(1.0), tf.constant(2)],
            sum=collections.OrderedDict(count_1=1, count_2=1.0),
        ),
        collections.OrderedDict(
            divide=[tf.constant(3.0), tf.constant(6)],
            sum=collections.OrderedDict(count_1=1, count_2=1.0),
        ),
    ],
    'expected_aggregated_metrics': collections.OrderedDict(
        divide=(1.0 + 3.0) / (2.0 + 6.0), sum=4.0
    ),
}

_TEST_ARGUMENTS_INVALID_INPUTS = [
    {
        'testcase_name': 'metric_finalizers_not_ordereddict',
        'metric_finalizers': {
            'accuracy': tf.function(func=lambda x: x[0] / x[1])
        },
        'local_unfinalized_metrics': _UNUSED_UNFINALIZED_METRICS,
        'error_type': TypeError,
        'error_message': 'Expected .*collections.OrderedDict',
    },
    {
        'testcase_name': 'metric_finalizers_key_not_string',
        'metric_finalizers': collections.OrderedDict(
            [(1, tf.function(func=lambda x: x))]
        ),
        'local_unfinalized_metrics': _UNUSED_UNFINALIZED_METRICS,
        'error_type': TypeError,
        'error_message': 'Expected .*str',
    },
    {
        'testcase_name': 'metric_finalizers_value_not_callable',
        'metric_finalizers': collections.OrderedDict(accuracy=tf.constant(1.0)),
        'local_unfinalized_metrics': _UNUSED_UNFINALIZED_METRICS,
        'error_type': TypeError,
        'error_message': 'Expected .*callable',
    },
    {
        'testcase_name': 'unfinalized_metrics_not_structure_type',
        'metric_finalizers': _UNUSED_METRICS_FINALIZERS,
        'local_unfinalized_metrics': tf.constant(1.0),
        'error_type': TypeError,
        'error_message': 'Expected .*`tff.types.StructWithPythonType`',
    },
    {
        'testcase_name': 'unfinalized_metrics_not_ordereddict',
        'metric_finalizers': _UNUSED_METRICS_FINALIZERS,
        'local_unfinalized_metrics': [tf.constant(1.0), tf.constant(2.0)],
        'error_type': TypeError,
        'error_message': (
            'with `collections.OrderedDict` as the Python container'
        ),
    },
    {
        'testcase_name': 'unmatched_metric_names',
        'metric_finalizers': collections.OrderedDict(
            accuracy=tf.function(func=lambda x: x[0] / x[1])
        ),
        'local_unfinalized_metrics': collections.OrderedDict(
            loss=[tf.constant(1.0), tf.constant(2.0)]
        ),
        'error_type': ValueError,
        'error_message': 'The metric names in `metric_finalizers` do not match',
    },
]


class SumThenFinalizeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      _TEST_ARGUMENTS_KERAS_METRICS,
      _TEST_ARGUMENTS_NON_KERAS_METRICS,
      _TEST_CALLABLE_ARGUMENTS_KERAS_METRICS,
  )
  def test_returns_correct_results(
      self,
      metric_finalizers,
      local_unfinalized_metrics_at_clients,
      expected_aggregated_metrics,
  ):
    aggregator_computation = aggregator.sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]
        ),
    )
    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients
    )
    self.assertAllEqual(aggregated_metrics, expected_aggregated_metrics)

  @parameterized.named_parameters(_TEST_ARGUMENTS_INVALID_INPUTS)
  def test_fails_with_invalid_inputs(
      self,
      metric_finalizers,
      local_unfinalized_metrics,
      error_type,
      error_message,
  ):
    with self.assertRaisesRegex(error_type, error_message):
      aggregator.sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics
          ),
      )


class SecureSumThenFinalizeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      _TEST_ARGUMENTS_KERAS_METRICS,
      _TEST_ARGUMENTS_NON_KERAS_METRICS,
      _TEST_CALLABLE_ARGUMENTS_KERAS_METRICS,
      _TEST_METRICS_MIXED_DTYPES,
  )
  def test_default_value_ranges_returns_correct_results(
      self,
      metric_finalizers,
      local_unfinalized_metrics_at_clients,
      expected_aggregated_metrics,
  ):
    aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]
        ),
    )
    try:
      static_assert.assert_not_contains_unsecure_aggregation(
          aggregator_computation
      )
    except:  # pylint: disable=bare-except
      self.fail('Metric aggregation contains non-secure summation aggregation')

    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients
    )

    no_clipped_values = collections.OrderedDict(
        secure_upper_clipped_count=0,
        secure_lower_clipped_count=0,
        secure_upper_threshold=aggregator.DEFAULT_SECURE_UPPER_BOUND,
        secure_lower_threshold=aggregator.DEFAULT_SECURE_LOWER_BOUND,
    )

    factory_keys = collections.OrderedDict()
    for value in tf.nest.flatten(local_unfinalized_metrics_at_clients[0]):
      tensor = tf.constant(value)
      if tensor.dtype.is_floating:
        lower = float(aggregator.DEFAULT_SECURE_LOWER_BOUND)
        upper = float(aggregator.DEFAULT_SECURE_UPPER_BOUND)
      elif tensor.dtype.is_integer:
        lower = int(aggregator.DEFAULT_SECURE_LOWER_BOUND)
        upper = int(aggregator.DEFAULT_SECURE_UPPER_BOUND)
      else:
        raise TypeError(
            f'Expected float or int, found tensors of dtype {tensor.dtype}.'
        )
      factory_key = aggregation_factory.create_factory_key(
          lower, upper, tensor.dtype
      )
      factory_keys[factory_key] = 1

    expected_measurements = collections.OrderedDict(
        (factory_key, no_clipped_values) for factory_key in factory_keys
    )
    secure_sum_measurements = aggregated_metrics.pop('secure_sum_measurements')
    self.assertAllClose(secure_sum_measurements, expected_measurements)
    self.assertAllClose(
        aggregated_metrics, expected_aggregated_metrics, rtol=1e-5, atol=1e-5
    )

  def _clipped_values(
      self, clipped_count, threshold=aggregator.DEFAULT_SECURE_UPPER_BOUND
  ):
    return collections.OrderedDict(
        secure_upper_clipped_count=clipped_count,
        secure_lower_clipped_count=0,
        secure_upper_threshold=threshold,
        secure_lower_threshold=0.0,
    )

  def test_user_value_ranges_returns_correct_results(self):
    metric_finalizers = collections.OrderedDict(
        accuracy=keras_finalizer.create_keras_metric_finalizer(
            tf.keras.metrics.SparseCategoricalAccuracy
        ),
        custom_sum=keras_finalizer.create_keras_metric_finalizer(
            CustomSumMetric
        ),
    )
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(
            # The unfinalized `accuracy` has two values: `total` and `count`.
            accuracy=[tf.constant(1.0), tf.constant(2.0)],
            # The unfinalized `custom_sum` has three values: `total`, `scalar`,
            # and `vector`.
            custom_sum=[tf.constant(1), tf.constant(1), tf.constant([1, 1])],
        ),
        collections.OrderedDict(
            accuracy=[tf.constant(3.0), tf.constant(6.0)],
            custom_sum=[tf.constant(1), tf.constant(1), tf.constant([1, 1])],
        ),
    ]
    aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]
        ),
        # Note: Partial specification, only the `accuracy` metrics denominator
        # variable has a different range; all others get the default.
        metric_value_ranges=collections.OrderedDict(
            accuracy=[
                None,
                (0.0, 1.0),
            ]
        ),
    )
    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients
    )

    expected_secure_sum_measurements = collections.OrderedDict()
    # The metric values are grouped into three `factory_key`s. The first group
    # only has `accoracy/0`.
    factory_key = aggregation_factory.create_factory_key(
        0.0, float(aggregator.DEFAULT_SECURE_UPPER_BOUND), tf.float32
    )
    expected_secure_sum_measurements[factory_key] = self._clipped_values(0)
    # The second `factory_key` only has `accuracy/1`. Both clients get clipped.
    factory_key = aggregation_factory.create_factory_key(0.0, 1.0, tf.float32)
    expected_secure_sum_measurements[factory_key] = self._clipped_values(2, 1.0)
    # The third `factory_key` covers 3 values in `custom_sum`.
    factory_key = aggregation_factory.create_factory_key(
        0, int(aggregator.DEFAULT_SECURE_UPPER_BOUND), tf.int32
    )
    expected_secure_sum_measurements[factory_key] = self._clipped_values(0)
    secure_sum_measurements = aggregated_metrics.pop('secure_sum_measurements')
    self.assertAllEqual(
        secure_sum_measurements, expected_secure_sum_measurements
    )

    expected_aggregated_metrics = collections.OrderedDict(
        accuracy=(1.0 + 3.0) /
        # The accuracy denominator is clipped to the range [0.0, 1.0]
        (1.0 + 1.0),
        custom_sum=8.0,
    )
    self.assertAllClose(
        aggregated_metrics, expected_aggregated_metrics, rtol=1e-5, atol=1e-5
    )

  def test_user_value_ranges_mixed_dtypes_returns_correct_results(self):
    metric_finalizers = collections.OrderedDict(
        divide=tf.function(func=lambda x: x[0] / tf.cast(x[1], tf.float32)),
        sum=tf.function(
            func=lambda x: tf.cast(x['count_1'], tf.float32) + x['count_2']  # pylint:disable=g-long-lambda
        ),
    )
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(
            divide=[tf.constant(1.0), tf.constant(2)],
            sum=collections.OrderedDict(count_1=2, count_2=1.0),
        ),
        collections.OrderedDict(
            divide=[tf.constant(3.0), tf.constant(1)],
            sum=collections.OrderedDict(count_1=3, count_2=3.0),
        ),
    ]
    aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]
        ),
        # Note: Partial specification, `divide/1` and `sum/count_1` gets the
        # same range (0, 1); `sum/count_2` gets range (0.0, 2.0); and `divide/0`
        # gets the default range.
        metric_value_ranges=collections.OrderedDict(
            divide=[
                None,
                (0, 1),
            ],
            sum=collections.OrderedDict(count_1=(0, 1), count_2=(0.0, 2.0)),
        ),
    )
    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients
    )

    expected_secure_sum_measurements = collections.OrderedDict()
    # The metric values are grouped into three `factory_key`s. The first group
    # only has `divide/0`.
    factory_key = aggregation_factory.create_factory_key(
        0.0, float(aggregator.DEFAULT_SECURE_UPPER_BOUND), tf.float32
    )
    expected_secure_sum_measurements[factory_key] = self._clipped_values(0)
    # The second `factory_key` has `divide/1` and `sum/count_1`. For the first
    # client, both `divide/1` and `sum/count_1` get clipped; for the second
    # client, `sum/count_1` gets clipped. As a result, the number of clipped
    # clients for this group is 2.
    factory_key = aggregation_factory.create_factory_key(0, 1, tf.int32)
    expected_secure_sum_measurements[factory_key] = self._clipped_values(2, 1)
    # The third `factory_key` has `sum/count_2`. One client gets clipped.
    factory_key = aggregation_factory.create_factory_key(0.0, 2.0, tf.float32)
    expected_secure_sum_measurements[factory_key] = self._clipped_values(1, 2.0)
    secure_sum_measurements = aggregated_metrics.pop('secure_sum_measurements')
    self.assertAllEqual(
        secure_sum_measurements, expected_secure_sum_measurements
    )

    expected_aggregated_metrics = collections.OrderedDict(
        divide=(1.0 + 3.0) /
        # `divide/1` is clipped to the range [0, 1]
        (1.0 + 1.0),
        # `sum/count_1` is clipped to the range [0, 1],
        # `sum/count_2` is clipped to the range [0.0, 2.0].
        sum=(1.0 + 1.0 + 1.0 + 2.0),
    )
    self.assertAllClose(
        aggregated_metrics, expected_aggregated_metrics, rtol=1e-5, atol=1e-5
    )

  def test_user_value_ranges_fails_invalid_dtype(self):
    class TestConcatMetric(tf.keras.metrics.Metric):
      """A custom metric that concatenates strings."""

      def __init__(self, name='custom_concat_metric'):
        super().__init__(name=name, dtype=tf.string)
        self._value = self.add_weight(
            name='value', shape=(), initializer='zeros', dtype=tf.string
        )

      def update_state(self, value):
        self._value.assign(tf.concat([self._value, value], axis=0))

      def result(self):
        return self._value.read_value()

    metric_finalizers = collections.OrderedDict(
        custom_sum=keras_finalizer.create_keras_metric_finalizer(
            TestConcatMetric
        )
    )
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(custom_sum=[tf.constant('abc')])
    ]
    with self.assertRaises(aggregation_factory.UnquantizableDTypeError):
      aggregator.secure_sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics_at_clients[0]
          ),
      )

  def test_user_value_ranges_fails_not_2_tuple(self):
    metric_finalizers = collections.OrderedDict(
        accuracy=keras_finalizer.create_keras_metric_finalizer(
            tf.keras.metrics.SparseCategoricalAccuracy
        )
    )
    local_unfinalized_metrics_at_clients = [
        collections.OrderedDict(accuracy=[tf.constant(1.0), tf.constant(2.0)])
    ]
    with self.assertRaisesRegex(ValueError, 'must be defined as a 2-tuple'):
      aggregator.secure_sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics_at_clients[0]
          ),
          metric_value_ranges=collections.OrderedDict(
              accuracy=[
                  # Invalid specification
                  (0.0, 1.0, 2.0),
                  None,
              ]
          ),
      )

  @parameterized.named_parameters(_TEST_ARGUMENTS_INVALID_INPUTS)
  def test_fails_with_invalid_inputs(
      self,
      metric_finalizers,
      local_unfinalized_metrics,
      error_type,
      error_message,
  ):
    with self.assertRaisesRegex(error_type, error_message):
      aggregator.secure_sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics
          ),
      )


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
