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
import collections
from typing import Any

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import keras_finalizer
from tensorflow_federated.python.learning.metrics import sum_aggregation_factory


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
    'local_unfinalized_metrics_type': computation_types.to_type(
        collections.OrderedDict(
            accuracy=[np.float32, np.float32],
            custom_sum=[
                np.int32,
                np.int32,
                computation_types.TensorType(np.int32, [2]),
            ],
        ),
    ),
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
    'local_unfinalized_metrics_type': computation_types.to_type(
        collections.OrderedDict(
            accuracy=[np.float32, np.float32],
            custom_sum=[
                np.int32,
                np.int32,
                computation_types.TensorType(np.int32, [2]),
            ],
        )
    ),
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
    'local_unfinalized_metrics_type': computation_types.to_type(
        collections.OrderedDict(
            divide=[np.float32, np.float32],
            sum=collections.OrderedDict(count_1=np.int32, count_2=np.int32),
        ),
    ),
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
            func=lambda x: tf.cast(x['count_1'], tf.float32) + x['count_2']
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
    'local_unfinalized_metrics_type': computation_types.to_type(
        collections.OrderedDict(
            divide=[np.float32, np.int32],
            sum=collections.OrderedDict(count_1=np.int32, count_2=np.float32),
        ),
    ),
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
      local_unfinalized_metrics_type,
      expected_aggregated_metrics,
  ):
    aggregator_computation = aggregator.sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=local_unfinalized_metrics_type,
    )

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            local_unfinalized_metrics_type, placements.CLIENTS
        )
    )
    def wrapped_aggregator_computation(unfinalized_metrics):
      return aggregator_computation(unfinalized_metrics)

    aggregated_metrics = wrapped_aggregator_computation(
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
    local_unfinalized_metrics_type = computation_types.tensorflow_to_type(
        tf.nest.map_structure(
            tf.TensorSpec.from_tensor,
            local_unfinalized_metrics,
        )
    )

    with self.assertRaisesRegex(error_type, error_message):
      # Concretize on a federated type with CLIENTS placement so that the method
      # invocation understands to interpret python lists of values as CLIENTS
      # placed values.
      @federated_computation.federated_computation(
          computation_types.FederatedType(
              local_unfinalized_metrics_type,
              placements.CLIENTS,
          )
      )
      def _aggregator_computation(unfinalized_metrics):
        return aggregator.sum_then_finalize(
            metric_finalizers=metric_finalizers,
        )(unfinalized_metrics)


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
      local_unfinalized_metrics_type,
      expected_aggregated_metrics,
  ):
    polymorphic_aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers
    )

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            local_unfinalized_metrics_type, placements.CLIENTS
        )
    )
    def aggregator_computation(unfinalized_metrics):
      return polymorphic_aggregator_computation(unfinalized_metrics)

    static_assert.assert_not_contains_unsecure_aggregation(
        aggregator_computation
    )

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
      factory_key = sum_aggregation_factory.create_factory_key(
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
    polymorphic_aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
        # Note: Partial specification, only the `accuracy` metrics denominator
        # variable has a different range; all others get the default.
        metric_value_ranges=collections.OrderedDict(
            accuracy=[
                None,
                (0.0, 1.0),
            ]
        ),
    )

    # Concretize on a federated type with CLIENTS placement so that the method
    # invocation understands to interpret python lists of values as CLIENTS
    # placed values.
    @federated_computation.federated_computation(
        computation_types.FederatedType(
            collections.OrderedDict(
                accuracy=[
                    computation_types.TensorType(np.float32),
                    computation_types.TensorType(np.float32),
                ],
                custom_sum=[
                    computation_types.TensorType(np.int32),
                    computation_types.TensorType(np.int32),
                    computation_types.TensorType(dtype=np.int32, shape=(2,)),
                ],
            ),
            placements.CLIENTS,
        )
    )
    def aggregator_computation(unfinalized_metrics):
      return polymorphic_aggregator_computation(unfinalized_metrics)

    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients
    )

    expected_secure_sum_measurements = collections.OrderedDict()
    # The metric values are grouped into three `factory_key`s. The first group
    # only has `accoracy/0`.
    factory_key = sum_aggregation_factory.create_factory_key(
        0.0, float(aggregator.DEFAULT_SECURE_UPPER_BOUND), tf.float32
    )
    expected_secure_sum_measurements[factory_key] = self._clipped_values(0)
    # The second `factory_key` only has `accuracy/1`. Both clients get clipped.
    factory_key = sum_aggregation_factory.create_factory_key(
        0.0, 1.0, tf.float32
    )
    expected_secure_sum_measurements[factory_key] = self._clipped_values(2, 1.0)
    # The third `factory_key` covers 3 values in `custom_sum`.
    factory_key = sum_aggregation_factory.create_factory_key(
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
            func=lambda x: tf.cast(x['count_1'], tf.float32) + x['count_2']
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
    polymorphic_aggregator_computation = aggregator.secure_sum_then_finalize(
        metric_finalizers=metric_finalizers,
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

    # Concretize on a federated type with CLIENTS placement so that the method
    # invocation understands to interpret python lists of values as CLIENTS
    # placed values.
    @federated_computation.federated_computation(
        computation_types.FederatedType(
            collections.OrderedDict(
                divide=[
                    computation_types.TensorType(np.float32),
                    computation_types.TensorType(np.int32),
                ],
                sum=collections.OrderedDict(
                    count_1=computation_types.TensorType(np.int32),
                    count_2=computation_types.TensorType(np.float32),
                ),
            ),
            placements.CLIENTS,
        )
    )
    def aggregator_computation(unfinalized_metrics):
      return polymorphic_aggregator_computation(unfinalized_metrics)

    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients
    )

    expected_secure_sum_measurements = collections.OrderedDict()
    # The metric values are grouped into three `factory_key`s. The first group
    # only has `divide/0`.
    factory_key = sum_aggregation_factory.create_factory_key(
        0.0, float(aggregator.DEFAULT_SECURE_UPPER_BOUND), tf.float32
    )
    expected_secure_sum_measurements[factory_key] = self._clipped_values(0)
    # The second `factory_key` has `divide/1` and `sum/count_1`. For the first
    # client, both `divide/1` and `sum/count_1` get clipped; for the second
    # client, `sum/count_1` gets clipped. As a result, the number of clipped
    # clients for this group is 2.
    factory_key = sum_aggregation_factory.create_factory_key(0, 1, tf.int32)
    expected_secure_sum_measurements[factory_key] = self._clipped_values(2, 1)
    # The third `factory_key` has `sum/count_2`. One client gets clipped.
    factory_key = sum_aggregation_factory.create_factory_key(
        0.0, 2.0, tf.float32
    )
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
    with self.assertRaises(sum_aggregation_factory.UnquantizableDTypeError):

      # Concretize on a federated type with CLIENTS placement so that the method
      # invocation understands to interpret python lists of values as CLIENTS
      # placed values.
      @federated_computation.federated_computation(
          computation_types.FederatedType(
              collections.OrderedDict(
                  custom_sum=computation_types.TensorType(tf.string),
              ),
              placements.CLIENTS,
          )
      )
      def _aggregator_computation(unfinalized_metrics):
        return aggregator.secure_sum_then_finalize(
            metric_finalizers=metric_finalizers
        )(unfinalized_metrics)

  def test_user_value_ranges_fails_not_2_tuple(self):
    metric_finalizers = collections.OrderedDict(
        accuracy=keras_finalizer.create_keras_metric_finalizer(
            tf.keras.metrics.SparseCategoricalAccuracy
        )
    )
    with self.assertRaisesRegex(ValueError, 'must be defined as a 2-tuple'):
      # Concretize on a federated type with CLIENTS placement so that the method
      # invocation understands to interpret python lists of values as CLIENTS
      # placed values.
      @federated_computation.federated_computation(
          computation_types.FederatedType(
              collections.OrderedDict(
                  accuracy=[
                      computation_types.TensorType(np.float32),
                      computation_types.TensorType(np.float32),
                  ],
              ),
              placements.CLIENTS,
          )
      )
      def _aggregator_computation(unfinalized_metrics):
        return aggregator.secure_sum_then_finalize(
            metric_finalizers=metric_finalizers,
            metric_value_ranges=collections.OrderedDict(
                accuracy=[
                    # Invalid specification
                    (0.0, 1.0, 2.0),
                    None,
                ]
            ),
        )(unfinalized_metrics)

  @parameterized.named_parameters(_TEST_ARGUMENTS_INVALID_INPUTS)
  def test_fails_with_invalid_inputs(
      self,
      metric_finalizers,
      local_unfinalized_metrics,
      error_type,
      error_message,
  ):
    local_unfinalized_metrics_type = computation_types.tensorflow_to_type(
        tf.nest.map_structure(
            tf.TensorSpec.from_tensor,
            local_unfinalized_metrics,
        )
    )

    with self.assertRaisesRegex(error_type, error_message):
      # Concretize on a federated type with CLIENTS placement so that the method
      # invocation understands to interpret python lists of values as CLIENTS
      # placed values.
      @federated_computation.federated_computation(
          computation_types.FederatedType(
              local_unfinalized_metrics_type,
              placements.CLIENTS,
          )
      )
      def _aggregator_computation(unfinalized_metrics):
        return aggregator.secure_sum_then_finalize(
            metric_finalizers=metric_finalizers
        )(unfinalized_metrics)


class FinalizeThenSampleTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('zero', 0, ValueError, 'must be positive'),
      ('negative', -1, ValueError, 'must be positive'),
      ('none', None, TypeError, 'sample_size'),
      ('string', '5', TypeError, 'sample_size'),
  )
  def test_fails_with_invalid_sample_size(
      self, bad_sample_size, expected_error_type, expected_error_message
  ):
    with self.assertRaisesRegex(expected_error_type, expected_error_message):
      # Concretize on a federated type with CLIENTS placement so that the method
      # invocation understands to interpret python lists of values as CLIENTS
      # placed values.
      @federated_computation.federated_computation(
          computation_types.FederatedType(
              collections.OrderedDict(
                  accuracy=[
                      computation_types.TensorType(np.float32),
                      computation_types.TensorType(np.float32),
                  ]
              ),
              placements.CLIENTS,
          )
      )
      def _aggregator_computation(unfinalized_metrics):
        return aggregator.finalize_then_sample(
            metric_finalizers=_UNUSED_METRICS_FINALIZERS,
            sample_size=bad_sample_size,
        )(unfinalized_metrics)

  @parameterized.named_parameters(_TEST_ARGUMENTS_INVALID_INPUTS)
  def test_fails_with_invalid_inputs(
      self,
      metric_finalizers,
      local_unfinalized_metrics,
      error_type,
      error_message,
  ):
    local_unfinalized_metrics_type = computation_types.tensorflow_to_type(
        tf.nest.map_structure(
            tf.TensorSpec.from_tensor,
            local_unfinalized_metrics,
        )
    )

    with self.assertRaisesRegex(error_type, error_message):
      # Concretize on a federated type with CLIENTS placement so that the method
      # invocation understands to interpret python lists of values as CLIENTS
      # placed values.
      @federated_computation.federated_computation(
          computation_types.FederatedType(
              local_unfinalized_metrics_type,
              placements.CLIENTS,
          )
      )
      def _aggregator_computation(unfinalized_metrics):
        return aggregator.finalize_then_sample(
            metric_finalizers=metric_finalizers
        )(unfinalized_metrics)

  @parameterized.named_parameters(
      ('sample_size_larger_then_num_clients', 4, 3, 3),
      ('sample_size_equal_num_clients', 2, 2, 2),
      ('sample_size_smaller_than_num_clients', 2, 5, 2),
  )
  def test_returns_correct_num_samples(
      self, sample_size, num_clients, expected_num_samples
  ):
    polymorphic_aggregator_computation = aggregator.finalize_then_sample(
        metric_finalizers=_UNUSED_METRICS_FINALIZERS,
        sample_size=sample_size,
    )

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            collections.OrderedDict(
                accuracy=[
                    computation_types.TensorType(np.float32),
                    computation_types.TensorType(np.float32),
                ],
            ),
            placements.CLIENTS,
        )
    )
    def aggregator_computation(unfinalized_metrics):
      return polymorphic_aggregator_computation(unfinalized_metrics)

    local_metrics_at_clients = [_UNUSED_UNFINALIZED_METRICS] * num_clients
    aggregated_metrics = aggregator_computation(local_metrics_at_clients)
    tf.nest.map_structure(
        lambda v: self.assertLen(v, expected_num_samples), aggregated_metrics
    )


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
