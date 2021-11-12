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

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import finalizer

_UNUSED_METRICS_FINALIZERS = collections.OrderedDict(
    accuracy=tf.function(func=lambda x: x[0] / x[1]))
_UNUSED_UNFINALIZED_METRICS = collections.OrderedDict(
    accuracy=[tf.constant(1.0), tf.constant(2.0)])


class CustomSumMetric(tf.keras.metrics.Sum):
  """A custom metric whose result is total + extra scalar and vector values."""

  def __init__(self, name='custom_sum_metric', dtype=None):
    super().__init__(name=name, dtype=dtype)
    self.scalar = self.add_weight(
        name='scalar', shape=(), initializer='zeros', dtype=tf.float32)
    self.vector = self.add_weight(
        name='vector', shape=(2,), initializer='zeros', dtype=tf.float32)

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
            custom_sum=[
                tf.constant(1.0),
                tf.constant(1.0),
                tf.constant([1.0, 1.0])
            ]),
        collections.OrderedDict(
            accuracy=[tf.constant(3.0), tf.constant(6.0)],
            custom_sum=[
                tf.constant(1.0),
                tf.constant(1.0),
                tf.constant([1.0, 1.0])
            ])
    ],
    # The finalized metrics are computed by first summing the unfinalized values
    # from clients, and run the corresponding finalizers (a division for
    # `accuracy`, and a sum for `custom_sum`) at the server.
    'expected_aggregated_metrics':
        collections.OrderedDict(
            accuracy=(1.0 + 3.0) / (2.0 + 6.0), custom_sum=8.0)
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
            sum=collections.OrderedDict(count_1=1.0, count_2=1.0)),
        collections.OrderedDict(
            divide=[tf.constant(3.0), tf.constant(6.0)],
            sum=collections.OrderedDict(count_1=1.0, count_2=1.0))
    ],
    'expected_aggregated_metrics':
        collections.OrderedDict(divide=(1.0 + 3.0) / (2.0 + 6.0), sum=4.0)
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


class AggregatorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(_TEST_ARGUMENTS_KERAS_METRICS,
                                  _TEST_ARGUMENTS_NON_KERAS_METRICS)
  def test_sum_then_finalize_returns_correct_results(
      self, metric_finalizers, local_unfinalized_metrics_at_clients,
      expected_aggregated_metrics):
    aggregator_computation = aggregator.sum_then_finalize(
        metric_finalizers=metric_finalizers,
        local_unfinalized_metrics_type=type_conversions.type_from_tensors(
            local_unfinalized_metrics_at_clients[0]))
    aggregated_metrics = aggregator_computation(
        local_unfinalized_metrics_at_clients)
    tf.nest.map_structure(self.assertAllEqual, aggregated_metrics,
                          expected_aggregated_metrics)

  @parameterized.named_parameters(_TEST_ARGUMENTS_INVALID_INPUTS)
  def test_sum_then_finalize_fails_with_invalid_inputs(
      self, metric_finalizers, local_unfinalized_metrics, error_type,
      error_message):
    with self.assertRaisesRegex(error_type, error_message):
      aggregator.sum_then_finalize(
          metric_finalizers=metric_finalizers,
          local_unfinalized_metrics_type=type_conversions.type_from_tensors(
              local_unfinalized_metrics))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
