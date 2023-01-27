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
"""Tests for aggregation_utils."""

import collections
from typing import Any, OrderedDict

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning.metrics import aggregation_utils
from tensorflow_federated.python.learning.metrics import keras_finalizer


@tf.function
def _tf_mean(x):
  return tf.math.divide_no_nan(x[0], x[1])


def _test_finalize_metrics(
    unfinalized_metrics: OrderedDict[str, Any]
) -> OrderedDict[str, Any]:
  return collections.OrderedDict(
      accuracy=keras_finalizer.create_keras_metric_finalizer(
          tf.keras.metrics.SparseCategoricalAccuracy
      )(unfinalized_metrics['accuracy'])
  )


class CheckMetricFinalizersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('ordereddict', collections.OrderedDict(mean=_tf_mean)),
      ('functional_finalizers', _test_finalize_metrics),
  )
  def test_valid_finalizers_does_not_raise(self, metric_finalizers):
    aggregation_utils.check_metric_finalizers(metric_finalizers)

  @parameterized.named_parameters(
      ('non_ordereddict', [_tf_mean], 'metric_finalizers'),
      (
          'non_str_key',
          collections.OrderedDict([(1.0, _tf_mean)]),
          'metric_finalizers key',
      ),
      (
          'non_callable',
          collections.OrderedDict(mean=_tf_mean([2.0, 1.0])),
          'metric_finalizers value',
      ),
  )
  def test_invalid_finalizers_raises(self, metric_finalizers, expected_regex):
    with self.assertRaisesRegex(TypeError, expected_regex):
      aggregation_utils.check_metric_finalizers(metric_finalizers)


class CheckUnfinalizedMetricsTypeTest(tf.test.TestCase, parameterized.TestCase):

  def test_valid_type_does_not_raise(self):
    local_unfianlized_metrics = collections.OrderedDict(
        num_examples=1, mean=[2.0, 1.0]
    )
    aggregation_utils.check_local_unfinalzied_metrics_type(
        type_conversions.type_from_tensors(local_unfianlized_metrics)
    )

  @parameterized.named_parameters(
      (
          'struct_type',
          computation_types.StructType([(None, tf.int32)]),
          '`tff.types.StructWithPythonType`',
      ),
      (
          'ordereddict',
          collections.OrderedDict(num_example=tf.int32),
          '`tff.types.StructWithPythonType`',
      ),
      (
          'list_container',
          type_conversions.type_from_tensors([1.0, 2.0]),
          'Python container',
      ),
  )
  def test_invalid_type_raises(self, unfinalized_metrics_type, expected_regex):
    with self.assertRaisesRegex(TypeError, expected_regex):
      aggregation_utils.check_local_unfinalzied_metrics_type(
          unfinalized_metrics_type
      )


class CheckFinalizersMatchUnfinalizedMetricsTypeTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_match_does_not_raise(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x), mean=_tf_mean
    )
    local_unfianlized_metrics = collections.OrderedDict(
        num_examples=1, mean=[2.0, 1.0]
    )
    aggregation_utils.check_finalizers_matches_unfinalized_metrics(
        metric_finalizers,
        type_conversions.type_from_tensors(local_unfianlized_metrics),
    )

  @parameterized.named_parameters(
      (
          'more_metrics_in_finalizers',
          collections.OrderedDict(
              num_examples=tf.function(func=lambda x: x), mean=_tf_mean
          ),
          collections.OrderedDict(num_examples=1),
          'Metric names in the `metric_finalizers`',
      ),
      (
          'more_metrics_in_unfinalized_metrics_type',
          collections.OrderedDict(mean=_tf_mean),
          collections.OrderedDict(num_examples=1, mean=[2.0, 1.0]),
          'Metric names in the `local_unfinalized_metrics`',
      ),
  )
  def test_not_match_raises(
      self, metric_finalizers, local_unfianlized_metrics, expected_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      aggregation_utils.check_finalizers_matches_unfinalized_metrics(
          metric_finalizers,
          type_conversions.type_from_tensors(local_unfianlized_metrics),
      )


if __name__ == '__main__':
  tf.test.main()
