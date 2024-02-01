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
from typing import Any

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.metrics import aggregation_utils
from tensorflow_federated.python.learning.metrics import keras_finalizer


@tf.function
def _tf_mean(x):
  return tf.math.divide_no_nan(x[0], x[1])


def _test_functional_finalize_metrics(
    unfinalized_metrics: collections.OrderedDict[str, Any]
) -> collections.OrderedDict[str, Any]:
  return collections.OrderedDict(
      accuracy=keras_finalizer.create_keras_metric_finalizer(
          tf.keras.metrics.SparseCategoricalAccuracy
      )(unfinalized_metrics['accuracy'])
  )


class CheckMetricFinalizersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('ordereddict', collections.OrderedDict(mean=_tf_mean)),
      ('functional_finalizers', _test_functional_finalize_metrics),
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
  )
  def test_invalid_finalizers_raises(self, metric_finalizers, expected_regex):
    with self.assertRaisesRegex(TypeError, expected_regex):
      aggregation_utils.check_metric_finalizers(metric_finalizers)


class CheckUnfinalizedMetricsTypeTest(tf.test.TestCase, parameterized.TestCase):

  def test_valid_type_does_not_raise(self):
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        collections.OrderedDict(
            num_examples=np.int32, mean=[np.float32, np.float32]
        ),
        collections.OrderedDict,
    )
    aggregation_utils.check_local_unfinalized_metrics_type(
        local_unfinalized_metrics_type
    )

  @parameterized.named_parameters(
      (
          'struct_type',
          computation_types.StructType([(None, np.int32)]),
          '`tff.types.StructWithPythonType`',
      ),
      (
          'ordereddict',
          collections.OrderedDict(num_example=np.int32),
          '`tff.types.StructWithPythonType`',
      ),
      (
          'list_container',
          computation_types.StructWithPythonType(
              [np.float32, np.float32], list
          ),
          'Python container',
      ),
  )
  def test_invalid_type_raises(self, unfinalized_metrics_type, expected_regex):
    with self.assertRaisesRegex(TypeError, expected_regex):
      aggregation_utils.check_local_unfinalized_metrics_type(
          unfinalized_metrics_type
      )


class CheckFinalizersMatchUnfinalizedMetricsTypeTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_match_does_not_raise(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x), mean=_tf_mean
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        collections.OrderedDict(
            num_examples=np.int32, mean=[np.float32, np.float32]
        ),
        collections.OrderedDict,
    )
    aggregation_utils.check_finalizers_matches_unfinalized_metrics(
        metric_finalizers,
        local_unfinalized_metrics_type,
    )

  @parameterized.named_parameters(
      (
          'more_metrics_in_finalizers',
          collections.OrderedDict(
              num_examples=tf.function(func=lambda x: x), mean=_tf_mean
          ),
          computation_types.StructWithPythonType(
              collections.OrderedDict(num_examples=np.int32),
              collections.OrderedDict,
          ),
          'Metric names in the `metric_finalizers`',
      ),
      (
          'more_metrics_in_unfinalized_metrics_type',
          collections.OrderedDict(mean=_tf_mean),
          computation_types.StructWithPythonType(
              collections.OrderedDict(
                  num_examples=np.int32, mean=[np.float32, np.float32]
              ),
              collections.OrderedDict,
          ),
          'Metric names in the `local_unfinalized_metrics`',
      ),
  )
  def test_not_match_raises(
      self, metric_finalizers, local_unfinalized_metrics_type, expected_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      aggregation_utils.check_finalizers_matches_unfinalized_metrics(
          metric_finalizers,
          local_unfinalized_metrics_type,
      )


class CheckBuildFinalizerComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'non_functional',
          collections.OrderedDict(accuracy=_tf_mean),
          computation_types.StructWithPythonType(
              [('accuracy', computation_types.TensorType(np.float32, (2,)))],
              collections.OrderedDict,
          ),
          collections.OrderedDict(accuracy=[0.2, 5.0]),
          collections.OrderedDict(accuracy=0.2 / 5.0),
      ),
      (
          'functional',
          _test_functional_finalize_metrics,
          computation_types.StructWithPythonType(
              [(
                  'accuracy',
                  [
                      computation_types.TensorType(np.float32),
                      computation_types.TensorType(np.float32),
                  ],
              )],
              collections.OrderedDict,
          ),
          collections.OrderedDict(accuracy=[0.4, 2.0]),
          collections.OrderedDict(accuracy=0.4 / 2.0),
      ),
  )
  def test_finalizer_computation_gives_correct_result(
      self,
      metric_finalizers,
      unfinalized_metric_type,
      unfinalized_metric,
      expected_result,
  ):
    finalizer_computation = aggregation_utils.build_finalizer_computation(
        metric_finalizers, unfinalized_metric_type
    )
    result = finalizer_computation(unfinalized_metric)
    tf.nest.map_structure(self.assertAlmostEqual, result, expected_result)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
