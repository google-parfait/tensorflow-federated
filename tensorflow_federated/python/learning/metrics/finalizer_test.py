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
"""Tests for finalizer."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning.metrics import finalizer


# Wrapping the `tf.function` returned by `create_keras_metric_finalizer` as a
# TFF computation. This is needed because the `tf.function` under testing
# will create `tf.Variable`s on the non-first call (and hence, will throw an
# error if it is not wrapped as a TFF computation).
def wrap_tf_function_in_tff_tf_computation(metric, unfinalized_metric_type):

  @computations.tf_computation(unfinalized_metric_type)
  def finalizer_computation(unfinalized_metric):
    metric_finalizer = finalizer.create_keras_metric_finalizer(metric)
    return metric_finalizer(unfinalized_metric)

  return finalizer_computation


class CustomMeanMetric(tf.keras.metrics.Mean):
  """A custom metric whose result is total/(count + initial_count)."""

  def __init__(self,
               initial_count: float,
               name='custom_mean_metric',
               dtype=None):
    self._initial_count = initial_count
    super().__init__(name=name, dtype=dtype)
    self.count.assign(initial_count)

  def get_config(self):
    config = super().get_config()
    config['initial_count'] = self._initial_count
    return config


class CustomSumMetric(tf.keras.metrics.Sum):
  """A custom metric whose result is total + extra scalar and vector values."""

  def __init__(self,
               has_extra_variables: bool,
               name='custom_sum_metric',
               dtype=None):
    super().__init__(name=name, dtype=dtype)
    self._has_extra_variables = has_extra_variables
    if self._has_extra_variables:
      self.scalar = self.add_weight(
          name='scalar', shape=(), initializer='zeros', dtype=tf.float32)
      self.vector = self.add_weight(
          name='vector', shape=(2,), initializer='zeros', dtype=tf.float32)

  # The method `update_state` is omitted here because only the `result` method
  # is useful in the tests below.
  def result(self):
    if self._has_extra_variables:
      return self.total + self.scalar + tf.reduce_sum(self.vector)
    else:
      return self.total

  def get_config(self):
    config = super().get_config()
    config['has_extra_variables'] = self._has_extra_variables
    return config


class CustomCounter(tf.keras.metrics.Sum):
  """A custom `tf.keras.metrics.Metric` with extra arguments in `__init__`."""

  def __init__(self, name='new_metric', arg1=0, dtype=tf.int64):
    super().__init__(name, dtype)
    self._arg1 = arg1

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(1, sample_weight)


class FinalizerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('keras_metric', tf.keras.metrics.SparseCategoricalAccuracy()),
      ('custom_keras_metric', CustomMeanMetric(initial_count=0.0)),
      ('keras_metric_constructor', tf.keras.metrics.SparseCategoricalAccuracy))
  def test_keras_metric_finalizer_returns_correct_result(self, metric):
    # The unfinalized accuracy contains two tensors `total` and `count`.
    unfinalized_accuracy = [tf.constant(2.0), tf.constant(2.0)]
    finalizer_computation = wrap_tf_function_in_tff_tf_computation(
        metric, type_conversions.type_from_tensors(unfinalized_accuracy))
    finalized_accuracy = finalizer_computation(unfinalized_accuracy)
    self.assertEqual(
        # The expected value is computed by dividing `total` by `count`.
        finalized_accuracy,
        unfinalized_accuracy[0] / unfinalized_accuracy[1])

  @parameterized.named_parameters(
      ('one_variable', CustomSumMetric(has_extra_variables=False),
       [tf.constant(1.0)], 1.0),
      ('three_variables', CustomSumMetric(has_extra_variables=True),
       [tf.constant(1.0),
        tf.constant(1.0),
        tf.constant([1.0, 1.0])], 4.0))
  def test_keras_metric_finalizer_succeeds_with_different_metric_variables(
      self, metric, unfinalized_metric_values, expected_result):
    finalizer_computation = wrap_tf_function_in_tff_tf_computation(
        metric, type_conversions.type_from_tensors(unfinalized_metric_values))
    finalized_metric = finalizer_computation(unfinalized_metric_values)
    self.assertEqual(finalized_metric, expected_result)

  @parameterized.named_parameters(
      ('tensor', tf.constant(1.0), 'found a non-callable'),
      ('loss_constructor', tf.keras.losses.MeanSquaredError,
       'found a callable'),  # go/pyformat-break
      ('custom_metric_with_extra_init_args', CustomCounter(arg1=1),
       'extra arguments'))
  def test_create_keras_metric_finalizer_fails_with_invalid_input(
      self, invalid_metric, error_message):
    unused_type = [tf.TensorSpec(shape=[], dtype=tf.float32)]
    with self.assertRaisesRegex(TypeError, error_message):
      wrap_tf_function_in_tff_tf_computation(invalid_metric, unused_type)

  @parameterized.named_parameters(
      ('not_a_list', tf.constant(1.0), TypeError, 'Expected list'),
      ('not_a_list_of_tensors', [tf.constant(1.0), [tf.constant(1.0)]
                                ], TypeError, 'found list'),
      ('unmatched_length', [tf.constant(1.0)
                           ], ValueError, 'found a list of length 1'),
      ('unmatched_shape', [tf.constant([1.0, 2.0]),
                           tf.constant(1.0)], ValueError,
       r'found a `tf.Tensor` of shape \(2,\) and dtype tf.float32'),
      ('unmatched_dtype', [tf.constant(1, dtype=tf.int32),
                           tf.constant(1.0)], ValueError,
       r'found a `tf.Tensor` of shape \(\) and dtype tf.int32'))
  def test_keras_metric_finalizer_fails_with_unmatched_unfinalized_metric_values(
      self, invalid_unfinalized_metric_values, error_type, error_message):
    # The expected unfinalized metric values for `SparseCategoricalAccuracy` is
    # a list of two `tf.Tensor`s and each has shape () and dtype tf.float32.
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    with self.assertRaisesRegex(error_type, error_message):
      wrap_tf_function_in_tff_tf_computation(
          metric,
          type_conversions.type_from_tensors(invalid_unfinalized_metric_values))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
