# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for the end-to-end user experiences of the FedAvg algorithm.

This includes integrations wtih tff.aggregators, tf.keras, and other
dependencies. Tests that models trained on TFF provided datasets converge.
"""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.tests import learning_test_models


def _get_tff_optimizer(learning_rate=0.1):
  return tff.learning.optimizers.build_sgdm(learning_rate=learning_rate)


def _get_keras_optimizer_fn(learning_rate=0.1):
  return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self, name='num_examples', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_pred)[0], sample_weight)


class FederatedAveragingE2ETest(tff.test.TestCase, parameterized.TestCase):

  def _run_test(self, process, *, datasets, expected_num_examples):
    state = process.initialize()
    prev_loss = np.inf
    aggregation_metrics = collections.OrderedDict(mean_value=(), mean_weight=())
    for _ in range(3):
      state, metric_outputs = process.next(state, datasets)
      self.assertEqual(
          list(metric_outputs.keys()),
          ['broadcast', 'aggregation', 'train', 'stat'])
      self.assertEmpty(metric_outputs['broadcast'])
      self.assertEqual(aggregation_metrics, metric_outputs['aggregation'])
      train_metrics = metric_outputs['train']
      self.assertEqual(train_metrics['num_examples'], expected_num_examples)
      self.assertLess(train_metrics['loss'], prev_loss)
      prev_loss = train_metrics['loss']

  @parameterized.named_parameters([
      ('unweighted_keras_opt', tff.learning.ClientWeighting.UNIFORM,
       _get_keras_optimizer_fn),
      ('example_weighted_keras_opt', tff.learning.ClientWeighting.NUM_EXAMPLES,
       _get_keras_optimizer_fn),
      ('custom_weighted_keras_opt', lambda _: tf.constant(1.5),
       _get_keras_optimizer_fn),
      ('unweighted_tff_opt', tff.learning.ClientWeighting.UNIFORM,
       _get_tff_optimizer),
      ('example_weighted_tff_opt', tff.learning.ClientWeighting.NUM_EXAMPLES,
       _get_tff_optimizer),
      ('custom_weighted_tff_opt', lambda _: tf.constant(1.5),
       _get_tff_optimizer),
  ])
  def test_client_weighting_converges(self, client_weighting, client_optimizer):
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=learning_test_models.LinearRegression,
        client_optimizer_fn=client_optimizer(),
        client_weighting=client_weighting)

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    num_clients = 3
    self._run_test(
        iterative_process,
        datasets=[ds] * num_clients,
        expected_num_examples=2 * num_clients)

  @parameterized.named_parameters([
      ('functional_model_keras_opt',
       learning_test_models.build_linear_regression_keras_functional_model,
       _get_keras_optimizer_fn),
      ('sequential_model_keras_opt',
       learning_test_models.build_linear_regression_keras_sequential_model,
       _get_keras_optimizer_fn),
      ('functional_model_tff_opt',
       learning_test_models.build_linear_regression_keras_functional_model,
       _get_tff_optimizer),
      ('sequential_model_tff_opt',
       learning_test_models.build_linear_regression_keras_sequential_model,
       _get_tff_optimizer),
  ])
  def test_models_defined_in_keras_converge(self, build_keras_model_fn,
                                            client_optimizer):
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    def model_fn():
      keras_model = build_keras_model_fn(feature_dims=2)
      return tff.learning.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=ds.element_spec,
          metrics=[NumExamplesCounter()])

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer(learning_rate=0.01))

    num_clients = 3
    self._run_test(
        iterative_process,
        datasets=[ds] * num_clients,
        expected_num_examples=2 * num_clients)

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_keras_model_with_lookup_table_converges(self, client_optimizer):
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[['R'], ['G'], ['B']], y=[[1.0], [2.0], [3.0]])).batch(2)

    def model_fn():
      keras_model = learning_test_models.build_lookup_table_keras_model()
      return tff.learning.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=ds.element_spec,
          metrics=[NumExamplesCounter()])

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn, client_optimizer_fn=client_optimizer())

    num_clients = 3
    self._run_test(
        iterative_process,
        datasets=[ds] * num_clients,
        expected_num_examples=3 * num_clients)

  @parameterized.named_parameters([
      ('robust_tff_opt', tff.learning.robust_aggregator, _get_tff_optimizer),
      ('robust_keras_opt', tff.learning.robust_aggregator,
       _get_keras_optimizer_fn),
      ('dp_tff_opt', lambda: tff.learning.dp_aggregator(1e-3, 3),
       _get_tff_optimizer),
      ('dp_keras_opt', lambda: tff.learning.dp_aggregator(1e-3, 3),
       _get_keras_optimizer_fn),
      ('compression_tff_opt', tff.learning.compression_aggregator,
       _get_tff_optimizer),
      ('compression_keras_opt', tff.learning.compression_aggregator,
       _get_keras_optimizer_fn),
      ('secure_tff', tff.learning.secure_aggregator, _get_tff_optimizer),
      ('secure_keras_opt', tff.learning.secure_aggregator,
       _get_keras_optimizer_fn),
  ])
  def test_recommended_aggregations_converge(self, default_aggregation,
                                             client_optimizer):
    process = tff.learning.build_federated_averaging_process(
        model_fn=learning_test_models.LinearRegression,
        client_optimizer_fn=client_optimizer(),
        model_update_aggregation_factory=default_aggregation())

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    num_clients = 3
    state = process.initialize()
    state, metrics = process.next(state, [ds] * num_clients)
    self.assertNotEmpty(metrics['aggregation'])


if __name__ == '__main__':
  # We must use the test execution context for the secure intrinsics introduced
  # by tff.learning.secure_aggregator.
  tff.backends.test.set_test_execution_context()
  tff.test.main()
