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
"""Tests for the integrations of FedSGD algorithm from tff.learning.

This includes integrations wtih tff.aggregators, tf.keras, and other
dependencies.
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


class FederatedSGDE2ETest(tff.test.TestCase, parameterized.TestCase):

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
  def test_orchestration_execute(self, client_weighting, server_optimizer):
    iterative_process = tff.learning.build_federated_sgd_process(
        model_fn=learning_test_models.LinearRegression,
        server_optimizer_fn=server_optimizer(),
        client_weighting=client_weighting)

    # Some data points along [x_1 + 2*x_2 + 3 = y], expecting to learn
    # kernel = [1, 2], bias = [3].
    ds1 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [0.0, 1.0]],
            y=[[3.0], [5.0]],
        )).batch(2)
    ds2 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0], [1.0, 0.0], [-1.0, -1.0]],
            y=[[8.0], [14.0], [4.00], [0.0]],
        )).batch(2)
    federated_ds = [ds1, ds2]

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    num_iterations = 3
    for _ in range(num_iterations):
      server_state, metric_outputs = iterative_process.next(
          server_state, federated_ds)
      train_metrics = metric_outputs['train']
      self.assertEqual(train_metrics['num_examples'],
                       num_iterations * len(federated_ds))
      loss = train_metrics['loss']
      self.assertLess(loss, prev_loss)
      prev_loss = loss

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
  def test_orchestration_execute_from_keras(self, build_keras_model_fn,
                                            server_optimizer):
    # Some data points along [x_1 + 2*x_2 + 3 = y], expecting to learn
    # kernel = [1, 2], bias = [3].
    ds1 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [0.0, 1.0]],
            y=[[3.0], [5.0]],
        )).batch(2)
    ds2 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0], [1.0, 0.0], [-1.0, -1.0]],
            y=[[8.0], [14.0], [4.00], [0.0]],
        )).batch(2)
    federated_ds = [ds1, ds2]

    def model_fn():
      # Note: we don't compile with an optimizer here; FedSGD does not use it.
      keras_model = build_keras_model_fn(feature_dims=2)
      return tff.learning.from_keras_model(
          keras_model,
          input_spec=ds1.element_spec,
          loss=tf.keras.losses.MeanSquaredError())

    iterative_process = tff.learning.build_federated_sgd_process(
        model_fn=model_fn, server_optimizer_fn=server_optimizer())

    server_state = iterative_process.initialize()
    prev_loss = np.inf
    num_iterations = 3
    for _ in range(num_iterations):
      server_state, metrics = iterative_process.next(server_state, federated_ds)
      new_loss = metrics['train']['loss']
      self.assertLess(new_loss, prev_loss)
      prev_loss = new_loss

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_execute_empty_data(self, server_optimizer):
    iterative_process = tff.learning.build_federated_sgd_process(
        model_fn=learning_test_models.LinearRegression,
        server_optimizer_fn=server_optimizer())

    # Results in empty dataset with correct types and shapes.
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(x=[[1.0, 2.0]], y=[[5.0]])).batch(
            5, drop_remainder=True)  # No batches of size 5 can be created.
    federated_ds = [ds] * 2

    server_state = iterative_process.initialize()
    first_state, metric_outputs = iterative_process.next(
        server_state, federated_ds)
    self.assertAllClose(
        list(first_state.model.trainable), [[[0.0], [0.0]], 0.0])
    self.assertEqual(
        list(metric_outputs.keys()),
        ['broadcast', 'aggregation', 'train', 'stat'])
    self.assertEmpty(metric_outputs['broadcast'])
    self.assertEqual(metric_outputs['train']['num_examples'], 0)
    self.assertTrue(tf.math.is_nan(metric_outputs['train']['loss']))

    # Test aggregation metrics with default model update aggregator
    aggregation_metrics = collections.OrderedDict(mean_value=(), mean_weight=())
    debug_measurements_keys = [
        'average_client_norm', 'std_dev_client_norm', 'server_update_max',
        'server_update_norm', 'server_update_min'
    ]
    expected_aggregation_keys = list(
        aggregation_metrics.keys()) + debug_measurements_keys
    self.assertEqual(
        list(metric_outputs['aggregation'].keys()), expected_aggregation_keys)
    for k, v in aggregation_metrics.items():
      self.assertEqual(v, metric_outputs['aggregation'][k])

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_get_model_weights(self, server_optimizer):
    iterative_process = tff.learning.build_federated_sgd_process(
        model_fn=learning_test_models.LinearRegression,
        server_optimizer_fn=server_optimizer())

    num_clients = 3
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)
    datasets = [ds] * num_clients

    state = iterative_process.initialize()
    self.assertIsInstance(
        iterative_process.get_model_weights(state), tff.learning.ModelWeights)
    self.assertAllClose(state.model.trainable,
                        iterative_process.get_model_weights(state).trainable)

    for _ in range(3):
      state, _ = iterative_process.next(state, datasets)
      self.assertIsInstance(
          iterative_process.get_model_weights(state), tff.learning.ModelWeights)
      self.assertAllClose(state.model.trainable,
                          iterative_process.get_model_weights(state).trainable)


if __name__ == '__main__':
  # We must use the test execution context for the secure intrinsics introduced
  # by tff.learning.secure_aggregator.
  tff.backends.test.set_test_execution_context()
  tff.test.main()
