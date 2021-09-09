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
"""Integration tests regarding the behavior of the FedAvg algorithm."""

import collections

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.tests import learning_test_models


def _get_tff_optimizer(learning_rate=0.1):
  return tff.learning.optimizers.build_sgdm(learning_rate=learning_rate)


def _get_keras_optimizer_fn(learning_rate=0.1):
  return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


class FederatedAveragingIntegrationTest(tff.test.TestCase,
                                        parameterized.TestCase):

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_clients_without_data_do_affect_training(self, client_optimizer):
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=learning_test_models.LinearRegression,
        client_optimizer_fn=client_optimizer())

    # Results in empty dataset with correct types and shapes.
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0]],
            y=[[5.0]],
        )).batch(
            5, drop_remainder=True)

    server_state = iterative_process.initialize()

    first_state, metric_outputs = iterative_process.next(server_state, [ds] * 2)
    self.assertAllClose(
        list(first_state.model.trainable), [[[0.0], [0.0]], 0.0])
    self.assertEqual(metric_outputs['train']['num_examples'], 0)
    self.assertTrue(tf.math.is_nan(metric_outputs['train']['loss']))

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_get_model_weights_from_trained_model(self, client_optimizer):
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=learning_test_models.LinearRegression,
        client_optimizer_fn=client_optimizer())

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
  def test_recommended_aggregations_produce_measurements(
      self, default_aggregation, client_optimizer):
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
