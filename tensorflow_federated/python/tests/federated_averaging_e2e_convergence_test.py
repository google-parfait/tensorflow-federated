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
"""End-to-end convergence tests for the FedAvg algorithm on realistic data."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def _get_tff_optimizer(learning_rate=0.1):
  return tff.learning.optimizers.build_sgdm(learning_rate=learning_rate)


def _get_keras_optimizer_fn(learning_rate=0.1):
  return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


class FederatedAveragingE2ETest(tff.test.TestCase, parameterized.TestCase):

  def _build_and_run_process(self,
                             client_optimizer_fn,
                             aggregator_factory=None):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=32, shuffle_buffer_size=1)
    task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, model_id='cnn', only_digits=True)
    train_client_ids = sorted(task.datasets.train_data.client_ids)
    preprocessed_train_data = task.datasets.train_data.preprocess(
        task.datasets.train_preprocess_fn)

    def client_selection_fn(round_num):
      random_state = np.random.RandomState(round_num)
      client_ids = random_state.choice(train_client_ids, size=10, replace=False)
      return [
          preprocessed_train_data.create_tf_dataset_for_client(a)
          for a in client_ids
      ]

    process = tff.learning.build_federated_averaging_process(
        model_fn=task.model_fn,
        client_optimizer_fn=client_optimizer_fn,
        model_update_aggregation_factory=aggregator_factory)

    state = process.initialize()
    training_metrics = []
    for round_num in range(200):
      client_data = client_selection_fn(round_num)
      state, metrics = process.next(state, client_data)
      training_metrics.append(metrics['train'])

    loss_last_10_rounds = [a['loss'] for a in training_metrics[-10:]]
    accuracy_last_10_rounds = [
        a['sparse_categorical_accuracy'] for a in training_metrics[-10:]
    ]
    average_loss_last_10_rounds = np.mean(loss_last_10_rounds)
    average_accuracy_last_10_rounds = np.mean(accuracy_last_10_rounds)

    return average_loss_last_10_rounds, average_accuracy_last_10_rounds

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn()),
      ('tff_opt', _get_tff_optimizer()),
  ])
  def test_emnist10_cnn_convergence(self, client_optimizer_fn):
    loss, accuracy = self._build_and_run_process(client_optimizer_fn)
    self.assertLessEqual(loss, 0.15)
    self.assertGreater(accuracy, 0.95)

  @parameterized.named_parameters([
      ('robust_aggregator', tff.learning.robust_aggregator),
      ('secure_aggregator', tff.learning.secure_aggregator),
  ])
  def test_emnist10_cnn_convergence_with_aggregator(self,
                                                    aggregator_factory_fn):
    loss, accuracy = self._build_and_run_process(
        client_optimizer_fn=_get_keras_optimizer_fn(),
        aggregator_factory=aggregator_factory_fn())
    self.assertLessEqual(loss, 0.15)
    self.assertGreater(accuracy, 0.95)

  def test_emnist10_cnn_convergence_compression_aggregator(self):
    loss, accuracy = self._build_and_run_process(
        client_optimizer_fn=_get_keras_optimizer_fn(),
        aggregator_factory=tff.learning.compression_aggregator())

    self.assertLessEqual(loss, 0.16)
    self.assertGreater(accuracy, 0.92)

  def test_emnist10_cnn_convergence_dp_aggregator_low_noise(self):
    # Test with very small noise multiplier. Results should be good,
    # a bit worse than `robust_aggregator` due to lower adaptive clip params.
    loss, accuracy = self._build_and_run_process(
        client_optimizer_fn=_get_keras_optimizer_fn(),
        aggregator_factory=tff.learning.dp_aggregator(1e-8, 10))

    self.assertLessEqual(loss, 0.22)
    self.assertGreater(accuracy, 0.92)

  def test_emnist10_cnn_convergence_dp_aggregator_high_noise(self):
    # Test with larger noise multiplier. Results should be worse, but still
    # better than chance.
    loss, accuracy = self._build_and_run_process(
        client_optimizer_fn=_get_keras_optimizer_fn(),
        aggregator_factory=tff.learning.dp_aggregator(2e-1, 10))

    self.assertLessEqual(loss, 5)
    self.assertGreater(accuracy, 0.15)


if __name__ == '__main__':
  # We must use the test execution context for the secure intrinsics introduced
  # by tff.learning.secure_aggregator.
  tff.backends.test.set_test_execution_context()
  tff.test.main()
