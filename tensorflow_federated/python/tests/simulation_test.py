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
"""End-to-end tests for simulations using TFF."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff


def iterative_process_builder(model_fn):
  return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      server_optimizer_fn=tf.keras.optimizers.SGD)


class FederatedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar100_image_classification',
       tff.simulation.baselines.cifar100.create_image_classification_task),
      ('emnist_autoencoder',
       tff.simulation.baselines.emnist.create_autoencoder_task),
      ('emnist_character_recognition',
       tff.simulation.baselines.emnist.create_character_recognition_task),
      ('shakespeare_character_prediction',
       tff.simulation.baselines.shakespeare.create_character_prediction_task),
  )
  def test_run_federated(self, baseline_task_fn):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=32)
    baseline_task = baseline_task_fn(train_client_spec, use_synthetic_data=True)

    process = iterative_process_builder(baseline_task.model_fn)

    def client_selection_fn(round_num):
      del round_num
      return baseline_task.datasets.sample_train_clients(num_clients=1)

    tff.simulation.run_simulation(
        process=process,
        client_selection_fn=client_selection_fn,
        total_rounds=2)


if __name__ == '__main__':
  tf.test.main()
