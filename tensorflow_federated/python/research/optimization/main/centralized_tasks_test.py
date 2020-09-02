# Copyright 2020, The TensorFlow Federated Authors.
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
"""End-to-end tests for centralized trainer tasks."""

import os.path

from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.optimization.cifar100 import centralized_cifar100
from tensorflow_federated.python.research.optimization.emnist import centralized_emnist
from tensorflow_federated.python.research.optimization.emnist_ae import centralized_emnist_ae
from tensorflow_federated.python.research.optimization.shakespeare import centralized_shakespeare
from tensorflow_federated.python.research.optimization.stackoverflow import centralized_stackoverflow
from tensorflow_federated.python.research.optimization.stackoverflow_lr import centralized_stackoverflow_lr


class CentralizedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar100', centralized_cifar100.run_centralized),
      ('emnist_cr', centralized_emnist.run_centralized),
      ('emnist_ae', centralized_emnist_ae.run_centralized),
      ('shakespeare', centralized_shakespeare.run_centralized),
      ('stackoverflow_nwp', centralized_stackoverflow.run_centralized),
      ('stackoverflow_lr', centralized_stackoverflow_lr.run_centralized),
  )
  def test_run_centralized(self, run_centralized_fn):
    num_epochs = 1
    root_output_dir = self.get_temp_dir()
    exp_name = 'test_run_centralized'

    run_centralized_fn(
        num_epochs=num_epochs,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        batch_size=10,
        max_batches=2,
        experiment_name=exp_name,
        root_output_dir=root_output_dir)

    self.assertTrue(tf.io.gfile.exists(root_output_dir))
    log_dir = os.path.join(root_output_dir, 'logdir', exp_name)
    train_log_dir = os.path.join(log_dir, 'train')
    validation_log_dir = os.path.join(log_dir, 'validation')
    self.assertTrue(tf.io.gfile.exists(log_dir))
    self.assertTrue(tf.io.gfile.exists(train_log_dir))
    self.assertTrue(tf.io.gfile.exists(validation_log_dir))

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))
    metrics_file = os.path.join(results_dir, 'metric_results.csv')
    self.assertTrue(tf.io.gfile.exists(metrics_file))

    metrics_csv = pd.read_csv(metrics_file)
    self.assertLen(
        metrics_csv.index,
        num_epochs,
        msg='The output metrics CSV should have {} rows, equal to the number of'
        'training epochs.'.format(num_epochs))

    self.assertIn(
        'loss',
        metrics_csv.columns,
        msg='The output metrics CSV should have a column "loss" if training is'
        'successful.')
    self.assertIn(
        'val_loss',
        metrics_csv.columns,
        msg='The output metrics CSV should have a column "val_loss" if '
        'validation metric computation is successful.')


if __name__ == '__main__':
  tf.test.main()
