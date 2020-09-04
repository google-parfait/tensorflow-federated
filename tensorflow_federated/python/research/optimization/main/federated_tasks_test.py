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
"""End-to-end tests for federated trainer tasks."""

import collections
import os.path

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.research.optimization.cifar100 import federated_cifar100
from tensorflow_federated.python.research.optimization.emnist import federated_emnist
from tensorflow_federated.python.research.optimization.emnist_ae import federated_emnist_ae
from tensorflow_federated.python.research.optimization.shakespeare import federated_shakespeare
from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule
from tensorflow_federated.python.research.optimization.stackoverflow import federated_stackoverflow
from tensorflow_federated.python.research.optimization.stackoverflow_lr import federated_stackoverflow_lr
from tensorflow_federated.python.research.utils import metrics_manager


def iterative_process_builder(model_fn, client_weight_fn=None):
  return fed_avg_schedule.build_fed_avg_process(
      model_fn=model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      client_lr=0.1,
      server_optimizer_fn=tf.keras.optimizers.SGD,
      server_lr=1.0,
      client_weight_fn=client_weight_fn)


def assign_weights_fn(reference_model, keras_model):
  reference_model.assign_weights_to(keras_model)


class FederatedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar100', federated_cifar100.run_federated),
      ('emnist_cr', federated_emnist.run_federated),
      ('emnist_ae', federated_emnist_ae.run_federated),
      ('shakespeare', federated_shakespeare.run_federated),
      ('stackoverflow_nwp', federated_stackoverflow.run_federated),
      ('stackoverflow_lr', federated_stackoverflow_lr.run_federated),
  )
  def test_run_federated(self, run_federated_fn):
    total_rounds = 1
    shared_args = collections.OrderedDict(
        client_epochs_per_round=1,
        client_batch_size=10,
        clients_per_round=1,
        client_datasets_random_seed=1,
        total_rounds=total_rounds,
        max_batches_per_client=2,
        iterative_process_builder=iterative_process_builder,
        assign_weights_fn=fed_avg_schedule.ServerState
        .assign_weights_to_keras_model,
        rounds_per_checkpoint=10,
        rounds_per_eval=10,
        rounds_per_train_eval=10,
        max_eval_batches=2)
    root_output_dir = self.get_temp_dir()
    exp_name = 'test_run_federated'
    shared_args['root_output_dir'] = root_output_dir
    shared_args['experiment_name'] = exp_name

    run_federated_fn(**shared_args)

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))

    scalar_manager = metrics_manager.ScalarMetricsManager(results_dir)
    metrics = scalar_manager.get_metrics()

    self.assertIn(
        'train/loss',
        metrics.columns,
        msg='The output metrics should have a `train/loss` column if training '
        'is successful.')
    self.assertIn(
        'eval/loss',
        metrics.columns,
        msg='The output metrics should have a `train/loss` column if validation'
        ' metrics computation is successful.')
    self.assertIn(
        'test/loss',
        metrics.columns,
        msg='The output metrics should have a `test/loss` column if test '
        'metrics computation is successful.')
    self.assertLen(
        metrics.index,
        total_rounds + 1,
        msg='The number of rows in the metrics CSV should be the number of '
        'training rounds + 1 (as there is an extra row for validation/test set'
        'metrics after training has completed.')


if __name__ == '__main__':
  tf.test.main()
