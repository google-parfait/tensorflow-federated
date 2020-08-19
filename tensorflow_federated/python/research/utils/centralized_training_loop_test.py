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

import os.path

import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.utils import centralized_training_loop


def create_dataset():
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices((
      [
          [1.0, 2.0],
          [3.0, 4.0],
      ],
      [
          [5.0],
          [6.0],
      ],
  ))
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  return dataset.repeat(3).batch(2)


def create_sequential_model(input_dims=2):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(input_dims,))
  ])


def compiled_keras_model(input_dims=2,
                         optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)):
  model = create_sequential_model(input_dims)
  model.compile(
      loss=tf.keras.losses.MeanSquaredError(),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.MeanSquaredError()])
  return model


class CentralizedTrainingLoopTest(tf.test.TestCase):

  def assertMetricDecreases(self, metric, expected_len):
    self.assertLen(metric, expected_len)
    self.assertLess(metric[-1], metric[0])

  def test_training_reduces_loss(self):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    history = centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        experiment_name='test_experiment',
        root_output_dir=self.get_temp_dir(),
        num_epochs=5,
        validation_dataset=dataset)

    self.assertCountEqual(
        history.history.keys(),
        ['loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error'])

    self.assertMetricDecreases(history.history['loss'], expected_len=5)
    self.assertMetricDecreases(history.history['val_loss'], expected_len=5)
    self.assertMetricDecreases(
        history.history['mean_squared_error'], expected_len=5)
    self.assertMetricDecreases(
        history.history['val_mean_squared_error'], expected_len=5)

  def test_lr_callback(self):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    keras_model = compiled_keras_model(optimizer=optimizer)
    dataset = create_dataset()
    history = centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        experiment_name='test_experiment',
        root_output_dir=self.get_temp_dir(),
        num_epochs=10,
        decay_epochs=8,
        lr_decay=0.5,
        validation_dataset=dataset)

    self.assertCountEqual(history.history.keys(), [
        'loss', 'mean_squared_error', 'val_loss', 'val_mean_squared_error', 'lr'
    ])
    self.assertAllClose(history.history['lr'], [0.1] * 7 + [0.05] * 3)

  def test_metric_writing(self):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    exp_name = 'write_eval_metrics'
    temp_filepath = self.get_temp_dir()
    root_output_dir = temp_filepath

    centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        experiment_name=exp_name,
        root_output_dir=root_output_dir,
        num_epochs=2,
        validation_dataset=dataset)

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

    hparams_file = os.path.join(results_dir, 'hparams.csv')
    self.assertFalse(tf.io.gfile.exists(hparams_file))

    metrics_csv = pd.read_csv(metrics_file)
    self.assertEqual(metrics_csv.shape, (2, 5))
    self.assertCountEqual(metrics_csv.columns, [
        'Unnamed: 0', 'loss', 'mean_squared_error', 'val_loss',
        'val_mean_squared_error'
    ])

  def test_metric_writing_without_validation(self):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    exp_name = 'write_metrics'
    temp_filepath = self.get_temp_dir()
    root_output_dir = temp_filepath

    centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        experiment_name=exp_name,
        root_output_dir=root_output_dir,
        num_epochs=3)

    self.assertTrue(tf.io.gfile.exists(root_output_dir))

    log_dir = os.path.join(root_output_dir, 'logdir', exp_name)
    train_log_dir = os.path.join(log_dir, 'train')
    validation_log_dir = os.path.join(log_dir, 'validation')
    self.assertTrue(tf.io.gfile.exists(log_dir))
    self.assertTrue(tf.io.gfile.exists(train_log_dir))
    self.assertFalse(tf.io.gfile.exists(validation_log_dir))

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))
    metrics_file = os.path.join(results_dir, 'metric_results.csv')
    self.assertTrue(tf.io.gfile.exists(metrics_file))

    hparams_file = os.path.join(results_dir, 'hparams.csv')
    self.assertFalse(tf.io.gfile.exists(hparams_file))

    metrics_csv = pd.read_csv(metrics_file)
    self.assertEqual(metrics_csv.shape, (3, 3))
    self.assertCountEqual(metrics_csv.columns,
                          ['Unnamed: 0', 'loss', 'mean_squared_error'])

  def test_hparam_writing(self):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    exp_name = 'write_hparams'
    temp_filepath = self.get_temp_dir()
    root_output_dir = temp_filepath

    hparams_dict = {
        'param1': 0,
        'param2': 5.02,
        'param3': 'sample',
        'param4': True
    }

    centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        experiment_name=exp_name,
        root_output_dir=root_output_dir,
        num_epochs=1,
        hparams_dict=hparams_dict)

    self.assertTrue(tf.io.gfile.exists(root_output_dir))

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    self.assertTrue(tf.io.gfile.exists(hparams_file))

    hparams_csv = pd.read_csv(hparams_file, index_col=0)
    expected_csv = pd.DataFrame(hparams_dict, index=[0])

    pd.testing.assert_frame_equal(hparams_csv, expected_csv)


if __name__ == '__main__':
  tf.test.main()
