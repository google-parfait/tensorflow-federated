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

import tensorflow as tf

from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.simulation.baselines import task
from tensorflow_federated.python.simulation.baselines import task_data
from tensorflow_federated.python.simulation.datasets import client_data


def create_dataset(client_id):
  del client_id
  # Create data satisfying y = 2*x + 1
  x = [[1.0], [2.0]]
  y = [[3.0], [5.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_client_data():
  return client_data.ClientData.from_clients_and_tf_fn(
      client_ids=['1', '2', '3'], serializable_dataset_fn=create_dataset)


def create_task_data():
  return task_data.BaselineTaskDatasets(
      train_data=create_client_data(), test_data=create_client_data())


def keras_model_builder():
  inputs = tf.keras.layers.Input(shape=(3,), name='input_layer')
  outputs = tf.keras.layers.Dense(
      2, kernel_initializer='ones', use_bias=False, name='dense_layer')(
          inputs)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='model')


def loss_builder():
  return tf.keras.losses.MeanSquaredError(name='loss')


def loss_list_builder():
  return [
      tf.keras.losses.MeanSquaredError(name='loss1'),
      tf.keras.losses.MeanAbsoluteError(name='loss2'),
  ]


def metrics_builder():
  return [
      tf.keras.metrics.MeanSquaredError(name='mse'),
      tf.keras.metrics.MeanAbsoluteError(name='mae')
  ]


class PrintKerasModelSummaryTest(tf.test.TestCase):

  def test_model_summary_matches_keras_model_summary(self):
    keras_model = keras_model_builder()
    model_summary = []
    task._print_keras_model_summary(
        keras_model=keras_model,
        loss=loss_builder(),
        print_fn=model_summary.append)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    model_summary_len = len(model_summary)
    tf_model_summary_len = len(tf_model_summary)
    self.assertLess(tf_model_summary_len, model_summary_len)
    self.assertEqual(tf_model_summary, model_summary[:tf_model_summary_len])

  def test_single_loss_summary(self):
    keras_model = keras_model_builder()
    loss = loss_builder()
    model_summary = []
    task._print_keras_model_summary(
        keras_model=keras_model, loss=loss, print_fn=model_summary.append)
    model_summary_len = len(model_summary)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    tf_model_summary_len = len(tf_model_summary)

    self.assertEqual(model_summary_len, tf_model_summary_len + 1)
    expected_loss_summary = 'Loss: {}'.format(loss.name)
    actual_loss_summary = model_summary[-1]
    self.assertEqual(actual_loss_summary, expected_loss_summary)

  def test_loss_list_summary(self):
    keras_model = keras_model_builder()
    loss = loss_list_builder()
    model_summary = []
    task._print_keras_model_summary(
        keras_model=keras_model, loss=loss, print_fn=model_summary.append)
    model_summary_len = len(model_summary)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    tf_model_summary_len = len(tf_model_summary)

    self.assertEqual(model_summary_len, tf_model_summary_len + 1)
    expected_loss_summary = 'Losses: {}'.format([x.name for x in loss])
    actual_loss_summary = model_summary[-1]
    self.assertEqual(actual_loss_summary, expected_loss_summary)

  def test_loss_and_metrics_summary(self):
    keras_model = keras_model_builder()
    loss = loss_builder()
    metrics = metrics_builder()
    model_summary = []
    task._print_keras_model_summary(
        keras_model=keras_model,
        loss=loss,
        metrics=metrics,
        print_fn=model_summary.append)
    model_summary_len = len(model_summary)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    tf_model_summary_len = len(tf_model_summary)

    self.assertEqual(model_summary_len, tf_model_summary_len + 2)
    expected_loss_summary = 'Loss: {}'.format(loss.name)
    actual_loss_summary = model_summary[-2]
    self.assertEqual(actual_loss_summary, expected_loss_summary)
    expected_metrics_summary = 'Metrics: {}'.format([x.name for x in metrics])
    actual_metrics_summary = model_summary[-1]
    self.assertEqual(actual_metrics_summary, expected_metrics_summary)

  def test_loss_weights_summary(self):
    keras_model = keras_model_builder()
    loss = loss_list_builder()
    loss_weights = [0.5, 1.0]
    model_summary = []
    task._print_keras_model_summary(
        keras_model=keras_model,
        loss=loss,
        loss_weights=loss_weights,
        print_fn=model_summary.append)
    model_summary_len = len(model_summary)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    tf_model_summary_len = len(tf_model_summary)

    self.assertEqual(model_summary_len, tf_model_summary_len + 2)
    expected_loss_summary = 'Losses: {}'.format([x.name for x in loss])
    actual_loss_summary = model_summary[-2]
    self.assertEqual(actual_loss_summary, expected_loss_summary)
    expected_loss_weights_summary = 'Loss Weights: {}'.format(loss_weights)
    actual_loss_weights_summary = model_summary[-1]
    self.assertEqual(actual_loss_weights_summary, expected_loss_weights_summary)

  def test_loss_weights_and_metrics_summary(self):
    keras_model = keras_model_builder()
    loss = loss_list_builder()
    loss_weights = [0.5, 1.0]
    metrics = metrics_builder()
    model_summary = []
    task._print_keras_model_summary(
        keras_model=keras_model,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
        print_fn=model_summary.append)
    model_summary_len = len(model_summary)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    tf_model_summary_len = len(tf_model_summary)

    self.assertEqual(model_summary_len, tf_model_summary_len + 3)
    expected_loss_summary = 'Losses: {}'.format([x.name for x in loss])
    actual_loss_summary = model_summary[-3]
    self.assertEqual(actual_loss_summary, expected_loss_summary)
    expected_loss_weights_summary = 'Loss Weights: {}'.format(loss_weights)
    actual_loss_weights_summary = model_summary[-2]
    self.assertEqual(actual_loss_weights_summary, expected_loss_weights_summary)
    expected_metrics_summary = 'Metrics: {}'.format([x.name for x in metrics])
    actual_metrics_summary = model_summary[-1]
    self.assertEqual(actual_metrics_summary, expected_metrics_summary)


class BaselineTaskTest(tf.test.TestCase):

  def test_task_constructs_with_tff_model_fn(self):
    x_type = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    input_spec = (x_type, x_type)

    def model_builder():
      return keras_utils.from_keras_model(
          keras_model=keras_model_builder(),
          loss=loss_builder(),
          metrics=metrics_builder(),
          input_spec=input_spec)

    baseline_task_data = create_task_data()
    baseline_task = task.BaselineTask(baseline_task_data, model_builder)

    actual_data_summary = []
    baseline_task.dataset_summary(print_fn=actual_data_summary.append)
    expected_data_summary = []
    baseline_task_data.summary(expected_data_summary.append)
    self.assertEqual(actual_data_summary, expected_data_summary)

    actual_model_summary = []
    baseline_task.model_summary(print_fn=actual_model_summary.append)
    expected_model_summary = ['Model input spec: {}'.format(input_spec)]
    self.assertEqual(actual_model_summary, expected_model_summary)

  def test_raises_on_different_data_and_model_spec(self):
    baseline_task_data = create_task_data()

    model_input_spec = (
        tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32, name=None),
        tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32, name=None),
    )
    inputs = tf.keras.layers.Input(shape=(4, 2))
    outputs = tf.keras.layers.Dense(
        2, kernel_initializer='ones', use_bias=False)(
            inputs)
    keras_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def tff_model_fn():
      return keras_utils.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=model_input_spec)

    with self.assertRaisesRegex(
        ValueError, 'The element type structure of task_datasets must match the'
        ' input spec of the tff.learning.Model provided.'):
      task.BaselineTask(baseline_task_data, tff_model_fn)

  def test_task_with_custom_model_summary(self):
    x_type = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    input_spec = (x_type, x_type)

    def model_builder():
      return keras_utils.from_keras_model(
          keras_model=keras_model_builder(),
          loss=loss_builder(),
          metrics=metrics_builder(),
          input_spec=input_spec)

    baseline_task_data = create_task_data()

    def model_summary_fn(print_fn):
      print_fn('This is a test')

    baseline_task = task.BaselineTask(
        baseline_task_data, model_builder, model_summary_fn=model_summary_fn)
    actual_model_summary = []
    baseline_task.model_summary(print_fn=actual_model_summary.append)
    expected_model_summary = ['This is a test']
    self.assertEqual(actual_model_summary, expected_model_summary)

  def construct_task_from_keras_model(self):
    baseline_task_data = create_task_data()
    keras_model = keras_model_builder()
    loss = loss_builder()
    baseline_task = task.BaselineTask.from_data_and_keras_model(
        task_datasets=baseline_task_data, keras_model=keras_model, loss=loss)
    actual_model_summary = []
    baseline_task.model_summary(print_fn=actual_model_summary.append)

    tf_model_summary = []
    keras_model.summary(print_fn=tf_model_summary.append)
    self.assertEqual(actual_model_summary[:-1], tf_model_summary)
    self.assertEqual(actual_model_summary[-1], 'Loss: loss')


if __name__ == '__main__':
  tf.test.main()
