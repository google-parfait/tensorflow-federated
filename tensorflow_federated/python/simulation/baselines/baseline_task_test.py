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

import keras
import tensorflow as tf

from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.simulation.baselines import baseline_task
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
      client_ids=['1', '2', '3'], serializable_dataset_fn=create_dataset
  )


def create_task_data():
  return task_data.BaselineTaskDatasets(
      train_data=create_client_data(), test_data=create_client_data()
  )


def keras_model_builder():
  inputs = keras.layers.Input(shape=(1,), name='input_layer')
  outputs = keras.layers.Dense(
      1, kernel_initializer='ones', use_bias=False, name='dense_layer'
  )(inputs)
  return keras.models.Model(inputs=inputs, outputs=outputs, name='model')


def tff_model_builder():
  x_type = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
  input_spec = (x_type, x_type)
  keras_model = keras_model_builder()
  loss = keras.losses.MeanSquaredError()
  return functional.functional_model_from_keras(
      keras_model, loss_fn=loss, input_spec=input_spec
  )


class TaskUtilsTest(tf.test.TestCase):

  def test_constructs_with_matching_dataset_and_model_spec(self):
    baseline_task_spec = baseline_task.BaselineTask(
        create_task_data(), tff_model_builder()
    )
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  def test_construct_raises_on_non_baseline_task_spec_datasets(self):
    with self.assertRaises(TypeError):
      baseline_task.BaselineTask(create_dataset(0), tff_model_builder())

  def test_construct_raises_on_non_tff_model(self):
    with self.assertRaises(TypeError):
      baseline_task.BaselineTask(create_task_data(), keras_model_builder())

  def test_raises_on_different_data_and_model_spec(self):
    baseline_task_spec_data = create_task_data()
    model_input_spec = (
        tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32, name=None),
        tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32, name=None),
    )
    inputs = keras.layers.Input(shape=(4, 2))
    outputs = keras.layers.Dense(2, kernel_initializer='ones', use_bias=False)(
        inputs
    )
    keras_model = keras.models.Model(inputs=inputs, outputs=outputs)
    tff_model = functional.functional_model_from_keras(
        keras_model,
        loss_fn=keras.losses.MeanSquaredError(),
        input_spec=model_input_spec,
    )

    with self.assertRaisesRegex(
        ValueError, 'Dataset element spec and model input spec do not match'
    ):
      baseline_task.BaselineTask(baseline_task_spec_data, tff_model)


if __name__ == '__main__':
  tf.test.main()
