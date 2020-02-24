# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for shared training utilities."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule
from tensorflow_federated.python.research.utils import training_utils


def model_builder():
  # Create a simple linear regression model, single output.
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(1,))
  ])
  return model


@tf.function
def get_sample_batch():
  return next(iter(create_tf_dataset_for_client(0)))


def create_tf_dataset_for_client(client_id, batch_data=True):
  # Create client data for y = 2*x+3
  np.random.seed(client_id)
  x = np.random.rand(6, 1).astype(np.float32)
  y = 2 * x + 3
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict([('x', x), ('y', y)]))
  if batch_data:
    dataset = dataset.batch(2)
  return dataset


class TrainingUtilsTest(tf.test.TestCase):

  def test_build_client_datasets_fn(self):
    tff_dataset = tff.simulation.client_data.ConcreteClientData(
        [2], create_tf_dataset_for_client)
    client_datasets_fn = training_utils.build_client_datasets_fn(tff_dataset, 1)
    client_datasets = client_datasets_fn(7)
    sample_batch = next(iter(client_datasets[0]))
    reference_batch = next(iter(create_tf_dataset_for_client(2)))
    self.assertAllClose(sample_batch, reference_batch)

  def test_build_evaluate_fn(self):

    loss_builder = tf.keras.losses.MeanSquaredError
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    def tff_model_fn():
      return tff.learning.from_keras_model(
          keras_model=model_builder(),
          dummy_batch=get_sample_batch(),
          loss=loss_builder(),
          metrics=metrics_builder())

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        tff_model_fn, client_optimizer_fn=tf.keras.optimizers.SGD)

    state = iterative_process.initialize()

    test_dataset = create_tf_dataset_for_client(1)

    evaluate_fn = training_utils.build_evaluate_fn(test_dataset, model_builder,
                                                   loss_builder,
                                                   metrics_builder)
    test_metrics = evaluate_fn(state)
    self.assertIn('loss', test_metrics)

  def test_tuple_conversion_from_tuple_datset(self):
    x = np.random.rand(6, 1)
    y = 2 * x + 3
    tuple_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    converted_dataset = training_utils.convert_to_tuple_dataset(tuple_dataset)
    tuple_batch = next(iter(tuple_dataset))
    converted_batch = next(iter(converted_dataset))
    self.assertAllClose(tuple_batch, converted_batch)

  def test_tuple_conversion_from_dict(self):
    x = np.random.rand(6, 1)
    y = 2 * x + 3
    tuple_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dict_dataset = tf.data.Dataset.from_tensor_slices({'x': x, 'y': y})
    converted_dataset = training_utils.convert_to_tuple_dataset(dict_dataset)
    tuple_batch = next(iter(tuple_dataset))
    converted_batch = next(iter(converted_dataset))
    self.assertAllClose(tuple_batch, converted_batch)

  def test_tuple_conversion_from_ordered_dict(self):
    x = np.random.rand(6, 1)
    y = 2 * x + 3
    tuple_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    ordered_dict_dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([('x', x), ('y', y)]))
    converted_dataset = training_utils.convert_to_tuple_dataset(
        ordered_dict_dataset)
    tuple_batch = next(iter(tuple_dataset))
    converted_batch = next(iter(converted_dataset))
    self.assertAllClose(tuple_batch, converted_batch)

  def test_tuple_conversion_from_named_tuple(self):
    x = np.random.rand(6, 1)
    y = 2 * x + 3
    tuple_dataset = tf.data.Dataset.from_tensor_slices((x, y))

    example_tuple = collections.namedtuple('Example', ['x', 'y'])
    named_tuple_dataset = tf.data.Dataset.from_tensor_slices(
        example_tuple(x=x, y=y))
    converted_dataset = training_utils.convert_to_tuple_dataset(
        named_tuple_dataset)
    tuple_batch = next(iter(tuple_dataset))
    converted_batch = next(iter(converted_dataset))
    self.assertAllClose(tuple_batch, converted_batch)

  # TODO(b/143440780): Add more robust tests for dataset tuple conversion.


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
