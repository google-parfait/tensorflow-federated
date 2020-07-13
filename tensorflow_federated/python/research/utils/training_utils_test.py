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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

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


def get_input_spec():
  return create_tf_dataset_for_client(0).element_spec


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


class TrainingUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '_int_no_replace',
          'a': 100,
          'replace': False
      }, {
          'testcase_name': '_int_replace',
          'a': 5,
          'replace': True
      }, {
          'testcase_name': '_sequence_no_replace',
          'a': [str(i) for i in range(100)],
          'replace': False
      }, {
          'testcase_name': '_sequence_replace',
          'a': [str(i) for i in range(5)],
          'replace': True
      })
  def test_build_sample_fn_with_random_seed(self, a, replace):
    size = 10
    random_seed = 1
    round_num = 5

    sample_fn_1 = training_utils.build_sample_fn(
        a, size, replace=replace, random_seed=random_seed)
    sample_1 = sample_fn_1(round_num)

    sample_fn_2 = training_utils.build_sample_fn(
        a, size, replace=replace, random_seed=random_seed)
    sample_2 = sample_fn_2(round_num)

    self.assertAllEqual(sample_1, sample_2)

  @parameterized.named_parameters(
      {
          'testcase_name': '_int_no_replace',
          'a': 100,
          'replace': False
      }, {
          'testcase_name': '_int_replace',
          'a': 5,
          'replace': True
      }, {
          'testcase_name': '_sequence_no_replace',
          'a': [str(i) for i in range(100)],
          'replace': False
      }, {
          'testcase_name': '_sequence_replace',
          'a': [str(i) for i in range(5)],
          'replace': True
      })
  def test_build_sample_fn_without_random_seed(self, a, replace):
    size = 10
    round_num = 5

    sample_fn_1 = training_utils.build_sample_fn(a, size, replace=replace)
    sample_1 = sample_fn_1(round_num)

    sample_fn_2 = training_utils.build_sample_fn(a, size, replace=replace)
    sample_2 = sample_fn_2(round_num)

    self.assertNotAllEqual(sample_1, sample_2)

  def test_build_client_datasets_fn(self):
    tff_dataset = tff.simulation.client_data.ConcreteClientData(
        [2], create_tf_dataset_for_client)
    client_datasets_fn = training_utils.build_client_datasets_fn(
        tff_dataset, train_clients_per_round=1)
    client_datasets = client_datasets_fn(round_num=7)
    sample_batch = next(iter(client_datasets[0]))
    reference_batch = next(iter(create_tf_dataset_for_client(2)))
    self.assertAllClose(sample_batch, reference_batch)

  def test_client_datasets_fn_with_random_seed(self):
    tff_dataset = tff.simulation.client_data.ConcreteClientData(
        [0, 1, 2, 3, 4], create_tf_dataset_for_client)

    client_datasets_fn_1 = training_utils.build_client_datasets_fn(
        tff_dataset, train_clients_per_round=1, random_seed=363)
    client_datasets_1 = client_datasets_fn_1(round_num=5)
    sample_batch_1 = next(iter(client_datasets_1[0]))

    client_datasets_fn_2 = training_utils.build_client_datasets_fn(
        tff_dataset, train_clients_per_round=1, random_seed=363)
    client_datasets_2 = client_datasets_fn_2(round_num=5)
    sample_batch_2 = next(iter(client_datasets_2[0]))

    self.assertAllClose(sample_batch_1, sample_batch_2)

  def test_different_random_seed_give_different_clients(self):
    tff_dataset = tff.simulation.client_data.ConcreteClientData(
        list(range(100)), create_tf_dataset_for_client)

    client_datasets_fn_1 = training_utils.build_client_datasets_fn(
        tff_dataset, train_clients_per_round=50, random_seed=1)
    client_datasets_1 = client_datasets_fn_1(round_num=1001)
    sample_batches_1 = [
        next(iter(client_dataset)) for client_dataset in client_datasets_1
    ]

    client_datasets_fn_2 = training_utils.build_client_datasets_fn(
        tff_dataset, train_clients_per_round=50, random_seed=2)
    client_datasets_2 = client_datasets_fn_2(round_num=1001)
    sample_batches_2 = [
        next(iter(client_dataset)) for client_dataset in client_datasets_2
    ]

    self.assertNotAllClose(sample_batches_1, sample_batches_2)

  def test_client_datasets_fn_without_random_seed(self):
    tff_dataset = tff.simulation.client_data.ConcreteClientData(
        list(range(100)), create_tf_dataset_for_client)
    client_datasets_fn = training_utils.build_client_datasets_fn(
        tff_dataset, train_clients_per_round=50)
    client_datasets_1 = client_datasets_fn(round_num=0)
    sample_batches_1 = [
        next(iter(client_dataset)) for client_dataset in client_datasets_1
    ]

    client_datasets_2 = client_datasets_fn(round_num=0)
    sample_batches_2 = [
        next(iter(client_dataset)) for client_dataset in client_datasets_2
    ]
    self.assertNotAllClose(sample_batches_1, sample_batches_2)

  def test_build_evaluate_fn(self):

    loss_builder = tf.keras.losses.MeanSquaredError
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    def tff_model_fn():
      return tff.learning.from_keras_model(
          keras_model=model_builder(),
          input_spec=get_input_spec(),
          loss=loss_builder(),
          metrics=metrics_builder())

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn, client_optimizer_fn=tf.keras.optimizers.SGD)
    state = iterative_process.initialize()
    test_dataset = create_tf_dataset_for_client(1)

    reference_model = tff.learning.ModelWeights(
        trainable=list(state.model.trainable),
        non_trainable=list(state.model.non_trainable))

    def assign_weights_to_keras_model(model, keras_model):
      model.assign_weights_to(keras_model)

    evaluate_fn = training_utils.build_evaluate_fn(
        test_dataset, model_builder, loss_builder, metrics_builder,
        assign_weights_to_keras_model)

    test_metrics = evaluate_fn(reference_model)
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
  tf.test.main()
