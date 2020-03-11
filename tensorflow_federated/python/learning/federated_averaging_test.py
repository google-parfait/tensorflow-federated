# Lint as: python3
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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils

tf.compat.v1.enable_v2_behavior()


class FederatedAveragingClientWithModelTest(test.TestCase,
                                            parameterized.TestCase):
  """Tests of _ClientFedAvg that use a common model and data."""

  def create_dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.LinearRegression.make_batch(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def create_model(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[tf.zeros((2, 1)), tf.constant(0.0)],
        non_trainable=[0.0],
    )

  def test_client_tf(self):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging._ClientFedAvg(
        model, tf.keras.optimizers.SGD(learning_rate=0.1))
    client_outputs = self.evaluate(client_tf(dataset, self.initial_weights()))

    # Both trainable parameters should have been updated,
    # and we don't return the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_outputs.weights_delta, axis=-1), 0.1)
    self.assertEqual(client_outputs.weights_delta_weight, 8.0)
    self.assertEqual(client_outputs.optimizer_output['num_examples'], 8)
    self.assertEqual(client_outputs.optimizer_output['has_non_finite_delta'], 0)

    self.assertDictContainsSubset(
        {
            'num_examples': 8,
            'num_examples_float': 8.0,
            'num_batches': 3,
        }, client_outputs.model_output)
    self.assertBetween(client_outputs.model_output['loss'],
                       np.finfo(np.float32).eps, 10.0)

  def test_client_tf_custom_delta_weight(self):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging._ClientFedAvg(
        model,
        tf.keras.optimizers.SGD(learning_rate=0.1),
        client_weight_fn=lambda _: tf.constant(1.5))
    client_outputs = client_tf(dataset, self.initial_weights())
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 1.5)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging._ClientFedAvg(
        model, tf.keras.optimizers.SGD(learning_rate=0.1))
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(dataset, init_weights)
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.weights_delta), [[[0.0], [0.0]], 0.0])
    self.assertEqual(
        self.evaluate(client_outputs.optimizer_output['has_non_finite_delta']),
        1)


class FederatedAveragingModelTffTest(test.TestCase, parameterized.TestCase):

  def test_orchestration_execute(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)
    federated_ds = [ds] * 3

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    for _ in range(3):
      server_state, metric_outputs = iterative_process.next(
          server_state, federated_ds)
      self.assertEqual(metric_outputs.num_examples, 2 * len(federated_ds))
      self.assertLess(metric_outputs.loss, prev_loss)
      prev_loss = metric_outputs.loss

  @parameterized.named_parameters([
      ('functional_model',
       model_examples.build_linear_regression_keras_functional_model),
      ('sequential_model',
       model_examples.build_linear_regression_keras_sequential_model),
  ])
  def test_orchestration_execute_from_keras(self, build_keras_model_fn):
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)

    def model_fn():
      keras_model = build_keras_model_fn(feature_dims=2)
      return keras_utils.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=ds.element_spec,
          metrics=[])

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01))
    federated_ds = [ds] * 3

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    for _ in range(3):
      server_state, metrics = iterative_process.next(server_state, federated_ds)
      self.assertLess(metrics.loss, prev_loss)
      prev_loss = metrics.loss

  def test_orchestration_execute_from_keras_with_lookup(self):
    self.skipTest('https://github.com/tensorflow/federated/issues/783')

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[['R'], ['G'], ['B']], y=[[1.0], [2.0], [3.0]])).batch(2)

    def model_fn():
      keras_model = model_examples.build_lookup_table_keras_model()
      return keras_utils.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=ds.element_spec,
          metrics=[])

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    federated_ds = [ds] * 3

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    for _ in range(3):
      server_state, metrics = iterative_process.next(server_state, federated_ds)
      self.assertLess(metrics.loss, prev_loss)
      prev_loss = metrics.loss

  def test_execute_empty_data(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    # Results in empty dataset with correct types and shapes.
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0]],
            y=[[5.0]],
        )).batch(
            5, drop_remainder=True)
    federated_ds = [ds] * 2

    server_state = iterative_process.initialize()

    first_state, metric_outputs = iterative_process.next(
        server_state, federated_ds)
    self.assertAllClose(
        list(first_state.model.trainable), [[[0.0], [0.0]], 0.0])
    self.assertEqual(metric_outputs.num_examples, 0)
    self.assertTrue(tf.math.is_nan(metric_outputs.loss))


if __name__ == '__main__':
  test.main()
