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
"""Tests for local client training implemented in ClientFedAvg."""

import collections
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce


class FederatedAveragingClientTest(tf.test.TestCase, parameterized.TestCase):
  """Tests of ClientFedAvg that use a common model and data."""

  def create_dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
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

  @parameterized.named_parameters(
      ('non-simulation_noclip', True, False, {}, 0.1),
      ('unweighted_non-simulation_noclip', False, False, {}, 0.1),
      ('simulation_noclip', True, True, {}, 0.1),
      ('non-simulation_clipnorm', True, False, {
          'clipnorm': 0.2
      }, 0.05), ('non-simulation_clipvalue', True, False, {
          'clipvalue': 0.1
      }, 0.02))
  def test_client_tf(self, weighted, simulation, optimizer_kwargs,
                     expected_norm):
    model = self.create_model()
    dataset = self.create_dataset()
    if weighted:
      client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES
    else:
      client_weighting = client_weight_lib.ClientWeighting.UNIFORM
    client_tf = federated_averaging.ClientFedAvg(
        model,
        lambda: tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs),
        client_weighting=client_weighting,
        use_experimental_simulation_loop=simulation)
    client_outputs = self.evaluate(client_tf(dataset, self.initial_weights()))
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_outputs.weights_delta, axis=-1), expected_norm)
    if weighted:
      self.assertEqual(client_outputs.weights_delta_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8},
                                  client_outputs.model_output)
    self.assertBetween(client_outputs.model_output['loss'][0],
                       np.finfo(np.float32).eps, 10.0)
    self.assertEqual(client_outputs.model_output['loss'][1], 8.0)

  def test_client_tf_custom_delta_weight(self):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model,
        lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        client_weighting=lambda _: tf.constant(1.5))
    client_outputs = client_tf(dataset, self.initial_weights())
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 1.5)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model, lambda: tf.keras.optimizers.SGD(learning_rate=0.1))
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(dataset, init_weights)
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.weights_delta), [[[0.0], [0.0]], 0.0])

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    model = self.create_model()
    dataset = self.create_dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model,
        lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        use_experimental_simulation_loop=simulation)
    client_tf(dataset, self.initial_weights())
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

if __name__ == '__main__':
  tf.test.main()
