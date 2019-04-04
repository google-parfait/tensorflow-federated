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
"""Tests for learning.federated_averaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils


class FederatedAveragingClientTest(test.TestCase, parameterized.TestCase):
  """Tests of ClientFedAvg that use a common model and data."""

  def dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.TrainableLinearRegression.make_batch(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def model(self):
    return model_examples.TrainableLinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable={
            'a': tf.constant([[0.0], [0.0]]),
            'b': tf.constant(0.0)
        },
        non_trainable={'c': 0.0})

  @test.graph_mode_test
  def test_client_tf(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_averaging.ClientFedAvg(model)
    init_op = tf.group(
        model_utils.model_initializer(model),
        tf.variables_initializer(client_tf.variables),
        name='fedavg_initializer')
    client_outputs = client_tf(dataset, self.initial_weights())

    tf.get_default_graph().finalize()
    with self.session() as sess:
      sess.run(init_op)
      out = sess.run(client_outputs)

      # Both trainable parameters should have been updated,
      # and we don't return the non-trainable 'c'.
      self.assertCountEqual(['a', 'b'], list(out.weights_delta.keys()))
      self.assertGreater(np.linalg.norm(out.weights_delta['a']), 0.1)
      self.assertGreater(np.linalg.norm(out.weights_delta['b']), 0.1)
      self.assertEqual(out.weights_delta_weight, 8.0)
      self.assertEqual(out.optimizer_output['num_examples'], 8)
      self.assertEqual(out.optimizer_output['has_non_finite_delta'], 0)

      self.assertEqual(out.model_output['num_examples'], 8)
      self.assertEqual(out.model_output['num_batches'], 3)
      self.assertBetween(out.model_output['loss'],
                         np.finfo(np.float32).eps, 10.0)

  def test_client_tf_custom_delta_weight(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model, client_weight_fn=lambda _: tf.constant(1.5))
    out = client_tf(dataset, self.initial_weights())
    self.assertEqual(self.evaluate(out.weights_delta_weight), 1.5)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_averaging.ClientFedAvg(model)
    init_weights = self.initial_weights()
    init_weights.trainable['b'] = bad_value
    out = client_tf(dataset, init_weights)
    self.assertEqual(self.evaluate(out.weights_delta_weight), 0.0)
    self.assertAllClose(
        self.evaluate(out.weights_delta['a']), np.array([[0.0], [0.0]]))
    self.assertAllClose(self.evaluate(out.weights_delta['b']), 0.0)
    self.assertEqual(
        self.evaluate(out.optimizer_output['has_non_finite_delta']), 1)


class FederatedAveragingTffTest(test.TestCase, parameterized.TestCase):

  def test_orchestration_execute(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.TrainableLinearRegression)

    ds = tf.data.Dataset.from_tensor_slices({
        'x': [[1., 2.], [3., 4.]],
        'y': [[5.], [6.]]
    }).batch(2)

    federated_ds = [ds] * 3

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    for _ in range(3):
      server_state, metric_outputs = iterative_process.next(
          server_state, federated_ds)
      self.assertEqual(metric_outputs.num_examples, 2 * len(federated_ds))
      self.assertLess(metric_outputs.loss, prev_loss)
      prev_loss = metric_outputs.loss

  @parameterized.parameters([
      model_examples.build_linear_regresion_keras_functional_model,
      model_examples.build_linear_regresion_keras_sequential_model,
      model_examples.build_linear_regresion_keras_subclass_model,
  ])
  def test_orchestration_execute_from_keras(self, build_keras_model_fn):
    dummy_batch = collections.OrderedDict([
        ('x', np.zeros([1, 2], np.float32)),
        ('y', np.zeros([1, 1], np.float32)),
    ])

    def model_fn():
      keras_model = build_keras_model_fn(feature_dims=2)
      keras_model.compile(
          optimizer=gradient_descent.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[])
      return model_utils.from_compiled_keras_model(keras_model, dummy_batch)

    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_fn)

    ds = tf.data.Dataset.from_tensor_slices({
        'x': [[1., 2.], [3., 4.]],
        'y': [[5.], [6.]]
    }).batch(2)
    federated_ds = [ds] * 3

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    for _ in range(3):
      server_state, metrics = iterative_process.next(server_state, federated_ds)
      self.assertLess(metrics.loss, prev_loss)
      prev_loss = metrics.loss

  def test_execute_empty_data(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.TrainableLinearRegression)

    # Results in empty dataset with correct types and shapes.
    ds = tf.data.Dataset.from_tensor_slices({
        'x': [[1., 2.]],
        'y': [[5.]]
    }).batch(
        5, drop_remainder=True)

    federated_ds = [ds] * 2

    server_state = iterative_process.initialize()

    first_state, metric_outputs = iterative_process.next(
        server_state, federated_ds)
    self.assertEqual(
        self.evaluate(tf.reduce_sum(first_state.model.trainable.a)) +
        self.evaluate(tf.reduce_sum(first_state.model.trainable.b)), 0)
    self.assertEqual(metric_outputs.num_examples, 0)
    self.assertTrue(tf.is_nan(metric_outputs.loss))


if __name__ == '__main__':
  # We default to TF 2.0 behavior, including eager execution, and use the
  # @graph_mode_test annotation for graph-mode (sess.run) tests.
  tf.compat.v1.enable_v2_behavior()
  test.main()
