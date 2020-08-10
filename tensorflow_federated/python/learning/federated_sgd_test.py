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
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning import federated_sgd
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils


class FederatedSgdTest(test.TestCase, parameterized.TestCase):

  def dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.LinearRegression.make_batch(
            x=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            y=[[1.0], [1.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def model(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[
            tf.constant([[0.0], [0.0]]),
            tf.constant(0.0),
        ],
        non_trainable=[0.0])

  def test_client_tf(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(model)
    client_outputs = self.evaluate(client_tf(dataset, self.initial_weights()))

    # Both trainable parameters should have gradients, and we don't return the
    # non-trainable 'c'. Model deltas for squared error:
    self.assertAllClose(client_outputs.weights_delta, [[[1.0], [0.0]], 1.0])
    self.assertAllClose(client_outputs.weights_delta_weight, 8.0)

    self.assertEqual(
        client_outputs.model_output, {
            'num_examples': 8,
            'num_examples_float': 8.0,
            'num_batches': 3,
            'loss': 0.5,
        })
    self.assertEqual(client_outputs.optimizer_output, {
        'client_weight': 8.0,
        'has_non_finite_delta': 0,
    })

  def test_client_tf_custom_batch_weight(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(
        model, batch_weight_fn=lambda batch: 2.0 * tf.reduce_sum(batch.x))
    client_outputs = client_tf(dataset, self.initial_weights())
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight),
                     16.0)  # 2 * 8

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(model)
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(dataset, init_weights)
    self.assertEqual(self.evaluate(client_outputs.weights_delta_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.weights_delta), [[[0.0], [0.0]], 0.0])
    self.assertEqual(
        self.evaluate(client_outputs.optimizer_output['has_non_finite_delta']),
        1)


class FederatedSGDTffTest(test.TestCase, parameterized.TestCase):

  def test_orchestration_execute(self):
    iterative_process = federated_sgd.build_federated_sgd_process(
        model_fn=model_examples.LinearRegression)

    # Some data points along [x_1 + 2*x_2 + 3 = y], expecting to learn
    # kernel = [1, 2], bias = [3].
    ds1 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [0.0, 1.0]],
            y=[[3.0], [5.0]],
        )).batch(2)
    ds2 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0], [1.0, 0.0], [-1.0, -1.0]],
            y=[[8.0], [14.0], [4.00], [0.0]],
        )).batch(2)
    federated_ds = [ds1, ds2]

    server_state = iterative_process.initialize()

    prev_loss = np.inf
    num_iterations = 3
    for _ in range(num_iterations):
      server_state, metric_outputs = iterative_process.next(
          server_state, federated_ds)
      train_metrics = metric_outputs['train']
      self.assertEqual(train_metrics['num_examples'],
                       num_iterations * len(federated_ds))
      loss = train_metrics['loss']
      self.assertLess(loss, prev_loss)
      prev_loss = loss

  @parameterized.named_parameters([
      ('functional_model',
       model_examples.build_linear_regression_keras_functional_model),
      ('sequential_model',
       model_examples.build_linear_regression_keras_sequential_model),
  ])
  def test_orchestration_execute_from_keras(self, build_keras_model_fn):
    # Some data points along [x_1 + 2*x_2 + 3 = y], expecting to learn
    # kernel = [1, 2], bias = [3].
    ds1 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [0.0, 1.0]],
            y=[[3.0], [5.0]],
        )).batch(2)
    ds2 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0], [1.0, 0.0], [-1.0, -1.0]],
            y=[[8.0], [14.0], [4.00], [0.0]],
        )).batch(2)
    federated_ds = [ds1, ds2]

    def model_fn():
      # Note: we don't compile with an optimizer here; FedSGD does not use it.
      keras_model = build_keras_model_fn(feature_dims=2)
      return keras_utils.from_keras_model(
          keras_model,
          input_spec=ds1.element_spec,
          loss=tf.keras.losses.MeanSquaredError())

    iterative_process = federated_sgd.build_federated_sgd_process(
        model_fn=model_fn)

    server_state = iterative_process.initialize()
    prev_loss = np.inf
    for _ in range(0):
      server_state, metrics = iterative_process.next(server_state, federated_ds)
      self.assertLess(metrics.train.loss, prev_loss)
      prev_loss = metrics.train.loss

  def test_execute_empty_data(self):
    iterative_process = federated_sgd.build_federated_sgd_process(
        model_fn=model_examples.LinearRegression)

    # Results in empty dataset with correct types and shapes.
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(x=[[1.0, 2.0]], y=[[5.0]])).batch(
            5, drop_remainder=True)  # No batches of size 5 can be created.
    federated_ds = [ds] * 2

    server_state = iterative_process.initialize()
    first_state, metric_outputs = iterative_process.next(
        server_state, federated_ds)
    self.assertAllClose(
        list(first_state.model.trainable), [[[0.0], [0.0]], 0.0])
    self.assertEqual(
        list(metric_outputs.keys()), ['broadcast', 'aggregation', 'train'])
    self.assertEmpty(metric_outputs['broadcast'])
    self.assertEmpty(metric_outputs['aggregation'])
    self.assertEqual(metric_outputs['train']['num_examples'], 0)
    self.assertTrue(tf.math.is_nan(metric_outputs['train']['loss']))


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
