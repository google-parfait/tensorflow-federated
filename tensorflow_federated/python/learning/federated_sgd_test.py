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
from unittest import mock
import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import federated_sgd
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce


class FederatedSgdTest(test_case.TestCase, parameterized.TestCase):

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

  @parameterized.named_parameters(
      ('non-simulation_weighted', False, True),
      ('non-simulation_unweighted', False, False),
      ('simulation_weighted', True, True),
      ('simulation_unweighted', True, False),
  )
  @test_utils.skip_test_for_multi_gpu
  def test_client_tf(self, simulation, weighted):
    model = self.model()
    dataset = self.dataset()
    if weighted:
      client_weighting = client_weight_lib.ClientWeighting.NUM_EXAMPLES
    else:
      client_weighting = client_weight_lib.ClientWeighting.UNIFORM
    client_tf = federated_sgd.ClientSgd(
        model,
        client_weighting=client_weighting,
        use_experimental_simulation_loop=simulation)
    client_outputs = self.evaluate(client_tf(dataset, self.initial_weights()))

    # Both trainable parameters should have gradients, and we don't return the
    # non-trainable 'c'. Model deltas for squared error:
    self.assertAllClose(client_outputs.weights_delta, [[[1.0], [0.0]], 1.0])
    if weighted:
      self.assertAllClose(client_outputs.weights_delta_weight, 8.0)
    else:
      self.assertAllClose(client_outputs.weights_delta_weight, 1.0)

    self.assertDictContainsSubset(
        client_outputs.model_output, {
            'num_examples': 8,
            'num_examples_float': 8.0,
            'num_batches': 3,
            'loss': 0.5,
        })
    self.assertEqual(client_outputs.optimizer_output['has_non_finite_delta'], 0)

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

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  @test_utils.skip_test_for_multi_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(
        model, use_experimental_simulation_loop=simulation)
    client_tf(dataset, self.initial_weights())
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()


class FederatedSGDTffTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('unweighted', client_weight_lib.ClientWeighting.UNIFORM),
      ('example_weighted', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('custom_weighted', lambda _: tf.constant(1.5)),
  ])
  @test_utils.skip_test_for_multi_gpu
  def test_orchestration_execute(self, client_weighting):
    iterative_process = federated_sgd.build_federated_sgd_process(
        model_fn=model_examples.LinearRegression,
        client_weighting=client_weighting)

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
  @test_utils.skip_test_for_multi_gpu
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

  @test_utils.skip_test_for_multi_gpu
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
        list(metric_outputs.keys()),
        ['broadcast', 'aggregation', 'train', 'stat'])
    self.assertEmpty(metric_outputs['broadcast'])
    self.assertEqual(metric_outputs['aggregation'],
                     collections.OrderedDict(mean_value=(), mean_weight=()))
    self.assertEqual(metric_outputs['train']['num_examples'], 0)
    self.assertTrue(tf.math.is_nan(metric_outputs['train']['loss']))

  @test_utils.skip_test_for_multi_gpu
  def test_get_model_weights(self):
    iterative_process = federated_sgd.build_federated_sgd_process(
        model_fn=model_examples.LinearRegression)

    num_clients = 3
    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)
    datasets = [ds] * num_clients

    state = iterative_process.initialize()
    self.assertIsInstance(
        iterative_process.get_model_weights(state), model_utils.ModelWeights)
    self.assertAllClose(state.model.trainable,
                        iterative_process.get_model_weights(state).trainable)

    for _ in range(3):
      state, _ = iterative_process.next(state, datasets)
      self.assertIsInstance(
          iterative_process.get_model_weights(state), model_utils.ModelWeights)
      self.assertAllClose(state.model.trainable,
                          iterative_process.get_model_weights(state).trainable)

  def test_aggregation_process_deprecation(self):
    aggregation_process = mean.MeanFactory().create(
        computation_types.to_type([(tf.float32, (2, 1)), tf.float32]),
        computation_types.TensorType(tf.float32))
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      federated_sgd.build_federated_sgd_process(
          model_fn=model_examples.LinearRegression,
          aggregation_process=aggregation_process)
      self.assertNotEmpty(w)
      self.assertEqual(w[0].category, DeprecationWarning)
      self.assertRegex(
          str(w[0].message), 'aggregation_process .* is deprecated')

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    federated_sgd.build_federated_sgd_process(model_fn=mock_model_fn)
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 4)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
