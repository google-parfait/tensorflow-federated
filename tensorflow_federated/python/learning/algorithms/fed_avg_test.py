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
import itertools
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.algorithms import fed_avg
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import distributors


class ClientFedAvgTest(test_case.TestCase, parameterized.TestCase):
  """Tests of the client work of FedAvg using a common model and data."""

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
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES))
  def test_keras_tff_client_work_equal(self, weighting):
    dataset = self.create_dataset()
    client_update_keras = fed_avg.build_client_update_with_keras_optimizer(
        model_fn=self.create_model, weighting=weighting)
    client_update_tff = fed_avg.build_client_update_with_tff_optimizer(
        model_fn=self.create_model, weighting=weighting)
    keras_result = client_update_keras(
        tf.keras.optimizers.SGD(learning_rate=0.1), self.initial_weights(),
        dataset)
    tff_result = client_update_tff(
        sgdm.build_sgdm(learning_rate=0.1), self.initial_weights(), dataset)
    self.assertAllClose(keras_result[0].update, tff_result[0].update)
    self.assertEqual(keras_result[0].update_weight, tff_result[0].update_weight)
    self.assertAllClose(keras_result[1], tff_result[1])
    self.assertDictEqual(keras_result[2], tff_result[2])

  @parameterized.named_parameters(
      ('non-simulation_noclip_uniform', False, {}, 0.1,
       client_weight_lib.ClientWeighting.UNIFORM),
      ('non-simulation_noclip_num_examples', False, {}, 0.1,
       client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('simulation_noclip_uniform', True, {}, 0.1,
       client_weight_lib.ClientWeighting.UNIFORM),
      ('simulation_noclip_num_examples', True, {}, 0.1,
       client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('non-simulation_clipnorm', False, {
          'clipnorm': 0.2
      }, 0.05, client_weight_lib.ClientWeighting.NUM_EXAMPLES),
      ('non-simulation_clipvalue', False, {
          'clipvalue': 0.1
      }, 0.02, client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_client_tf(self, simulation, optimizer_kwargs, expected_norm,
                     weighting):
    client_tf = fed_avg.build_client_update_with_keras_optimizer(
        model_fn=self.create_model,
        weighting=weighting,
        use_experimental_simulation_loop=simulation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    dataset = self.create_dataset()
    client_result, model_output, stat_output = self.evaluate(
        client_tf(optimizer, self.initial_weights(), dataset))
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    self.assertAllGreater(
        np.linalg.norm(client_result.update, axis=-1), expected_norm)
    if weighting == client_weight_lib.ClientWeighting.UNIFORM:
      self.assertEqual(client_result.update_weight, 1.0)
    else:
      self.assertEqual(client_result.update_weight, 8.0)
    self.assertEqual(stat_output['num_examples'], 8)
    self.assertDictContainsSubset(
        {
            'num_examples': 8,
            'num_examples_float': 8.0,
            'num_batches': 3,
        }, model_output)
    self.assertBetween(model_output['loss'], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = fed_avg.build_client_update_with_keras_optimizer(
        model_fn=self.create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = self.create_dataset()
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(optimizer, init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs[0].update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs[0].update), [[[0.0], [0.0]], 0.0])

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    client_tf = fed_avg.build_client_update_with_keras_optimizer(
        model_fn=self.create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        use_experimental_simulation_loop=simulation)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = self.create_dataset()
    client_tf(optimizer, self.initial_weights(), dataset)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()


class FedAvgTest(test_case.TestCase, parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters((
      '_'.join(name for name, _ in named_params),
      *(param for _, param in named_params),
  ) for named_params in itertools.product([
      ('keras_optimizer', tf.keras.optimizers.SGD),
      ('tff_optimizer', sgdm.build_sgdm(learning_rate=0.1)),
  ], [
      ('robust_aggregator', model_update_aggregator.robust_aggregator),
      ('compression_aggregator',
       model_update_aggregator.compression_aggregator),
      ('secure_aggreagtor', model_update_aggregator.secure_aggregator),
  ]))
  # pylint: enable=g-complex-comprehension
  def test_construction_calls_model_fn(self, optimizer_fn, aggregation_factory):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_avg.weighted_fed_avg(
        model_fn=mock_model_fn,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory())
    self.assertEqual(mock_model_fn.call_count, 3)

  @mock.patch('tensorflow_federated.python.learning.'
              'algorithms.fed_avg.weighted_fed_avg')
  def test_weighted_fed_avg_called_by_unweighted_fed_avg(self, mock_fed_avg):
    fed_avg.unweighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0))
    self.assertEqual(mock_fed_avg.call_count, 1)

  @mock.patch('tensorflow_federated.python.learning.'
              'algorithms.fed_avg.weighted_fed_avg')
  @mock.patch('tensorflow_federated.python.learning.'
              'algorithms.aggregation.as_weighted_aggregator')
  def test_aggregation_wrapper_called_by_unweighted(self, _, mock_as_weighted):
    fed_avg.unweighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0))
    self.assertEqual(mock_as_weighted.call_count, 1)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      fed_avg.weighted_fed_avg(
          model_fn=model_examples.LinearRegression(),
          client_optimizer_fn=tf.keras.optimizers.SGD)

  def test_raises_on_invalid_client_weighting(self):
    with self.assertRaises(TypeError):
      fed_avg.weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          client_weighting='uniform')

  def test_raises_on_invalid_distributor(self):
    model_weights_type = type_conversions.type_from_tensors(
        model_utils.ModelWeights.from_model(model_examples.LinearRegression()))
    distributor = distributors.build_broadcast_process(model_weights_type)
    invalid_distributor = iterative_process.IterativeProcess(
        distributor.initialize, distributor.next)
    with self.assertRaises(TypeError):
      fed_avg.weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_distributor=invalid_distributor)

  def test_weighted_fed_avg_raises_on_unweighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      fed_avg.weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=aggregator)

  def test_unweighted_fed_avg_raises_on_weighted_aggregator(self):
    aggregator = model_update_aggregator.robust_aggregator(weighted=True)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      fed_avg.unweighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=aggregator)


if __name__ == '__main__':
  test_case.main()
