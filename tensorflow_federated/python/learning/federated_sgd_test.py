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
"""Tests for local client training implemented in ClientSgd.

Integration tests that include server averaging and alternative tff.aggregator
factories are in found in
tensorflow_federated/python/tests/federated_sgd_integration_test.py.
"""

import collections
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import federated_sgd
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce


class FederatedSgdTest(test_case.TestCase, parameterized.TestCase):

  def dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
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

  def test_clietsgd_fails_for_non_tff_model(self):
    keras_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    with self.assertRaisesRegex(TypeError, 'Model'):
      federated_sgd.ClientSgd(keras_model)

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


class FederatedSGDTest(test_case.TestCase, parameterized.TestCase):
  """Tests construction of FedSGD training process."""

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
      ('robust_aggregator', model_update_aggregator.robust_aggregator),
      ('dp_aggregator', lambda: model_update_aggregator.dp_aggregator(1e-3, 3)),
      ('compression_aggregator',
       model_update_aggregator.compression_aggregator),
      ('secure_aggreagtor', model_update_aggregator.secure_aggregator),
  )
  # pylint: enable=g-complex-comprehension
  def test_construction_calls_model_fn(self, aggregation_factory):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    federated_sgd.build_federated_sgd_process(
        model_fn=mock_model_fn,
        model_update_aggregation_factory=aggregation_factory())
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 3)


if __name__ == '__main__':
  test_case.main()
