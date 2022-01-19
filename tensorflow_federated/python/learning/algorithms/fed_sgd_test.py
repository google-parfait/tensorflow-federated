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
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.algorithms import fed_sgd
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

  def model_fn(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable=[
            tf.constant([[0.0], [0.0]]),
            tf.constant(0.0),
        ],
        non_trainable=[0.0])

  @parameterized.named_parameters(
      ('non-simulation', False),
      ('simulation', True),
  )
  @test_utils.skip_test_for_multi_gpu
  def test_client_update(self, simulation):
    dataset = self.dataset()
    client_update = fed_sgd._build_client_update(
        self.model_fn(), use_experimental_simulation_loop=simulation)
    client_result, model_output, stat_output = client_update(
        self.initial_weights(), dataset)

    # Both trainable parameters should have gradients, and we don't return the
    # non-trainable 'c'. Model deltas for squared error:
    self.assertAllClose(client_result.update, [[[-1.0], [0.0]], -1.0])
    self.assertAllClose(client_result.update_weight, 8.0)
    self.assertEqual(stat_output['num_examples'], 8.0)
    self.assertDictContainsSubset({
        'num_examples': 8,
    }, model_output)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    dataset = self.dataset()
    client_update = fed_sgd._build_client_update(self.model_fn())
    init_weights = self.initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs, _, _ = client_update(init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs.update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.update), [[[0.0], [0.0]], 0.0])

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  @test_utils.skip_test_for_multi_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    dataset = self.dataset()
    client_update = fed_sgd._build_client_update(
        self.model_fn(), use_experimental_simulation_loop=simulation)
    client_update(self.initial_weights(), dataset)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()


class FederatedSGDTest(test_case.TestCase, parameterized.TestCase):
  """Tests construction of FedSGD training process."""

  def test_raises_on_non_callable_model_fn(self):
    non_callable_model_fn = model_examples.LinearRegression()
    with self.assertRaisesRegex(TypeError, 'found non-callable'):
      fed_sgd.build_fed_sgd(non_callable_model_fn)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
      ('robust_aggregator', model_update_aggregator.robust_aggregator),
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
    fed_sgd.build_fed_sgd(
        model_fn=mock_model_fn, model_aggregator=aggregation_factory())
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 3)


if __name__ == '__main__':
  test_case.main()
