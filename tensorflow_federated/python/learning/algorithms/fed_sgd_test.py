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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_sgd
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.tensorflow_libs import tensorflow_test_utils


def _dataset() -> tf.data.Dataset:
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
          y=[[1.0], [1.0], [1.0], [1.0]],
      )
  )
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  # Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(2).batch(3)


def _model_fn() -> model_examples.LinearRegression:
  return model_examples.LinearRegression(feature_dim=2)


def _model_fn_unconnected() -> model_examples.LinearRegression:
  return model_examples.LinearRegression(feature_dim=2, has_unconnected=True)


def _build_functional_model() -> functional.FunctionalModel:
  return test_models.build_functional_linear_regression(feature_dim=2)


def _build_functional_model_unconnected() -> functional.FunctionalModel:
  return test_models.build_functional_linear_regression(
      feature_dim=2, has_unconnected=True
  )


def _initial_weights() -> model_weights.ModelWeights:
  return model_weights.ModelWeights(
      trainable=[
          tf.constant([[0.0], [0.0]]),
          tf.constant(0.0),
      ],
      non_trainable=[0.0],
  )


def _initial_weights_unconnected() -> model_weights.ModelWeights:
  return model_weights.ModelWeights(
      trainable=[
          tf.constant([[0.0], [0.0]]),
          tf.constant(0.0),
          tf.constant(0.0),
      ],
      non_trainable=[0.0],
  )


class FederatedSgdTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('non-simulation', False),
      ('simulation', True),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_update(self, simulation):
    dataset = _dataset()
    client_update = fed_sgd._build_client_update(
        _model_fn(), use_experimental_simulation_loop=simulation
    )
    client_result, model_output = client_update(_initial_weights(), dataset)

    # Both trainable parameters should have gradients, and we don't return the
    # non-trainable 'c'. Model deltas for squared error:
    self.assertAllClose(client_result.update, [[[-1.0], [0.0]], -1.0])
    self.assertAllClose(client_result.update_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8}, model_output)

  @parameterized.named_parameters(
      ('non-simulation', False),
      ('simulation', True),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_update_with_unconnected_weights(self, simulation):
    dataset = _dataset()
    client_update = fed_sgd._build_client_update(
        _model_fn_unconnected(), use_experimental_simulation_loop=simulation
    )
    client_result, model_output = client_update(
        _initial_weights_unconnected(), dataset
    )

    # Both trainable parameters should have gradients, and we don't return the
    # non-trainable 'c'. Model deltas for squared error:
    self.assertAllClose(client_result.update, [[[-1.0], [0.0]], -1.0, 0.0])
    self.assertAllClose(client_result.update_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8}, model_output)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    dataset = _dataset()
    client_update = fed_sgd._build_client_update(
        _model_fn(), use_experimental_simulation_loop=False
    )
    init_weights = _initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs, _ = client_update(init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs.update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.update), [[[0.0], [0.0]], 0.0]
    )

  @parameterized.named_parameters(
      ('non-simulation', False), ('simulation', True)
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    dataset = _dataset()
    client_update = fed_sgd._build_client_update(
        _model_fn(), use_experimental_simulation_loop=simulation
    )
    client_update(_initial_weights(), dataset)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('robust_aggregator', model_update_aggregator.robust_aggregator),
      (
          'compression_aggregator',
          model_update_aggregator.compression_aggregator,
      ),
      ('secure_aggreagtor', model_update_aggregator.secure_aggregator),
  )
  def test_construction_calls_model_fn(self, aggregation_factory):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_sgd.build_fed_sgd(
        model_fn=mock_model_fn, model_aggregator=aggregation_factory()
    )
    # TODO: b/186451541 - reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 3)

  def test_no_unsecure_aggregation_with_secure_aggregator(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_sgd.build_fed_sgd(
        model_fn,
        model_aggregator=model_update_aggregator.secure_aggregator(),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )


class FunctionalFederatedSgdTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('non-simulation', False),
      ('simulation', True),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_update(self, simulation):
    dataset = _dataset()
    model = _build_functional_model()
    client_update = fed_sgd._build_functional_client_update(
        model, use_experimental_simulation_loop=simulation
    )
    client_result, model_output = client_update(_initial_weights(), dataset)
    # Both trainable parameters should have gradients. Model deltas for squared
    # error:
    self.assertAllClose(client_result.update, [[[-2.0], [0.0]], -2.0])
    self.assertAllClose(client_result.update_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8}, model_output)

  @parameterized.named_parameters(
      ('non-simulation', False),
      ('simulation', True),
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_update_with_unconnected_weights(self, simulation):
    dataset = _dataset()
    model = _build_functional_model_unconnected()
    client_update = fed_sgd._build_functional_client_update(
        model, use_experimental_simulation_loop=simulation
    )
    client_result, model_output = client_update(
        _initial_weights_unconnected(), dataset
    )
    # Both trainable parameters should have gradients. Model deltas for squared
    # error:
    self.assertAllClose(client_result.update, [[[-2.0], [0.0]], -2.0, 0.0])
    self.assertAllClose(client_result.update_weight, 8.0)
    self.assertDictContainsSubset({'num_examples': 8}, model_output)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    dataset = _dataset()
    model = _build_functional_model()
    client_update = fed_sgd._build_functional_client_update(
        model, use_experimental_simulation_loop=False
    )
    init_weights = _initial_weights()
    # Set a non-finite bias to force non-finite gradients.
    init_weights.trainable[1] = tf.constant(bad_value)
    client_outputs, _ = client_update(init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs.update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs.update), [[[0.0], [0.0]], 0.0]
    )

  @parameterized.named_parameters(
      ('non-simulation', False), ('simulation', True)
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    dataset = _dataset()
    model = _build_functional_model()
    client_update = fed_sgd._build_functional_client_update(
        model, use_experimental_simulation_loop=simulation
    )
    client_update(_initial_weights(), dataset)
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  def test_build_functional_client_work_without_functional_model_fails(self):
    with self.assertRaisesRegex(TypeError, 'FunctionalModel'):
      fed_sgd._build_functional_fed_sgd_client_work(
          model=lambda: 0,
          metrics_aggregator=aggregator.secure_sum_then_finalize,
      )

  def test_build_functional_fed_sgd_succeeds(self):
    model = _build_functional_model()
    fed_sgd.build_fed_sgd(model_fn=model)

  def test_no_unsecure_aggregation_with_secure_aggregator(self):
    model = _build_functional_model()
    learning_process = fed_sgd.build_fed_sgd(
        model,
        model_aggregator=model_update_aggregator.secure_aggregator(),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )


if __name__ == '__main__':
  tf.test.main()
