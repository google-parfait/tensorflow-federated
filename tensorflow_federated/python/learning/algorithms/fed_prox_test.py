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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_prox
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import distributors


class FedProxConstructionTest(parameterized.TestCase):
  """Tests construction of the FedProx training process."""

  @parameterized.product(
      optimizer_fn=[
          tf.keras.optimizers.SGD,
          sgdm.build_sgdm(learning_rate=0.1),
      ],
      aggregation_factory=[
          model_update_aggregator.robust_aggregator,
          model_update_aggregator.compression_aggregator,
          model_update_aggregator.secure_aggregator,
      ],
  )
  def test_construction_calls_model_fn(self, optimizer_fn, aggregation_factory):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_prox.build_weighted_fed_prox(
        model_fn=mock_model_fn,
        proximal_strength=1.0,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory(),
    )
    self.assertEqual(mock_model_fn.call_count, 3)

  @parameterized.named_parameters(
      ('non-simulation_tff_optimizer', False),
      ('simulation_tff_optimizer', True),
  )
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  def test_client_tf_dataset_reduce_fn(self, simulation, mock_method):
    fed_prox.build_weighted_fed_prox(
        model_fn=model_examples.LinearRegression,
        proximal_strength=1.0,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
        use_experimental_simulation_loop=simulation,
    )
    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  @parameterized.named_parameters(
      ('weighted', fed_prox.build_unweighted_fed_prox),
      ('unweighted', fed_prox.build_weighted_fed_prox),
  )
  def test_build_functional_model_fed_prox(self, build_fed_prox):
    model = test_models.build_functional_linear_regression(feature_dim=2)
    build_fed_prox(
        model_fn=model,
        proximal_strength=1.0,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )

  def test_build_functional_model_fed_prox_non_tff_optimizer_fails(self):
    model = test_models.build_functional_linear_regression(feature_dim=2)
    with self.assertRaisesRegex(
        TypeError,
        'client_optimizer_fn` must be a `tff.learning.optimizers.Optimizer',
    ):
      fed_prox.build_weighted_fed_prox(
          model_fn=model,
          proximal_strength=1.0,
          client_optimizer_fn=tf.keras.optimizers.SGD,
      )

  @mock.patch.object(fed_prox, 'build_weighted_fed_prox')
  def test_build_weighted_fed_prox_called_by_unweighted_fed_prox(
      self, mock_fed_avg
  ):
    fed_prox.build_unweighted_fed_prox(
        model_fn=model_examples.LinearRegression,
        proximal_strength=1.0,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )
    self.assertEqual(mock_fed_avg.call_count, 1)

  @mock.patch.object(fed_prox, 'build_weighted_fed_prox')
  @mock.patch.object(factory_utils, 'as_weighted_aggregator')
  def test_aggregation_wrapper_called_by_unweighted(self, _, mock_as_weighted):
    fed_prox.build_unweighted_fed_prox(
        model_fn=model_examples.LinearRegression,
        proximal_strength=1.0,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )
    self.assertEqual(mock_as_weighted.call_count, 1)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      fed_prox.build_weighted_fed_prox(
          model_fn=model_examples.LinearRegression(),
          proximal_strength=1.0,
          client_optimizer_fn=tf.keras.optimizers.SGD,
      )

  def test_raises_on_negative_proximal_strength(self):
    with self.assertRaises(ValueError):
      fed_prox.build_weighted_fed_prox(
          model_fn=model_examples.LinearRegression,
          proximal_strength=-1.0,
          client_optimizer_fn=tf.keras.optimizers.SGD,
      )

  def test_raises_on_invalid_distributor(self):
    model_weights_type = model_weights.weights_type_from_model(
        model_examples.LinearRegression
    )
    distributor = distributors.build_broadcast_process(model_weights_type)
    invalid_distributor = iterative_process.IterativeProcess(
        distributor.initialize, distributor.next
    )
    with self.assertRaises(TypeError):
      fed_prox.build_weighted_fed_prox(
          model_fn=model_examples.LinearRegression,
          proximal_strength=1.0,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_distributor=invalid_distributor,
      )

  def test_weighted_fed_avg_raises_on_unweighted_aggregator(self):
    model_aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      fed_prox.build_weighted_fed_prox(
          model_fn=model_examples.LinearRegression,
          proximal_strength=1.0,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=model_aggregator,
      )

  def test_unweighted_fed_avg_raises_on_weighted_aggregator(self):
    model_aggregator = model_update_aggregator.robust_aggregator(weighted=True)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      fed_prox.build_unweighted_fed_prox(
          model_fn=model_examples.LinearRegression,
          proximal_strength=1.0,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=model_aggregator,
      )

  def test_weighted_fed_prox_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_prox.build_weighted_fed_prox(
        model_fn,
        proximal_strength=1.0,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=True
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )

  def test_unweighted_fed_prox_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_prox.build_unweighted_fed_prox(
        model_fn,
        proximal_strength=1.0,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=False
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )


if __name__ == '__main__':
  absltest.main()
