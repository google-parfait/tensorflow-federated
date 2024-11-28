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

from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_avg
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.optimizers import sgdm


class FedAvgTest(parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

  def test_construction_calls_model_fn(self):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing, etc
    # ).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_avg.build_weighted_fed_avg(
        model_fn=mock_model_fn,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1),
        model_aggregator=model_update_aggregator.robust_aggregator(),
    )
    self.assertEqual(mock_model_fn.call_count, 3)

  @parameterized.named_parameters(
      ('dataset_reduce', loop_builder.LoopImplementation.DATASET_REDUCE),
      ('dataset_iterator', loop_builder.LoopImplementation.DATASET_ITERATOR),
  )
  @mock.patch.object(
      loop_builder,
      'build_training_loop',
      wraps=loop_builder.build_training_loop,
  )
  def test_client_tf_dataset_reduce_fn(self, loop_implementation, mock_method):
    fed_avg.build_weighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
        loop_implementation=loop_implementation,
    )
    mock_method.assert_called_once_with(loop_implementation=loop_implementation)

  @mock.patch.object(fed_avg, 'build_weighted_fed_avg')
  def test_build_weighted_fed_avg_called_by_unweighted_fed_avg(
      self, mock_fed_avg
  ):
    fed_avg.build_unweighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )
    self.assertEqual(mock_fed_avg.call_count, 1)

  @mock.patch.object(fed_avg, 'build_weighted_fed_avg')
  @mock.patch.object(factory_utils, 'as_weighted_aggregator')
  def test_aggregation_wrapper_called_by_unweighted(self, _, mock_as_weighted):
    fed_avg.build_unweighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
    )
    self.assertEqual(mock_as_weighted.call_count, 1)

  def test_raises_on_callable_non_model_fn(self):
    with self.assertRaisesRegex(TypeError, 'callable returned type:'):
      fed_avg.build_weighted_fed_avg(
          model_fn=lambda: 0, client_optimizer_fn=sgdm.build_sgdm()
      )
    with self.assertRaisesRegex(TypeError, 'callable returned type:'):
      fed_avg.build_unweighted_fed_avg(
          model_fn=lambda: 0, client_optimizer_fn=sgdm.build_sgdm()
      )

  def test_raises_on_invalid_client_weighting(self):
    with self.assertRaisesRegex(TypeError, 'client_weighting'):
      fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          client_weighting='uniform',
      )

  def test_weighted_fed_avg_raises_on_unweighted_aggregator(self):
    model_aggregator = model_update_aggregator.robust_aggregator(weighted=False)
    with self.assertRaisesRegex(TypeError, 'WeightedAggregationFactory'):
      fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=model_aggregator,
      )

  def test_unweighted_fed_avg_raises_on_weighted_aggregator(self):
    model_aggregator = model_update_aggregator.robust_aggregator(weighted=True)
    with self.assertRaisesRegex(TypeError, 'UnweightedAggregationFactory'):
      fed_avg.build_unweighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          model_aggregator=model_aggregator,
      )


class FunctionalFedAvgTest(parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

  @parameterized.named_parameters(
      ('weighted', fed_avg.build_weighted_fed_avg),
      ('unweighted', fed_avg.build_unweighted_fed_avg),
  )
  def test_raises_on_non_callable_or_functional_model(self, constructor):
    with self.assertRaisesRegex(TypeError, 'is not a callable'):
      constructor(
          model_fn=0, client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1)
      )


if __name__ == '__main__':
  absltest.main()
