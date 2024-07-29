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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory_utils
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_avg
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.optimizers import sgdm


def _create_client_dataset(*, num_batches: int, batch_size: int):
  return {
      'x': np.ones(shape=[num_batches, batch_size, 2], dtype=np.float32),
      'y': (
          np.zeros(shape=[num_batches, batch_size, 1], dtype=np.float32) + 0.01
      ),
  }


class FedAvgTest(tf.test.TestCase, parameterized.TestCase):
  """Tests construction of the FedAvg training process."""

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
    fed_avg.build_weighted_fed_avg(
        model_fn=mock_model_fn,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory(),
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

  def test_client_slice_foldl_reduce(self):
    learning_process = fed_avg.build_weighted_fed_avg(
        model_fn=model_examples.LinearRegression,
        client_optimizer_fn=sgdm.build_sgdm(1.0),
        loop_implementation=loop_builder.LoopImplementation.SLICE_FOLDL,
    )
    type_test_utils.assert_types_equivalent(
        learning_process.next.type_signature.parameter[1],
        computation_types.FederatedType(
            {
                'x': computation_types.TensorType(
                    shape=(None, None, 2), dtype=np.float32
                ),
                'y': computation_types.TensorType(
                    shape=(None, None, 1), dtype=np.float32
                ),
            },
            placements.CLIENTS,
        ),
    )
    num_batches = 4
    batch_size = 8
    dataset_as_arrays = {
        'x': np.ones(shape=[num_batches, batch_size, 2], dtype=np.float32),
        'y': (
            np.zeros(shape=[num_batches, batch_size, 1], dtype=np.float32)
            + 0.01
        ),
    }
    state = learning_process.initialize()
    output = learning_process.next(state, [dataset_as_arrays])
    self.assertEqual(output.metrics['client_work']['train']['num_examples'], 32)
    self.assertGreater(output.metrics['client_work']['train']['loss'], 0.0)

  def test_training_loops_produce_same_result(self):
    num_rounds = 3
    client_dataset_arrays = [
        _create_client_dataset(num_batches=4, batch_size=2),
        _create_client_dataset(num_batches=2, batch_size=2),
    ]
    client_datasets = [
        tf.data.Dataset.from_tensor_slices(arrays)
        for arrays in client_dataset_arrays
    ]
    outputs = []
    for loop_implementation in [
        loop_builder.LoopImplementation.DATASET_REDUCE,
        loop_builder.LoopImplementation.DATASET_ITERATOR,
    ]:
      learning_process = fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          loop_implementation=loop_implementation,
      )
      state = learning_process.initialize()
      for _ in range(num_rounds):
        output = learning_process.next(state, client_datasets)
        state = output.state
      outputs.append(output)
    for loop_implementation in [
        loop_builder.LoopImplementation.SLICE_FOLDL,
    ]:
      learning_process = fed_avg.build_weighted_fed_avg(
          model_fn=model_examples.LinearRegression,
          client_optimizer_fn=sgdm.build_sgdm(1.0),
          loop_implementation=loop_implementation,
      )
      state = learning_process.initialize()
      for _ in range(num_rounds):
        output = learning_process.next(state, client_dataset_arrays)
        state = output.state
      outputs.append(output)
    for a, b in zip(outputs, outputs[1:]):
      self.assertAllClose(a, b)

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
          model_fn=lambda: 0, client_optimizer_fn=tf.keras.optimizers.SGD
      )
    with self.assertRaisesRegex(TypeError, 'callable returned type:'):
      fed_avg.build_unweighted_fed_avg(
          model_fn=lambda: 0, client_optimizer_fn=tf.keras.optimizers.SGD
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

  def test_weighted_fed_avg_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_avg.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=True
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )

  def test_unweighted_fed_avg_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_avg.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=False
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
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

  @parameterized.named_parameters(
      ('weighted', fed_avg.build_weighted_fed_avg),
      ('unweighted', fed_avg.build_unweighted_fed_avg),
  )
  def test_raises_on_non_tff_optimizer(self, constructor):
    model = test_models.build_functional_linear_regression()
    with self.subTest('client_optimizer'):
      with self.assertRaisesRegex(TypeError, 'client_optimizer_fn'):
        constructor(
            model_fn=model,
            client_optimizer_fn=tf.keras.optimizers.SGD,
            server_optimizer_fn=sgdm.build_sgdm(),
        )
    with self.subTest('server_optimizer'):
      with self.assertRaisesRegex(TypeError, 'server_optimizer_fn'):
        constructor(
            model_fn=model,
            client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1),
            server_optimizer_fn=tf.keras.optimizers.SGD,
        )

  @parameterized.named_parameters(
      ('weighted', fed_avg.build_weighted_fed_avg),
      ('unweighted', fed_avg.build_unweighted_fed_avg),
  )
  def test_weighted_fed_avg_with_only_secure_aggregation(self, constructor):
    model = test_models.build_functional_linear_regression()
    learning_process = constructor(
        model_fn=model,
        client_optimizer_fn=sgdm.build_sgdm(learning_rate=0.1),
        server_optimizer_fn=sgdm.build_sgdm(),
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=constructor is fed_avg.build_weighted_fed_avg
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
