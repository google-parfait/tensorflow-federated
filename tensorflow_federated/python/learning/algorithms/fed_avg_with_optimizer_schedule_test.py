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

from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_avg_with_optimizer_schedule
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.optimizers import sgdm


class ClientScheduledFedAvgTest(parameterized.TestCase):

  @parameterized.product(
      optimizer_fn=[
          lambda x: tf.keras.optimizers.SGD(learning_rate=x),
          lambda x: sgdm.build_sgdm(learning_rate=x),
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
    learning_rate_fn = lambda x: 0.1
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=mock_model_fn,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory(),
    )
    self.assertEqual(mock_model_fn.call_count, 3)

  def test_construction_of_functional_model(self):
    learning_rate_fn = lambda x: 0.1
    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=test_models.build_functional_linear_regression(),
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=lambda x: sgdm.build_sgdm(learning_rate=x),
    )

  @parameterized.named_parameters(
      ('dataset_reduce', loop_builder.LoopImplementation.DATASET_REDUCE),
      ('dataset_iterator', loop_builder.LoopImplementation.DATASET_ITERATOR),
  )
  @mock.patch.object(
      loop_builder,
      'build_training_loop',
      wraps=loop_builder.build_training_loop,
  )
  def test_client_tf_dataset_reduce_fn(self, loop_implementation, mock_reduce):
    client_learning_rate_fn = lambda x: 0.5
    client_optimizer_fn = tf.keras.optimizers.SGD
    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=client_learning_rate_fn,
        client_optimizer_fn=client_optimizer_fn,
        loop_implementation=loop_implementation,
    )
    mock_reduce.assert_called_once_with(loop_implementation=loop_implementation)

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD()),
      ('tff_optimizer', lambda x: sgdm.build_sgdm()),
  ])
  def test_construction_calls_client_learning_rate_fn(self, optimizer_fn):
    mock_learning_rate_fn = mock.Mock(side_effect=lambda x: 1.0)
    optimizer_fn = tf.keras.optimizers.SGD
    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=mock_learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
    )

    # TODO: b/268530457 - Investigate if we can/should reduce this to 1 call.
    # Called twice, once at the clients (to build the relevant optimizer) and
    # once at the server (to create the relevant measurement.)
    self.assertEqual(mock_learning_rate_fn.call_count, 2)

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD()),
      ('tff_optimizer', lambda x: sgdm.build_sgdm()),
  ])
  def test_construction_calls_client_optimizer_fn(self, optimizer_fn):
    learning_rate_fn = lambda x: 0.5
    mock_optimizer_fn = mock.Mock(side_effect=optimizer_fn)
    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=mock_optimizer_fn,
    )

    # The optimizer function should be called twice. The first invocation uses a
    # placeholder of 1.0. The second uses `learning_rate_fn(0)`.
    self.assertEqual(mock_optimizer_fn.call_count, 2)
    self.assertEqual(mock_optimizer_fn.call_args_list[0][0][0], 1.0)
    self.assertEqual(mock_optimizer_fn.call_args_list[1][0][0], 0.5)

  def test_construction_calls_server_optimizer_fn(self):
    learning_rate_fn = lambda x: 0.5
    client_optimizer_fn = tf.keras.optimizers.SGD
    mock_server_optimizer_fn = mock.Mock(side_effect=tf.keras.optimizers.SGD)

    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=mock_server_optimizer_fn,
    )

    mock_server_optimizer_fn.assert_called()

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD()),
      ('tff_optimizer', lambda x: sgdm.build_sgdm()),
  ])
  def test_constructs_with_non_constant_learning_rate(self, optimizer_fn):
    def learning_rate_fn(round_num):
      return tf.cond(tf.less(round_num, 2), lambda: 0.1, lambda: 0.01)

    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
    )

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD()),
      ('tff_optimizer', lambda x: sgdm.build_sgdm()),
  ])
  def test_constructs_with_tf_function(self, optimizer_fn):
    @tf.function
    def learning_rate_fn(round_num):
      if round_num < 2:
        return 0.1
      else:
        return 0.01

    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
    )

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression(),
          client_learning_rate_fn=lambda x: 0.1,
          client_optimizer_fn=tf.keras.optimizers.SGD,
      )

  def test_construction_with_only_secure_aggregation(self):
    model_fn = model_examples.LinearRegression
    learning_process = fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn,
        client_learning_rate_fn=lambda x: 0.5,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        model_aggregator=model_update_aggregator.secure_aggregator(
            weighted=True
        ),
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(
        learning_process.next
    )

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD()),
      ('tff_optimizer', lambda x: sgdm.build_sgdm()),
  ])
  def test_measurements_include_client_learning_rate(self, optimizer_fn):
    client_work = fed_avg_with_optimizer_schedule.build_scheduled_client_work(
        model_fn=model_examples.LinearRegression,
        learning_rate_fn=lambda x: 1.0,
        optimizer_fn=optimizer_fn,
        metrics_aggregator=aggregator.sum_then_finalize,
    )
    output_type_signature = client_work.next.type_signature.result
    self.assertTrue(
        hasattr(output_type_signature[2].member, 'client_learning_rate')
    )


if __name__ == '__main__':
  absltest.main()
