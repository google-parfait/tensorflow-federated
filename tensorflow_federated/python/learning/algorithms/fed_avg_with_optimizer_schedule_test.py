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

import itertools
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_update_aggregator
from tensorflow_federated.python.learning.algorithms import fed_avg_with_optimizer_schedule
from tensorflow_federated.python.learning.optimizers import sgdm


class ClientScheduledFedAvgTest(test_case.TestCase, parameterized.TestCase):

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters((
      '_'.join(name for name, _ in named_params),
      *(param for _, param in named_params),
  ) for named_params in itertools.product([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD(learning_rate=x)),
      ('tff_optimizer', lambda x: sgdm.build_sgdm(learning_rate=x)),
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
    learning_rate_fn = lambda x: 0.1
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=mock_model_fn,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory())
    self.assertEqual(mock_model_fn.call_count, 3)

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
        client_optimizer_fn=optimizer_fn)

    self.assertEqual(mock_learning_rate_fn.call_count, 1)

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
        client_optimizer_fn=mock_optimizer_fn)

    # The optimizer function should be called twice. The first invocation uses a
    # placeholder of 1.0. The second uses `learning_rate_fn(0)`.
    self.assertEqual(mock_optimizer_fn.call_count, 2)
    self.assertEqual(mock_optimizer_fn.call_args_list[0][0][0], 1.0)
    self.assertEqual(mock_optimizer_fn.call_args_list[1][0][0], 0.5)

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD()),
      ('tff_optimizer', lambda x: sgdm.build_sgdm()),
  ])
  def test_constructs_with_non_constant_learning_rate(self, optimizer_fn):

    def learning_rate_fn(round_num):
      tf.cond(tf.less(round_num, 2), lambda: 0.1, lambda: 0.01)

    fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
        model_fn=model_examples.LinearRegression,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      fed_avg_with_optimizer_schedule.build_weighted_fed_avg_with_optimizer_schedule(
          model_fn=model_examples.LinearRegression(),
          client_learning_rate_fn=lambda x: 0.1,
          client_optimizer_fn=tf.keras.optimizers.SGD)


if __name__ == '__main__':
  test_case.main()
