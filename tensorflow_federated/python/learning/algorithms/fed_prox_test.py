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
from tensorflow_federated.python.learning.algorithms import fed_prox
from tensorflow_federated.python.learning.optimizers import sgdm


class FedProxConstructionTest(test_case.TestCase, parameterized.TestCase):
  """Tests construction of the FedProx training process."""

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
    fed_prox.build_example_weighted_fed_prox_process(
        model_fn=mock_model_fn,
        proximal_strength=1.0,
        client_optimizer_fn=optimizer_fn,
        model_aggregator=aggregation_factory())
    self.assertEqual(mock_model_fn.call_count, 3)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      fed_prox.build_example_weighted_fed_prox_process(
          model_fn=model_examples.LinearRegression(),
          proximal_strength=1.0,
          client_optimizer_fn=tf.keras.optimizers.SGD)

  def test_raises_on_negative_proximal_strength(self):
    with self.assertRaises(ValueError):
      fed_prox.build_example_weighted_fed_prox_process(
          model_fn=model_examples.LinearRegression,
          proximal_strength=-1.0,
          client_optimizer_fn=tf.keras.optimizers.SGD)


if __name__ == '__main__':
  test_case.main()
