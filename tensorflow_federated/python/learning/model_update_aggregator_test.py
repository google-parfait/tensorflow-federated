# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for model_update_aggregator."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.learning import model_update_aggregator

_test_qe_config = model_update_aggregator.QuantileEstimationConfig(
    initial_estimate=1.0, target_quantile=0.5, learning_rate=1.0)


_test_type = computation_types.TensorType(tf.float32)


class ModelUpdateAggregatorTest(test_case.TestCase, parameterized.TestCase):

  def _check_value(self, fn, args, key=None, value=None, error=None):
    print(f'key: {key}  value: {value}')
    print(f'args: {args}')
    if key:
      args[key] = value
    if error is None:
      fn(**args)
    else:
      with self.assertRaises(error):
        fn(**args)

  @parameterized.named_parameters(
      ('good', None, None, None),
      ('initial_estimate_type', 'initial_estimate', 'bad', TypeError),
      ('initial_estimate_value', 'initial_estimate', 0.0, ValueError),
      ('target_quantile_type', 'target_quantile', 'bad', TypeError),
      ('target_quantile_value', 'target_quantile', -1.0, ValueError),
      ('learning_rate_type', 'learning_rate', 'bad', TypeError),
      ('learning_rate_value', 'learning_rate', 0.0, ValueError),
  )
  def test_quantile_estimation_config_args(self, key, value, error):
    good_args = dict(
        initial_estimate=1.0, target_quantile=0.5, learning_rate=1.0)
    self._check_value(model_update_aggregator.QuantileEstimationConfig,
                      good_args, key, value, error)

  def test_quantile_estimation_config_to_process(self):
    process = _test_qe_config.to_quantile_estimation_process()
    self.assertIsInstance(process, estimation_process.EstimationProcess)

  @parameterized.named_parameters(
      ('good', None, None, None),
      ('quantile_type', 'quantile', 'bad', TypeError),
      ('quantile_value', 'quantile', None, None),
      ('multiplier_type', 'multiplier', 'bad', TypeError),
      ('multiplier_value', 'multiplier', 0.0, ValueError),
      ('increment_type', 'increment', 'bad', TypeError),
      ('increment_value', 'increment', -1.0, ValueError),
  )
  def test_zeroing_config_args(self, key, value, error):
    good_args = dict(quantile=_test_qe_config, multiplier=10.0, increment=0.5)
    self._check_value(model_update_aggregator.ZeroingConfig, good_args, key,
                      value, error)

  def test_zeroing_config_to_factory(self):
    qe_config = model_update_aggregator.QuantileEstimationConfig(
        initial_estimate=1.0, target_quantile=0.5, learning_rate=1.0)
    zeroing_config = model_update_aggregator.ZeroingConfig(
        quantile=qe_config, multiplier=10.0, increment=0.5)
    factory_ = zeroing_config.to_factory(mean_factory.MeanFactory())
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  @parameterized.named_parameters(
      ('good', None, None, None),
      ('clip_type', 'clip', 'bad', TypeError),
      ('clip_float', 'clip', 1.0, None),
      ('clip_value', 'clip', 0.0, ValueError),
      ('clip_quantile', 'clip', _test_qe_config, None),
  )
  def test_clipping_config_args(self, key, value, error):
    self._check_value(model_update_aggregator.ClippingConfig, {}, key, value,
                      error)

  def test_clipping_config_to_factory(self):
    clipping_config = model_update_aggregator.ClippingConfig()
    factory_ = clipping_config.to_factory(mean_factory.MeanFactory())
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  @parameterized.named_parameters(
      ('good', None, None, None),
      ('noise_multiplier_type', 'noise_multiplier', 'bad', TypeError),
      ('noise_multiplier_value', 'noise_multiplier', -1.0, ValueError),
      ('clients_per_round_type', 'clients_per_round', 'bad', TypeError),
      ('clients_per_round_value', 'clients_per_round', 0.0, ValueError),
      ('clipping_type', 'clipping', 'bad', TypeError),
      ('clipped_count_stddev_type', 'clipped_count_stddev', 'bad', TypeError),
      ('clipped_count_stddev_value', 'clipped_count_stddev', -1.0, ValueError),
  )
  def test_dp_config_args(self, key, value, error):
    good_args = dict(noise_multiplier=1.0, clients_per_round=10.0)
    self._check_value(model_update_aggregator.DPConfig, good_args, key, value,
                      error)

  def test_dp_config_to_factory(self):
    dp_config = model_update_aggregator.DPConfig(
        noise_multiplier=1.0, clients_per_round=10.0)
    factory_ = dp_config.to_factory()
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create_unweighted(_test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  def test_model_update_aggregator(self):
    factory_ = model_update_aggregator.model_update_aggregator()
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)


if __name__ == '__main__':
  test_case.main()
