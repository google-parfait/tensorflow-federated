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

import tensorflow as tf

from tensorflow_federated.python.aggregators import clipping_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.learning import model_update_aggregator


_test_type = computation_types.TensorType(tf.float32)


class ModelUpdateAggregatorTest(test_case.TestCase):

  def test_apply_zeroing(self):
    factory_ = model_update_aggregator._apply_zeroing(
        model_update_aggregator.AdaptiveZeroingConfig(),
        mean_factory.MeanFactory())
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  def test_apply_fixed_clipping(self):
    factory_ = model_update_aggregator._apply_clipping(
        model_update_aggregator.FixedClippingConfig(1.0),
        mean_factory.MeanFactory())
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  def test_apply_adaptive_clipping(self):
    factory_ = model_update_aggregator._apply_clipping(
        model_update_aggregator.AdaptiveClippingConfig(),
        mean_factory.MeanFactory())
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  def test_dp_factory(self):
    dp_config = model_update_aggregator.DifferentialPrivacyConfig(
        noise_multiplier=1.0, clients_per_round=10.0)
    factory_ = model_update_aggregator._dp_factory(dp_config)
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create_unweighted(_test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  def test_model_update_aggregator(self):
    factory_ = model_update_aggregator.model_update_aggregator()
    self.assertIsInstance(factory_, clipping_factory.ZeroingFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

  def test_model_update_aggregator_no_zeroing(self):
    factory_ = model_update_aggregator.model_update_aggregator(zeroing=None)
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    self.assertNotIsInstance(factory_, clipping_factory.ZeroingFactory)
    process = factory_.create_weighted(_test_type, _test_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)


if __name__ == '__main__':
  test_case.main()
