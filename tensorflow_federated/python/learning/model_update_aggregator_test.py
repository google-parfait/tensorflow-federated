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
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.learning import model_update_aggregator


_float_type = computation_types.TensorType(tf.float32)


class ModelUpdateAggregatorTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', False, False),
      ('zeroing', True, False),
      ('clipping', False, True),
      ('zeroing_and_clipping', False, False),
  )
  def test_robust_aggregator(self, zeroing, clipping):
    factory_ = model_update_aggregator.robust_aggregator(zeroing, clipping)

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_float_type, _float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertLen(process.next.type_signature.parameter, 3)

  @parameterized.named_parameters(
      ('simple', False),
      ('zeroing', True),
  )
  def test_dp_aggregator(self, zeroing):
    factory_ = model_update_aggregator.dp_aggregator(
        noise_multiplier=1.0, clients_per_round=10, zeroing=zeroing)

    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(_float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertLen(process.next.type_signature.parameter, 2)

  @parameterized.named_parameters(
      ('simple', False, False),
      ('zeroing', True, False),
      ('clipping', False, True),
      ('zeroing_and_clipping', False, False),
  )
  def test_secure_aggregator(self, zeroing, clipping):
    factory_ = model_update_aggregator.secure_aggregator(zeroing, clipping)

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_float_type, _float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertLen(process.next.type_signature.parameter, 3)

  @parameterized.named_parameters(
      ('simple', False, False),
      ('zeroing', True, False),
      ('clipping', False, True),
      ('zeroing_and_clipping', False, False),
  )
  def test_compression_aggregator(self, zeroing, clipping):
    factory_ = model_update_aggregator.compression_aggregator(zeroing, clipping)

    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(_float_type, _float_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)
    self.assertLen(process.next.type_signature.parameter, 3)


if __name__ == '__main__':
  test_case.main()
