# Copyright 2021, The TensorFlow Federated Authors.
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

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning.algorithms import aggregation

_TEST_VALUE_TYPE = computation_types.TensorType(tf.float32, (2,))
_TEST_WEIGHT_TYPE = computation_types.TensorType(tf.float32)


class UnweightedAsWeightedAggregationTest(test_case.TestCase):

  def test_returns_weighted_factory(self):
    wrapped_factory = aggregation.as_weighted_aggregator(
        sum_factory.SumFactory())
    self.assertIsInstance(wrapped_factory, factory.WeightedAggregationFactory)

  def test_wrapped_aggregator_same_as_unweighted_aggregator(self):
    unweighted_factory = sum_factory.SumFactory()
    wrapped_factory = aggregation.as_weighted_aggregator(unweighted_factory)

    unweighted_aggregator = unweighted_factory.create(_TEST_VALUE_TYPE)
    weighted_aggregator = wrapped_factory.create(_TEST_VALUE_TYPE,
                                                 _TEST_WEIGHT_TYPE)

    test_data = [(1.0, 2.0), (3.0, 4.0), (0.0, 5.0)]
    test_weights = [1.0, 1.0, 1.0]

    unweighted_output = unweighted_aggregator.next(
        unweighted_aggregator.initialize(), test_data)
    weighted_output = weighted_aggregator.next(weighted_aggregator.initialize(),
                                               test_data, test_weights)

    self.assertAllEqual(unweighted_output.state, weighted_output.state)
    self.assertAllEqual(unweighted_output.result, weighted_output.result)
    self.assertAllEqual(unweighted_output.measurements,
                        weighted_output.measurements)

  def test_wrapped_aggregator_independent_of_weights(self):
    aggregator = aggregation.as_weighted_aggregator(
        sum_factory.SumFactory()).create(_TEST_VALUE_TYPE, _TEST_WEIGHT_TYPE)

    test_data = [(1.0, 2.0), (3.0, 4.0), (0.0, 5.0)]
    uniform_weights = [1.0, 1.0, 1.0]

    state = aggregator.initialize()
    uniform_output = aggregator.next(state, test_data, uniform_weights)

    for seed in range(0, 30, 3):
      random_weights = [
          tf.random.stateless_uniform((), [seed + i, 0]) for i in range(3)
      ]
      random_output = aggregator.next(state, test_data, random_weights)
      self.assertAllEqual(uniform_output.state, random_output.state)
      self.assertAllEqual(uniform_output.result, random_output.result)
      self.assertAllEqual(uniform_output.measurements,
                          random_output.measurements)

  def test_as_weighted_raises(self):
    with self.assertRaises(TypeError):
      aggregation.as_weighted_aggregator(mean.MeanFactory())


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
