# Copyright 2019, The TensorFlow Federated Authors.
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

import collections

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_privacy

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.utils import differential_privacy


def wrap_aggregate_fn(dp_aggregate_fn, sample_value):
  tff_types = type_conversions.type_from_tensors(sample_value)

  @computations.federated_computation
  def run_initialize():
    return intrinsics.federated_value(dp_aggregate_fn.initialize(),
                                      placement_literals.SERVER)

  @computations.federated_computation(run_initialize.type_signature.result,
                                      computation_types.FederatedType(
                                          tff_types,
                                          placement_literals.CLIENTS))
  def run_aggregate(global_state, client_values):
    return dp_aggregate_fn(global_state, client_values)

  return run_initialize, run_aggregate


class BuildDpQueryTest(test.TestCase):

  def test_build_dp_query_basic(self):
    query = differential_privacy.build_dp_query(1.0, 2.0, 3.0)
    self.assertIsInstance(query, tensorflow_privacy.GaussianAverageQuery)

    self.assertEqual(query._numerator._l2_norm_clip, 1.0)
    self.assertEqual(query._numerator._stddev, 2.0)
    self.assertEqual(query._denominator, 3.0)

  def test_build_dp_query_adaptive(self):
    ccba = 0.1

    query = differential_privacy.build_dp_query(
        1.0,
        2.0,
        3.0,
        adaptive_clip_learning_rate=0.05,
        target_unclipped_quantile=0.5,
        clipped_count_budget_allocation=ccba,
        expected_clients_per_round=10)
    self.assertIsInstance(query,
                          tensorflow_privacy.QuantileAdaptiveClipAverageQuery)
    self.assertIsInstance(query._numerator,
                          tensorflow_privacy.QuantileAdaptiveClipSumQuery)

    expected_sum_query_noise_multiplier = 2.0 * (1.0 - ccba)**(-0.5)
    self.assertAlmostEqual(query._numerator._noise_multiplier,
                           expected_sum_query_noise_multiplier)
    self.assertEqual(query._denominator, 3.0)

  def test_build_dp_query_per_vector(self):

    class MockTensor():

      def __init__(self, shape):
        self.shape = shape

    mock_shape = collections.namedtuple('MockShape', ['dims'])
    mock_dim = collections.namedtuple('MockDim', ['value'])
    mock_model = collections.namedtuple('MockModel', ['weights'])
    mock_weights = collections.namedtuple('MockWeights', ['trainable'])

    def make_mock_tensor(*dims):
      return MockTensor(mock_shape([mock_dim(dim) for dim in dims]))

    vectors = collections.OrderedDict(
        a=make_mock_tensor(2),
        b=make_mock_tensor(2, 3),
        c=make_mock_tensor(1, 3, 4))
    model = mock_model(mock_weights(vectors))

    query = differential_privacy.build_dp_query(
        1.0, 2.0, 3.0, per_vector_clipping=True, model=model)

    self.assertIsInstance(query, tensorflow_privacy.NestedQuery)

    def check(subquery):
      self.assertIsInstance(subquery, tensorflow_privacy.GaussianAverageQuery)
      self.assertEqual(subquery._denominator, 3.0)

    tf.nest.map_structure(check, query._queries)

    noise_multipliers = tf.nest.flatten(
        tf.nest.map_structure(
            lambda query: query._numerator._stddev / query._numerator.
            _l2_norm_clip, query._queries))

    effective_noise_multiplier = sum([x**-2.0 for x in noise_multipliers])**-0.5
    self.assertAlmostEqual(effective_noise_multiplier, 2.0)


class BuildDpAggregateTest(test.TestCase):

  def test_dp_sum(self):
    query = tensorflow_privacy.GaussianSumQuery(4.0, 0.0)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(query)

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, 0.0)
    global_state = initialize()

    global_state, result = aggregate(global_state, [1.0, 3.0, 5.0])

    self.assertEqual(global_state.l2_norm_clip, 4.0)
    self.assertEqual(global_state.stddev, 0.0)
    self.assertEqual(result, 8.0)

  def test_dp_sum_structure_odict(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(query)

    def datapoint(a, b):
      return collections.OrderedDict(a=(a,), b=[b])

    data = [
        datapoint(1.0, 2.0),
        datapoint(2.0, 3.0),
        datapoint(6.0, 8.0),  # Clipped to 3.0, 4.0
    ]

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, data[0])
    global_state = initialize()

    global_state, result = aggregate(global_state, data)

    self.assertEqual(global_state.l2_norm_clip, 5.0)
    self.assertEqual(global_state.stddev, 0.0)

    self.assertEqual(result['a'][0], 6.0)
    self.assertEqual(result['b'][0], 9.0)

  def test_dp_sum_structure_list(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    def _value_type_fn(value):
      del value
      return [
          computation_types.TensorType(tf.float32),
          computation_types.TensorType(tf.float32),
      ]

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(
        query, value_type_fn=_value_type_fn)

    def datapoint(a, b):
      return [tf.Variable(a, name='a'), tf.Variable(b, name='b')]

    data = [
        datapoint(1.0, 2.0),
        datapoint(2.0, 3.0),
        datapoint(6.0, 8.0),  # Clipped to 3.0, 4.0
    ]

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, data[0])
    global_state = initialize()

    global_state, result = aggregate(global_state, data)

    self.assertEqual(global_state.l2_norm_clip, 5.0)
    self.assertEqual(global_state.stddev, 0.0)

    result = list(result)
    self.assertEqual(result[0], 6.0)
    self.assertEqual(result[1], 9.0)

  def test_dp_stateful_mean(self):

    class ShrinkingSumQuery(tensorflow_privacy.GaussianSumQuery):

      def get_noised_result(self, sample_state, global_state):
        global_state = self._GlobalState(
            tf.maximum(global_state.l2_norm_clip - 1, 0.0), global_state.stddev)

        return sample_state, global_state

    query = ShrinkingSumQuery(4.0, 0.0)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(query)

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, 0.0)
    global_state = initialize()

    records = [1.0, 3.0, 5.0]

    def run_and_check(global_state, expected_l2_norm_clip, expected_result):
      global_state, result = aggregate(global_state, records)
      self.assertEqual(global_state.l2_norm_clip, expected_l2_norm_clip)
      self.assertEqual(result, expected_result)
      return global_state

    self.assertEqual(global_state.l2_norm_clip, 4.0)
    global_state = run_and_check(global_state, 3.0, 8.0)
    global_state = run_and_check(global_state, 2.0, 7.0)
    global_state = run_and_check(global_state, 1.0, 5.0)
    global_state = run_and_check(global_state, 0.0, 3.0)
    global_state = run_and_check(global_state, 0.0, 0.0)

  def test_dp_global_state_type(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    _, dp_global_state_type = differential_privacy.build_dp_aggregate(query)

    self.assertIsInstance(dp_global_state_type,
                          computation_types.StructWithPythonType)


class BuildDpAggregateProcessTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float', 0.0), ('list', [0.0, 0.0]),
      ('odict', collections.OrderedDict(a=0.0, b=0.0)))
  def test_process_type_signature(self, value_template):
    query = tensorflow_privacy.GaussianSumQuery(4.0, 0.0)
    value_type = type_conversions.type_from_tensors(value_template)
    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = query.initial_global_state()
    server_state_type = computation_types.FederatedType(
        type_conversions.type_from_tensors(global_state), placements.SERVER)
    self.assertEqual(
        dp_aggregate_process.initialize.type_signature,
        computation_types.FunctionType(
            parameter=None, result=server_state_type))

    metrics_type = type_conversions.type_from_tensors(
        query.derive_metrics(global_state))

    client_value_type = computation_types.FederatedType(value_type,
                                                        placements.CLIENTS)
    client_value_weight_type = computation_types.FederatedType(
        tf.float32, placements.CLIENTS)
    server_result_type = computation_types.FederatedType(
        value_type, placements.SERVER)
    server_metrics_type = computation_types.FederatedType(
        metrics_type, placements.SERVER)
    self.assertTrue(
        dp_aggregate_process.next.type_signature.is_equivalent_to(
            computation_types.FunctionType(
                parameter=collections.OrderedDict(
                    global_state=server_state_type,
                    value=client_value_type,
                    weight=client_value_weight_type),
                result=measured_process.MeasuredProcessOutput(
                    state=server_state_type,
                    result=server_result_type,
                    measurements=server_metrics_type))))

  def test_dp_sum(self):
    query = tensorflow_privacy.GaussianSumQuery(4.0, 0.0)

    value_type = type_conversions.type_from_tensors(0.0)
    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = dp_aggregate_process.initialize()

    output = dp_aggregate_process.next(global_state, [1.0, 3.0, 5.0],
                                       [1.0, 1.0, 1.0])

    self.assertEqual(output.state.l2_norm_clip, 4.0)
    self.assertEqual(output.state.stddev, 0.0)
    self.assertEqual(output.result, 8.0)

  def test_dp_sum_structure_odict(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    def datapoint(a, b):
      return collections.OrderedDict(a=(a,), b=[b])

    data = [
        datapoint(1.0, 2.0),
        datapoint(2.0, 3.0),
        datapoint(6.0, 8.0),  # Clipped to 3.0, 4.0
    ]

    value_type = type_conversions.type_from_tensors(data[0])
    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = dp_aggregate_process.initialize()

    output = dp_aggregate_process.next(global_state, data, [1.0, 1.0, 1.0])

    self.assertEqual(output.state.l2_norm_clip, 5.0)
    self.assertEqual(output.state.stddev, 0.0)

    self.assertEqual(output.result['a'][0], 6.0)
    self.assertEqual(output.result['b'][0], 9.0)

  def test_dp_sum_structure_nested_odict(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    def datapoint(a, b, c):
      return collections.OrderedDict(
          a=(a,), bc=collections.OrderedDict(b=[b], c=(c,)))

    data = [
        datapoint(1.0, 2.0, 1.0),
        datapoint(2.0, 3.0, 1.0),
        datapoint(6.0, 8.0, 0.0),  # Clipped to 3.0, 4.0, 0.0
    ]

    value_type = type_conversions.type_from_tensors(data[0])
    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = dp_aggregate_process.initialize()

    output = dp_aggregate_process.next(global_state, data, [1.0, 1.0, 1.0])

    self.assertEqual(output.state.l2_norm_clip, 5.0)
    self.assertEqual(output.state.stddev, 0.0)

    self.assertEqual(output.result['a'][0], 6.0)
    self.assertEqual(output.result['bc']['b'][0], 9.0)
    self.assertEqual(output.result['bc']['c'][0], 2.0)

  def test_dp_sum_structure_complex(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    def datapoint(a, b, c):
      return collections.OrderedDict(a=(a,), bc=([b], (c,)))

    data = [
        datapoint(1.0, 2.0, 1.0),
        datapoint(2.0, 3.0, 1.0),
        datapoint(6.0, 8.0, 0.0),  # Clipped to 3.0, 4.0, 0.0
    ]

    value_type = type_conversions.type_from_tensors(data[0])
    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = dp_aggregate_process.initialize()

    output = dp_aggregate_process.next(global_state, data, [1.0, 1.0, 1.0])

    self.assertEqual(output.state.l2_norm_clip, 5.0)
    self.assertEqual(output.state.stddev, 0.0)

    self.assertEqual(output.result['a'][0], 6.0)
    self.assertEqual(output.result['bc'][0][0], 9.0)
    self.assertEqual(output.result['bc'][1][0], 2.0)

  def test_dp_sum_structure_list(self):
    query = tensorflow_privacy.GaussianSumQuery(5.0, 0.0)

    def datapoint(a, b):
      return [tf.Variable(a, name='a'), tf.Variable(b, name='b')]

    data = [
        datapoint(1.0, 2.0),
        datapoint(2.0, 3.0),
        datapoint(6.0, 8.0),  # Clipped to 3.0, 4.0
    ]

    value_type = type_conversions.type_from_tensors(data[0])

    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = dp_aggregate_process.initialize()

    output = dp_aggregate_process.next(global_state, data, [1.0, 1.0, 1.0])

    self.assertEqual(output.state.l2_norm_clip, 5.0)
    self.assertEqual(output.state.stddev, 0.0)

    result = list(output.result)
    self.assertEqual(result[0], 6.0)
    self.assertEqual(result[1], 9.0)

  def test_dp_stateful_mean(self):

    class ShrinkingSumQuery(tensorflow_privacy.GaussianSumQuery):

      def get_noised_result(self, sample_state, global_state):
        global_state = self._GlobalState(
            tf.maximum(global_state.l2_norm_clip - 1, 0.0), global_state.stddev)

        return sample_state, global_state

    query = ShrinkingSumQuery(4.0, 0.0)

    value_type = type_conversions.type_from_tensors(0.0)
    dp_aggregate_process = differential_privacy.build_dp_aggregate_process(
        value_type, query)

    global_state = dp_aggregate_process.initialize()

    records = [1.0, 3.0, 5.0]

    def run_and_check(global_state, expected_l2_norm_clip, expected_result):
      output = dp_aggregate_process.next(global_state, records, [1.0, 1.0, 1.0])
      self.assertEqual(output.state.l2_norm_clip, expected_l2_norm_clip)
      self.assertEqual(output.result, expected_result)
      return output.state

    self.assertEqual(global_state.l2_norm_clip, 4.0)
    global_state = run_and_check(global_state, 3.0, 8.0)
    global_state = run_and_check(global_state, 2.0, 7.0)
    global_state = run_and_check(global_state, 1.0, 5.0)
    global_state = run_and_check(global_state, 0.0, 3.0)
    global_state = run_and_check(global_state, 0.0, 0.0)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
