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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

_test_dp_query = tfp.GaussianSumQuery(l2_norm_clip=1.0, stddev=0.0)

_test_struct_type = [(tf.float32, (2,)), tf.float32]
_test_inner_agg_factory = test_utils.SumPlusOneFactory()


class DPFactoryComputationTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_simple', tf.float32, None),
      ('struct_simple', _test_struct_type, None),
      ('float_inner', tf.float32, _test_inner_agg_factory),
      ('struct_inner', _test_struct_type, _test_inner_agg_factory))
  def test_type_properties(self, value_type, inner_agg_factory):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(
        _test_dp_query, inner_agg_factory)
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = factory_.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    query_state = _test_dp_query.initial_global_state()
    query_state_type = type_conversions.type_from_tensors(query_state)
    query_metrics_type = type_conversions.type_from_tensors(
        _test_dp_query.derive_metrics(query_state))

    inner_state_type = tf.int32 if inner_agg_factory else ()

    server_state_type = computation_types.at_server(
        (query_state_type, inner_state_type))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    inner_measurements_type = tf.int32 if inner_agg_factory else ()
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            dp_query_metrics=query_metrics_type, dp=inner_measurements_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  def test_init_non_dp_query_raises(self):
    with self.assertRaises(TypeError):
      differential_privacy.DifferentiallyPrivateFactory('not a dp_query')

  def test_init_non_agg_factory_raises(self):
    with self.assertRaises(TypeError):
      differential_privacy.DifferentiallyPrivateFactory(_test_dp_query,
                                                        'not an agg factory')

  @parameterized.named_parameters(
      ('federated_type',
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)))
  def test_incorrect_value_type_raises(self, bad_value_type):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    with self.assertRaises(TypeError):
      factory_.create(bad_value_type)


class DPFactoryExecutionTest(test_case.TestCase):

  def test_simple_sum(self):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    value_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type)

    # The test query has clip 1.0 and no noise, so this computes clipped sum.

    state = process.initialize()

    client_data = [0.5, 1.0, 1.5]
    output = process.next(state, client_data)
    self.assertAllClose(2.5, output.result)

  def test_structure_sum(self):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(_test_dp_query)
    value_type = computation_types.to_type([tf.float32, tf.float32])
    process = factory_.create(value_type)

    # The test query has clip 1.0 and no noise, so this computes clipped sum.

    state = process.initialize()

    # pyformat: disable
    client_data = [
        [0.1, 0.2],         # not clipped (norm < 1)
        [5 / 13, 12 / 13],  # not clipped (norm == 1)
        [3.0, 4.0]          # clipped to 3/5, 4/5
    ]
    output = process.next(state, client_data)

    expected_result = [0.1 +  5 / 13 + 3 / 5,
                       0.2 + 12 / 13 + 4 / 5]
    # pyformat: enable
    self.assertAllClose(expected_result, output.result)

  def test_inner_sum(self):
    factory_ = differential_privacy.DifferentiallyPrivateFactory(
        _test_dp_query, _test_inner_agg_factory)
    value_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type)

    # The test query has clip 1.0 and no noise, so this computes clipped sum.
    # Inner agg adds another 1.0 (post-clipping).

    state = process.initialize()
    self.assertAllEqual(0, state[1])

    client_data = [0.5, 1.0, 1.5]
    output = process.next(state, client_data)
    self.assertAllEqual(1, output.state[1])
    self.assertAllClose(3.5, output.result)
    self.assertAllEqual(test_utils.MEASUREMENT_CONSTANT,
                        output.measurements['dp'])

  def test_adaptive_query(self):
    query = tfp.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=1.0,
        noise_multiplier=0.0,
        target_unclipped_quantile=1.0,
        learning_rate=1.0,
        clipped_count_stddev=0.0,
        expected_num_records=3.0,
        geometric_update=False)
    factory_ = differential_privacy.DifferentiallyPrivateFactory(query)
    value_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type)

    state = process.initialize()

    client_data = [0.5, 1.5, 2.0]  # Two clipped on first round.
    expected_result = 0.5 + 1.0 + 1.0
    output = process.next(state, client_data)
    self.assertAllClose(expected_result, output.result)

    # Clip is increased by 2/3 to 5/3.
    expected_result = 0.5 + 1.5 + 5 / 3
    output = process.next(output.state, client_data)
    self.assertAllClose(expected_result, output.result)

  def test_noise(self):
    noise = 3.14159
    factory_ = differential_privacy.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise, clients_per_round=1.0, clip=1.0)
    value_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type)

    state = process.initialize()
    client_data = [0.36788]
    outputs = []
    # num_iterations empirically chosen to flake < 1 in 10000 runs of this test.
    num_iterations = 500

    for _ in range(num_iterations):
      output = process.next(state, client_data)
      outputs.append(output.result)
      state = output.state

    stddev = np.std(outputs, ddof=1.0)
    self.assertAllClose(stddev, noise, rtol=0.15)

  def test_gaussian_adaptive_cls(self):
    process = differential_privacy.DifferentiallyPrivateFactory.gaussian_adaptive(
        noise_multiplier=1e-2, clients_per_round=10)
    self.assertIsInstance(process,
                          differential_privacy.DifferentiallyPrivateFactory)

  def test_gaussian_fixed_cls(self):
    process = differential_privacy.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=1.0, clients_per_round=10, clip=1.0)
    self.assertIsInstance(process,
                          differential_privacy.DifferentiallyPrivateFactory)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
