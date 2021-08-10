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
"""Tests for modular_clipping_factory."""
import collections
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import modular_clipping_factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class ModularClippingSumFactoryComputationTest(test_case.TestCase,
                                               parameterized.TestCase):

  def test_type_properties_simple(self):
    value_type = computation_types.to_type((tf.int32, (2,)))
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower=-2, clip_range_upper=2)
    process = agg_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    # Inner SumFactory has no state.
    server_state_type = computation_types.at_server(())

    expected_init_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(expected_init_type))

    expected_measurements_type = collections.OrderedDict(modclip=())

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=computation_types.at_server(
                expected_measurements_type)))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('lower_is_larger', 1, -5), ('lower_is_larger_negative', -1, -2),
      ('lower_is_larger_positive', 3, 2), ('overflow', -2**30, 2**30 + 5),
      ('overflow_positive', 0, 2**31), ('overflow_negative', -2**31 - 1, 0))
  def test_raise_on_clip_range(self, lower, upper):
    with self.assertRaises(ValueError):
      modular_clipping_factory.ModularClippingSumFactory(
          clip_range_lower=lower, clip_range_upper=upper)

  @parameterized.named_parameters(('string', 'invalid'), ('float', 10.0),
                                  ('tensor', tf.constant(10)))
  def test_raise_on_invalid_clip_type(self, value):
    with self.assertRaises(TypeError):
      modular_clipping_factory.ModularClippingSumFactory(
          clip_range_lower=value, clip_range_upper=2)
    with self.assertRaises(TypeError):
      modular_clipping_factory.ModularClippingSumFactory(
          clip_range_lower=-2, clip_range_upper=value)

  @parameterized.named_parameters(
      ('plain_struct', [('a', tf.int32)]),
      ('sequence', computation_types.SequenceType(tf.int32)),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('nested_sequence', [[[computation_types.SequenceType(tf.int32)]]]))
  def test_tff_value_types_raise_on(self, value_type):
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower=-2, clip_range_upper=2)
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'Expected `value_type` to be'):
      agg_factory.create(value_type)

  @parameterized.named_parameters(('bool', tf.bool), ('string', tf.string))
  def test_component_tensor_dtypes_raise_on(self, value_type):
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower=-2, clip_range_upper=2)
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'must all be integers'):
      agg_factory.create(value_type)


class ModularClippingSumFactoryExecutionTest(test_case.TestCase,
                                             parameterized.TestCase):

  @parameterized.named_parameters([
      ('in_range', -5, 10, [5], [5]), ('out_range_left', -5, 10, [-15], [0]),
      ('out_range_right', -5, 10, [20], [5]),
      ('boundary_left', -5, 10, [-5], [-5]),
      ('boundary_right', -5, 10, [10], [-5]),
      ('negative_in_range', -20, -10, [-15], [-15]),
      ('negative_out_range_left', -20, -10, [-25], [-15]),
      ('negative_out_range_right', -20, -10, [-5], [-15]),
      ('positive_in_range', 20, 40, [30], [30]),
      ('positive_out_range_left', 20, 40, [10], [30]),
      ('positive_out_range_right', 20, 40, [50], [30]),
      ('large_range_symmetric', -2**30, 2**30 - 1, [2**30 + 5], [-2**30 + 6]),
      ('large_range_left', -2**31 + 1, 0, [5], [-2**31 + 6]),
      ('large_range_right', 0, 2**31 - 1, [-5], [2**31 - 6])
  ])
  def test_clip_individual_values(self, clip_range_lower, clip_range_upper,
                                  client_data, expected_sum):
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower, clip_range_upper)
    value_type = computation_types.to_type(tf.int32)
    process = agg_factory.create(value_type)
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(output.result, expected_sum)

  @parameterized.named_parameters([('in_range_clip', -3, 3, [1, -2, 1, -2], -2),
                                   ('boundary_clip', -3, 3, [-3, 3, 3, 3], 0),
                                   ('out_range_clip', -2, 2, [-3, 3, 5], 1),
                                   ('mixed_clip', -2, 2, [-4, -2, 1, 2, 7], 0)])
  def test_clip_sum(self, clip_range_lower, clip_range_upper, client_data,
                    expected_sum):
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower, clip_range_upper)
    value_type = computation_types.to_type(tf.int32)
    process = agg_factory.create(value_type)
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(output.result, expected_sum)

  @parameterized.named_parameters([('in_range_clip', -3, 3, [1, -2, 1, -2], -2),
                                   ('boundary_clip', -3, 3, [-3, 3, 3, 3], 0),
                                   ('out_range_clip', -2, 2, [-3, 3, 5], 1),
                                   ('mixed_clip', -2, 2, [-4, -2, 1, 2, 7], 0)])
  def test_clip_sum_struct(self, clip_range_lower, clip_range_upper,
                           client_data, expected_sum):
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower, clip_range_upper)
    value_type = computation_types.to_type((tf.int32, (2,)))
    process = agg_factory.create(value_type)
    state = process.initialize()
    client_tensor_data = [
        tf.constant(v, dtype=tf.int32, shape=(2,)) for v in client_data
    ]
    output = process.next(state, client_tensor_data)
    self.assertAllClose(
        tf.constant(expected_sum, dtype=tf.int32, shape=(2,)), output.result)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
