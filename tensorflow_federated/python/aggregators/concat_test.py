# Copyright 2021, Google LLC.
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

from tensorflow_federated.python.aggregators import concat
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

_test_struct_type_int_mixed = [tf.int32, (tf.int32, (2,)), (tf.int32, (3, 3))]
_test_struct_type_float_mixed = [
    tf.float32, (tf.float32, (2,)), (tf.float32, (3, 3))
]
_test_struct_type_nested = collections.OrderedDict(
    a=[tf.float32, [(tf.float32, (2, 2, 2))]], b=(tf.float32, (3, 3)))


def _concat_mean():
  return concat.concat_factory(mean.MeanFactory())


def _concat_sum():
  return concat.concat_factory(sum_factory.SumFactory())


def _make_test_struct_nested(value):
  return collections.OrderedDict(
      a=[tf.cast(value, tf.float32), [tf.ones([2, 2, 2]) * value]],
      b=tf.ones((3, 3)) * value)


class ConcatFactoryComputationTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float', tf.float32),
      ('struct_list_int_scalars', [tf.int32, tf.int32, tf.int32]),
      ('struct_list_int_mixed', _test_struct_type_int_mixed),
      ('struct_list_float_mixed', _test_struct_type_float_mixed),
      ('struct_nested', _test_struct_type_nested))
  def test_concat_type_properties_unweighted(self, value_type):
    factory = _concat_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    # Inner SumFactory has no state.
    server_state_type = computation_types.at_server(())

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assert_types_equivalent(process.initialize.type_signature,
                                 expected_initialize_type)

    # Inner SumFactory has no measurements.
    expected_measurements_type = computation_types.at_server(())
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assert_types_equivalent(process.next.type_signature,
                                 expected_next_type)

  @parameterized.named_parameters(
      ('float_value_float32_weight', tf.float32, tf.float32),
      ('struct_value_float32_weight', _test_struct_type_nested, tf.float32),
      ('float_value_float64_weight', tf.float32, tf.float64),
      ('struct_value_float64_weight', _test_struct_type_nested, tf.float64))
  def test_clip_type_properties_weighted(self, value_type, weight_type):
    factory = _concat_mean()
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)
    process = factory.create(value_type, weight_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    # State comes from the inner MeanFactory.
    server_state_type = computation_types.at_server(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assert_types_equivalent(process.initialize.type_signature,
                                 expected_initialize_type)

    # Measurements come from the inner mean factory.
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(mean_value=(), mean_weight=()))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type),
            weight=computation_types.at_clients(weight_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assert_types_equivalent(process.next.type_signature,
                                 expected_next_type)

  @parameterized.named_parameters(('bool', tf.bool), ('string', tf.string),
                                  ('string_list', [tf.string, tf.string]),
                                  ('string_nested', [tf.string, [tf.string]]),
                                  ('variant', tf.variant))
  def test_raises_on_non_numeric_dtypes(self, value_type):
    factory = _concat_sum()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'must all be integers or floats'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('int32_int64_list', [tf.int32, tf.int64]),
      ('float32_float64_list', [tf.float32, tf.float64]),
      ('float32_float64_nested', [tf.float32, (tf.float64, [tf.float64])]),
      ('int32_float32_list', [tf.int32, tf.float32]),
      ('int32_float32_list_mixed_rank', [(tf.int32, (2, 3, 4)), tf.float32]),
      ('int32_float32_nested', [tf.int32, (tf.float32, [tf.float32])]),
      ('int32_string_list', [tf.int32, tf.string]))
  def test_raises_on_mixed_dtypes(self, value_type):
    factory = _concat_sum()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'should have the same dtype'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('plain_struct', [('a', tf.int32)]),
      ('sequence', computation_types.SequenceType(tf.int32)),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('nested_sequence', [[[computation_types.SequenceType(tf.int32)]]]),
      ('nested_function', [computation_types.FunctionType(tf.int32, tf.int32)]))
  def test_raises_on_bad_tff_value_types(self, value_type):
    factory = _concat_sum()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'Expected `value_type` to be'):
      factory.create(value_type)


class ConcatFactoryExecutionTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar', tf.int32, [1, 2, 3], 6),
      ('rank_1_tensor', (tf.int32, [3]), [(1, 1, 1), (2, 2, 2)],
       (3, 3, 3)), ('rank_2_tensor', (tf.int32, [2, 2]), [((1, 1), (1, 1)),
                                                          ((2, 2), (2, 2))],
                    ((3, 3), (3, 3))),
      ('nested', _test_struct_type_nested,
       [_make_test_struct_nested(1),
        _make_test_struct_nested(2)], _make_test_struct_nested(3)))
  def test_concat_sum(self, value_type, client_data, expected_sum):
    factory = _concat_sum()
    process = factory.create(computation_types.to_type(value_type))

    state = process.initialize()
    self.assertEqual(state, ())

    output = process.next(state, client_data)
    self.assertEqual(output.state, ())
    self.assertEqual(output.measurements, ())
    self.assertAllClose(output.result, expected_sum, atol=0)

  @parameterized.named_parameters(
      ('scalar', tf.float32, [1, 2, 3], [3, 4, 5], 26.0 / 12),
      ('rank_1_tensor', (tf.float32, [2]), [(1, 1), (5, 5)], [3, 1], (2, 2)),
      ('rank_2_tensor', (tf.float32, [2, 2]), [((1, 1), (1, 1)),
                                               ((5, 5), (5, 5))], [3, 1],
       ((2, 2), (2, 2))),
      ('nested', _test_struct_type_nested,
       [_make_test_struct_nested(1),
        _make_test_struct_nested(5)], [3, 1], _make_test_struct_nested(2)))
  def test_concat_mean(self, value_type, client_data, client_weight,
                       expected_mean):
    factory = _concat_mean()
    process = factory.create(
        computation_types.to_type(value_type),
        computation_types.to_type(tf.float32))

    expected_state = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=())
    expected_measurements = collections.OrderedDict(
        mean_value=(), mean_weight=())

    state = process.initialize()
    self.assertEqual(state, expected_state)

    output = process.next(state, client_data, client_weight)
    self.assertEqual(output.state, expected_state)
    self.assertEqual(output.measurements, expected_measurements)
    self.assertAllClose(output.result, expected_mean)

  @parameterized.named_parameters([
      ('scalars', [np.int32(0), np.int32(1),
                   np.int32(2)], np.arange(3)),
      ('rank_1_tensor', np.arange(10), np.arange(10)),
      ('rank_3_tensor', np.arange(24).reshape(3, 2, 4), np.arange(24)),
      ('rank_1_tensor_list', [np.arange(2),
                              np.arange(3)], np.array([0, 1, 0, 1, 2])),
      ('mixed_rank_tensor_list', [np.array([[0, 1], [2, 3]]),
                                  np.array([4, 5])], np.arange(6)),
      ('nested_tensors',
       (np.array([0]),
        [np.array([[1], [2]]),
         dict(a=np.array([3]), b=np.array([4, 5]))]), np.arange(6)),
      ('large_rank_1_tensor_list', [np.arange(100),
                                    np.arange(100, 500)], np.arange(500))
  ])
  def test_concat_impl(self, value, expected_concat_value):
    """Checks the structure gets flattened/concatenated and packed correctly."""
    # Need to convert np arrays to tensors first.
    value = tf.nest.map_structure(tf.constant, value)
    concat_value = concat._concat_impl(value)
    self.assertAllEqual(concat_value, expected_concat_value)

  @parameterized.named_parameters([
      ('scalars', [np.int32(0), np.int32(1),
                   np.int32(2)], np.arange(3)),
      ('rank_1_tensor', np.arange(10), np.arange(10)),
      ('rank_3_tensor', np.arange(24).reshape(3, 2, 4), np.arange(24)),
      ('rank_1_tensor_list', [np.arange(2),
                              np.arange(3)], np.array([0, 1, 0, 1, 2])),
      ('mixed_rank_tensor_list', [np.array([[0, 1], [2, 3]]),
                                  np.array([4, 5])], np.arange(6)),
      ('nested_tensors',
       (np.array([0]),
        [np.array([[1], [2]]),
         dict(a=np.array([3]), b=np.array([4, 5]))]), np.arange(6)),
      ('large_rank_1_tensor_list', [np.arange(100),
                                    np.arange(100, 500)], np.arange(500))
  ])
  def test_unconcat_impl(self, original_structure, concat_value):
    # Need to convert np arrays to tensors first.
    original_structure = tf.nest.map_structure(tf.constant, original_structure)
    concat_value = tf.constant(concat_value)

    packed_record = concat._unconcat_impl(concat_value, original_structure)
    tf.nest.assert_same_structure(packed_record, original_structure)
    self.assertAllClose(packed_record, original_structure, atol=0)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
