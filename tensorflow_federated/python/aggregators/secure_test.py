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

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert

_float_at_server = computation_types.at_server(tf.float32)
_float_at_clients = computation_types.at_clients(tf.float32)
_int_at_server = computation_types.at_server(tf.int32)
_int_at_clients = computation_types.at_clients(tf.int32)


def _test_struct_type(dtype):
  return ((dtype, (2,)), dtype)


def _test_float_init_fn(factor):
  return computations.federated_computation(
      lambda: intrinsics.federated_value(factor * 1.0, placements.SERVER))


def _test_float_next_fn(factor):
  shift_one = computations.tf_computation(lambda x: x + (factor * 1.0))
  return computations.federated_computation(
      lambda state, value: intrinsics.federated_map(shift_one, state),
      _float_at_server, _float_at_clients)


_test_float_report_fn = computations.federated_computation(
    lambda state: state, _float_at_server)


def _test_estimation_process(factor):
  return estimation_process.EstimationProcess(
      _test_float_init_fn(factor), _test_float_next_fn(factor),
      _test_float_report_fn)


def _measurements_type(bound_type):
  return computation_types.at_server(
      collections.OrderedDict(
          secure_upper_clipped_count=secure.COUNT_TF_TYPE,
          secure_lower_clipped_count=secure.COUNT_TF_TYPE,
          secure_upper_threshold=bound_type,
          secure_lower_threshold=bound_type))


class SecureModularSumFactoryComputationTest(test_case.TestCase,
                                             parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_non_symmetric_int32', 8, tf.int32, False),
      ('scalar_non_symmetric_int64', 8, tf.int64, False),
      ('struct_non_symmetric', 8, _test_struct_type(tf.int32), False),
      ('scalar_symmetric_int32', 8, tf.int32, True),
      ('scalar_symmetric_int64', 8, tf.int64, True),
      ('struct_symmetric', 8, _test_struct_type(tf.int32), True),
      ('numpy_modulus_non_symmetric', np.int32(8), tf.int32, False),
      ('numpy_modulus_symmetric', np.int32(8), tf.int32, True),
  )
  def test_type_properties(self, modulus, value_type, symmetric_range):
    factory_ = secure.SecureModularSumFactory(
        modulus=8, symmetric_range=symmetric_range)
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = factory_.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = computation_types.at_server(
        computation_types.to_type(()))
    expected_measurements_type = expected_state_type

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Factory returned an AggregationProcess containing '
                'non-secure aggregation.')

  def test_float_modulus_raises(self):
    with self.assertRaises(TypeError):
      secure.SecureModularSumFactory(modulus=8.0)
    with self.assertRaises(TypeError):
      secure.SecureModularSumFactory(modulus=np.float32(8.0))

  def test_modulus_not_positive_raises(self):
    with self.assertRaises(ValueError):
      secure.SecureModularSumFactory(modulus=0)
    with self.assertRaises(ValueError):
      secure.SecureModularSumFactory(modulus=-1)

  def test_symmetric_range_not_bool_raises(self):
    with self.assertRaises(TypeError):
      secure.SecureModularSumFactory(modulus=8, symmetric_range='True')

  @parameterized.named_parameters(
      ('float_type', computation_types.TensorType(tf.float32)),
      ('mixed_type', computation_types.to_type([tf.float32, tf.int32])),
      ('federated_type',
       computation_types.FederatedType(tf.int32, placements.SERVER)),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)))
  def test_incorrect_value_type_raises(self, bad_value_type):
    with self.assertRaises(TypeError):
      secure.SecureModularSumFactory(8).create(bad_value_type)


class SecureModularSumFactoryExecutionTest(test_case.TestCase,
                                           parameterized.TestCase):

  @parameterized.named_parameters(
      ('a', 2, [0, 1], 1),
      ('b', 2, [0, 1, 1], 0),
      ('c', 2, [0, 1, 1, 1], 1),
      ('d', 8, [1, 2, 3], 6),
      ('e', 8, [1, 2, 3, 4], 2),
      ('f', 22, [22], 0),
      ('g', 22, [220], 0),
      ('h', 22, [-1, 7], 6),
  )
  def test_non_symmetric(self, modulus, client_data, expected_sum):
    factory_ = secure.SecureModularSumFactory(modulus, symmetric_range=False)
    process = factory_.create(computation_types.to_type(tf.int32))
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(expected_sum, output.result)

  @parameterized.named_parameters(
      ('a', 2, [0, 1], 1),
      ('b', 2, [0, 1, 1], -1),
      ('c', 2, [0, 1, 1, 1], 0),
      ('d', 2, [0, -1], -1),
      ('e', 2, [0, -1, -1], 1),
      ('f', 2, [0, -1, -1, -1], 0),
      ('g', 8, [1, 2, 3], 6),
      ('h', 8, [1, 2, 3, 4], -5),
      ('i', 8, [-1, -2, -3, -4], 5),
      ('j', 22, [22], -21),
      ('k', 22, [-22], 21),
      ('l', 22, [43 * 5], 0),
      ('m', 22, [112, 123], 20),
  )
  def test_symmetric(self, modulus, client_data, expected_sum):
    factory_ = secure.SecureModularSumFactory(modulus, symmetric_range=True)
    process = factory_.create(computation_types.to_type(tf.int32))
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(expected_sum, output.result)

  def test_struct_type(self):
    factory_ = secure.SecureModularSumFactory(8)
    process = factory_.create(
        computation_types.to_type(_test_struct_type(tf.int32)))
    state = process.initialize()
    client_data = [(tf.constant([1, 2]), tf.constant(3)),
                   (tf.constant([4, 5]), tf.constant(6))]
    output = process.next(state, client_data)
    self.assertAllEqual([5, 7], output.result[0])
    self.assertEqual(1, output.result[1])


class SecureSumFactoryComputationTest(test_case.TestCase,
                                      parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_scalar', tf.float32, 1.0, -1.0, tf.float32),
      ('float_np', tf.float32, np.float32(1.0), np.float32(-1.0), tf.float32),
      ('float_struct', _test_struct_type(tf.float32), 1.0, -1.0, tf.float32),
      ('int_scalar', tf.int32, 1, -1, tf.int32),
      ('int_np', tf.int32, np.int32(1), np.int32(-1), tf.int32),
      ('int_struct', _test_struct_type(tf.int32), 1, -1, tf.int32),
      ('py_int_bounds_int64_value', tf.int64, 1, 0, tf.int64),
      ('numpy_int32_bounds_int64_value', tf.int64, np.int32(1), np.int32(0),
       tf.int64),
      ('numpy_int64_bounds_int32_value', tf.int32, np.int64(1), np.int64(0),
       tf.int32),
      ('py_float_bounds_float64_value', tf.float64, 1.0, 0.0, tf.float64),
      ('numpy_float32_bounds_float64_value', tf.float64, np.float32(1.0),
       np.float32(0.0), tf.float64),
      ('numpy_float64_bounds_float32_value', tf.float32, np.float64(1.0),
       np.float64(0.0), tf.float32),
  )
  def test_type_properties_constant_bounds(self, value_type, upper_bound,
                                           lower_bound, measurements_dtype):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=upper_bound, lower_bound_threshold=lower_bound)
    self.assertIsInstance(secure_sum_f, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = secure_sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = computation_types.at_server(
        computation_types.to_type(()))
    expected_measurements_type = _measurements_type(measurements_dtype)

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Factory returned an AggregationProcess containing '
                'non-secure aggregation.')

  @parameterized.named_parameters(
      ('float32_scalar', tf.float32, tf.float32),
      ('float64_scalar', tf.float64, tf.float64),
      ('float32_struct', _test_struct_type(tf.float32), tf.float32),
      ('float64_struct', _test_struct_type(tf.float64), tf.float64),
  )
  def test_type_properties_single_bound(self, value_type, dtype):
    upper_bound_process = _test_estimation_process(1)
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=upper_bound_process)
    self.assertIsInstance(secure_sum_f, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = secure_sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    threshold_type = upper_bound_process.report.type_signature.result.member
    expected_state_type = computation_types.at_server(
        computation_types.to_type(threshold_type))
    expected_measurements_type = _measurements_type(dtype)

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Factory returned an AggregationProcess containing '
                'non-secure aggregation.')

  @parameterized.named_parameters(
      ('float32_scalar', tf.float32, tf.float32),
      ('float64_scalar', tf.float64, tf.float64),
      ('float32_struct', _test_struct_type(tf.float32), tf.float32),
      ('float64_struct', _test_struct_type(tf.float64), tf.float64),
  )
  def test_type_properties_adaptive_bounds(self, value_type, dtype):
    upper_bound_process = _test_estimation_process(1)
    lower_bound_process = _test_estimation_process(-1)
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=upper_bound_process,
        lower_bound_threshold=lower_bound_process)
    self.assertIsInstance(secure_sum_f, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = secure_sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    threshold_type = upper_bound_process.report.type_signature.result.member
    expected_state_type = computation_types.at_server(
        computation_types.to_type((threshold_type, threshold_type)))
    expected_measurements_type = _measurements_type(dtype)

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Factory returned an AggregationProcess containing '
                'non-secure aggregation.')

  @parameterized.named_parameters(('int_smaller', -1, 1), ('int_equal', 1, 1),
                                  ('float_smaller', -1.0, 1.0),
                                  ('float_equal', 1.0, 1.0))
  def test_upper_bound_not_larger_than_lower_bound_raises(self, upper, lower):
    with self.assertRaises(ValueError):
      secure.SecureSumFactory(upper, lower)

  def test_int_ranges_beyond_2_pow_32(self):
    secure_sum_f = secure.SecureSumFactory(2**33, -2**33)
    # Bounds this large should be provided only with tf.int64 value_type.
    process = secure_sum_f.create(computation_types.TensorType(tf.int64))
    self.assertEqual(process.next.type_signature.result.result.member.dtype,
                     tf.int64)

  @parameterized.named_parameters(('py', 1, -1),
                                  ('np', np.int32(1), np.int32(-1)))
  def test_value_type_incompatible_with_config_mode_raises_int(
      self, upper, lower):
    secure_sum_f = secure.SecureSumFactory(upper, lower)
    with self.assertRaises(TypeError):
      secure_sum_f.create(computation_types.TensorType(tf.float32))

  @parameterized.named_parameters(('py', 1.0, -1.0),
                                  ('np', np.float32(1.0), np.float32(-1.0)))
  def test_value_type_incompatible_with_config_mode_raises_float(
      self, upper, lower):
    secure_sum_f = secure.SecureSumFactory(upper, lower)
    with self.assertRaises(TypeError):
      secure_sum_f.create(computation_types.TensorType(tf.int32))

  def test_value_type_incompatible_with_config_mode_raises_single_process(self):
    secure_sum_f = secure.SecureSumFactory(_test_estimation_process(1))
    with self.assertRaises(TypeError):
      secure_sum_f.create(computation_types.TensorType(tf.int32))

  def test_value_type_incompatible_with_config_mode_raises_two_processes(self):
    secure_sum_f = secure.SecureSumFactory(
        _test_estimation_process(1), _test_estimation_process(-1))
    with self.assertRaises(TypeError):
      secure_sum_f.create(computation_types.TensorType(tf.int32))

  @parameterized.named_parameters(
      ('federated_type',
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)))
  def test_incorrect_value_type_raises(self, bad_value_type):
    secure_sum_f = secure.SecureSumFactory(1.0, -1.0)
    with self.assertRaises(TypeError):
      secure_sum_f.create(bad_value_type)


class SecureSumFactoryExecutionTest(test_case.TestCase):

  def test_int_constant_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=1, lower_bound_threshold=-1)
    process = secure_sum_f.create(computation_types.to_type(tf.int32))
    client_data = [-2, -1, 0, 1, 2, 3]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(1, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1,
        expected_secure_lower_threshold=-1)

  def test_float_constant_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=1.0, lower_bound_threshold=-1.0)
    process = secure_sum_f.create(computation_types.to_type(tf.float32))
    client_data = [-2.5, -0.5, 0.0, 1.0, 1.5, 2.5]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(1.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1.0,
        expected_secure_lower_threshold=-1.0)

  def test_float_single_process_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=_test_estimation_process(1))
    process = secure_sum_f.create(computation_types.to_type(tf.float32))
    client_data = [-2.5, -0.5, 0.0, 1.0, 1.5, 3.5]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(1.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1.0,
        expected_secure_lower_threshold=-1.0)

    output = process.next(output.state, client_data)
    self.assertAllClose(2.0, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=2.0,
        expected_secure_lower_threshold=-2.0)

    output = process.next(output.state, client_data)
    self.assertAllClose(2.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=3.0,
        expected_secure_lower_threshold=-3.0)

  def test_float_two_processes_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=_test_estimation_process(1),
        lower_bound_threshold=_test_estimation_process(-1))
    process = secure_sum_f.create(computation_types.to_type(tf.float32))
    client_data = [-2.5, -0.5, 0.0, 1.0, 1.5, 3.5]

    state = process.initialize()
    output = process.next(state, client_data)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1.0,
        expected_secure_lower_threshold=-1.0)

    output = process.next(output.state, client_data)
    self.assertAllClose(2.0, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=2.0,
        expected_secure_lower_threshold=-2.0)

    output = process.next(output.state, client_data)
    self.assertAllClose(2.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=3.0,
        expected_secure_lower_threshold=-3.0)

  def test_float_32_larger_than_2_pow_32(self):
    secure_sum_f = secure.SecureSumFactory(upper_bound_threshold=float(2**34))
    process = secure_sum_f.create(computation_types.to_type(tf.float32))
    client_data = [float(2**33), float(2**33), float(2**34)]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(float(2**35), output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=0,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=float(2**34),
        expected_secure_lower_threshold=float(-2**34))

  def test_float_64_larger_than_2_pow_64(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=np.float64(2**66))
    process = secure_sum_f.create(computation_types.to_type(tf.float64))
    client_data = [np.float64(2**65), np.float64(2**65), np.float64(2**66)]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(np.array(2**67, np.float64), output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=0,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=np.float64(2**66),
        expected_secure_lower_threshold=np.float64(-2**66))

  def _check_measurements(self, measurements,
                          expected_secure_upper_clipped_count,
                          expected_secure_lower_clipped_count,
                          expected_secure_upper_threshold,
                          expected_secure_lower_threshold):
    self.assertEqual(expected_secure_upper_clipped_count,
                     measurements['secure_upper_clipped_count'])
    self.assertEqual(expected_secure_lower_clipped_count,
                     measurements['secure_lower_clipped_count'])
    self.assertAllClose(expected_secure_upper_threshold,
                        measurements['secure_upper_threshold'])
    self.assertAllClose(expected_secure_lower_threshold,
                        measurements['secure_lower_threshold'])


class IsStructureOfSingleDtypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('bool', computation_types.TensorType(tf.bool)),
      ('int', computation_types.TensorType(tf.int32)),
      ('ints', computation_types.StructType([tf.int32, tf.int32])),
      ('floats', computation_types.StructType([tf.float32, tf.float32])),
      ('nested_struct',
       computation_types.StructType([
           computation_types.TensorType(tf.int32),
           computation_types.StructType([tf.int32, tf.int32])
       ])),
      ('federated_floats_at_clients',
       computation_types.FederatedType(
           computation_types.StructType([tf.float32, tf.float32]),
           placements.CLIENTS)),
  )
  def test_returns_true(self, type_spec):
    self.assertTrue(secure._is_structure_of_single_dtype(type_spec))

  @parameterized.named_parameters(
      ('empty_struct', computation_types.StructType([])),
      ('int_and_float', computation_types.StructType([tf.int32, tf.float32])),
      ('int32_and_int64', computation_types.StructType([tf.int32, tf.int64])),
      ('float32_and_float64',
       computation_types.StructType([tf.float32, tf.float64])),
      ('nested_struct',
       computation_types.StructType([
           computation_types.TensorType(tf.int32),
           computation_types.StructType([tf.float32, tf.float32])
       ])),
      ('sequence_of_ints', computation_types.SequenceType(tf.int32)),
      ('placement', computation_types.PlacementType()),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('abstract', computation_types.AbstractType('T')),
  )
  def test_returns_false(self, type_spec):
    self.assertFalse(secure._is_structure_of_single_dtype(type_spec))


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
