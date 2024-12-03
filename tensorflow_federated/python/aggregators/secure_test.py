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
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

_float_at_server = federated_language.FederatedType(
    np.float32, federated_language.SERVER
)
_float_at_clients = federated_language.FederatedType(
    np.float32, federated_language.CLIENTS
)
_int_at_server = federated_language.FederatedType(
    np.int32, federated_language.SERVER
)
_int_at_clients = federated_language.FederatedType(
    np.int32, federated_language.CLIENTS
)


def _test_struct_type(dtype):
  return (dtype, (2,)), dtype


def _test_float_init_fn(factor):
  return federated_language.federated_computation(
      lambda: federated_language.federated_value(
          factor * 1.0, federated_language.SERVER
      )
  )


def _test_float_next_fn(factor):
  @tensorflow_computation.tf_computation
  def shift_one(x):
    return x + (factor * 1.0)

  return federated_language.federated_computation(
      lambda state, value: federated_language.federated_map(shift_one, state),
      _float_at_server,
      _float_at_clients,
  )


_test_float_report_fn = federated_language.federated_computation(
    lambda state: state, _float_at_server
)


def _test_estimation_process(factor):
  return estimation_process.EstimationProcess(
      _test_float_init_fn(factor),
      _test_float_next_fn(factor),
      _test_float_report_fn,
  )


def _measurements_type(bound_type):
  return federated_language.FederatedType(
      collections.OrderedDict(
          secure_upper_clipped_count=secure.COUNT_TYPE,
          secure_lower_clipped_count=secure.COUNT_TYPE,
          secure_upper_threshold=bound_type,
          secure_lower_threshold=bound_type,
      ),
      federated_language.SERVER,
  )


class SecureModularSumFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('scalar_non_symmetric_int32', 8, np.int32, False),
      ('scalar_non_symmetric_int64', 8, np.int64, False),
      ('struct_non_symmetric', 8, _test_struct_type(np.int32), False),
      ('scalar_symmetric_int32', 8, np.int32, True),
      ('scalar_symmetric_int64', 8, np.int64, True),
      ('struct_symmetric', 8, _test_struct_type(np.int32), True),
      ('numpy_modulus_non_symmetric', np.int32(8), np.int32, False),
      ('numpy_modulus_symmetric', np.int32(8), np.int32, True),
  )
  def test_type_properties(self, modulus, value_type, symmetric_range):
    factory_ = secure.SecureModularSumFactory(
        modulus=modulus, symmetric_range=symmetric_range
    )
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type(value_type)
    process = factory_.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = federated_language.FederatedType(
        (), federated_language.SERVER
    )
    expected_measurements_type = expected_state_type

    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=federated_language.FederatedType(
                value_type, federated_language.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=federated_language.FederatedType(
                value_type, federated_language.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )
    federated_language.framework.assert_not_contains_unsecure_aggregation(
        process.next
    )

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
      ('float_type', federated_language.TensorType(np.float32)),
      ('mixed_type', federated_language.to_type([np.float32, np.int32])),
      (
          'federated_type',
          federated_language.FederatedType(np.int32, federated_language.SERVER),
      ),
      ('function_type', federated_language.FunctionType(None, ())),
      ('sequence_type', federated_language.SequenceType(np.float32)),
  )
  def test_incorrect_value_type_raises(self, bad_value_type):
    with self.assertRaises(TypeError):
      secure.SecureModularSumFactory(8).create(bad_value_type)


class SecureModularSumFactoryExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

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
    process = factory_.create(federated_language.TensorType(np.int32))
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
    process = factory_.create(federated_language.TensorType(np.int32))
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(expected_sum, output.result)

  def test_struct_type(self):
    factory_ = secure.SecureModularSumFactory(8)
    process = factory_.create(
        federated_language.to_type(_test_struct_type(np.int32))
    )
    state = process.initialize()
    client_data = [
        (np.array([1, 2], np.int32), np.array(3, np.int32)),
        (np.array([4, 5], np.int32), np.array(6, np.int32)),
    ]
    output = process.next(state, client_data)
    self.assertAllEqual([5, 7], output.result[0])
    self.assertEqual(1, output.result[1])


class SecureSumFactoryComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_scalar', np.float32, 1.0, -1.0, np.float32),
      ('float_np', np.float32, np.float32(1.0), np.float32(-1.0), np.float32),
      ('float_struct', _test_struct_type(np.float32), 1.0, -1.0, np.float32),
      ('int_scalar', np.int32, 1, -1, np.int32),
      ('int_np', np.int32, np.int32(1), np.int32(-1), np.int32),
      ('int_struct', _test_struct_type(np.int32), 1, -1, np.int32),
      ('py_int_bounds_int64_value', np.int64, 1, 0, np.int64),
      (
          'numpy_int32_bounds_int64_value',
          np.int64,
          np.int32(1),
          np.int32(0),
          np.int64,
      ),
      (
          'numpy_int64_bounds_int32_value',
          np.int32,
          np.int64(1),
          np.int64(0),
          np.int32,
      ),
      ('py_float_bounds_float64_value', np.float64, 1.0, 0.0, np.float64),
      (
          'numpy_float32_bounds_float64_value',
          np.float64,
          np.float32(1.0),
          np.float32(0.0),
          np.float64,
      ),
      (
          'numpy_float64_bounds_float32_value',
          np.float32,
          np.float64(1.0),
          np.float64(0.0),
          np.float32,
      ),
  )
  def test_type_properties_constant_bounds(
      self, value_type, upper_bound, lower_bound, measurements_dtype
  ):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=upper_bound, lower_bound_threshold=lower_bound
    )
    self.assertIsInstance(secure_sum_f, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type(value_type)
    process = secure_sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = federated_language.FederatedType(
        (), federated_language.SERVER
    )
    expected_measurements_type = _measurements_type(measurements_dtype)

    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=federated_language.FederatedType(
                value_type, federated_language.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=federated_language.FederatedType(
                value_type, federated_language.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )
    federated_language.framework.assert_not_contains_unsecure_aggregation(
        process.next
    )

  @parameterized.named_parameters(
      ('float32_scalar', np.float32, np.float32),
      ('float64_scalar', np.float64, np.float64),
      ('float32_struct', _test_struct_type(np.float32), np.float32),
      ('float64_struct', _test_struct_type(np.float64), np.float64),
  )
  def test_type_properties_single_bound(self, value_type, dtype):
    upper_bound_process = _test_estimation_process(1)
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=upper_bound_process
    )
    self.assertIsInstance(secure_sum_f, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type(value_type)
    process = secure_sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    threshold_type = upper_bound_process.report.type_signature.result.member
    expected_state_type = federated_language.FederatedType(
        federated_language.to_type(threshold_type), federated_language.SERVER
    )
    expected_measurements_type = _measurements_type(dtype)

    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=federated_language.FederatedType(
                value_type, federated_language.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=federated_language.FederatedType(
                value_type, federated_language.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )
    federated_language.framework.assert_not_contains_unsecure_aggregation(
        process.next
    )

  @parameterized.named_parameters(
      ('float32_scalar', np.float32, np.float32),
      ('float64_scalar', np.float64, np.float64),
      ('float32_struct', _test_struct_type(np.float32), np.float32),
      ('float64_struct', _test_struct_type(np.float64), np.float64),
  )
  def test_type_properties_adaptive_bounds(self, value_type, dtype):
    upper_bound_process = _test_estimation_process(1)
    lower_bound_process = _test_estimation_process(-1)
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=upper_bound_process,
        lower_bound_threshold=lower_bound_process,
    )
    self.assertIsInstance(secure_sum_f, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type(value_type)
    process = secure_sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    threshold_type = upper_bound_process.report.type_signature.result.member
    expected_state_type = federated_language.FederatedType(
        federated_language.to_type((threshold_type, threshold_type)),
        federated_language.SERVER,
    )
    expected_measurements_type = _measurements_type(dtype)

    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=federated_language.FederatedType(
                value_type, federated_language.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=federated_language.FederatedType(
                value_type, federated_language.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )
    federated_language.framework.assert_not_contains_unsecure_aggregation(
        process.next
    )

  @parameterized.named_parameters(
      ('int_smaller', -1, 1),
      ('int_equal', 1, 1),
      ('float_smaller', -1.0, 1.0),
      ('float_equal', 1.0, 1.0),
  )
  def test_upper_bound_not_larger_than_lower_bound_raises(self, upper, lower):
    with self.assertRaises(ValueError):
      secure.SecureSumFactory(upper, lower)

  def test_int_ranges_beyond_2_pow_32(self):
    secure_sum_f = secure.SecureSumFactory(2**33, -(2**33))
    # Bounds this large should be provided only with np.int64 value_type.
    process = secure_sum_f.create(federated_language.TensorType(np.int64))
    self.assertEqual(
        process.next.type_signature.result.result.member.dtype, np.int64
    )

  @parameterized.named_parameters(
      ('py', 1, -1), ('np', np.int32(1), np.int32(-1))
  )
  def test_value_type_incompatible_with_config_mode_raises_int(
      self, upper, lower
  ):
    secure_sum_f = secure.SecureSumFactory(upper, lower)
    with self.assertRaises(TypeError):
      secure_sum_f.create(federated_language.TensorType(np.float32))

  @parameterized.named_parameters(
      ('py', 1.0, -1.0), ('np', np.float32(1.0), np.float32(-1.0))
  )
  def test_value_type_incompatible_with_config_mode_raises_float(
      self, upper, lower
  ):
    secure_sum_f = secure.SecureSumFactory(upper, lower)
    with self.assertRaises(TypeError):
      secure_sum_f.create(federated_language.TensorType(np.int32))

  def test_value_type_incompatible_with_config_mode_raises_single_process(self):
    secure_sum_f = secure.SecureSumFactory(_test_estimation_process(1))
    with self.assertRaises(TypeError):
      secure_sum_f.create(federated_language.TensorType(np.int32))

  def test_value_type_incompatible_with_config_mode_raises_two_processes(self):
    secure_sum_f = secure.SecureSumFactory(
        _test_estimation_process(1), _test_estimation_process(-1)
    )
    with self.assertRaises(TypeError):
      secure_sum_f.create(federated_language.TensorType(np.int32))

  @parameterized.named_parameters(
      (
          'federated_type',
          federated_language.FederatedType(
              np.float32, federated_language.SERVER
          ),
      ),
      ('function_type', federated_language.FunctionType(None, ())),
      ('sequence_type', federated_language.SequenceType(np.float32)),
  )
  def test_incorrect_value_type_raises(self, bad_value_type):
    secure_sum_f = secure.SecureSumFactory(1.0, -1.0)
    with self.assertRaises(TypeError):
      secure_sum_f.create(bad_value_type)


class SecureSumFactoryExecutionTest(tf.test.TestCase):

  def test_int_constant_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=1, lower_bound_threshold=-1
    )
    process = secure_sum_f.create(federated_language.TensorType(np.int32))
    client_data = [-2, -1, 0, 1, 2, 3]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(1, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1,
        expected_secure_lower_threshold=-1,
    )

  def test_float_constant_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=1.0, lower_bound_threshold=-1.0
    )
    process = secure_sum_f.create(federated_language.TensorType(np.float32))
    client_data = [-2.5, -0.5, 0.0, 1.0, 1.5, 2.5]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(1.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1.0,
        expected_secure_lower_threshold=-1.0,
    )

  def test_float_single_process_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=_test_estimation_process(1)
    )
    process = secure_sum_f.create(federated_language.TensorType(np.float32))
    client_data = [-2.5, -0.5, 0.0, 1.0, 1.5, 3.5]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(1.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1.0,
        expected_secure_lower_threshold=-1.0,
    )

    output = process.next(output.state, client_data)
    self.assertAllClose(2.0, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=2.0,
        expected_secure_lower_threshold=-2.0,
    )

    output = process.next(output.state, client_data)
    self.assertAllClose(2.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=3.0,
        expected_secure_lower_threshold=-3.0,
    )

  def test_float_two_processes_bounds(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=_test_estimation_process(1),
        lower_bound_threshold=_test_estimation_process(-1),
    )
    process = secure_sum_f.create(federated_language.TensorType(np.float32))
    client_data = [-2.5, -0.5, 0.0, 1.0, 1.5, 3.5]

    state = process.initialize()
    output = process.next(state, client_data)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=2,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=1.0,
        expected_secure_lower_threshold=-1.0,
    )

    output = process.next(output.state, client_data)
    self.assertAllClose(2.0, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=1,
        expected_secure_upper_threshold=2.0,
        expected_secure_lower_threshold=-2.0,
    )

    output = process.next(output.state, client_data)
    self.assertAllClose(2.5, output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=1,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=3.0,
        expected_secure_lower_threshold=-3.0,
    )

  def test_float_32_larger_than_2_pow_32(self):
    secure_sum_f = secure.SecureSumFactory(upper_bound_threshold=float(2**34))
    process = secure_sum_f.create(federated_language.TensorType(np.float32))
    client_data = [float(2**33), float(2**33), float(2**34)]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(float(2**35), output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=0,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=float(2**34),
        expected_secure_lower_threshold=float(-(2**34)),
    )

  def test_float_64_larger_than_2_pow_64(self):
    secure_sum_f = secure.SecureSumFactory(
        upper_bound_threshold=np.float64(2**66)
    )
    process = secure_sum_f.create(federated_language.TensorType(np.float64))
    client_data = [
        np.float64(2**65),
        np.float64(2**65),
        np.float64(2**66),
    ]

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertAllClose(np.array(2**67, np.float64), output.result)
    self._check_measurements(
        output.measurements,
        expected_secure_upper_clipped_count=0,
        expected_secure_lower_clipped_count=0,
        expected_secure_upper_threshold=np.float64(2**66),
        expected_secure_lower_threshold=np.float64(-(2**66)),
    )

  def _check_measurements(
      self,
      measurements,
      expected_secure_upper_clipped_count,
      expected_secure_lower_clipped_count,
      expected_secure_upper_threshold,
      expected_secure_lower_threshold,
  ):
    self.assertEqual(
        expected_secure_upper_clipped_count,
        measurements['secure_upper_clipped_count'],
    )
    self.assertEqual(
        expected_secure_lower_clipped_count,
        measurements['secure_lower_clipped_count'],
    )
    self.assertAllClose(
        expected_secure_upper_threshold, measurements['secure_upper_threshold']
    )
    self.assertAllClose(
        expected_secure_lower_threshold, measurements['secure_lower_threshold']
    )


class IsStructureOfSingleDtypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('bool', federated_language.TensorType(np.bool_)),
      ('int', federated_language.TensorType(np.int32)),
      ('ints', federated_language.StructType([np.int32, np.int32])),
      ('floats', federated_language.StructType([np.float32, np.float32])),
      (
          'nested_struct',
          federated_language.StructType([
              federated_language.TensorType(np.int32),
              federated_language.StructType([np.int32, np.int32]),
          ]),
      ),
      (
          'federated_floats_at_clients',
          federated_language.FederatedType(
              federated_language.StructType([np.float32, np.float32]),
              federated_language.CLIENTS,
          ),
      ),
  )
  def test_returns_true(self, type_spec):
    self.assertTrue(secure._is_structure_of_single_dtype(type_spec))

  @parameterized.named_parameters(
      ('empty_struct', federated_language.StructType([])),
      ('int_and_float', federated_language.StructType([np.int32, np.float32])),
      ('int32_and_int64', federated_language.StructType([np.int32, np.int64])),
      (
          'float32_and_float64',
          federated_language.StructType([np.float32, np.float64]),
      ),
      (
          'nested_struct',
          federated_language.StructType([
              federated_language.TensorType(np.int32),
              federated_language.StructType([np.float32, np.float32]),
          ]),
      ),
      ('sequence_of_ints', federated_language.SequenceType(np.int32)),
      ('placement', federated_language.PlacementType()),
      ('function', federated_language.FunctionType(np.int32, np.int32)),
      ('abstract', federated_language.AbstractType('T')),
  )
  def test_returns_false(self, type_spec):
    self.assertFalse(secure._is_structure_of_single_dtype(type_spec))


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
