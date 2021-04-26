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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import primitives
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.test import static_assert

_MIN_MAX_TEST_DTYPES = [('int16', tf.int16), ('int32', tf.int32),
                        ('int64', tf.int64), ('float16', tf.float16),
                        ('float32', tf.float32), ('float64', tf.float64),
                        ('bfloat16', tf.bfloat16)]


class FederatedMinTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_min_scalar(self, dtype):

    @computations.federated_computation(computation_types.at_clients(dtype))
    def call_federated_min(value):
      return primitives.federated_min(value)

    self.assertEqual(
        computation_types.at_server(dtype),
        call_federated_min.type_signature.result)

    value = call_federated_min([
        tf.cast(1.0, dtype),
        tf.cast(2.0, dtype),
        tf.cast(5.0, dtype),
    ])
    self.assertAllClose(value, 1.0)

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_min_struct(self, dtype):
    struct_type = computation_types.at_clients(
        computation_types.to_type([dtype, (dtype, [2])]))

    @computations.federated_computation(struct_type)
    def call_federated_min(value):
      return primitives.federated_min(value)

    self.assertEqual(
        computation_types.at_server(struct_type.member),
        call_federated_min.type_signature.result)

    value = call_federated_min([
        [tf.cast(1.0, dtype), tf.cast([2.0, 8.0], dtype)],
        [tf.cast(2.0, dtype), tf.cast([6.0, -12.0], dtype)],
        [tf.cast(5.0, dtype), tf.cast([-1.0, 0.0], dtype)],
    ])
    self.assertAllClose(value, [1.0, [-1.0, -12.0]])

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_min_named_struct(self, dtype):
    struct_type = computation_types.at_clients(
        computation_types.to_type(collections.OrderedDict(x=dtype, y=dtype)))

    @computations.federated_computation(struct_type)
    def call_federated_min(value):
      return primitives.federated_min(value)

    self.assertEqual(
        computation_types.at_server(struct_type.member),
        call_federated_min.type_signature.result)

    value = call_federated_min([
        collections.OrderedDict(x=tf.cast(1.0, dtype), y=tf.cast(2.0, dtype)),
        collections.OrderedDict(x=tf.cast(2.0, dtype), y=tf.cast(6.0, dtype)),
        collections.OrderedDict(x=tf.cast(5.0, dtype), y=tf.cast(-1.0, dtype)),
    ])
    self.assertAllClose(value, collections.OrderedDict(x=1.0, y=-1.0))

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_min_nested_struct(self, dtype):
    struct_type = computation_types.at_clients(
        computation_types.to_type([[dtype, dtype], dtype]))

    @computations.federated_computation(struct_type)
    def call_federated_min(value):
      return primitives.federated_min(value)

    self.assertEqual(
        computation_types.at_server(struct_type.member),
        call_federated_min.type_signature.result)

    value = call_federated_min([
        [[tf.cast(1.0, dtype), tf.cast(2.0, dtype)],
         tf.cast(8.0, dtype)],
        [[tf.cast(2.0, dtype), tf.cast(6.0, dtype)],
         tf.cast(-12.0, dtype)],
        [[tf.cast(5.0, dtype), tf.cast(-1.0, dtype)],
         tf.cast(0.0, dtype)],
    ])
    self.assertAllClose(value, [[1.0, -1.0], -12.0])

  def test_federated_min_wrong_type(self):
    with self.assertRaisesRegex(TypeError, 'Unsupported dtype.'):

      @computations.federated_computation(computation_types.at_clients(tf.bool))
      def call_federated_min(value):
        return primitives.federated_min(value)

      call_federated_min([False])

  def test_federated_min_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.* argument must be a tff.Value placed at CLIENTS'):

      @computations.federated_computation(computation_types.at_server(tf.int32))
      def call_federated_min(value):
        return primitives.federated_min(value)

      call_federated_min([1, 2, 3])


class FederatedMaxTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_max_scalar(self, dtype):

    @computations.federated_computation(computation_types.at_clients(dtype))
    def call_federated_max(value):
      return primitives.federated_max(value)

    self.assertEqual(
        computation_types.at_server(dtype),
        call_federated_max.type_signature.result)

    value = call_federated_max([
        tf.cast(1.0, dtype),
        tf.cast(2.0, dtype),
        tf.cast(5.0, dtype),
    ])
    self.assertEqual(value, tf.cast(5.0, dtype))

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_max_struct(self, dtype):
    struct_type = computation_types.at_clients(
        computation_types.to_type([dtype, (dtype, [2])]))

    @computations.federated_computation(struct_type)
    def call_federated_max(value):
      return primitives.federated_max(value)

    self.assertEqual(
        computation_types.at_server(struct_type.member),
        call_federated_max.type_signature.result)

    value = call_federated_max([
        [tf.cast(1.0, dtype), tf.cast([2.0, 8.0], dtype)],
        [tf.cast(2.0, dtype), tf.cast([6.0, -12.0], dtype)],
        [tf.cast(5.0, dtype), tf.cast([-1.0, 0.0], dtype)],
    ])
    self.assertAllClose(value, [5.0, [6.0, 8.0]])

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_max_named_struct(self, dtype):
    struct_type = computation_types.at_clients(
        computation_types.to_type(collections.OrderedDict(x=dtype, y=dtype)))

    @computations.federated_computation(struct_type)
    def call_federated_max(value):
      return primitives.federated_max(value)

    self.assertEqual(
        computation_types.at_server(struct_type.member),
        call_federated_max.type_signature.result)

    value = call_federated_max([
        collections.OrderedDict(x=tf.cast(1.0, dtype), y=tf.cast(2.0, dtype)),
        collections.OrderedDict(x=tf.cast(2.0, dtype), y=tf.cast(6.0, dtype)),
        collections.OrderedDict(x=tf.cast(5.0, dtype), y=tf.cast(-1.0, dtype)),
    ])
    self.assertAllClose(value, collections.OrderedDict(x=5.0, y=6.0))

  @parameterized.named_parameters(_MIN_MAX_TEST_DTYPES)
  def test_federated_max_nested_struct(self, dtype):
    struct_type = computation_types.at_clients(
        computation_types.to_type([[dtype, dtype], dtype]))

    @computations.federated_computation(struct_type)
    def call_federated_max(value):
      return primitives.federated_max(value)

    self.assertEqual(
        computation_types.at_server(struct_type.member),
        call_federated_max.type_signature.result)

    value = call_federated_max([
        [[tf.cast(1.0, dtype), tf.cast(2.0, dtype)],
         tf.cast(8.0, dtype)],
        [[tf.cast(2.0, dtype), tf.cast(6.0, dtype)],
         tf.cast(-12.0, dtype)],
        [[tf.cast(5.0, dtype), tf.cast(-1.0, dtype)],
         tf.cast(0.0, dtype)],
    ])
    self.assertAllClose(value, [[5.0, 6.0], 8.0])

  def test_federated_max_wrong_type(self):
    with self.assertRaisesRegex(TypeError, 'Unsupported dtype.'):

      @computations.federated_computation(computation_types.at_clients(tf.bool))
      def call_federated_max(value):
        return primitives.federated_max(value)

      call_federated_max([True, False])

  def test_federated_max_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.*argument must be a tff.Value placed at CLIENTS.*'):

      @computations.federated_computation(
          computation_types.at_server(tf.float32))
      def call_federated_max(value):
        return primitives.federated_max(value)

      call_federated_max([1.0, 2.0, 3.0])


class FederatedSampleTest(tf.test.TestCase):

  def test_federated_sample_single_value(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    value = call_federated_sample([1.0, 2.0, 5.0])
    self.assertCountEqual(value, [1.0, 2.0, 5.0])

  def test_federated_sample_on_nested_scalars(self):
    tuple_type = collections.OrderedDict([
        ('x', tf.float32),
        ('y', tf.float32),
    ])

    @computations.federated_computation(
        computation_types.FederatedType(tuple_type, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    x0 = 0.0
    y0 = 1.0
    x1 = -1.0
    y1 = 5.0
    test_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    value = call_federated_sample(
        [test_type(x0, y0),
         test_type(x1, y1),
         test_type(2.0, -10.0)])
    result = value._asdict()
    i0 = list(result['x']).index(x0)
    i1 = list(result['y']).index(y1)

    # Assert shuffled in unison.
    self.assertEqual(result['y'][i0], y0)
    self.assertEqual(result['x'][i1], x1)

  def test_federated_sample_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.*argument must be a tff.Value placed at CLIENTS.*'):

      @computations.federated_computation(
          computation_types.FederatedType(tf.bool, placements.SERVER))
      def call_federated_sample(value):
        return primitives.federated_sample(value)

      call_federated_sample([True, False, True, True])

  def test_federated_sample_max_size_is_100(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    value = call_federated_sample([1.0] * 100 + [0.0] * 100)
    self.assertLen(value, 100)
    self.assertAlmostEqual(len(np.nonzero(value)[0]), 50, delta=20)

  def test_federated_sample_preserves_nan_percentage(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    value = call_federated_sample([1.0] * 100 + [np.nan] * 100)
    self.assertAlmostEqual(np.count_nonzero(np.isnan(value)), 50, delta=20)

  def test_federated_sample_preserves_inf_percentage(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    value = call_federated_sample([1.0] * 100 + [np.inf] * 100)
    self.assertAlmostEqual(np.count_nonzero(np.isinf(value)), 50, delta=20)

  def test_federated_sample_named_tuple_type_of_ordered_dict(self):
    dict_type = computation_types.to_type(
        collections.OrderedDict([('x', tf.float32), ('y', tf.float32)]))

    @computations.federated_computation(
        computation_types.FederatedType(dict_type, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    x = 0.0
    y = 5.0
    test_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    value = call_federated_sample(
        [test_type(x, y),
         test_type(3.4, 5.6),
         test_type(1.0, 1.0)])
    result = value._asdict()

    self.assertIn(y, result['y'])
    self.assertIn(x, result['x'])

  def test_federated_sample_nested_named_tuples(self):
    tuple_test_type = (
        collections.OrderedDict([('x', tf.float32), ('y', tf.float32)]))
    dict_test_type = (
        computation_types.to_type(
            collections.OrderedDict([('a', tf.float32), ('b', tf.float32)])))
    nested_tuple_type = collections.OrderedDict([('tuple_1', tuple_test_type),
                                                 ('tuple_2', dict_test_type)])
    nested_test_type = collections.namedtuple('Nested', ['tuple_1', 'tuple_2'])

    @computations.federated_computation(
        computation_types.FederatedType(nested_tuple_type, placements.CLIENTS))
    def call_federated_sample(value):
      return primitives.federated_sample(value)

    tuple_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    dict_type = collections.namedtuple('NestedScalars', ['a', 'b'])
    value = call_federated_sample([
        nested_test_type(tuple_type(1.2, 2.2), dict_type(1.3, 8.8)),
        nested_test_type(tuple_type(-9.1, 3.1), dict_type(1.2, -5.4))
    ])._asdict(recursive=True)

    self.assertIn(1.2, value['tuple_1']['x'])
    self.assertIn(8.8, value['tuple_2']['b'])


class SecureQuantizedSumStaticAssertsTest(tf.test.TestCase,
                                          parameterized.TestCase):

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64),
                                  ('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_contains_static_aggregation(self, dtype):
    """Tests that built computation contains at least one secure sum call."""

    # Bounds provided as Python constants.
    @computations.federated_computation(
        computation_types.FederatedType((dtype, (2,)), placements.CLIENTS))
    def comp_py_bounds(value):
      return primitives.secure_quantized_sum(
          value, np.array(-1.0, dtype.as_numpy_dtype),
          np.array(1.0, dtype.as_numpy_dtype))

    static_assert.assert_contains_secure_aggregation(comp_py_bounds)

    # Bounds provided as tff values.
    @computations.federated_computation(
        computation_types.FederatedType((dtype, (2,)), placements.CLIENTS),
        computation_types.FederatedType(dtype, placements.SERVER),
        computation_types.FederatedType(dtype, placements.SERVER))
    def comp_tff_bounds(value, upper_bound, lower_bound):
      return primitives.secure_quantized_sum(value, upper_bound, lower_bound)

    static_assert.assert_contains_secure_aggregation(comp_tff_bounds)


class SecureQuantizedSumTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_client_tensor_shift_int_range(self, int_type):
    """Tests that client shift produces ints in expected range for SecAgg."""
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant([0, 1, 2, 3, 5, -254, -255, -256, 255, 256, 257],
                    dtype=int_type), tf.constant(-255, dtype=int_type),
        tf.constant(256, dtype=int_type))
    self.assertAllEqual([255, 256, 257, 258, 260, 1, 0, 0, 510, 511, 511],
                        encoded)

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_client_tensor_shift_int_range_bounds_equal(self, int_type):
    """Tests that ints are in expected range for SecAgg for equal bounds."""
    # Ensure all outputs are exactly lower_bound == upper_bound and there are no
    # off by 1 errors.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant([-1, 0, 1, 2, 3, 5, -254, -255, -256, 255, 256, 257],
                    dtype=int_type), tf.constant(0, dtype=int_type),
        tf.constant(0, dtype=int_type))
    self.assertAllEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], encoded)

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_client_tensor_shift_int_range_shifted_bounds_equal(self, int_type):
    """Tests that ints are in expected range for SecAgg for equal nonzero bounds."""
    # Ensure all outputs are exactly lower_bound == upper_bound and there are no
    # off by 1 errors.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant([-1, 0, 1, 2, 3, 5, -254, -255, -256, 255, 256, 257],
                    dtype=int_type), tf.constant(1, dtype=int_type),
        tf.constant(1, dtype=int_type))
    self.assertAllEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], encoded)

  def test_client_tensor_shift_int64_bound_max(self):
    """Tests that client shift handles edge-cases with tf.int64 properly."""
    # Check upper_bound - lower_bound == _SECAGG_MAX doesn't cause off by
    # 1 issue in calculating scale factor. We expect quantization with a scale
    # factor of 2.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant(
            [-1, 0, 2**31, 2**32, 2**32 - 1, 2**32 - 2, 2**32 - 3, 2**63 - 1],
            dtype=tf.int64), tf.constant(0, dtype=tf.int64),
        tf.constant(2**32 - 1, dtype=tf.int64))
    self.assertAllEqual(
        [0, 0, 2**30, 2**31 - 1, 2**31 - 1, 2**31 - 1, 2**31 - 2, 2**31 - 1],
        encoded)

  def test_client_tensor_shift_int64_bound_max_decremented(self):
    """Tests that client shift handles edge-cases with tf.int64 properly."""
    # Check upper_bound - lower_bound == _SECAGG_MAX - 1 doesn't cause off by
    # 1 issue in calculating scale factor. We expect exact results without
    # scaling.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant(
            [-1, 0, 2**31, 2**32, 2**32 - 1, 2**32 - 2, 2**32 - 3, 2**63 - 1],
            dtype=tf.int64), tf.constant(0, dtype=tf.int64),
        tf.constant(2**32 - 2, dtype=tf.int64))
    self.assertAllEqual(
        [0, 0, 2**31, 2**32 - 2, 2**32 - 2, 2**32 - 2, 2**32 - 3, 2**32 - 2],
        encoded)

  def test_client_tensor_shift_int64_bound_max_incremented(self):
    """Tests that client shift handles edge-cases with tf.int64 properly."""
    # Check upper_bound - lower_bound == _SECAGG_MAX + 1 doesn't cause off by
    # 1 issue in calculating scale factor. We expect exact results without
    # scaling.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant(
            [-1, 0, 2**31, 2**32, 2**32 + 1, 2**32 - 2, 2**32 - 3, 2**63 - 1],
            dtype=tf.int64), tf.constant(0, dtype=tf.int64),
        tf.constant(2**32, dtype=tf.int64))
    self.assertAllEqual(
        [0, 0, 2**30, 2**31, 2**31, 2**31 - 1, 2**31 - 2, 2**31], encoded)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_client_tensor_shift_float_range(self, float_type):
    """Tests that float client shift produces ints in expected range for SecAgg."""
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant([-2.0, -1.0, 1.0, 2.0], dtype=float_type),
        tf.constant(-1.0, dtype=float_type), tf.constant(1.0, dtype=float_type))
    self.assertAllEqual([0, 0, 2**32 - 1, 2**32 - 1], encoded)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_client_tensor_shift_float_range_bounds_equal(self, float_type):
    """Tests that float client shift produces ints in expected range for SecAgg."""
    # Ensure lower_bound == upper_bound doesn't cause division by zero issues.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant([-1, 0, 1, 2, 3, 5, -254, -255, -256, 255, 256, 257],
                    dtype=float_type), tf.constant(0.0, dtype=float_type),
        tf.constant(0.0, dtype=float_type))
    self.assertAllEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], encoded)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_client_tensor_shift_float_shifted_range_bounds_equal(
      self, float_type):
    """Tests that float client shift produces ints in expected range for SecAgg."""
    # Ensure lower_bound == upper_bound doesn't cause division by zero issues.
    encoded = primitives._client_tensor_shift_for_secure_sum(
        tf.constant([-1, 0, 1, 2, 3, 5, -254, -255, -256, 255, 256, 257],
                    dtype=float_type), tf.constant(1.1, dtype=float_type),
        tf.constant(1.1, dtype=float_type))
    self.assertAllEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], encoded)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_client_tensor_shift_invalid_bounds_float(self, float_type):
    """Ensures lower_bound > upper_bound causes error."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._client_tensor_shift_for_secure_sum(
          tf.constant([-2.0, -1.0, 1.0, 2.0], dtype=float_type),
          tf.constant(1.0, dtype=float_type),
          tf.constant(-1.0, dtype=float_type))

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_client_tensor_shift_invalid_bounds_int(self, int_type):
    """Ensures lower_bound > upper_bound causes error."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._client_tensor_shift_for_secure_sum(
          tf.constant([-2, -1, 1, 2], dtype=int_type),
          tf.constant(1, dtype=int_type), tf.constant(-1, dtype=int_type))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_server_tensor_shift_invalid_float_value_raises_error(
      self, float_type):
    """Ensures out-of-range summed values cause an error during server decode."""
    num_summands = tf.constant(1, tf.int32)

    # 2**32 is outside [0, 2**32 - 1], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(
          num_summands, tf.constant(2**32, tf.int64),
          tf.constant(-1.0, float_type), tf.constant(1.0, float_type),
          float_type)

    # -1 is outside [0, 2**32 - 1], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(
          num_summands, tf.constant(-1,
                                    tf.int64), tf.constant(-1.0, float_type),
          tf.constant(1.0, float_type), float_type)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_server_tensor_shift_invalid_float_value_multiple_summands_raises_error(
      self, float_type):
    """Ensures multiple summed values cause an error during server decode."""
    num_summands = tf.constant(2, tf.int32)

    # 2**33 - 1 is outside [0, (2**32 - 1) * 2], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(
          num_summands, tf.constant(2**33 - 1, tf.int64),
          tf.constant(-1.0, float_type), tf.constant(1.0, float_type),
          float_type)

    # -1 is outside [0, (2**32 - 1) * 2], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(
          num_summands, tf.constant(-1,
                                    tf.int64), tf.constant(-1.0, float_type),
          tf.constant(1.0, float_type), float_type)

    # This should work with 2 summands since 2**32 is within
    # [0, (2**32 - 1) * 2]. Note that this would cause an error with 1 summand.
    primitives._server_tensor_shift_for_secure_sum(
        num_summands, tf.constant(2**32, tf.int64),
        tf.constant(-1.0, float_type), tf.constant(1.0, float_type), float_type)

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_server_tensor_shift_invalid_int_value_raises_error(self, int_type):
    """Ensures out-of-range summed values cause an error during server decode."""
    num_summands = tf.constant(1, tf.int32)

    # 2**32 is outside [0, 2**32 - 1], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(
          num_summands, tf.constant(2**32, tf.int64), tf.constant(-1, int_type),
          tf.constant(1, int_type), int_type)

    # -1 is outside [0, 2**32 - 1], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(num_summands,
                                                     tf.constant(-1, tf.int64),
                                                     tf.constant(-1, int_type),
                                                     tf.constant(1, int_type),
                                                     int_type)

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_server_tensor_shift_invalid_int_value_multiple_summands_raises_error(
      self, int_type):
    """Ensures multiple summed values cause an error during server decode."""
    num_summands = tf.constant(2, tf.int32)

    # 2**33 - 1 is outside [0, (2**32 - 1) * 2], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(
          num_summands, tf.constant(2**33 - 1, tf.int64),
          tf.constant(-1, int_type), tf.constant(1, int_type), int_type)

    # -1 is outside [0, (2**32 - 1) * 2], so expect an error.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      primitives._server_tensor_shift_for_secure_sum(num_summands,
                                                     tf.constant(-1, tf.int64),
                                                     tf.constant(-1, int_type),
                                                     tf.constant(1, int_type),
                                                     int_type)

    # This should work with 2 summands since 2**32 is within
    # [0, (2**32 - 1) * 2]. Note that this would cause an error with 1 summand.
    primitives._server_tensor_shift_for_secure_sum(num_summands,
                                                   tf.constant(2**32, tf.int64),
                                                   tf.constant(-1, int_type),
                                                   tf.constant(1, int_type),
                                                   int_type)

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_scalar_int_type_py_range(self, int_type):
    """Tests value of integer scalar type and scalar np range."""
    call_secure_sum = _build_test_sum_fn_py_bounds(int_type,
                                                   _np_val_fn(0, int_type),
                                                   _np_val_fn(255, int_type))
    self.assertEqual(0, call_secure_sum([0]))
    self.assertEqual(278, call_secure_sum([0, 1, 255, 22]))

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_tensor_int_type_py_range(self, int_type):
    """Tests value of integer tensor type and scalar np range."""
    t_type = computation_types.TensorType(int_type, (2,))
    call_secure_sum = _build_test_sum_fn_py_bounds(t_type,
                                                   _np_val_fn(0, int_type),
                                                   _np_val_fn(255, int_type))
    data = [(0, 0), (1, 2), (255, 5), (22, 123)]
    self.assertAllEqual((278, 130), call_secure_sum(data))

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_composite_int_type_py_range(self, int_type):
    """Tests value of integer composite type and scalar np range."""
    t_type = computation_types.to_type(((int_type, (2,)), (int_type, (3,))))
    call_secure_sum = _build_test_sum_fn_py_bounds(t_type,
                                                   _np_val_fn(0, int_type),
                                                   _np_val_fn(255, int_type))
    data = [((0, 0), (0, 0, 0)),
            ((1, 2), (3, 4, 5)),
            ((255, 5), (71, 11, 201)),
            ((22, 123), (255, 0, 64))]  # pyformat: disable
    result = call_secure_sum(data)
    self.assertAllEqual((278, 130), result[0])
    self.assertAllEqual((329, 15, 270), result[1])

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_composite_int_type_composite_py_range(self, int_type):
    """Tests value of integer composite type and composite np range."""
    t_type = computation_types.to_type(((int_type, (2,)), (int_type, (3,))))
    call_secure_sum = _build_test_sum_fn_py_bounds(
        t_type, (_np_val_fn(0, int_type), _np_val_fn(63, int_type)),
        (_np_val_fn(255, int_type), _np_val_fn(64, int_type)))
    data = [((0, 0), (0, 0, 0)),
            ((1, 2), (3, 4, 5)),
            ((255, 5), (71, 11, 201)),
            ((22, 123), (255, 0, 64))]  # pyformat: disable
    result = call_secure_sum(data)
    self.assertAllEqual((278, 130), result[0])
    self.assertAllEqual((254, 252, 254), result[1])

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_scalar_int_type_tff_range(self, int_type):
    """Tests value of integer scalar type and scalar tff range."""
    call_secure_sum = _build_test_sum_fn_tff_bounds(int_type, int_type,
                                                    int_type)
    self.assertEqual(0, call_secure_sum([0], 0, 255))
    self.assertEqual(258, call_secure_sum([0, 255, 1, 2], 0, 255))

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_tensor_int_type_tff_range(self, int_type):
    """Tests value of integer tensor type and scalar tff range."""
    call_secure_sum = _build_test_sum_fn_tff_bounds(
        computation_types.TensorType(int_type, (2,)), int_type, int_type)
    self.assertAllEqual([256, 7],
                        call_secure_sum([[0, 0], [1, 2], [255, 5]], 0, 255))

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_composite_int_type_tff_range(self, int_type):
    """Tests value of integer composite type and scalar tff range."""
    t_type = computation_types.to_type(((int_type, (2,)), (int_type, (3,))))
    call_secure_sum = _build_test_sum_fn_tff_bounds(t_type, int_type, int_type)
    data = [((0, 0), (0, 0, 0)),
            ((1, 2), (3, 4, 5)),
            ((255, 5), (71, 11, 201)),
            ((22, 123), (255, 0, 64))]  # pyformat: disable
    result = call_secure_sum(data, 0, 255)
    self.assertAllEqual((278, 130), result[0])
    self.assertAllEqual((329, 15, 270), result[1])

  @parameterized.named_parameters(('int32', tf.int32), ('int64', tf.int64))
  def test_composite_int_type_composite_tff_range(self, int_type):
    """Tests value of integer composite type and composite tff range."""
    t_type = computation_types.to_type(((int_type, (2,)), (int_type, (3,))))
    call_secure_sum = _build_test_sum_fn_tff_bounds(t_type,
                                                    (int_type, int_type),
                                                    (int_type, int_type))
    data = [((0, 0), (0, 0, 0)),
            ((1, 2), (3, 4, 5)),
            ((255, 5), (71, 11, 201)),
            ((22, 123), (255, 0, 64))]  # pyformat: disable
    result = call_secure_sum(data, (0, 63), (255, 64))
    self.assertAllEqual((278, 130), result[0])
    self.assertAllEqual((254, 252, 254), result[1])

  def test_int_type_non_zero_lower_bound(self):
    """Tests that the integer logic is correct when lower bound is not zero."""
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.int32, 4, 20)
    self.assertEqual(4, call_secure_sum([3]))
    self.assertEqual(4, call_secure_sum([4]))
    self.assertEqual(5, call_secure_sum([5]))
    self.assertEqual(20, call_secure_sum([20]))
    self.assertEqual(20, call_secure_sum([21]))

    self.assertEqual(8, call_secure_sum([4, 4]))
    self.assertEqual(24, call_secure_sum([4, 20]))
    self.assertEqual(40, call_secure_sum([4] * 10))
    self.assertEqual(42, call_secure_sum([10, 12, 20]))

  def test_int_type_range_more_than_2_power_32(self):
    """Tests that summation is not exact if range spans more than 2**32."""
    # Range of size 2 * 2**32 - 1.
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.int64, -2**32 + 1,
                                                   2**32 - 2)
    result_0 = call_secure_sum([0])
    result_1 = call_secure_sum([1])
    result_2 = call_secure_sum([2])

    # Because (upper_bound - lower_bound) is 2**33 - 1, and the maximal bitwidth
    # is 32, the accuracy of summation is limited to 2 "units". A unit is
    # smallest difference between different inputs, in this case 1. As such, any
    # element should be represented with accuracy of 1, and any three
    # consecutive elements should map to two distinct values. This way, this
    # test is not sensitive to changes in implementation details.
    self.assertLessEqual(np.abs(result_0 - 0), 1)
    self.assertLessEqual(np.abs(result_1 - 1), 1)
    self.assertLessEqual(np.abs(result_2 - 2), 1)
    self.assertLen(set((result_0, result_1, result_2)), 2)

    result = call_secure_sum([-2**32 + 1, 2**32 - 2])
    # Each element can be represented incorrectly up to accuracy of 1.
    self.assertLessEqual(np.abs(result - 0), 2)

  def test_int_type_out_of_range(self):
    """Tests that integer value is clipped if outside of specified range."""
    # Range specified as Python constants.
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.int32, 0, 255)
    self.assertEqual(0, call_secure_sum([-1]))
    self.assertEqual(0, call_secure_sum([-255]))
    self.assertEqual(0, call_secure_sum([-256]))
    self.assertEqual(255, call_secure_sum([256]))
    self.assertEqual(255, call_secure_sum([12345]))
    self.assertEqual(255, call_secure_sum([-10, 256]))

    # Range specified as tff values.
    call_secure_sum = _build_test_sum_fn_tff_bounds(tf.int32, tf.int32,
                                                    tf.int32)
    self.assertEqual(0, call_secure_sum([-1], 0, 255))
    self.assertEqual(0, call_secure_sum([-255], 0, 255))
    self.assertEqual(0, call_secure_sum([-256], 0, 255))
    self.assertEqual(255, call_secure_sum([256], 0, 255))
    self.assertEqual(255, call_secure_sum([12345], 0, 255))
    self.assertEqual(255, call_secure_sum([-10, 256], 0, 255))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_scalar_float_type_py_range(self, float_type):
    """Tests value of float scalar type and scalar np range."""
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type,
                                                   _np_val_fn(0.0, float_type),
                                                   _np_val_fn(1.0, float_type))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(1.0, call_secure_sum([1.0]))
    self.assertAllClose(1.0, call_secure_sum([0.0, 1.0]))
    self.assertAllClose(1.8, call_secure_sum([0.0, 0.1, 0.7, 1.0]))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_tensor_float_type_py_range(self, float_type):
    """Tests value of float tensor type and scalar np range."""
    t_type = computation_types.TensorType(float_type, (2,))
    call_secure_sum = _build_test_sum_fn_py_bounds(t_type,
                                                   _np_val_fn(0.0, float_type),
                                                   _np_val_fn(1.0, float_type))
    data = [(0.0, 0.0), (0.1, 0.55), (0.7, 0.15), (1.0, 0.99)]
    self.assertAllClose((1.8, 1.69), call_secure_sum(data))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_composite_float_type_py_range(self, float_type):
    """Tests value of float composite type and scalar np range."""
    t_type = computation_types.to_type(((float_type, (2,)), (float_type, (3,))))
    call_secure_sum = _build_test_sum_fn_py_bounds(t_type,
                                                   _np_val_fn(0.0, float_type),
                                                   _np_val_fn(1.0, float_type))
    data = [((0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.1, 0.55), (0.3, 0.4, 0.5)),
            ((0.7, 0.15), (0.1234, 0.0001, 0.9999)),
            ((1.0, 0.99), (0.5, 0.7, 0.33))]  # pyformat: disable
    result = call_secure_sum(data)
    self.assertAllClose((1.8, 1.69), result[0])
    self.assertAllClose((0.9234, 1.1001, 1.8299), result[1])

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_composite_float_type_composite_py_range(self, float_type):
    """Tests value of float composite type and composite np range."""
    t_type = computation_types.to_type(((float_type, (2,)), (float_type, (3,))))
    call_secure_sum = _build_test_sum_fn_py_bounds(
        t_type, (_np_val_fn(0.0, float_type), _np_val_fn(0.2, float_type)),
        (_np_val_fn(1.0, float_type), _np_val_fn(0.9, float_type)))
    data = [((0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.1, 0.55), (0.3, 0.4, 0.5)),
            ((0.7, 0.15), (0.1234, 0.0001, 0.9999)),
            ((1.0, 0.99), (0.5, 0.7, 0.33))]  # pyformat: disable
    result = call_secure_sum(data)
    self.assertAllClose((1.8, 1.69), result[0])
    self.assertAllClose((1.2, 1.5, 1.93), result[1])

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_scalar_float_type_tff_range(self, float_type):
    """Tests value of float scalar type and scalar tff range."""
    call_secure_sum = _build_test_sum_fn_tff_bounds(float_type, float_type,
                                                    float_type)
    self.assertEqual(0.0, call_secure_sum([0.0], 0.0, 1.0))
    self.assertEqual(1.0, call_secure_sum([1.0], 0.0, 1.0))
    self.assertAllClose(1.0, call_secure_sum([0.0, 1.0], 0.0, 1.0))
    self.assertAllClose(1.8, call_secure_sum([0.0, 0.1, 0.7, 1.0], 0.0, 1.0))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_tensor_float_type_tff_range(self, float_type):
    """Tests value of float tensor type and scalar tff range."""
    t_type = computation_types.TensorType(float_type, (2,))
    call_secure_sum = _build_test_sum_fn_tff_bounds(t_type, float_type,
                                                    float_type)
    data = [(0.0, 0.0), (0.1, 0.55), (0.7, 0.15), (1.0, 0.99)]
    self.assertAllClose((1.8, 1.69), call_secure_sum(data, 0.0, 1.0))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_composite_float_type_tff_range(self, float_type):
    """Tests value of float composite type and scalar tff range."""
    t_type = computation_types.to_type(((float_type, (2,)), (float_type, (3,))))
    call_secure_sum = _build_test_sum_fn_tff_bounds(t_type, float_type,
                                                    float_type)
    data = [((0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.1, 0.55), (0.3, 0.4, 0.5)),
            ((0.7, 0.15), (0.1234, 0.0001, 0.9999)),
            ((1.0, 0.99), (0.5, 0.7, 0.33))]  # pyformat: disable
    result = call_secure_sum(data, 0.0, 1.0)
    self.assertAllClose((1.8, 1.69), result[0])
    self.assertAllClose((0.9234, 1.1001, 1.8299), result[1])

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_composite_float_type_composite_tff_range(self, float_type):
    """Tests value of float composite type and composite tff range."""
    t_type = computation_types.to_type(((float_type, (2,)), (float_type, (3,))))
    call_secure_sum = _build_test_sum_fn_tff_bounds(t_type,
                                                    (float_type, float_type),
                                                    (float_type, float_type))
    data = [((0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.1, 0.55), (0.3, 0.4, 0.5)),
            ((0.7, 0.15), (0.1234, 0.0001, 0.9999)),
            ((1.0, 0.99), (0.5, 0.7, 0.33))]  # pyformat: disable
    result = call_secure_sum(data, (0.0, 0.2), (1.0, 0.9))
    self.assertAllClose((1.8, 1.69), result[0])
    self.assertAllClose((1.2, 1.5, 1.93), result[1])

  def test_float_type_out_of_range(self):
    """Tests that float value is clipped if outside of specified range."""
    # Range specified as Python constants.
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.float32, 0.0, 1.0)
    self.assertAllClose(0.0, call_secure_sum([-1.0]))
    self.assertAllClose(0.0, call_secure_sum([-0.001]))
    self.assertAllClose(1.0, call_secure_sum([2.9]))
    self.assertAllClose(1.0, call_secure_sum([1.0001]))
    self.assertAllClose(1.0, call_secure_sum([-0.5, 1.9]))

    # Range specified as tff values.
    call_secure_sum = _build_test_sum_fn_tff_bounds(tf.float32, tf.float32,
                                                    tf.float32)
    self.assertAllClose(0.0, call_secure_sum([-1.0], 0.0, 1.0))
    self.assertAllClose(0.0, call_secure_sum([-0.001], 0.0, 1.0))
    self.assertAllClose(1.0, call_secure_sum([2.9], 0.0, 1.0))
    self.assertAllClose(1.0, call_secure_sum([1.0001], 0.0, 1.0))
    self.assertAllClose(1.0, call_secure_sum([-0.5, 1.9], 0.0, 1.0))

  def test_float_type_non_zero_lower_bound(self):
    """Tests that the float logic is correct when lower bound is not zero."""
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.float32, 4.0, 20.0)
    self.assertAllClose(4.0, call_secure_sum([3.0]))
    self.assertAllClose(4.0, call_secure_sum([4.0]))
    self.assertAllClose(5.0, call_secure_sum([5.0]))
    self.assertAllClose(20.0, call_secure_sum([20.0]))
    self.assertEqual(20.0, call_secure_sum([21.0]))

    self.assertAllClose(8.0, call_secure_sum([4.0, 4.0]))
    self.assertAllClose(24.0, call_secure_sum([4.0, 20.0]))
    self.assertAllClose(40.0, call_secure_sum([4.0] * 10))
    self.assertAllClose(42.0, call_secure_sum([10.0, 12.0, 20.0]))

  def test_mixed_type_structure(self):
    """Tests a structure consisting of different dtypes can be aggregted."""
    t_type = computation_types.to_type(((tf.int32, (2,)), (tf.float32, (3,))))
    call_secure_sum = _build_test_sum_fn_py_bounds(t_type, (0, 0.0), (255, 1.0))
    data = [((0, 0), (0.0, 0.0, 0.0)),
            ((1, 2), (0.3, 0.4, 0.5)),
            ((255, 5), (0.1234, 0.0001, 0.9999)),
            ((22, 123), (0.5, 0.7, 0.33))]  # pyformat: disable
    result = call_secure_sum(data)
    self.assertAllEqual((278, 130), result[0])
    self.assertAllClose((0.9234, 1.1001, 1.8299), result[1])

  def test_numeric_border_conditions_int(self):
    """Tests that certain border conditions do not cause numeric issues."""
    # Ensure division by zero does not occur.
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.int32, 0, 0)
    self.assertEqual(0, call_secure_sum([-1]))
    self.assertEqual(0, call_secure_sum([0]))
    self.assertEqual(0, call_secure_sum([1]))
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.int32, 1, 1)
    self.assertEqual(1, call_secure_sum([0]))
    self.assertEqual(1, call_secure_sum([1]))
    self.assertEqual(1, call_secure_sum([2]))

    # Ensure that using entire int32 range does not somehow break down.
    call_secure_sum = _build_test_sum_fn_py_bounds(tf.int32, -2**31, -1 + 2**31)
    self.assertEqual(0, call_secure_sum([0]))
    self.assertEqual(-2**31, call_secure_sum([-2**31]))
    self.assertEqual(-1 + 2**31, call_secure_sum([-1 + 2**31]))
    self.assertEqual(6, call_secure_sum([1, 2, 3]))
    self.assertEqual(-1, call_secure_sum([-2**31, -1 + 2**31]))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_numeric_border_conditions_float(self, float_type):
    """Tests that certain border conditions do not cause numeric issues."""
    np_val_fn = lambda v: np.array(v, float_type.as_numpy_dtype)

    # Ensure division by zero does not occur.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type, np_val_fn(0.0),
                                                   np_val_fn(0.0))
    self.assertEqual(0.0, call_secure_sum([-1.0]))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([1.0]))
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type, np_val_fn(1.0),
                                                   np_val_fn(1.0))
    self.assertEqual(1.0, call_secure_sum([0.0]))
    self.assertEqual(1.0, call_secure_sum([1.0]))
    self.assertEqual(1.0, call_secure_sum([2.0]))
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type, np_val_fn(1.1),
                                                   np_val_fn(1.1))
    self.assertAlmostEqual(1.1, call_secure_sum([0.0]))
    self.assertAlmostEqual(1.1, call_secure_sum([1.1]))
    self.assertAlmostEqual(1.1, call_secure_sum([2.0]))

    # Ensure that even a large range leads to sum with a high accuracy.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type,
                                                   np_val_fn(-10**6 * 1.0),
                                                   np_val_fn(10**6 * 1.0))
    self.assertAllClose(1111001.0,
                        call_secure_sum([10**6, 10**5, 10**4, 1000, 1]))
    self.assertAllClose(1011001.0,
                        call_secure_sum([10**6, 10**5, 10**4, 1000, 1, -10**5]))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_numeric_precision_zeros_float(self, float_type):
    """Ensure that 0's are represented exactly regardless of clipping range."""
    np_val_fn = lambda v: np.array(v, float_type.as_numpy_dtype)

    # Small symmetric clipping range. Test multiple summands as well to ensure
    # dequantization shifting handles this without introducing additional
    # precision errors.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type, np_val_fn(-1.0),
                                                   np_val_fn(1.0))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([0.0, 0.0, 0.0]))
    self.assertEqual(
        call_secure_sum([0.552]), call_secure_sum([0.552, 0.0, 0.0]))

    # Medium-size symmetric clipping range.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type,
                                                   np_val_fn(-100.0),
                                                   np_val_fn(100.0))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([0.0, 0.0, 0.0]))
    # This ensures that any float precision errors introduced don't come from
    # adding 0.0.
    self.assertEqual(
        call_secure_sum([0.552]), call_secure_sum([0.552, 0.0, 0.0]))

    # Medium-size symmetric clipping range.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type,
                                                   np_val_fn(-10**6 * 1.0),
                                                   np_val_fn(10**6 * 1.0))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([0.0, 0.0, 0.0]))
    # This ensures that any float precision errors introduced don't come from
    # adding 0.0.
    self.assertEqual(
        call_secure_sum([0.552]), call_secure_sum([0.552, 0.0, 0.0]))

    # Non-symmetric clipping range.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type,
                                                   np_val_fn(-100.0),
                                                   np_val_fn(200.0))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([0.0, 0.0, 0.0]))
    # This ensures that any float precision errors introduced don't come from
    # adding 0.0.
    self.assertEqual(
        call_secure_sum([0.552]), call_secure_sum([0.552, 0.0, 0.0]))

    # Non-symmetric clipping range with lower bound close to 0.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type, np_val_fn(-1e-5),
                                                   np_val_fn(200.0))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([0.0, 0.0, 0.0]))
    # This ensures that any float precision errors introduced don't come from
    # adding 0.0.
    self.assertEqual(
        call_secure_sum([0.552]), call_secure_sum([0.552, 0.0, 0.0]))

    # Non-symmetric clipping range with lower bound very close to 0.
    call_secure_sum = _build_test_sum_fn_py_bounds(float_type,
                                                   np_val_fn(-1e-16),
                                                   np_val_fn(200.0))
    self.assertEqual(0.0, call_secure_sum([0.0]))
    self.assertEqual(0.0, call_secure_sum([0.0, 0.0, 0.0]))
    # This ensures that any float precision errors introduced don't come from
    # adding 0.0.
    self.assertEqual(
        call_secure_sum([0.552]), call_secure_sum([0.552, 0.0, 0.0]))

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_numeric_precision_small_values_float(self, float_type):
    """Ensure that small values are as accurate as expected."""
    np_val_fn = lambda v: np.array(v, float_type.as_numpy_dtype)

    def call_secure_sum_with_small_values(call_secure_sum_fn, orders):
      """Checks sum with multiple orders of magnitude of small inputs."""
      for i in range(1, orders + 1):
        value = 10**-i
        summed_value = call_secure_sum_fn([value, 0.0])
        # Ensure relative tolerance is within 10%. This means the sum is within
        # 0.1 * 10**-i of the expected value.
        self.assertLessEqual(np.abs(summed_value - value) / value, 0.1)

    # Small clipping range. Expect 9 significant figures relative to range, so
    # values as small as 10**-9 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(float_type,
                                                      np_val_fn(-1.0),
                                                      np_val_fn(1.0))
    call_secure_sum_with_small_values(call_secure_sum_fn, orders=9)

    # Large clipping range. Expect 9 significant figures relative to range, so
    # values as small as 10**-3 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(float_type,
                                                      np_val_fn(-10**6 * 1.0),
                                                      np_val_fn(10**6 * 1.0))
    call_secure_sum_with_small_values(call_secure_sum_fn, orders=3)

    # Asymmetric clipping range. Expect 9 significant figures relative to range,
    # so values as small as 10**-7 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(float_type,
                                                      np_val_fn(-1.0),
                                                      np_val_fn(100.0))
    call_secure_sum_with_small_values(call_secure_sum_fn, orders=7)

  def test_numeric_precision_represent_sums_float32(self):
    """Ensure that upper_bound + value for small float32 values is as accurate as expected."""
    np_val_fn = lambda v: np.array(v, tf.float32.as_numpy_dtype)

    def call_secure_sum_with_small_values(call_secure_sum_fn, orders,
                                          upper_bound):
      """Checks sum with multiple orders of magnitude of small inputs."""
      for i in range(1, orders + 1):
        # Test sums slightly above `upper_bound` since this uses additional bits
        # of the float mantissa, which can be a problem for `tf.float32`
        # especially in representing the final output.
        value = 10**-i
        summed_value = call_secure_sum_fn([value, upper_bound])
        # Ensure relative tolerance is within 50% relative to value. Note
        # that Python will incur numerical stability issues at 1e-16 but this
        # isn't a problem for the larger orders we test.
        self.assertLessEqual(
            np.abs(summed_value - value - upper_bound) / value, 0.5)

    # Small clipping range. Expect 7 significant figures relative to range, so
    # values as small as 10**-7 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(tf.float32,
                                                      np_val_fn(-1.0),
                                                      np_val_fn(1.0))
    call_secure_sum_with_small_values(
        call_secure_sum_fn, orders=7, upper_bound=1.0)

    # Large clipping range. Expect 7 significant figures relative to range, so
    # values as small as 10**-1 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(tf.float32,
                                                      np_val_fn(-10**6 * 1.0),
                                                      np_val_fn(10**6 * 1.0))
    call_secure_sum_with_small_values(
        call_secure_sum_fn, orders=1, upper_bound=10**6 * 1.0)

    # Asymmetric clipping range. Expect 7 significant figures relative to range,
    # so values as small as 10**-5 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(tf.float32,
                                                      np_val_fn(-1.0),
                                                      np_val_fn(100.0))
    call_secure_sum_with_small_values(
        call_secure_sum_fn, orders=5, upper_bound=100.0)

  def test_numeric_precision_represent_sums_float64(self):
    """Ensure that upper_bound + value for small float64 values is as accurate as expected."""
    np_val_fn = lambda v: np.array(v, tf.float64.as_numpy_dtype)

    def call_secure_sum_with_small_values(call_secure_sum_fn, orders,
                                          upper_bound):
      """Checks sum with multiple orders of magnitude of small inputs."""
      for i in range(1, orders + 1):
        # Test sums slightly above `upper_bound` since this uses additional bits
        # of the float mantissa, which can be a problem for `tf.float64` in
        # representing the final output.
        value = 10**-i
        summed_value = call_secure_sum_fn([value, upper_bound])
        # Ensure relative tolerance is within 50% relative to value. Note
        # that Python will incur numerical stability issues at 1e-16 but this
        # isn't a problem for the larger orders we test.
        self.assertLessEqual(
            np.abs(summed_value - value - upper_bound) / value, 0.5)

    # Small clipping range. Expect 9 significant figures relative to range, so
    # values as small as 10**-9 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(tf.float64,
                                                      np_val_fn(-1.0),
                                                      np_val_fn(1.0))
    call_secure_sum_with_small_values(
        call_secure_sum_fn, orders=9, upper_bound=1.0)

    # Large clipping range. Expect 9 significant figures relative to range, so
    # values as small as 10**-3 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(tf.float64,
                                                      np_val_fn(-10**6 * 1.0),
                                                      np_val_fn(10**6 * 1.0))
    call_secure_sum_with_small_values(
        call_secure_sum_fn, orders=3, upper_bound=10**6 * 1.0)

    # Asymmetric clipping range. Expect 9 significant figures relative to range,
    # so values as small as 10**-7 should be captured.
    call_secure_sum_fn = _build_test_sum_fn_py_bounds(tf.float64,
                                                      np_val_fn(-1.0),
                                                      np_val_fn(100.0))
    call_secure_sum_with_small_values(
        call_secure_sum_fn, orders=7, upper_bound=100.0)

  @parameterized.named_parameters(('int8', tf.int8), ('int16', tf.int16),
                                  ('float16', tf.float16))
  def test_client_value_bad_dtype_raises(self, bad_dtype):
    with self.assertRaises(primitives.UnsupportedDTypeError):
      _build_test_sum_fn_py_bounds(bad_dtype,
                                   np.array(0, bad_dtype.as_numpy_dtype),
                                   np.array(1, bad_dtype.as_numpy_dtype))

  def test_range_type_mismatch_raises(self):
    with self.assertRaises(primitives.ScalarBoundSimpleValueDTypeError):
      _build_test_sum_fn_py_bounds(tf.float32, 0, 1)
    with self.assertRaises(primitives.ScalarBoundSimpleValueDTypeError):
      _build_test_sum_fn_py_bounds(tf.int32, 0.0, 1.0)
    with self.assertRaises(primitives.ScalarBoundStructValueDTypeError):
      _build_test_sum_fn_py_bounds((tf.float32, tf.float64), 0.0, 1.0)
    with self.assertRaises(primitives.ScalarBoundStructValueDTypeError):
      _build_test_sum_fn_py_bounds((tf.int32, tf.int64), 0, 1)
    with self.assertRaises(primitives.BoundsDifferentSignaturesError):
      _build_test_sum_fn_py_bounds(tf.int32, 0, 1.0)
    with self.assertRaises(primitives.BoundsDifferentSignaturesError):
      _build_test_sum_fn_py_bounds(tf.int32, 0.0, 1)

  def test_bounds_different_types_raises(self):
    with self.assertRaises(primitives.BoundsDifferentTypesError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.CLIENTS))
      def call_secure_sum(value):  # pylint: disable=unused-variable
        lower_bound = intrinsics.federated_value(0, placements.SERVER)
        upper_bound = 1
        summed_value = primitives.secure_quantized_sum(value, lower_bound,
                                                       upper_bound)
        return summed_value

  def test_clients_placed_bounds_raises(self):
    with self.assertRaises(primitives.BoundsNotPlacedAtServerError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.CLIENTS))
      def call_secure_sum(value):  # pylint: disable=unused-variable
        lower_bound = intrinsics.federated_value(0, placements.CLIENTS)
        upper_bound = intrinsics.federated_value(1, placements.CLIENTS)
        summed_value = primitives.secure_quantized_sum(value, lower_bound,
                                                       upper_bound)
        return summed_value

  def test_range_structure_mismatch_raises(self):
    with self.assertRaises(primitives.StructuredBoundsTypeMismatchError):
      _build_test_sum_fn_py_bounds((tf.int32, tf.int32, tf.int32), (0, 0),
                                   (1, 1))
    with self.assertRaises(primitives.StructuredBoundsTypeMismatchError):
      _build_test_sum_fn_py_bounds((tf.int32, tf.int32, tf.int32),
                                   (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _build_test_sum_fn_py_bounds(value_type, lower_bound, upper_bound):
  """Example `federated_computation` using `secure_quantized_sum`.

  The provided `lower_bound` and `upper_bound` should be a Python constant or a
  numpy array which will be directly provided to the `secure_quantized_sum`
  operator.

  Args:
    value_type: A `tff.Type` of value to be used.
    lower_bound: A Python numeric constant or a numpy array.
    upper_bound: A Python numeric constant or a numpy array.

  Returns:
    A `tff.federated_computation` with type signature `(value_type@CLIENTS ->
    value_type@SERVER)`.
  """

  @computations.federated_computation(
      computation_types.FederatedType(value_type, placements.CLIENTS))
  def call_secure_sum(value):
    summed_value = primitives.secure_quantized_sum(value, lower_bound,
                                                   upper_bound)
    return summed_value

  return call_secure_sum


def _build_test_sum_fn_tff_bounds(value_type, lower_bound_type,
                                  upper_bound_type):
  """Example `federated_computation` using `secure_quantized_sum`.

  The provided `lower_bound_type` and `upper_bound_type` describes the
  `tff.Type` of the federated value which will be passed to the returned
  `federated_computation` and to the `secure_quantized_sum` operator.

  Args:
    value_type: A `tff.Type` of value to be used.
    lower_bound_type: A `tff.Type` of lower_bound to be used.
    upper_bound_type: A `tff.Type` of upper_bound to be used.

  Returns:
    A `tff.federated_computation` with type signature `((value_type@CLIENTS,
    lower_bound_type@SERVER, upper_bound_type@SERVER) -> value_type@SERVER)`.
  """

  @computations.federated_computation(
      computation_types.FederatedType(value_type, placements.CLIENTS),
      computation_types.FederatedType(lower_bound_type, placements.SERVER),
      computation_types.FederatedType(upper_bound_type, placements.SERVER))
  def call_secure_sum(value, lower_bound, upper_bound):
    summed_value = primitives.secure_quantized_sum(value, lower_bound,
                                                   upper_bound)
    return summed_value

  return call_secure_sum


def _np_val_fn(value, tf_dtype):
  """Converts `value` to numpy array of dtype corresponding to `tf_dtype`."""
  return np.array(value, tf_dtype.as_numpy_dtype)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
