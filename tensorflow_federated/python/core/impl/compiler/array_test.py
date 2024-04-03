# Copyright 2023, The TensorFlow Federated Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.proto.v0 import data_type_pb2
from tensorflow_federated.python.core.impl.compiler import array


class ArrayTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'bool',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BOOL,
              shape=array_pb2.ArrayShape(dim=[]),
              bool_list=array_pb2.Array.BoolList(value=[True]),
          ),
          np.bool_(True),
      ),
      (
          'int8',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT8,
              shape=array_pb2.ArrayShape(dim=[]),
              int8_list=array_pb2.Array.IntList(value=[1]),
          ),
          np.int8(1),
      ),
      (
          'int16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT16,
              shape=array_pb2.ArrayShape(dim=[]),
              int16_list=array_pb2.Array.IntList(value=[1]),
          ),
          np.int16(1),
      ),
      (
          'int32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(value=[1]),
          ),
          np.int32(1),
      ),
      (
          'int64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              int64_list=array_pb2.Array.Int64List(value=[1]),
          ),
          np.int64(1),
      ),
      (
          'uint8',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT8,
              shape=array_pb2.ArrayShape(dim=[]),
              uint8_list=array_pb2.Array.IntList(value=[1]),
          ),
          np.uint8(1),
      ),
      (
          'uint16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT16,
              shape=array_pb2.ArrayShape(dim=[]),
              uint16_list=array_pb2.Array.IntList(value=[1]),
          ),
          np.uint16(1),
      ),
      (
          'uint32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT32,
              shape=array_pb2.ArrayShape(dim=[]),
              uint32_list=array_pb2.Array.Uint32List(value=[1]),
          ),
          np.uint32(1),
      ),
      (
          'uint64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT64,
              shape=array_pb2.ArrayShape(dim=[]),
              uint64_list=array_pb2.Array.Uint64List(value=[1]),
          ),
          np.uint64(1),
      ),
      (
          'float16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_HALF,
              shape=array_pb2.ArrayShape(dim=[]),
              float16_list=array_pb2.Array.IntList(
                  value=[
                      np.array(1.0, dtype=np.float16).astype(np.uint16).item(),
                  ]
              ),
          ),
          np.float16(1.0),
      ),
      (
          'float32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[1.0]),
          ),
          np.float32(1.0),
      ),
      (
          'float64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_DOUBLE,
              shape=array_pb2.ArrayShape(dim=[]),
              float64_list=array_pb2.Array.DoubleList(value=[1.0]),
          ),
          np.float64(1.0),
      ),
      (
          'complex64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX64,
              shape=array_pb2.ArrayShape(dim=[]),
              complex64_list=array_pb2.Array.FloatList(value=[1.0, 1.0]),
          ),
          np.complex64((1.0 + 1.0j)),
      ),
      (
          'complex128',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              complex128_list=array_pb2.Array.DoubleList(value=[1.0, 1.0]),
          ),
          np.complex128((1.0 + 1.0j)),
      ),
      (
          'string',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'a']),
          ),
          np.str_(b'a'),
      ),
      (
          'array',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int32_list=array_pb2.Array.IntList(value=[1, 2, 3, 4, 5, 6]),
          ),
          np.array([[1, 2, 3], [4, 5, 6]]),
      ),
  )
  def test_from_proto_returns_value(self, proto, expected_value):
    actual_value = array.from_proto(proto)

    if isinstance(expected_value, np.ndarray):
      np.testing.assert_array_equal(actual_value, expected_value)
    else:
      self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'complex64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX64,
              shape=array_pb2.ArrayShape(dim=[]),
              complex64_list=array_pb2.Array.FloatList(value=[1.0]),
          ),
      ),
      (
          'complex128',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              complex128_list=array_pb2.Array.DoubleList(value=[1.0]),
          ),
      ),
  )
  def test_from_proto_raises_value_error_with_wrong_value(self, proto):
    with self.assertRaises(ValueError):
      array.from_proto(proto)

  @parameterized.named_parameters(
      ('None', None),
      ('object', object()),
  )
  def test_from_proto_raises_not_implemented_error(self, value):
    with self.assertRaises(NotImplementedError):
      array.to_proto(value)

  @parameterized.named_parameters(
      (
          'bool',
          True,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BOOL,
              shape=array_pb2.ArrayShape(dim=[]),
              bool_list=array_pb2.Array.BoolList(value=[True]),
          ),
      ),
      (
          'int32',
          np.iinfo(np.int32).max,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(
                  value=[np.iinfo(np.int32).max]
              ),
          ),
      ),
      (
          'int64',
          np.iinfo(np.int64).max,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              int64_list=array_pb2.Array.Int64List(
                  value=[np.iinfo(np.int64).max]
              ),
          ),
      ),
      (
          'float',
          1.0,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[1.0]),
          ),
      ),
      (
          'complex',
          complex(1.0, 1.0),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              complex128_list=array_pb2.Array.DoubleList(value=[1.0, 1.0]),
          ),
      ),
      (
          'string',
          'a',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'a']),
          ),
      ),
      (
          'generic',
          np.int32(1),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'array',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int32_list=array_pb2.Array.IntList(value=[1, 2, 3, 4, 5, 6]),
          ),
      ),
  )
  def test_to_proto_returns_value_with_no_dtype_hint(
      self, value, expected_value
  ):
    actual_value = array.to_proto(value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'bool',
          True,
          np.bool_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BOOL,
              shape=array_pb2.ArrayShape(dim=[]),
              bool_list=array_pb2.Array.BoolList(value=[True]),
          ),
      ),
      (
          'int8',
          1,
          np.int8,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT8,
              shape=array_pb2.ArrayShape(dim=[]),
              int8_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'int16',
          1,
          np.int16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT16,
              shape=array_pb2.ArrayShape(dim=[]),
              int16_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'int32',
          1,
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'int64',
          1,
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              int64_list=array_pb2.Array.Int64List(value=[1]),
          ),
      ),
      (
          'uint8',
          1,
          np.uint8,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT8,
              shape=array_pb2.ArrayShape(dim=[]),
              uint8_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'uint16',
          1,
          np.uint16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT16,
              shape=array_pb2.ArrayShape(dim=[]),
              uint16_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'uint32',
          1,
          np.uint32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT32,
              shape=array_pb2.ArrayShape(dim=[]),
              uint32_list=array_pb2.Array.Uint32List(value=[1]),
          ),
      ),
      (
          'uint64',
          1,
          np.uint64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT64,
              shape=array_pb2.ArrayShape(dim=[]),
              uint64_list=array_pb2.Array.Uint64List(value=[1]),
          ),
      ),
      (
          'float16',
          1.0,
          np.float16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_HALF,
              shape=array_pb2.ArrayShape(dim=[]),
              float16_list=array_pb2.Array.IntList(
                  value=[
                      np.array(1.0, dtype=np.float16).astype(np.uint16).item(),
                  ]
              ),
          ),
      ),
      (
          'float32',
          1.0,
          np.float32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[1.0]),
          ),
      ),
      (
          'float64',
          1.0,
          np.float64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_DOUBLE,
              shape=array_pb2.ArrayShape(dim=[]),
              float64_list=array_pb2.Array.DoubleList(value=[1.0]),
          ),
      ),
      (
          'complex64',
          (1.0 + 1.0j),
          np.complex64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX64,
              shape=array_pb2.ArrayShape(dim=[]),
              complex64_list=array_pb2.Array.FloatList(value=[1.0, 1.0]),
          ),
      ),
      (
          'complex128',
          (1.0 + 1.0j),
          np.complex128,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              complex128_list=array_pb2.Array.DoubleList(value=[1.0, 1.0]),
          ),
      ),
      (
          'string',
          'a',
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'a']),
          ),
      ),
      (
          'generic',
          np.int32(1),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'array',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int32_list=array_pb2.Array.IntList(value=[1, 2, 3, 4, 5, 6]),
          ),
      ),
      (
          'scalar_different_dtype',
          1,
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              int64_list=array_pb2.Array.Int64List(value=[1]),
          ),
      ),
      (
          'generic_different_dtype',
          np.int32(1),
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              int64_list=array_pb2.Array.Int64List(value=[1]),
          ),
      ),
      (
          'array_different_dtype',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int64_list=array_pb2.Array.Int64List(value=[1, 2, 3, 4, 5, 6]),
          ),
      ),
  )
  def test_to_proto_returns_value_with_dtype_hint(
      self, value, dtype, expected_value
  ):
    actual_value = array.to_proto(value, dtype_hint=dtype)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('scalar', np.iinfo(np.int64).max, np.int32),
      ('generic', np.int64(np.iinfo(np.int64).max), np.int32),
      ('array', np.array([np.iinfo(np.int64).max] * 3, np.int64), np.int32),
  )
  def test_to_proto_raises_value_error_with_incompatible_dtype_hint(
      self, value, dtype_hint
  ):
    with self.assertRaises(ValueError):
      array.to_proto(value, dtype_hint=dtype_hint)

  @parameterized.named_parameters(
      ('complex64', 1.0, np.complex64),
      ('complex128', 1.0, np.complex128),
  )
  def test_to_proto_raises_value_error_with_wrong_value(self, value, dtype):
    with self.assertRaises(ValueError):
      array.to_proto(value, dtype_hint=dtype)

  @parameterized.named_parameters(
      ('None', None),
      ('object', object()),
  )
  def test_to_proto_raises_not_implemented_error(self, value):
    with self.assertRaises(NotImplementedError):
      array.to_proto(value)

  @parameterized.named_parameters(
      ('scalar', 1, np.int64),
      ('str', 'abc', np.str_),
      ('generic', np.int32(1), np.int64),
      ('array', np.array([[1, 2, 3], [4, 5, 6]], np.int32), np.int64),
  )
  def test_can_cast_returns_true(self, value, dtype):
    result = array._can_cast(value, dtype)
    self.assertTrue(result)

  @parameterized.named_parameters(
      ('scalar', np.iinfo(np.int64).max, np.int32),
      ('str', 'abc', np.int32),
      ('generic', np.int64(np.iinfo(np.int64).max), np.int32),
      ('array', np.array([np.iinfo(np.int64).max] * 3, np.int64), np.int32),
  )
  def test_can_cast_returns_false(self, value, dtype):
    result = array._can_cast(value, dtype)
    self.assertFalse(result)

  @parameterized.named_parameters(
      ('bool', True, np.bool_),
      ('int', 1, np.int32),
      ('float', 1.0, np.float32),
      ('complex', (1.0 + 1.0j), np.complex64),
      ('str', 'abc', np.str_),
      ('bytes', b'abc', np.bytes_),
      ('generic', np.int32(1), np.int32),
      ('generic_smaller_size', np.int32(1), np.int16),
      ('generic_larger_size', np.int32(1), np.int64),
      ('array', np.array([[1, 2, 3], [4, 5, 6]], np.int32), np.int32),
      (
          'array_smaller_size',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int16,
      ),
      (
          'array_larger_size',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int32,
      ),
  )
  def test_is_compatible_dtype_returns_true(self, value, dtype):
    result = array.is_compatible_dtype(value, dtype)
    self.assertTrue(result)

  @parameterized.named_parameters(
      ('scalar_and_incompatible_dtype_kind', 1, np.float32),
      ('scalar_and_incompatible_dtype_size', np.iinfo(np.int64).max, np.int32),
      ('generic_and_incompatible_dtype_kind', np.int32(1), np.float32),
      (
          'generic_and_incompatible_dtype_size',
          np.int64(np.iinfo(np.int64).max),
          np.int32,
      ),
      (
          'array_and_incompatible_dtype_kind',
          np.array([1, 2, 3], np.int32),
          np.float32,
      ),
      (
          'array_and_incompatible_dtype_size',
          np.array([np.iinfo(np.int64).max] * 3, np.int64),
          np.float32,
      ),
  )
  def test_is_compatible_dtype_returns_false(self, value, dtype):
    result = array.is_compatible_dtype(value, dtype)
    self.assertFalse(result)

  @parameterized.named_parameters(
      ('scalar', 1, []),
      ('generic', np.int32(1), []),
      ('array', np.array([[1, 2, 3], [4, 5, 6]], np.int32), [2, 3]),
  )
  def test_is_compatible_shape_returns_true(self, value, shape):
    result = array.is_compatible_shape(value, shape)
    self.assertTrue(result)

  @parameterized.named_parameters(
      ('scalar', 1, [3]),
      ('generic', np.int32(1), [3]),
      ('array', np.array([[1, 2, 3], [4, 5, 6]], np.int32), [3]),
  )
  def test_is_compatible_shape_returns_false(self, value, shape):
    result = array.is_compatible_shape(value, shape)
    self.assertFalse(result)


if __name__ == '__main__':
  absltest.main()
