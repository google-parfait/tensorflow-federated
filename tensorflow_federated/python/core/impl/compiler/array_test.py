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

import math

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.proto.v0 import data_type_pb2
from tensorflow_federated.python.core.impl.compiler import array


class FromProtoTest(parameterized.TestCase):

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
                  value=[np.asarray(1.0, np.float16).view(np.uint16).item()]
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
          np.complex64(1.0 + 1.0j),
      ),
      (
          'complex128',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              complex128_list=array_pb2.Array.DoubleList(value=[1.0, 1.0]),
          ),
          np.complex128(1.0 + 1.0j),
      ),
      (
          'bfloat16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[]),
              bfloat16_list=array_pb2.Array.IntList(
                  value=[
                      np.asarray(1.0, ml_dtypes.bfloat16).view(np.uint16).item()
                  ]
              ),
          ),
          ml_dtypes.bfloat16(1.0),
      ),
      (
          'str',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
          b'abc',
      ),
      (
          'str_null_terminated',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc\x00\x00']),
          ),
          b'abc\x00\x00',
      ),
      (
          'array_int32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int32_list=array_pb2.Array.IntList(value=[1, 2, 3, 4, 5, 6]),
          ),
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
      ),
      (
          'array_int32_empty',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[0]),
              int32_list=array_pb2.Array.IntList(value=[]),
          ),
          np.array([], np.int32),
      ),
      (
          'array_str',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
          np.array([b'abc', b'def'], np.object_),
      ),
      (
          'array_str_null_terminated',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(
                  value=[b'abc\x00\x00', b'def\x00\x00']
              ),
          ),
          np.array([b'abc\x00\x00', b'def\x00\x00'], np.object_),
      ),
      (
          'nan',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[np.nan]),
          ),
          np.float32(np.nan),
      ),
      (
          'inf',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[np.inf]),
          ),
          np.float32(np.inf),
      ),
  )
  def test_returns_value(self, proto, expected_value):
    actual_value = array.from_proto(proto)

    if isinstance(actual_value, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(actual_value, expected_value, strict=True)
    else:
      self.assertIsInstance(actual_value, type(expected_value))
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
  def test_raises_value_error_with_wrong_value(self, proto):
    with self.assertRaises(ValueError):
      array.from_proto(proto)


class ToProtoTest(parameterized.TestCase):

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
          'str',
          'abc',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'bytes',
          b'abc',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'bytes_null_terminated',
          b'abc\x00\x00',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc\x00\x00']),
          ),
      ),
      (
          'generic_int32',
          np.int32(1),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'generic_float16',
          np.float16(1.0),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_HALF,
              shape=array_pb2.ArrayShape(dim=[]),
              float16_list=array_pb2.Array.IntList(
                  value=[np.asarray(1.0, np.float16).view(np.uint16).item()]
              ),
          ),
      ),
      (
          'generic_bfloat16',
          ml_dtypes.bfloat16(1.0),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[]),
              bfloat16_list=array_pb2.Array.IntList(
                  value=[
                      np.asarray(1.0, ml_dtypes.bfloat16).view(np.uint16).item()
                  ]
              ),
          ),
      ),
      (
          'generic_str',
          np.str_('abc'),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'generic_bytes',
          np.bytes_(b'abc'),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'array_int32',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int32_list=array_pb2.Array.IntList(value=[1, 2, 3, 4, 5, 6]),
          ),
      ),
      (
          'array_int32_empty',
          np.array([], np.int32),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[0]),
              int32_list=array_pb2.Array.IntList(value=[]),
          ),
      ),
      (
          'array_float16',
          np.array([1.0], np.float16),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_HALF,
              shape=array_pb2.ArrayShape(dim=[1]),
              float16_list=array_pb2.Array.IntList(
                  value=[np.asarray(1.0, np.float16).view(np.uint16).item()]
              ),
          ),
      ),
      (
          'array_bfloat16',
          np.array([1.0], ml_dtypes.bfloat16),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[1]),
              bfloat16_list=array_pb2.Array.IntList(
                  value=[
                      np.asarray(1.0, ml_dtypes.bfloat16).view(np.uint16).item()
                  ]
              ),
          ),
      ),
      (
          'array_str',
          np.array(['abc', 'def'], np.str_),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_bytes',
          np.array([b'abc', b'def'], np.bytes_),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_object_str',
          np.array(['abc', 'def'], np.object_),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_object_bytes',
          np.array([b'abc', b'def'], np.object_),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_object_bytes_null_terminated',
          np.array([b'abc\x00\x00', b'def\x00\x00'], np.object_),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(
                  value=[b'abc\x00\x00', b'def\x00\x00']
              ),
          ),
      ),
      (
          'nan',
          np.nan,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[np.nan]),
          ),
      ),
      (
          'inf',
          np.inf,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[np.inf]),
          ),
      ),
  )
  def test_returns_value_with_no_dtype_hint(self, value, expected_value):
    actual_value = array.to_proto(value)

    # Externally protobuf does not compare NaN values as equal.
    if isinstance(value, float) and math.isnan(value):
      self.assertEqual(actual_value.dtype, expected_value.dtype)
      self.assertEqual(actual_value.shape, expected_value.shape)
      self.assertLen(actual_value.float32_list.value, 1)
      self.assertTrue(math.isnan(actual_value.float32_list.value[0]))
    else:
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
                  value=[np.asarray(1.0, np.float16).view(np.uint16).item()]
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
          'str',
          'abc',
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'bytes',
          b'abc',
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'bytes_null_terminated',
          b'abc\x00\x00',
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc\x00\x00']),
          ),
      ),
      (
          'generic_int32',
          np.int32(1),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              int32_list=array_pb2.Array.IntList(value=[1]),
          ),
      ),
      (
          'generic_bfloat16',
          # Note: we must not use Python `float` here because ml_dtypes.bfloat16
          # is declared as kind `V` (void) not `f` (float) to prevent numpy from
          # trying to equate float16 and bfloat16 (which are not compatible).
          ml_dtypes.bfloat16(1.0),
          ml_dtypes.bfloat16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[]),
              bfloat16_list=array_pb2.Array.IntList(
                  value=[
                      np.asarray(1.0, ml_dtypes.bfloat16).view(np.uint16).item()
                  ]
              ),
          ),
      ),
      (
          'generic_str',
          np.str_('abc'),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'generic_bytes',
          np.bytes_(b'abc'),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
              string_list=array_pb2.Array.BytesList(value=[b'abc']),
          ),
      ),
      (
          'array_int32',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              int32_list=array_pb2.Array.IntList(value=[1, 2, 3, 4, 5, 6]),
          ),
      ),
      (
          'array_int32_empty',
          np.array([], np.int32),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[0]),
              int32_list=array_pb2.Array.IntList(value=[]),
          ),
      ),
      (
          'array_str',
          np.array(['abc', 'def'], np.str_),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_bytes',
          np.array([b'abc', b'def'], np.bytes_),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_object_str',
          np.array(['abc', 'def'], np.object_),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_object_bytes',
          np.array([b'abc', b'def'], np.object_),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(value=[b'abc', b'def']),
          ),
      ),
      (
          'array_object_bytes_null_terminated',
          np.array([b'abc\x00\x00', b'def\x00\x00'], np.object_),
          np.str_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[2]),
              string_list=array_pb2.Array.BytesList(
                  value=[b'abc\x00\x00', b'def\x00\x00']
              ),
          ),
      ),
      (
          'nan',
          np.nan,
          np.float32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[np.nan]),
          ),
      ),
      (
          'inf',
          np.inf,
          np.float32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              float32_list=array_pb2.Array.FloatList(value=[np.inf]),
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
  def test_returns_value_with_dtype_hint(self, value, dtype, expected_value):
    actual_value = array.to_proto(value, dtype_hint=dtype)

    # Externally protobuf does not compare NaN values as equal.
    if isinstance(value, float) and math.isnan(value):
      self.assertEqual(actual_value.dtype, expected_value.dtype)
      self.assertEqual(actual_value.shape, expected_value.shape)
      self.assertLen(actual_value.float32_list.value, 1)
      self.assertTrue(math.isnan(actual_value.float32_list.value[0]))
    else:
      self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('bytes', b'abc', np.bytes_),
  )
  def test_raises_value_error_with_invalid_dtype_hint(self, value, dtype):
    with self.assertRaises(ValueError):
      array.to_proto(value, dtype_hint=dtype)

  @parameterized.named_parameters(
      ('scalar', np.iinfo(np.int64).max, np.int32),
      ('generic', np.int64(np.iinfo(np.int64).max), np.int32),
      ('array', np.array([np.iinfo(np.int64).max] * 3, np.int64), np.int32),
  )
  def test_raises_value_error_with_incompatible_dtype_hint(self, value, dtype):
    with self.assertRaises(ValueError):
      array.to_proto(value, dtype_hint=dtype)

  @parameterized.named_parameters(
      ('None', None),
      ('object', object()),
  )
  def test_raises_not_implemented_error(self, value):
    with self.assertRaises(NotImplementedError):
      array.to_proto(value)


class FromProtoContentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'bool',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BOOL,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.bool_(True).tobytes(),
          ),
          np.bool_(True),
      ),
      (
          'int8',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT8,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int8(1).tobytes(),
          ),
          np.int8(1),
      ),
      (
          'int16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int16(1).tobytes(),
          ),
          np.int16(1),
      ),
      (
          'int32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int32(1).tobytes(),
          ),
          np.int32(1),
      ),
      (
          'int64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int64(1).tobytes(),
          ),
          np.int64(1),
      ),
      (
          'uint8',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT8,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint8(1).tobytes(),
          ),
          np.uint8(1),
      ),
      (
          'uint16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint16(1).tobytes(),
          ),
          np.uint16(1),
      ),
      (
          'uint32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint32(1).tobytes(),
          ),
          np.uint32(1),
      ),
      (
          'uint64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint64(1).tobytes(),
          ),
          np.uint64(1),
      ),
      (
          'float16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_HALF,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float16(1.0).tobytes(),
          ),
          np.float16(1.0),
      ),
      (
          'float32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(1.0).tobytes(),
          ),
          np.float32(1.0),
      ),
      (
          'float64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_DOUBLE,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float64(1.0).tobytes(),
          ),
          np.float64(1.0),
      ),
      (
          'complex64',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.complex64(1.0 + 1.0j).tobytes(),
          ),
          np.complex64(1.0 + 1.0j),
      ),
      (
          'complex128',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.complex128(1.0 + 1.0j).tobytes(),
          ),
          np.complex128(1.0 + 1.0j),
      ),
      (
          'bfloat16',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=ml_dtypes.bfloat16(1.0).tobytes(),
          ),
          ml_dtypes.bfloat16(1.0),
      ),
      (
          'array_int32',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              content=np.array(
                  [[1, 2, 3], [4, 5, 6]], dtype=np.int32
              ).tobytes(),
          ),
          np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
      ),
      (
          'array_int32_empty',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[0]),
              int32_list=array_pb2.Array.IntList(value=[]),
          ),
          np.array([], np.int32),
      ),
      (
          'nan',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(np.nan).tobytes(),
          ),
          np.float32(np.nan),
      ),
      (
          'inf',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(np.inf).tobytes(),
          ),
          np.float32(np.inf),
      ),
  )
  def test_returns_value(self, proto, expected_value):
    actual_value = array.from_proto_content(proto)

    if isinstance(actual_value, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(actual_value, expected_value, strict=True)
    else:
      self.assertIsInstance(actual_value, type(expected_value))
      self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'str',
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_STRING,
              shape=array_pb2.ArrayShape(dim=[]),
          ),
      ),
  )
  def test_raises_value_error_with_invalid_dtype(self, proto):
    with self.assertRaises(ValueError):
      array.from_proto(proto)


class ToProtoContentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'bool',
          True,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BOOL,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.bool_(True).tobytes(),
          ),
      ),
      (
          'int32',
          np.iinfo(np.int32).max,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int32(np.iinfo(np.int32).max).tobytes(),
          ),
      ),
      (
          'int64',
          np.iinfo(np.int64).max,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int64(np.iinfo(np.int64).max).tobytes(),
          ),
      ),
      (
          'float',
          1.0,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(1.0).tobytes(),
          ),
      ),
      (
          'complex',
          (1.0 + 1.0j),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.complex128(1.0 + 1.0j).tobytes(),
          ),
      ),
      (
          'generic_int32',
          np.int32(1),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int32(1).tobytes(),
          ),
      ),
      (
          'generic_bfloat16',
          ml_dtypes.bfloat16(1.0),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=ml_dtypes.bfloat16(1.0).tobytes(),
          ),
      ),
      (
          'array_int32',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              content=np.array([[1, 2, 3], [4, 5, 6]], np.int32).tobytes(),
          ),
      ),
      (
          'array_int32_empty',
          np.array([], np.int32),
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[0]),
              content=np.array([], np.int32).tobytes(),
          ),
      ),
      (
          'nan',
          np.nan,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(np.nan).tobytes(),
          ),
      ),
      (
          'inf',
          np.inf,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(np.inf).tobytes(),
          ),
      ),
  )
  def test_returns_value_with_no_dtype_hint(self, value, expected_value):
    actual_value = array.to_proto_content(value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'bool',
          True,
          np.bool_,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BOOL,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.bool_(True).tobytes(),
          ),
      ),
      (
          'int8',
          1,
          np.int8,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT8,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int8(1).tobytes(),
          ),
      ),
      (
          'int16',
          1,
          np.int16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int16(1).tobytes(),
          ),
      ),
      (
          'int32',
          1,
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int32(1).tobytes(),
          ),
      ),
      (
          'int64',
          1,
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int64(1).tobytes(),
          ),
      ),
      (
          'uint8',
          1,
          np.uint8,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT8,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint8(1).tobytes(),
          ),
      ),
      (
          'uint16',
          1,
          np.uint16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint16(1).tobytes(),
          ),
      ),
      (
          'uint32',
          1,
          np.uint32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint32(1).tobytes(),
          ),
      ),
      (
          'uint64',
          1,
          np.uint64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_UINT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.uint64(1).tobytes(),
          ),
      ),
      (
          'float16',
          1.0,
          np.float16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_HALF,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float16(1.0).tobytes(),
          ),
      ),
      (
          'float32',
          1.0,
          np.float32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(1.0).tobytes(),
          ),
      ),
      (
          'float64',
          1.0,
          np.float64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_DOUBLE,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float64(1.0).tobytes(),
          ),
      ),
      (
          'complex64',
          (1.0 + 1.0j),
          np.complex64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.complex64(1.0 + 1.0j).tobytes(),
          ),
      ),
      (
          'complex128',
          (1.0 + 1.0j),
          np.complex128,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_COMPLEX128,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.complex128(1.0 + 1.0j).tobytes(),
          ),
      ),
      (
          'generic_int32',
          np.int32(1),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int32(1).tobytes(),
          ),
      ),
      (
          'generic_bfloat16',
          ml_dtypes.bfloat16(1.0),
          ml_dtypes.bfloat16,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_BFLOAT16,
              shape=array_pb2.ArrayShape(dim=[]),
              content=ml_dtypes.bfloat16(1.0).tobytes(),
          ),
      ),
      (
          'array_int32',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              content=np.array([[1, 2, 3], [4, 5, 6]], np.int32).tobytes(),
          ),
      ),
      (
          'array_int32_empty',
          np.array([], np.int32),
          np.int32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT32,
              shape=array_pb2.ArrayShape(dim=[0]),
              content=np.array([], np.int32).tobytes(),
          ),
      ),
      (
          'nan',
          np.nan,
          np.float32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(np.nan).tobytes(),
          ),
      ),
      (
          'inf',
          np.inf,
          np.float32,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.float32(np.inf).tobytes(),
          ),
      ),
      (
          'scalar_different_dtype',
          1,
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int64(1).tobytes(),
          ),
      ),
      (
          'generic_different_dtype',
          np.int32(1),
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[]),
              content=np.int64(1).tobytes(),
          ),
      ),
      (
          'array_different_dtype',
          np.array([[1, 2, 3], [4, 5, 6]], np.int32),
          np.int64,
          array_pb2.Array(
              dtype=data_type_pb2.DataType.DT_INT64,
              shape=array_pb2.ArrayShape(dim=[2, 3]),
              content=np.array([[1, 2, 3], [4, 5, 6]], np.int64).tobytes(),
          ),
      ),
  )
  def test_returns_value_with_dtype_hint(self, value, dtype, expected_value):
    actual_value = array.to_proto_content(value, dtype_hint=dtype)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('bytes', b'abc', np.bytes_),
  )
  def test_raises_value_error_with_invalid_dtype_hint(self, value, dtype):
    with self.assertRaises(ValueError):
      array.to_proto_content(value, dtype_hint=dtype)

  @parameterized.named_parameters(
      ('scalar', np.iinfo(np.int64).max, np.int32),
      ('generic', np.int64(np.iinfo(np.int64).max), np.int32),
      ('array', np.array([np.iinfo(np.int64).max] * 3, np.int64), np.int32),
  )
  def test_raises_value_error_with_incompatible_dtype_hint(self, value, dtype):
    with self.assertRaises(ValueError):
      array.to_proto(value, dtype_hint=dtype)

  @parameterized.named_parameters(
      ('None', None),
      ('object', object()),
  )
  def test_raises_not_implemented_error(self, value):
    with self.assertRaises(NotImplementedError):
      array.to_proto(value)


class IsCompatibleDtypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('bool', True, np.bool_),
      ('int8', 1, np.int8),
      ('int16', 1, np.int16),
      ('int32', 1, np.int32),
      ('int64', 1, np.int64),
      ('uint8', 1, np.uint8),
      ('uint16', 1, np.uint16),
      ('uint32', 1, np.uint32),
      ('uint64', 1, np.uint64),
      ('float16', 1.0, np.float16),
      ('float32', 1.0, np.float32),
      ('float64', 1.0, np.float64),
      ('complex64', (1.0 + 1.0j), np.complex64),
      ('complex128', (1.0 + 1.0j), np.complex128),
      ('bfloat16', ml_dtypes.bfloat16(1.0), ml_dtypes.bfloat16),
      ('str', 'abc', np.str_),
      ('bytes', b'abc', np.str_),
      ('generic_int32', np.int32(1), np.int32),
      ('array_int32', np.array([[1, 2, 3], [4, 5, 6]], np.int32), np.int32),
      ('array_str', np.array(['abc', 'def'], np.str_), np.str_),
      ('array_bytes', np.array([b'abc', b'def'], np.bytes_), np.str_),
      (
          'array_bytes_null_terminated',
          np.array([b'abc\x00\x00', b'def\x00\x00'], np.object_),
          np.str_,
      ),
      ('nan', np.nan, np.float32),
      ('inf', np.inf, np.float32),
  )
  def test_returns_true(self, value, dtype):
    result = array.is_compatible_dtype(value, dtype)
    self.assertTrue(result)

  @parameterized.named_parameters(
      ('scalar_incompatible_dtype_kind', 1, np.float32),
      ('scalar_incompatible_dtype_kind_bfloat16', 1.0, ml_dtypes.bfloat16),
      ('scalar_incompatible_dtype_size_int', np.iinfo(np.int64).max, np.int32),
      (
          'scalar_incompatible_dtype_size_float',
          np.finfo(np.float64).max,
          np.int32,
      ),
      (
          'scalar_incompatible_dtype_size_complex_real',
          complex(np.finfo(np.float64).max, 1),
          np.complex64,
      ),
      (
          'scalar_incompatible_dtype_size_complex_imaginary',
          complex(1, np.finfo(np.float64).max),
          np.complex64,
      ),
      ('generic_incompatible_dtype_kind', np.int32(1), np.float32),
      (
          'generic_incompatible_dtype_size',
          np.int64(np.iinfo(np.int64).max),
          np.int32,
      ),
      (
          'array_incompatible_dtype_kind',
          np.array([1, 2, 3], np.int32),
          np.float32,
      ),
      (
          'array_incompatible_dtype_size',
          np.array([np.iinfo(np.int64).max] * 3, np.int64),
          np.int32,
      ),
  )
  def test_returns_false(self, value, dtype):
    result = array.is_compatible_dtype(value, dtype)
    self.assertFalse(result)


class IsCompatibleShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar', 1, []),
      ('generic', np.int32(1), []),
      ('array', np.array([[1, 2, 3], [4, 5, 6]], np.int32), [2, 3]),
  )
  def test_returns_true(self, value, shape):
    result = array.is_compatible_shape(value, shape)
    self.assertTrue(result)

  @parameterized.named_parameters(
      ('scalar', 1, [3]),
      ('generic', np.int32(1), [3]),
      ('array', np.array([[1, 2, 3], [4, 5, 6]], np.int32), [3]),
  )
  def test_returns_false(self, value, shape):
    result = array.is_compatible_shape(value, shape)
    self.assertFalse(result)


if __name__ == '__main__':
  absltest.main()
