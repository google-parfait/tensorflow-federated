# Copyright 2024, The TensorFlow Federated Authors.
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
"""Utilities for working with arrays."""

from typing import Optional, Union

import numpy as np

from tensorflow_federated.proto.v0 import array_pb2
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import dtype_utils

# Array is the Python representation of the `Array` protobuf, and is the native
# representation of an array.
Array = Union[
    # Python types
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    # Numpy types
    np.generic,
    np.ndarray,
]


def from_proto(array_pb: array_pb2.Array) -> Array:
  """Returns an `Array` for the `array_pb`."""
  dtype = dtype_utils.from_proto(array_pb.dtype)
  shape = array_shape.from_proto(array_pb.shape)

  if dtype is np.bool_:
    value = array_pb.bool_list.value
  elif dtype is np.int8:
    value = array_pb.int8_list.value
  elif dtype is np.int16:
    value = array_pb.int16_list.value
  elif dtype is np.int32:
    value = array_pb.int32_list.value
  elif dtype is np.int64:
    value = array_pb.int64_list.value
  elif dtype is np.uint8:
    value = array_pb.uint8_list.value
  elif dtype is np.uint16:
    value = array_pb.uint16_list.value
  elif dtype is np.uint32:
    value = array_pb.uint32_list.value
  elif dtype is np.uint64:
    value = array_pb.uint64_list.value
  elif dtype is np.float16:
    value = array_pb.float16_list.value
    # Values of dtype np.float16 are packed to and unpacked from a protobuf
    # field of type int32 using the following logic in order to maintain
    # compatibility with how other external environments (e.g. TensorFlow, Jax)
    # represent values of np.float16.
    value = np.array(value, dtype=np.uint16).astype(np.float16)
  elif dtype is np.float32:
    value = array_pb.float32_list.value
  elif dtype is np.float64:
    value = array_pb.float64_list.value
  elif dtype is np.complex64:
    if len(array_pb.complex64_list.value) % 2 != 0:
      raise ValueError(
          'Expected the number of complex values to be even, one real and one'
          ' imaginary part for each complex value.'
      )
    value = iter(array_pb.complex64_list.value)
    value = [complex(real, imag) for real, imag in zip(value, value)]
  elif dtype is np.complex128:
    if len(array_pb.complex128_list.value) % 2 != 0:
      raise ValueError(
          'Expected the number of complex values to be even, one real and one'
          ' imaginary part for each complex value.'
      )
    value = iter(array_pb.complex128_list.value)
    value = [complex(real, imag) for real, imag in zip(value, value)]
  elif dtype is np.str_:
    value = array_pb.string_list.value
  else:
    raise NotImplementedError(f'Unexpected dtype found: {dtype}.')

  if not array_shape.is_shape_scalar(shape):
    value = np.array(value, dtype).reshape(shape)
  else:
    (value,) = value

  return value


def to_proto(
    array: Array, *, dtype_hint: Optional[type[np.generic]] = None
) -> array_pb2.Array:
  """Returns a `Array` for the `array`."""

  if dtype_hint is not None:
    if not np.can_cast(array, dtype_hint):
      raise ValueError(f'Expected that {array} can be cast to {dtype_hint}.')

  if isinstance(array, (np.ndarray, np.generic)):
    if dtype_hint is not None:
      dtype = dtype_hint
    else:
      dtype = array.dtype.type
    shape = array.shape
    if array.dtype.type is np.str_:
      array = array.astype(np.bytes_)
    value = array.flatten().tolist()
  elif isinstance(array, (bool, int, float, complex, str, bytes)):
    if dtype_hint is not None:
      dtype = dtype_hint
    else:
      dtype = dtype_utils.infer_dtype(array)
    shape = ()
    if isinstance(array, str):
      array = array.encode()
    value = [array]
  else:
    raise NotImplementedError(f'Unexpected value found: {array}.')

  dtype_pb = dtype_utils.to_proto(dtype)
  shape_pb = array_shape.to_proto(shape)

  if dtype is np.bool_:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        bool_list=array_pb2.Array.BoolList(value=value),
    )
  elif dtype is np.int8:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int8_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.int16:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int16_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.int32:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int32_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.int64:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        int64_list=array_pb2.Array.Int64List(value=value),
    )
  elif dtype is np.uint8:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint8_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.uint16:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint16_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.uint32:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint32_list=array_pb2.Array.Uint32List(value=value),
    )
  elif dtype is np.uint64:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        uint64_list=array_pb2.Array.Uint64List(value=value),
    )
  elif dtype is np.float16:
    # Values of dtype np.float16 are packed to and unpacked from a protobuf
    # field of type int32 using the following logic in order to maintain
    # compatibility with how other external environments (e.g. TensorFlow, Jax)
    # represent values of np.float16.
    value = np.array(value, dtype=np.float16).astype(np.uint16).tolist()
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        float16_list=array_pb2.Array.IntList(value=value),
    )
  elif dtype is np.float32:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        float32_list=array_pb2.Array.FloatList(value=value),
    )
  elif dtype is np.float64:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        float64_list=array_pb2.Array.DoubleList(value=value),
    )
  elif dtype is np.complex64:
    packed_value = []
    for x in value:
      if not isinstance(x, complex):
        raise ValueError(f'Expected a complex type, found {type(x)}.')
      packed_value.extend([x.real, x.imag])
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        complex64_list=array_pb2.Array.FloatList(value=packed_value),
    )
  elif dtype is np.complex128:
    packed_value = []
    for x in value:
      if not isinstance(x, complex):
        raise ValueError(f'Expected a complex type, found {type(x)}.')
      packed_value.extend([x.real, x.imag])
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        complex128_list=array_pb2.Array.DoubleList(value=packed_value),
    )
  elif dtype is np.str_:
    return array_pb2.Array(
        dtype=dtype_pb,
        shape=shape_pb,
        string_list=array_pb2.Array.BytesList(value=value),
    )
  else:
    raise NotImplementedError(f'Unexpected dtype found: {dtype}.')
