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
# limitations under the License.
"""Utilities for working with dtypes."""

from collections.abc import Mapping
from typing import Union
import warnings

import ml_dtypes
import numpy as np

from tensorflow_federated.proto.v0 import data_type_pb2


# Mapping from `DataType` to `type[np.generic]`.
_PROTO_TO_DTYPE: Mapping[data_type_pb2.DataType, type[np.generic]] = {
    data_type_pb2.DataType.DT_BOOL: np.bool_,
    data_type_pb2.DataType.DT_INT8: np.int8,
    data_type_pb2.DataType.DT_INT16: np.int16,
    data_type_pb2.DataType.DT_INT32: np.int32,
    data_type_pb2.DataType.DT_INT64: np.int64,
    data_type_pb2.DataType.DT_UINT8: np.uint8,
    data_type_pb2.DataType.DT_UINT16: np.uint16,
    data_type_pb2.DataType.DT_UINT32: np.uint32,
    data_type_pb2.DataType.DT_UINT64: np.uint64,
    data_type_pb2.DataType.DT_HALF: np.float16,
    data_type_pb2.DataType.DT_FLOAT: np.float32,
    data_type_pb2.DataType.DT_DOUBLE: np.float64,
    data_type_pb2.DataType.DT_COMPLEX64: np.complex64,
    data_type_pb2.DataType.DT_COMPLEX128: np.complex128,
    data_type_pb2.DataType.DT_BFLOAT16: ml_dtypes.bfloat16,
    data_type_pb2.DataType.DT_STRING: np.str_,
}


def from_proto(
    dtype_pb: data_type_pb2.DataType,
) -> type[np.generic]:
  """Returns a `type[np.generic]` for the `dtype_pb`."""
  if dtype_pb in _PROTO_TO_DTYPE:
    return _PROTO_TO_DTYPE[dtype_pb]
  else:
    raise NotImplementedError(f'Unexpected dtype found: {dtype_pb}.')


# Mapping from `type[np.generic]` to `DataType`.
_DTYPE_TO_PROTO: Mapping[type[np.generic], data_type_pb2.DataType] = {
    np.bool_: data_type_pb2.DataType.DT_BOOL,
    np.int8: data_type_pb2.DataType.DT_INT8,
    np.int16: data_type_pb2.DataType.DT_INT16,
    np.int32: data_type_pb2.DataType.DT_INT32,
    np.int64: data_type_pb2.DataType.DT_INT64,
    np.uint8: data_type_pb2.DataType.DT_UINT8,
    np.uint16: data_type_pb2.DataType.DT_UINT16,
    np.uint32: data_type_pb2.DataType.DT_UINT32,
    np.uint64: data_type_pb2.DataType.DT_UINT64,
    np.float16: data_type_pb2.DataType.DT_HALF,
    np.float32: data_type_pb2.DataType.DT_FLOAT,
    np.float64: data_type_pb2.DataType.DT_DOUBLE,
    np.complex64: data_type_pb2.DataType.DT_COMPLEX64,
    np.complex128: data_type_pb2.DataType.DT_COMPLEX128,
    ml_dtypes.bfloat16: data_type_pb2.DataType.DT_BFLOAT16,
    np.str_: data_type_pb2.DataType.DT_STRING,
}


def to_proto(dtype: type[np.generic]) -> data_type_pb2.DataType:
  """Returns a `DataType` for the `dtype`."""
  if dtype in _DTYPE_TO_PROTO:
    return _DTYPE_TO_PROTO[dtype]
  else:
    raise NotImplementedError(f'Unexpected dtype found: {dtype}.')


def is_valid_dtype(dtype: type[np.generic]) -> bool:
  """Returns `True` if `dtype` is valid, otherwise `False`."""
  return dtype in _DTYPE_TO_PROTO


def can_cast(
    value: Union[bool, int, float, complex, str, bytes],
    dtype: type[np.generic],
) -> bool:
  """Returns `True` if `value` can be cast to `dtype`, otherwise `False`.

  This function is intended to be used to determine if the size of the `dtype`
  is capable of holding the `value`. This is useful, for example, when trying to
  infer the dtype of the `value`. This function is not intended to be used to
  determine if you **should** cast a the `value` to `dtype`.

  Args:
    value: The value to check.
    dtype: The dtype to check against.
  """

  # `np.can_cast` does not support Python scalars (since version 2.0). Casting
  # the value to a numpy value and testing for an overflow is equivalent to
  # testing the Python value.
  numpy_version = tuple(int(x) for x in np.__version__.split('.'))
  if numpy_version >= (2, 0):
    # When encountering an overflow, numpy issues a `RuntimeWarning` for
    # floating dtypes and raises an `OverflowError` for integer dtypes.
    with warnings.catch_warnings(action='error', category=RuntimeWarning):
      try:
        np.asarray(value, dtype=dtype)
        return True
      except (OverflowError, RuntimeWarning):
        return False
  else:
    return np.can_cast(value, dtype)


def infer_dtype(
    obj: Union[bool, int, float, complex, str, bytes],
) -> type[np.generic]:
  """Returns a scalar numpy dtype for a Python scalar.

  Args:
    obj: A Python scalar.

  Returns:
    A scalar numpy dtype.
  """
  if isinstance(obj, bool):
    return np.bool_
  elif isinstance(obj, int):
    if can_cast(obj, np.int32):
      return np.int32
    elif can_cast(obj, np.int64):
      return np.int64
    else:
      raise ValueError(
          'Expected `obj` to be an `int` in the range'
          f' [{np.iinfo(np.int64).min}, {np.iinfo(np.int64).max}],'
          f' found: {obj}.'
      )
  elif isinstance(obj, float):
    return np.float32
  elif isinstance(obj, complex):
    return np.complex128
  elif isinstance(obj, (str, bytes)):
    return np.str_
  else:
    raise NotImplementedError(f'Unexpected type found: {type(obj)}.')
