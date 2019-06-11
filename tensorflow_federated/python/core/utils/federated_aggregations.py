# Lint as: python3
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
"""Contains implementations of extra federated aggregations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core import api as tff


def _validate_value_on_clients(value):
  py_typecheck.check_type(value, tff.Value)
  py_typecheck.check_type(value.type_signature, tff.FederatedType)
  if value.type_signature.placement is not tff.CLIENTS:
    raise TypeError('`value` argument must be a tff.Value placed at CLIENTS. '
                    'Got: {!s}'.format(value.type_signature))


def _validate_dtype_is_numeric(dtype):
  if dtype not in [tf.int32, tf.float32]:
    raise TypeError('Type must be int32 or float32. ' 'Got: {!s}'.format(dtype))


def _federated_reduce_with_func(value, tf_func, zeros):
  """Applies to `tf_func` to accumulated `value`s.

  This utility provides a generic aggregation for accumulating a value and
  applying a simple aggregation (like minimum or maximum aggregations).

  Args:
    value: A `tff.Value` placed on the `tff.CLIENTS`, that is a `tf.int32` or
      `tf.float32`.
    tf_func: A function to be applied to the accumulated values. Must be a
      binary operation where both parameters are of type `U` and the return type
      is also `U`.
    zeros: The zero of the same type as `value` in the algebra of reduction
      operators.

  Returns:
    A representation on the `tff.SERVER` of the result of aggregating `value`.
  """
  member_type = value.type_signature.member

  @tff.tf_computation(value.type_signature.member, value.type_signature.member)
  def accumulate(current, value):
    if isinstance(member_type, tff.NamedTupleType):
      return anonymous_tuple.map_structure(tf_func, current, value)
    return tf.nest.map_structure(tf_func, current, value)

  @tff.tf_computation(value.type_signature.member, value.type_signature.member)
  def merge(a, b):
    if isinstance(member_type, tff.NamedTupleType):
      return anonymous_tuple.map_structure(tf_func, a, b)
    return tf.nest.map_structure(tf_func, a, b)

  @tff.tf_computation(value.type_signature.member)
  def report(value):
    return value

  return tff.federated_aggregate(value, zeros, accumulate, merge, report)


def _initial_values(initial_value_fn, member_type):
  """Create a nested structure of initial values for the reduction.

  Args:
    initial_value_fn: A function that maps a tff.TensorType to a specific value
      constant for initialization.
    member_type: A `tff.Type` representing the member components of the
      federated type.

  Returns:
    A function of the result of reducing a value with no constituents.
  """

  @tff.tf_computation
  def zeros_fn():
    if isinstance(member_type, tff.NamedTupleType):
      anonymous_tuple.map_structure(
          lambda v: _validate_dtype_is_numeric(v.dtype), member_type)
      return anonymous_tuple.map_structure(
          lambda v: tf.fill(v.shape, value=initial_value_fn(v)), member_type)
    _validate_dtype_is_numeric(member_type.dtype)
    return tf.fill(member_type.shape, value=initial_value_fn(member_type))

  return zeros_fn()


def federated_min(value):
  """Aggregation to find the minimum value from the `tff.CLIENTS`.

  Args:
    value: A `tff.Value` placed on the `tff.CLIENTS`.

  Returns:
    In the degenerate scenario that the `value` is aggregated over an empty set
    of `tff.CLIENTS`, the tensor constituents of the result are set to the
    maximum of the underlying numeric data type.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  zeros = _initial_values(lambda v: v.dtype.max, member_type)
  return _federated_reduce_with_func(value, tf.minimum, zeros)


def federated_max(value):
  """Aggregation to find the maximum value from the `tff.CLIENTS`.

  Args:
    value: A `tff.Value` placed on the `tff.CLIENTS`.

  Returns:
    In the degenerate scenario that the `value` is aggregated over an empty set
    of `tff.CLIENTS`, the tensor constituents of the result are set to the
    minimum of the underlying numeric data type.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  zeros = _initial_values(lambda v: v.dtype.min, member_type)
  return _federated_reduce_with_func(value, tf.maximum, zeros)
