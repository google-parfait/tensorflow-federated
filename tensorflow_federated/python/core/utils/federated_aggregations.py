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

import collections
import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import value_base


def _validate_value_on_clients(value):
  py_typecheck.check_type(value, value_base.Value)
  py_typecheck.check_type(value.type_signature, computation_types.FederatedType)
  if value.type_signature.placement is not placements.CLIENTS:
    raise TypeError(
        '`value` argument must be a tff.Value placed at CLIENTS. Got: {!s}'
        .format(value.type_signature))


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

  @computations.tf_computation(value.type_signature.member,
                               value.type_signature.member)
  def accumulate(current, value):
    if member_type.is_struct():
      return structure.map_structure(tf_func, current, value)
    return tf.nest.map_structure(tf_func, current, value)

  @computations.tf_computation(value.type_signature.member,
                               value.type_signature.member)
  def merge(a, b):
    if member_type.is_struct():
      return structure.map_structure(tf_func, a, b)
    return tf.nest.map_structure(tf_func, a, b)

  @computations.tf_computation(value.type_signature.member)
  def report(value):
    return value

  return intrinsics.federated_aggregate(value, zeros, accumulate, merge, report)


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

  @computations.tf_computation
  def zeros_fn():
    if member_type.is_struct():
      structure.map_structure(lambda v: _validate_dtype_is_numeric(v.dtype),
                              member_type)
      return structure.map_structure(
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


@attr.s
class _Samples(object):
  """Class representing internal sample data structure.

  The class contains two parts, `accumulators` and `rands`, that are parallel
  lists (e.g. the i-th index in one corresponds to the i-th index in the other).
  These two lists are used to sample from the accumulators with equal
  probability.
  """
  accumulators = attr.ib()
  rands = attr.ib()


def _zeros_for_sample(member_type):
  """Create an empty nested structure for the sample aggregation.

  Args:
    member_type: A `tff.Type` representing the member components of the
      federated type.

  Returns:
    A function of the result of zeros to first concatenate.
  """

  @computations.tf_computation
  def accumlator_type_fn():
    """Gets the type for the accumulators."""
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if member_type.is_struct():
      a = structure.map_structure(
          lambda v: tf.zeros([0] + v.shape.dims, v.dtype), member_type)
      return _Samples(structure.to_odict(a, True), tf.zeros([0], tf.float32))
    if member_type.shape:
      s = [0] + member_type.shape.dims
    return _Samples(tf.zeros(s, member_type.dtype), tf.zeros([0], tf.float32))

  return accumlator_type_fn()


def _get_accumulator_type(member_type):
  """Constructs a `tff.Type` for the accumulator in sample aggregation.

  Args:
    member_type: A `tff.Type` representing the member components of the
      federated type.

  Returns:
    The `tff.StructType` associated with the accumulator. The tuple contains
    two parts, `accumulators` and `rands`, that are parallel lists (e.g. the
    i-th index in one corresponds to the i-th index in the other). These two
    lists are used to sample from the accumulators with equal probability.
  """
  # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
  if member_type.is_struct():
    a = structure.map_structure(
        lambda v: computation_types.TensorType(v.dtype, [None] + v.shape.dims),
        member_type)
    return computation_types.StructType(
        collections.OrderedDict({
            'accumulators':
                computation_types.StructType(structure.to_odict(a, True)),
            'rands':
                computation_types.TensorType(tf.float32, shape=[None]),
        }))
  return computation_types.StructType(
      collections.OrderedDict({
          'accumulators':
              computation_types.TensorType(
                  member_type.dtype, shape=[None] + member_type.shape.dims),
          'rands':
              computation_types.TensorType(tf.float32, shape=[None]),
      }))


def federated_sample(value, max_num_samples=100):
  """Aggregation to produce uniform sample of at most `max_num_samples` values.

  Each client value is assigned a random number when it is examined during each
  accumulation. Each accumulate and merge only keeps the top N values based
  on the random number. Report drops the random numbers and only returns the
  at most N values sampled from the accumulated client values using standard
  reservoir sampling (https://en.wikipedia.org/wiki/Reservoir_sampling), where
  N is user provided `max_num_samples`.

  Args:
    value: A `tff.Value` placed on the `tff.CLIENTS`.
    max_num_samples: The maximum number of samples to collect from client
      values. If fewer clients than the defined max sample size participated in
      the round of computation, the actual number of samples will equal the
      number of clients in the round.

  Returns:
    At most `max_num_samples` samples of the value from the `tff.CLIENTS`.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  accumulator_type = _get_accumulator_type(member_type)
  zeros = _zeros_for_sample(member_type)

  @tf.function
  def fed_concat_expand_dims(a, b):
    b = tf.expand_dims(b, axis=0)
    return tf.concat([a, b], axis=0)

  @tf.function
  def fed_concat(a, b):
    return tf.concat([a, b], axis=0)

  @tf.function
  def fed_gather(value, indices):
    return tf.gather(value, indices)

  def apply_sampling(accumulators, rands):
    size = tf.shape(rands)[0]
    k = tf.minimum(size, max_num_samples)
    indices = tf.math.top_k(rands, k=k).indices
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if member_type.is_struct():
      return structure.map_structure(lambda v: fed_gather(v, indices),
                                     accumulators), fed_gather(rands, indices)
    return fed_gather(accumulators, indices), fed_gather(rands, indices)

  @computations.tf_computation(accumulator_type, value.type_signature.member)
  def accumulate(current, value):
    """Accumulates samples through concatenation."""
    rands = fed_concat_expand_dims(current.rands, tf.random.uniform(shape=()))
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if member_type.is_struct():
      accumulators = structure.map_structure(
          fed_concat_expand_dims, _ensure_structure(current.accumulators),
          _ensure_structure(value))
    else:
      accumulators = fed_concat_expand_dims(current.accumulators, value)

    accumulators, rands = apply_sampling(accumulators, rands)
    return _Samples(accumulators, rands)

  @computations.tf_computation(accumulator_type, accumulator_type)
  def merge(a, b):
    """Merges accumulators through concatenation."""
    # TODO(b/121288403): Special-casing anonymous tuple shouldn't be needed.
    if accumulator_type.is_struct():
      samples = structure.map_structure(fed_concat, _ensure_structure(a),
                                        _ensure_structure(b))
    else:
      samples = fed_concat(a, b)
    accumulators, rands = apply_sampling(samples.accumulators, samples.rands)
    return _Samples(accumulators, rands)

  @computations.tf_computation(accumulator_type)
  def report(value):
    return value.accumulators

  return intrinsics.federated_aggregate(value, zeros, accumulate, merge, report)


def _ensure_structure(obj):
  return structure.from_container(obj, True)
