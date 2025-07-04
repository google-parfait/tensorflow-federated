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
"""A package of primitive (stateless) aggregations."""

from typing import Any, NamedTuple

import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation


def _validate_value_on_clients(value):
  py_typecheck.check_type(value, federated_language.Value)
  py_typecheck.check_type(
      value.type_signature, federated_language.FederatedType
  )
  if value.type_signature.placement is not federated_language.CLIENTS:
    raise TypeError(
        '`value` argument must be a federated_language.Value placed at CLIENTS.'
        ' Got: {!s}'.format(value.type_signature)
    )


class _Samples(NamedTuple):
  """Class representing internal sample data structure.

  The class contains two parts, `accumulators` and `rands`, that are parallel
  lists (e.g. the i-th index in one corresponds to the i-th index in the other).
  These two lists are used to sample from the accumulators with equal
  probability.
  """

  accumulators: Any
  rands: Any


def _zeros_for_sample(member_type):
  """Create an empty nested structure for the sample aggregation.

  Args:
    member_type: A `federated_language.Type` representing the member components
      of the federated type.

  Returns:
    A function of the result of zeros to first concatenate.
  """

  def zeros_for_tensor_type(tensor_type):
    return tf.zeros((0,) + tensor_type.shape, tensor_type.dtype)

  @tensorflow_computation.tf_computation
  def accumlator_type_fn():
    """Gets the type for the accumulators."""
    accumulator_zeros = type_conversions.structure_from_tensor_type_tree(
        zeros_for_tensor_type, member_type
    )
    return _Samples(accumulator_zeros, tf.zeros([0], tf.float32))

  return accumlator_type_fn()


def _get_accumulator_type(member_type):
  """Constructs a `federated_language.Type` for the accumulator in sample aggregation.

  Args:
    member_type: A `federated_language.Type` representing the member components
      of the federated type.

  Returns:
    The `federated_language.StructType` associated with the accumulator. The
    tuple contains
    two parts, `accumulators` and `rands`, that are parallel lists (e.g. the
    i-th index in one corresponds to the i-th index in the other). These two
    lists are used to sample from the accumulators with equal probability.
  """

  def add_unknown_first_dim(tensor_type):
    return federated_language.TensorType(
        tensor_type.dtype, (None,) + tensor_type.shape
    )

  accumulator_type = type_conversions.structure_from_tensor_type_tree(
      add_unknown_first_dim, member_type
  )
  return federated_language.to_type(
      _Samples(
          accumulators=accumulator_type,
          rands=federated_language.TensorType(np.float32, shape=[None]),
      )
  )


def federated_sample(value, max_num_samples=100):
  """Aggregation to produce uniform sample of at most `max_num_samples` values.

  Each client value is assigned a random number when it is examined during each
  accumulation. Each accumulate and merge only keeps the top N values based
  on the random number. Report drops the random numbers and only returns the
  at most N values sampled from the accumulated client values using standard
  reservoir sampling (https://en.wikipedia.org/wiki/Reservoir_sampling), where
  N is user provided `max_num_samples`.

  Args:
    value: A `federated_language.Value` placed on the
      `federated_language.CLIENTS`.
    max_num_samples: The maximum number of samples to collect from client
      values. If fewer clients than the defined max sample size participated in
      the round of computation, the actual number of samples will equal the
      number of clients in the round.

  Returns:
    At most `max_num_samples` samples of the value from the
    `federated_language.CLIENTS`.
  """
  _validate_value_on_clients(value)
  member_type = value.type_signature.member
  accumulator_type = _get_accumulator_type(member_type)
  zeros = _zeros_for_sample(member_type)

  def apply_sampling(accumulators, rands):
    size = tf.shape(rands)[0]
    k = tf.minimum(size, max_num_samples)
    indices = tf.math.top_k(rands, k=k).indices
    return tf.nest.map_structure(
        lambda v: tf.gather(v, indices), accumulators
    ), tf.gather(rands, indices)

  def concat_expand_dims(a, b):
    b = tf.expand_dims(b, axis=0)
    return tf.concat([a, b], axis=0)

  @tensorflow_computation.tf_computation(
      accumulator_type, value.type_signature.member
  )
  def accumulate(current, value):
    """Accumulates samples through concatenation."""
    rands = concat_expand_dims(current.rands, tf.random.uniform(shape=()))
    accumulators = tf.nest.map_structure(
        concat_expand_dims, current.accumulators, value
    )
    accumulators, rands = apply_sampling(accumulators, rands)
    return _Samples(accumulators, rands)

  @tensorflow_computation.tf_computation(accumulator_type, accumulator_type)
  def merge(a, b):
    """Merges accumulators through concatenation."""

    def zero_axis_concat(a, b):
      return tf.concat([a, b], axis=0)

    samples = tf.nest.map_structure(zero_axis_concat, a, b)
    accumulators, rands = apply_sampling(samples.accumulators, samples.rands)
    return _Samples(accumulators, rands)

  @tensorflow_computation.tf_computation(accumulator_type)
  def report(value):
    return value.accumulators

  return federated_language.federated_aggregate(
      value, zeros, accumulate, merge, report
  )


# Lower precision types are not supported to avoid potential hard to discover
# numerical issues in conversion to/from format compatible with secure sum.
_SECURE_QUANTIZED_SUM_ALLOWED_DTYPES = (
    np.int32,
    np.int64,
    np.float32,
    np.float64,
)

# The largest integer value provided to federated_secure_sum_bitwidth operator.
_SECAGG_MAX = 2**32 - 1


class BoundsDifferentTypesError(TypeError):

  def __init__(self, lower_bound, upper_bound):
    message = (
        'Both lower_bound and upper_bound must be either federated '
        'values or Python constants. Found: type(lower_bound): '
        f'{type(lower_bound)}, type(upper_bound): {type(upper_bound)}'
    )
    super().__init__(message)


class BoundsDifferentSignaturesError(TypeError):

  def __init__(self, lower_bound, upper_bound):
    message = (
        'Provided lower_bound and upper_bound must have the same '
        'type_signature. Found: lower_bound signature: '
        f'{lower_bound.type_signature}, upper_bound signature: '
        f'{upper_bound.type_signature}.'
    )
    super().__init__(message)


class BoundsNotPlacedAtServerError(TypeError):

  def __init__(self, placement):
    message = (
        'Provided lower_bound and upper_bound must be placed at '
        f'federated_language.SERVER. Placement found: {placement}'
    )
    super().__init__(message)


class StructuredBoundsTypeMismatchError(TypeError):

  def __init__(self, value_type, bounds_type):
    message = (
        'If bounds are specified as structures (not scalars), the '
        'structures must match the structure of provided client_value, '
        'with identical dtypes, but not necessarily shapes. Found: '
        f'client_value type: {value_type}, bounds type: {bounds_type}.'
    )
    super().__init__(message)


class ScalarBoundStructValueDTypeError(TypeError):

  def __init__(self, value_type, bounds_type):
    message = (
        'If scalar bounds are provided, all parts of client_value must '
        f'be of matching dtype. Found: client_value type: {value_type}, '
        f'bounds type: {bounds_type}.'
    )
    super().__init__(message)


class ScalarBoundSimpleValueDTypeError(TypeError):

  def __init__(self, value_type, bounds_type):
    message = (
        'Bounds must have the same dtype as client_value. Found: '
        f'client_value type: {value_type}, bounds type: {bounds_type}.'
    )
    super().__init__(message)


class UnsupportedDTypeError(TypeError):

  def __init__(self, dtype):
    message = (
        f'Value is of unsupported dtype {dtype}. Currently supported '
        f'types are {_SECURE_QUANTIZED_SUM_ALLOWED_DTYPES}.'
    )
    super().__init__(message)


def _check_secure_quantized_sum_dtype(dtype):
  if dtype not in _SECURE_QUANTIZED_SUM_ALLOWED_DTYPES:
    raise UnsupportedDTypeError(dtype)


def _normalize_secure_quantized_sum_args(
    client_value, lower_bound, upper_bound
):
  """Normalizes inputs to `secure_quantized_sum` method.

  Validates the epxected structure of arguments, as documented in the docstring
  of `secure_quantized_sum` method. The arguments provided are also returned,
  possibly normalized to meet those expectations. In particular, if
  `lower_bound` and `upper_bound` are Python constants, these are converted to
  `federated_language.SERVER`-placed federated values.

  Args:
    client_value: A `federated_language.Value` placed at
      `federated_language.CLIENTS`.
    lower_bound: The smallest possible value for `client_value` (inclusive).
      Values smaller than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `federated_language.Value` placed at
      `federated_language.SERVER`, with dtype matching that of `client_value`.
    upper_bound: The largest possible value for `client_value` (inclusive).
      Values greater than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `federated_language.Value` placed at
      `federated_language.SERVER`, with dtype matching that of `client_value`.

  Returns:
    Normalized `(client_value, lower_bound, upper_bound)` tuple.

  Raises:
    BoundsDifferentTypesError: If `lower_bound` and `upper_bound` are not both
      `federated_language.Value`s or both not `federated_language.Value`s.
    BoundsDifferentSignaturesError: If `lower_bound.type_signature` and
      `upper_bound.type_signature` are not equal.
    BoundsNotPlacedAtServerError: If `lower_bound.type_signature` and
      `upper_bound.type_signature` are not placed at
      `federated_language.SERVER`.
    StructuredBoundsTypeMismatchError: If `lower_bound` and `upper_bound` are
      structures and do not match the structure and dtypes of `client_value`.
    ScalarBoundStructValueDTypeError: If `lower_bound` and `upper_bound` are
      scalars and `client_value` is a structure and all dtypes in `client_value`
      are not equal to the dtype of the bounds.
    ScalarBoundSimpleValueDTypeError: If `lower_bound` and `upper_bound` are
      scalars and `client_value` is a scalars and the dtype fo `client_value`
      is not equal to the dtype of the bounds.
  """
  # Validation of client_value.
  _validate_value_on_clients(client_value)
  client_value_member = client_value.type_signature.member
  if isinstance(
      client_value.type_signature.member, federated_language.StructType
  ):
    dtypes = [v.dtype for v in structure.flatten(client_value_member)]
    for dtype in dtypes:
      _check_secure_quantized_sum_dtype(dtype)
  else:
    dtypes = client_value_member.dtype
    _check_secure_quantized_sum_dtype(dtypes)

  # Validation of bounds.
  if isinstance(lower_bound, federated_language.Value) != isinstance(
      upper_bound, federated_language.Value
  ):
    raise BoundsDifferentTypesError(lower_bound, upper_bound)
  elif not isinstance(lower_bound, federated_language.Value):
    # Normalization of bounds to federated values.
    lower_bound = federated_language.federated_value(
        lower_bound, federated_language.SERVER
    )
    upper_bound = federated_language.federated_value(
        upper_bound, federated_language.SERVER
    )

  if lower_bound.type_signature != upper_bound.type_signature:
    raise BoundsDifferentSignaturesError(lower_bound, upper_bound)
  # The remaining type checks only use lower_bound as the upper_bound has
  # itendical type_signature.
  if lower_bound.type_signature.placement != federated_language.SERVER:  # pytype: disable=attribute-error
    raise BoundsNotPlacedAtServerError(lower_bound.type_signature.placement)  # pytype: disable=attribute-error

  # Validation of client_value and bounds compatibility.
  bound_member = lower_bound.type_signature.member  # pytype: disable=attribute-error
  if isinstance(bound_member, federated_language.StructType):
    if not isinstance(client_value_member, federated_language.StructType) or (
        structure._map_structure(lambda v: v.dtype, bound_member)  # pylint: disable=protected-access
        != structure._map_structure(lambda v: v.dtype, client_value_member)  # pylint: disable=protected-access
    ):
      raise StructuredBoundsTypeMismatchError(client_value_member, bound_member)
  else:
    # If bounds are scalar, must be compatible with all tensors in client_value.
    if isinstance(client_value_member, federated_language.StructType):
      if len(set(dtypes)) > 1 or (bound_member.dtype != dtypes[0]):
        raise ScalarBoundStructValueDTypeError(
            client_value_member, bound_member
        )
    else:
      if bound_member.dtype != client_value_member.dtype:
        raise ScalarBoundSimpleValueDTypeError(
            client_value_member, bound_member
        )

  return client_value, lower_bound, upper_bound


def _client_tensor_shift_for_secure_sum(value, lower_bound, upper_bound):
  """Mapping to be applied to every tensor before secure sum.

  This operation is performed on `federated_language.CLIENTS` to prepare values
  to format
  compatible with `federated_language.federated_secure_sum_bitwidth` operator.

  This clips elements of `value` to `[lower_bound, upper_bound]`, shifts and
  scales it to range `[0, 2**32-1]` and casts it to `tf.int64`. The specific
  operation depends on dtype of `value`.

  Args:
    value: A Tensor to be shifted for compatibility with
      `federated_secure_sum_bitwidth`.
    lower_bound: The smallest value expected in `value`.
    upper_bound: The largest value expected in `value`.

  Returns:
    Shifted value of dtype `tf.int64`.
  """
  tf.debugging.assert_less_equal(
      lower_bound,
      upper_bound,
      message=(
          'lower_bound must be smaller than upper_bound for secagg quantization'
      ),
  )
  if value.dtype == tf.int32:
    clipped_val = tf.clip_by_value(value, lower_bound, upper_bound)
    # Cast BEFORE shift in order to avoid overflow if full int32 range is used.
    return tf.cast(clipped_val, tf.int64) - tf.cast(lower_bound, tf.int64)
  elif value.dtype == tf.int64:
    clipped_val = tf.clip_by_value(value, lower_bound, upper_bound)
    range_span = upper_bound - lower_bound
    scale_factor = tf.math.floordiv(range_span, _SECAGG_MAX) + 1
    shifted_value = tf.cond(
        tf.greater(scale_factor, 1),
        lambda: tf.math.floordiv(clipped_val - lower_bound, scale_factor),
        lambda: clipped_val - lower_bound,
    )
    return shifted_value
  else:
    # This should be ensured earlier and thus not user-facing.
    assert value.dtype in [tf.float32, tf.float64]
    clipped_value = tf.clip_by_value(value, lower_bound, upper_bound)
    # Prevent NaNs if `lower_bound` and `upper_bound` are the same.
    scale_factor = tf.math.divide_no_nan(
        tf.constant(_SECAGG_MAX, tf.float64),
        tf.cast(upper_bound - lower_bound, tf.float64),
    )
    scaled_value = tf.cast(clipped_value, tf.float64) * scale_factor
    # Perform deterministic rounding here, which may introduce bias as every
    # value may be rounded in the same direction for some input data.
    rounded_value = tf.saturate_cast(tf.round(scaled_value), tf.int64)
    # Perform shift in integer space to minimize float precision errors.
    shifted_value = rounded_value - tf.saturate_cast(
        tf.round(tf.cast(lower_bound, tf.float64) * scale_factor), tf.int64
    )
    # Clip to expected range in case of numerical stability issues.
    quantized_value = tf.clip_by_value(
        shifted_value,
        tf.constant(0, dtype=tf.int64),
        tf.constant(_SECAGG_MAX, dtype=tf.int64),
    )
    return quantized_value


@tf.function
def _server_tensor_shift_for_secure_sum(
    num_summands, value, lower_bound, upper_bound, output_dtype
):
  """Mapping to be applied to every tensor after secure sum.

  This operation is performed on `federated_language.SERVER` to dequantize
  outputs of the
  `federated_language.federated_secure_sum_bitwidth` operator.

  It is reverse of `_client_tensor_shift_for_secure_sum` taking into account
  that `num_summands` elements were summed, so the inverse shift needs to be
  appropriately scaled.

  Args:
    num_summands: The number of summands that formed `value`.
    value: A summed Tensor to be shifted to original representation.
    lower_bound: The smallest value expected in `value` before it was summed.
    upper_bound: The largest value expected in `value` before it was summed.
    output_dtype: The dtype of value after being shifted.

  Returns:
    Shifted value of dtype `output_dtype`.
  """
  # Ensure summed `value` is within the expected range given `num_summands`.
  min_valid_value = tf.constant(0, tf.int64)
  # Cast to tf.int64 before multiplication to prevent overflow.
  max_valid_value = tf.constant(_SECAGG_MAX, tf.int64) * tf.cast(
      num_summands, tf.int64
  )
  tf.Assert(
      tf.math.logical_and(
          tf.math.reduce_min(value) >= min_valid_value,
          tf.math.reduce_max(value) <= max_valid_value,
      ),
      [value, min_valid_value, max_valid_value],
  )

  if output_dtype == tf.int32:
    value = value + tf.cast(num_summands, tf.int64) * tf.cast(
        lower_bound, tf.int64
    )
  elif output_dtype == tf.int64:
    range_span = upper_bound - lower_bound
    scale_factor = tf.math.floordiv(range_span, _SECAGG_MAX) + 1
    num_summands = tf.cast(num_summands, tf.int64)
    value = tf.cond(
        scale_factor > 1,
        lambda: value * scale_factor + num_summands * lower_bound,
        lambda: value + num_summands * lower_bound,
    )
  else:
    # This should be ensured earlier and thus not user-facing.
    assert output_dtype in [tf.float32, tf.float64]
    # Use exactly the same `scale_factor` as during client quantization so that
    # float precision errors (which are deterministic) cancel out. This ensures
    # that the sum of [0] is exactly 0 for any clipping range.
    scale_factor = tf.math.divide_no_nan(
        tf.constant(_SECAGG_MAX, tf.float64),
        tf.cast(upper_bound - lower_bound, tf.float64),
    )
    # Scale the shift by `num_summands` as an integer to prevent additional
    # float precision errors for multiple summands. This also ensures that the
    # sum of [0] * num_summands is exactly 0 for any clipping range.
    value = value + tf.saturate_cast(
        tf.round(tf.cast(lower_bound, tf.float64) * scale_factor), tf.int64
    ) * tf.cast(num_summands, tf.int64)
    value = tf.cast(value, tf.float64)
    value = value * (
        tf.cast(upper_bound - lower_bound, tf.float64) / _SECAGG_MAX
    )
    # If `lower_bound` and `upper_bound` are the same, the above shift had no
    # effect since `scale_factor` is 0. Shift here instead.
    shifted_value = value + tf.cast(num_summands, tf.float64) * tf.cast(
        lower_bound, tf.float64
    )
    value = tf.cond(
        tf.equal(lower_bound, upper_bound), lambda: shifted_value, lambda: value
    )

  return tf.cast(value, output_dtype)


def secure_quantized_sum(client_value, lower_bound, upper_bound):
  """Quantizes and sums values securely.

  Provided `client_value` can be either a Tensor or a nested structure of
  Tensors. If it is a nested structure, `lower_bound` and `upper_bound` must be
  either both scalars, or both have the same structure as `client_value`, with
  each element being a scalar, representing the bounds to be used for each
  corresponding Tensor in `client_value`.

  This method converts each Tensor in provided `client_value` to appropriate
  format and uses the `federated_language.federated_secure_sum_bitwidth`
  operator to realize
  the sum.

  The dtype of Tensors in provided `client_value` can be one of `[tf.int32,
  tf.int64, tf.float32, tf.float64]`.

  If the dtype of `client_value` is `tf.int32` or `tf.int64`, the summation is
  possibly exact, depending on `lower_bound` and `upper_bound`: In the case that
  `upper_bound - lower_bound < 2**32`, the summation will be exact. If it is
  not, `client_value` will be quantized to precision of 32 bits, so the worst
  case error introduced for the value of each client will be approximately
  `(upper_bound - lower_bound) / 2**32`. Deterministic rounding to nearest value
  is used in such cases.

  If the dtype of `client_value` is `tf.float32` or `tf.float64`, the summation
  is generally *not* accurate up to full floating point precision. Instead, the
  values are first clipped to the `[lower_bound, upper_bound]` range. These
  values are then uniformly quantized to 32 bit resolution, using deterministic
  rounding to round the values to the quantization points. Rounding happens
  roughly as follows (implementation is a bit more complex to mitigate numerical
  stability issues):

  ```
  values = tf.round(
      (client_value - lower_bound) * ((2**32 - 1) / (upper_bound - lower_bound))
  ```

  After summation, the inverse operation if performed, so the return value
  is of the same dtype as the input `client_value`.

  In terms of accuracy, it is safe to assume accuracy within 7-8 significant
  digits for `tf.float32` inputs, and 8-9 significant digits for `tf.float64`
  inputs, where the significant digits refer to precision relative to the range
  of the provided bounds. Thus, these bounds should not be set extremely wide.
  Accuracy losses arise due to (1) quantization within the given clipping range,
  (2) float precision of final outputs (e.g. `tf.float32` has 23 bits in its
  mantissa), and (3) precision losses that arise in doing math on `tf.float32`
  and `tf.float64` inputs.

  As a concrete example, if the range is `+/- 1000`, errors up to `1e-4` per
  element should be expected for `tf.float32` and up to `1e-5` for `tf.float64`.

  Args:
    client_value: A `federated_language.Value` placed at
      `federated_language.CLIENTS`.
    lower_bound: The smallest possible value for `client_value` (inclusive).
      Values smaller than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `federated_language.Value` placed at
      `federated_language.SERVER`, with dtype matching that of `client_value`.
    upper_bound: The largest possible value for `client_value` (inclusive).
      Values greater than this bound will be clipped. Must be either a scalar or
      a nested structure of scalars, matching the structure of `client_value`.
      Must be either a Python constant or a `federated_language.Value` placed at
      `federated_language.SERVER`, with dtype matching that of `client_value`.

  Returns:
    Summed `client_value` placed at `federated_language.SERVER`, of the same
    dtype as
    `client_value`.

  Raises:
    TypeError (or its subclasses): If input arguments do not satisfy the type
      constraints specified above.
  """

  # Possibly converts Python constants to federated values.
  client_value, lower_bound, upper_bound = _normalize_secure_quantized_sum_args(
      client_value, lower_bound, upper_bound
  )

  # This object is used during decoration of the `client_shift` method, and the
  # value stored in this mutable container is used during decoration of the
  # `server_shift` method. The reason for this is that we cannot currently get
  # the needed information out of `client_value.type_signature.member` as we
  # need both the `TensorType` information as well as the Python container
  # attached to them.
  temp_box = []

  # These tf_computations assume the inputs were already validated. In
  # particular, that lower_bnd and upper_bnd have the same structure, and if not
  # scalar, the structure matches the structure of value.
  @tensorflow_computation.tf_computation()
  def client_shift(value, lower_bnd, upper_bnd):
    assert not temp_box
    temp_box.append(tf.nest.map_structure(lambda v: v.dtype, value))
    fn = _client_tensor_shift_for_secure_sum
    if tf.is_tensor(lower_bnd):
      return tf.nest.map_structure(lambda v: fn(v, lower_bnd, upper_bnd), value)
    else:
      return tf.nest.map_structure(fn, value, lower_bnd, upper_bnd)

  @tensorflow_computation.tf_computation()
  def server_shift(value, lower_bnd, upper_bnd, summands):
    fn = _server_tensor_shift_for_secure_sum
    if tf.is_tensor(lower_bnd):
      return tf.nest.map_structure(
          lambda v, dtype: fn(summands, v, lower_bnd, upper_bnd, dtype),
          value,
          temp_box[0],
      )
    else:
      return tf.nest.map_structure(
          lambda *args: fn(summands, *args),
          value,
          lower_bnd,
          upper_bnd,
          temp_box[0],
      )

  client_one = federated_language.federated_value(1, federated_language.CLIENTS)

  # Orchestration.
  client_lower_bound = federated_language.federated_broadcast(lower_bound)
  client_upper_bound = federated_language.federated_broadcast(upper_bound)

  value = federated_language.federated_map(
      client_shift, (client_value, client_lower_bound, client_upper_bound)
  )
  num_summands = federated_language.federated_secure_sum_bitwidth(
      client_one, bitwidth=1
  )

  secagg_value_type = value.type_signature.member  # pytype: disable=attribute-error
  assert isinstance(
      secagg_value_type, federated_language.TensorType
  ) or isinstance(secagg_value_type, federated_language.StructType)
  if isinstance(secagg_value_type, federated_language.TensorType):
    bitwidths = 32
  else:
    bitwidths = structure._map_structure(lambda t: 32, secagg_value_type)  # pylint: disable=protected-access

  value = federated_language.federated_secure_sum_bitwidth(
      value, bitwidth=bitwidths
  )
  value = federated_language.federated_map(
      server_shift, (value, lower_bound, upper_bound, num_summands)
  )
  return value
