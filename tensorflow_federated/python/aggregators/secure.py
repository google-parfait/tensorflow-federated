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
"""Factories for secure summation."""

import collections
import enum
import math
import typing
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import primitives
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.backends.mapreduce import intrinsics as mapreduce_intrinsics
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

NORM_TYPE = np.float32
COUNT_TYPE = np.int32

ThresholdEstType = Union[
    int, float, np.integer, np.floating, estimation_process.EstimationProcess
]


# Enum for internal tracking of configuration of `SecureSumFactory`.
class _Config(enum.Enum):
  INT = 1
  FLOAT = 2


class SecureModularSumFactory(factory.UnweightedAggregationFactory):
  """`AggregationProcess` factory for securely summing values under a modulus.

  The created `tff.templates.AggregationProcess` uses the
  `tff.backends.mapreduce.federated_secure_modular_sum` operator for movement of
  values from `tff.CLIENTS` to `tff.SERVER`.

  The aggregator requires integer types, and values in range `[0, modulus-1]`
  (if `symmetric_range` is `False`) or in range `[-(modulus-1), +(modulus-1)]`
  (if `symmetric_range` is `True`). In the latter case, additional computation
  is needed for correct reduction to the
  `tff.backends.mapreduce.federated_secure_modular_sum` with `2*modulus-1` as
  the modulus for actual secure summation.

  The aggregator always returns a value in those ranges, implemented as a
  modular summation, regardless of input values. That is, if an input value at
  runtime is outside of the specified range an error will not be raised, rather
  the value will "wrap around", according to modular arithmetic.

  For example, if `symmetric_range` is `False`, given client values `[1, 3, 6]`
  and modulus `4`, the sum will be
  `((1 % 4) + (3 % 4) + (6 % 4)) % 4 = (1 + 3 + 2) % 4 = 6 % 4 = 2`.

  If `symmetric_range` is `True`, given client values `[1, 3, 6]` and modulus
  `4`, the sum will be
  `((1 %s 4) + (3 %s 4) + (6 %s 4)) %s 4 = (1 + 3 + (-3)) %s 4 = 1`.
  The `* %s x` operator symbolizes modular "wrap around" to range `[-x, x]`.

  The implementation of the case of `symmetric_range` is `True` is by delegation
  to `tff.backends.mapreduce.federated_secure_modular_sum` with modulus
  `2*modulus-1` is, which is equivalent to modular clip to range `[-(modulus-1),
  +(modulus-1)]`, and then representing `x` in that range as `(x + 2*modulus-1)
  % 2*modulus-1`, which is congruent with `x` under the desired modulus, thus
  compatible with secure aggregation. This is reverted after summation by
  modular clip to the initial range `[-(modulus-1), +(modulus-1)]`.

  NOTE: Unlike `tff.backends.mapreduce.federated_secure_modular_sum`, the
  `modulus` cannot be a structure of integers. If different moduli are needed
  for different tensors, recommended way of to achieve it is using the
  compositon of aggregators.
  """

  def __init__(
      self, modulus: Union[int, np.ndarray], symmetric_range: bool = False
  ):
    """Initializes `SecureModularSumFactory`.

    Args:
      modulus: An integer modulus for the summation.
      symmetric_range: A bool indicating whether the summation is on symmetric
        range around `0` or not.

    Raises:
      TypeError: If `modulus` is not an `int` or Numpy scalar, or
        `symmetric_range` is not a `bool`.
      ValueError: If `modulus` is not positive.
    """
    py_typecheck.check_type(modulus, (int, np.integer))
    if modulus <= 0:
      raise ValueError(
          f'Provided modulus must be positive, but found: {modulus}'
      )
    py_typecheck.check_type(symmetric_range, bool)

    self._modulus = modulus
    self._symmetric_range = symmetric_range

  def create(self, value_type):
    type_args = typing.get_args(factory.ValueType)
    py_typecheck.check_type(value_type, type_args)
    if not type_analysis.is_structure_of_integers(value_type):
      raise TypeError(
          'Provided value_type must either be an integer type or'
          f'a structure of integer types, but found: {value_type}'
      )

    @federated_computation.federated_computation
    def init_fn():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.FederatedType(value_type, placements.CLIENTS),
    )
    def next_fn(state, value):
      if self._symmetric_range:
        # Sum in [-M+1, M-1].
        # Delegation to `tff.backends.mapreduce.federated_secure_modular_sum`
        # with modulus 2*M-1 is equivalent to modular clip to range [-M+1, M-1].
        # Then, represent `x` in that range as `(x + 2*M-1) % 2*M-1` which is
        # congruent with `x` under the desired modulus, thus compatible with
        # secure aggregation. This is reverted after summation by modular clip
        # to the initial range.
        summed_value = mapreduce_intrinsics.federated_secure_modular_sum(
            value, 2 * self._modulus - 1
        )
        summed_value = intrinsics.federated_map(
            tensorflow_computation.tf_computation(
                self._mod_clip_after_symmetric_range_sum
            ),
            summed_value,
        )
      else:
        summed_value = mapreduce_intrinsics.federated_secure_modular_sum(
            value, self._modulus
        )
      empty_measurements = intrinsics.federated_value((), placements.SERVER)
      return measured_process.MeasuredProcessOutput(
          state, summed_value, empty_measurements
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def _mod_clip_after_symmetric_range_sum(self, value):
    """Modular clip by value after summation with symmetric_range.

    Since we know the sum will be in range [0, 2*(modulus-1)], the modular
    clip to range [-(modulus-1), +(modulus-1)] is equivalent to shifting values
    larger or equal to the modulus.

    Args:
      value: The summed value to be clipped.

    Returns:
      The clipped value.
    """

    def shift_negative_values(v):
      where = tf.cast(tf.math.greater_equal(v, self._modulus), v.dtype)
      return v - (2 * self._modulus - 1) * where

    return tf.nest.map_structure(shift_negative_values, value)


def _check_bound_process(
    bound_process: estimation_process.EstimationProcess, name: str
):
  """Checks type properties for estimation process for bounds.

  The process must be an `EstimationProcess` with `next` function of type
  signature (<state@SERVER, NORM_TYPE@CLIENTS> -> state@SERVER), and `report`
  with type signature (state@SERVER -> NORM_TYPE@SERVER).

  Args:
    bound_process: A process to check.
    name: A string name for formatting error messages.
  """
  py_typecheck.check_type(bound_process, estimation_process.EstimationProcess)

  next_parameter_type = bound_process.next.type_signature.parameter
  if (
      not isinstance(next_parameter_type, computation_types.StructType)
      or len(next_parameter_type) != 2
  ):
    raise TypeError(
        f'`{name}.next` must take two arguments but found:\n'
        f'{next_parameter_type}'
    )

  float_type_at_clients = computation_types.FederatedType(
      NORM_TYPE, placements.CLIENTS
  )
  if not next_parameter_type[1].is_assignable_from(float_type_at_clients):  # pytype: disable=unsupported-operands
    raise TypeError(
        f'Second argument of `{name}.next` must be assignable from '
        f'{float_type_at_clients} but found {next_parameter_type[1]}'  # pytype: disable=unsupported-operands
    )

  next_result_type = bound_process.next.type_signature.result
  if not bound_process.state_type.is_assignable_from(next_result_type):
    raise TypeError(
        f'Result type of `{name}.next` must consist of state only '
        f'but found result type:\n{next_result_type}\n'
        f'while the state type is:\n{bound_process.state_type}'
    )

  report_type = bound_process.report.type_signature.result
  estimated_value_type_at_server = computation_types.FederatedType(
      next_parameter_type[1].member,  # pytype: disable=unsupported-operands
      placements.SERVER,
  )
  if not report_type.is_assignable_from(estimated_value_type_at_server):
    raise TypeError(
        f'Report type of `{name}.report` must be assignable from '
        f'{estimated_value_type_at_server} but found {report_type}.'
    )


class SecureSumFactory(factory.UnweightedAggregationFactory):
  """`AggregationProcess` factory for securely summing values.

  The created `tff.templates.AggregationProcess` uses the
  `tff.federated_secure_sum_bitwidth` operator for movement of all values from
  `tff.CLIENTS` to `tff.SERVER`.

  In order for values to be securely summed, their range needs to be known in
  advance and communicated to clients, so that clients can prepare the values in
  a form compatible with the `tff.federated_secure_sum_bitwidth` operator (that
  is, integers in range `[0, 2**b-1]` for some `b`), and for inverse mapping to
  be applied on the server. This will be done as specified by the
  `upper_bound_threshold` and `lower_bound_threshold` constructor arguments,
  with the following options:

  For integer values to be summed, these arguments must be `int` Python
  constants or integer Numpy scalars, and the values during execution will be
  clipped to these thresholds and then securely summed.

  For floating point values to be summed, the values during execution will be
  clipped to these thresholds, then uniformly quantized to integers in the range
  `[0, 2**32-1]`, and then securely summed.

  The `upper_bound_threshold` and `lower_bound_threshold` arguments can in this
  case be either `float` Python constants, float Numpy scalars, or instances of
  `tff.templates.EstimationProcess`, which adapts the thresholds between rounds
  of execution.

  In all cases, it is possible to specify only `upper_bound_threshold`, in which
  case this threshold will be treated as a bound on the absolute value of the
  value to be summed.

  For the case when floating point values are to be securely summed and more
  aggressive quantization is needed (i.e. less than 32 bits), the recommended
  pattern is to use `tff.aggregators.EncodedSumFactory` with this factory class
  as its inner aggregation factory.

  If the `value_type` passed to the `create` method is a structure, all its
  constituent `tff.TensorType`s must have the same dtype (i.e. mixing
  `tf.float32` and `tf.float64` is not allowed).

  The created process will report measurements
  * `secure_upper_threshold`: The upper constant used for clipping.
  * `secure_lower_threshold`: The lower constant used for clipping.
  * `secure_upper_clipped_count`: The number of aggregands clipped to the
    `secure_upper_threshold` before aggregation.
  * `secure_lower_clipped_count`: The number of aggregands clipped to the
    `secure_lower_threshold` before aggregation.
  """

  def __init__(
      self,
      upper_bound_threshold: ThresholdEstType,
      lower_bound_threshold: Optional[ThresholdEstType] = None,
  ):
    """Initializes `SecureSumFactory`.

    Args:
      upper_bound_threshold: Either a `int` or `float` Python constant, a Numpy
        scalar, or a `tff.templates.EstimationProcess`, used for determining the
        upper bound before summation.
      lower_bound_threshold: Optional. Either a `int` or `float` Python
        constant, a Numpy scalar, or a `tff.templates.EstimationProcess`, used
        for determining the lower bound before summation. If specified, must be
        the same type as `upper_bound_threshold`.

    Raises:
      TypeError: If `upper_bound_threshold` and `lower_bound_threshold` are not
        instances of one of (`int`, `float` or
        `tff.templates.EstimationProcess`).
      ValueError: If `upper_bound_threshold` is provided as a negative constant.
    """
    type_args = typing.get_args(ThresholdEstType)
    py_typecheck.check_type(upper_bound_threshold, type_args)
    if lower_bound_threshold is not None:
      if not isinstance(lower_bound_threshold, type(upper_bound_threshold)):
        raise TypeError(
            'Provided upper_bound_threshold and lower_bound_threshold '
            'must have the same types, but found:\n'
            f'type(upper_bound_threshold): {upper_bound_threshold}\n'
            f'type(lower_bound_threshold): {lower_bound_threshold}'
        )
    self._upper_bound_threshold = upper_bound_threshold
    self._lower_bound_threshold = lower_bound_threshold

    # Configuration specific for aggregating integer types.
    if _is_integer(upper_bound_threshold):
      self._config_mode = _Config.INT
      if lower_bound_threshold is None:
        _check_positive(upper_bound_threshold)
        lower_bound_threshold = -1 * upper_bound_threshold
      else:
        _check_upper_larger_than_lower(
            upper_bound_threshold, lower_bound_threshold
        )
      self._init_fn = _empty_state
      self._update_state = lambda _, __, ___: _empty_state()
      # We must add one because the size of inclusive range [0, threshold_range]
      # is threshold_range + 1. We ensure that threshold_range > 0 above.
      self._secagg_bitwidth = math.ceil(
          math.log2(upper_bound_threshold - lower_bound_threshold + 1)
      )

    # Configuration specific for aggregating floating point types.
    else:
      self._config_mode = _Config.FLOAT
      if _is_float(upper_bound_threshold):
        # Bounds specified as Python constants.
        if lower_bound_threshold is None:
          _check_positive(upper_bound_threshold)
        else:
          _check_upper_larger_than_lower(
              upper_bound_threshold, lower_bound_threshold
          )
        self._init_fn = _empty_state
        self._update_state = lambda _, __, ___: _empty_state()
      else:
        # Bounds specified as an EstimationProcess.
        _check_bound_process(upper_bound_threshold, 'upper_bound_threshold')
        upper_bound_threshold = typing.cast(
            estimation_process.EstimationProcess, upper_bound_threshold
        )
        if lower_bound_threshold is None:
          self._init_fn = upper_bound_threshold.initialize
          self._update_state = _create_update_state_single_process(
              upper_bound_threshold
          )
        else:
          _check_bound_process(lower_bound_threshold, 'lower_bound_threshold')
          lower_bound_threshold = typing.cast(
              estimation_process.EstimationProcess, lower_bound_threshold
          )
          self._init_fn = _create_initial_state_two_processes(
              upper_bound_threshold, lower_bound_threshold
          )
          self._update_state = _create_update_state_two_processes(
              upper_bound_threshold, lower_bound_threshold
          )

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    self._check_value_type_compatible_with_config_mode(value_type)

    @federated_computation.federated_computation(
        self._init_fn.type_signature.result,
        computation_types.FederatedType(value_type, placements.CLIENTS),
    )
    def next_fn(state, value):
      # Compute min and max *before* clipping and use it to update the state.
      value_max = intrinsics.federated_map(_reduce_nest_max, value)
      value_min = intrinsics.federated_map(_reduce_nest_min, value)
      upper_bound, lower_bound = self._get_bounds_from_state(
          state, value_max.type_signature.member.dtype  # pytype: disable=attribute-error
      )

      new_state = self._update_state(state, value_min, value_max)

      # Clips value to [lower_bound, upper_bound] and securely sums it.
      summed_value = self._sum_securely(value, upper_bound, lower_bound)

      # TODO: b/163880757 - pass upper_bound and lower_bound through clients.
      measurements = self._compute_measurements(
          upper_bound, lower_bound, value_max, value_min
      )
      return measured_process.MeasuredProcessOutput(
          new_state, summed_value, measurements
      )

    return aggregation_process.AggregationProcess(self._init_fn, next_fn)

  def _compute_measurements(
      self, upper_bound, lower_bound, value_max, value_min
  ):
    """Creates measurements to be reported. All values are summed securely."""
    is_max_clipped = intrinsics.federated_map(
        tensorflow_computation.tf_computation(
            lambda bound, value: tf.cast(bound < value, COUNT_TYPE)
        ),
        (intrinsics.federated_broadcast(upper_bound), value_max),
    )
    max_clipped_count = intrinsics.federated_secure_sum_bitwidth(
        is_max_clipped, bitwidth=1
    )
    is_min_clipped = intrinsics.federated_map(
        tensorflow_computation.tf_computation(
            lambda bound, value: tf.cast(bound > value, COUNT_TYPE)
        ),
        (intrinsics.federated_broadcast(lower_bound), value_min),
    )
    min_clipped_count = intrinsics.federated_secure_sum_bitwidth(
        is_min_clipped, bitwidth=1
    )
    measurements = collections.OrderedDict(
        secure_upper_clipped_count=max_clipped_count,
        secure_lower_clipped_count=min_clipped_count,
        secure_upper_threshold=upper_bound,
        secure_lower_threshold=lower_bound,
    )
    return intrinsics.federated_zip(measurements)

  def _sum_securely(self, value, upper_bound, lower_bound):
    """Securely sums `value` placed at CLIENTS."""
    if self._config_mode == _Config.INT:
      value = intrinsics.federated_map(
          _client_shift,
          (
              value,
              intrinsics.federated_broadcast(upper_bound),
              intrinsics.federated_broadcast(lower_bound),
          ),
      )
      value = intrinsics.federated_secure_sum_bitwidth(
          value, self._secagg_bitwidth
      )
      num_summands = intrinsics.federated_secure_sum_bitwidth(
          _client_one(), bitwidth=1
      )
      value = intrinsics.federated_map(
          _server_shift, (value, lower_bound, num_summands)
      )
      return value
    elif self._config_mode == _Config.FLOAT:
      return primitives.secure_quantized_sum(value, lower_bound, upper_bound)
    else:
      raise ValueError(f'Unexpected internal config type: {self._config_mode}')

  def _get_bounds_from_state(self, state, bound_dtype):
    if isinstance(
        self._upper_bound_threshold, estimation_process.EstimationProcess
    ):
      if self._lower_bound_threshold is not None:
        bounds_fn = _create_get_bounds_two_processes(
            self._upper_bound_threshold,
            self._lower_bound_threshold,
            bound_dtype,
        )
      else:
        bounds_fn = _create_get_bounds_single_process(
            self._upper_bound_threshold, bound_dtype
        )
    else:
      if self._lower_bound_threshold is not None:
        bounds_fn = _create_get_bounds_const(
            self._upper_bound_threshold,
            self._lower_bound_threshold,
            bound_dtype,
        )
      else:
        bounds_fn = _create_get_bounds_const(
            self._upper_bound_threshold,
            -self._upper_bound_threshold,
            bound_dtype,
        )
    return bounds_fn(state)

  def _check_value_type_compatible_with_config_mode(self, value_type):
    type_args = typing.get_args(factory.ValueType)
    py_typecheck.check_type(value_type, type_args)
    if not _is_structure_of_single_dtype(value_type):
      raise TypeError(
          'Expected a type which is a structure containing the same dtypes, '
          f'found {value_type}.'
      )

    if self._config_mode == _Config.INT:
      if not type_analysis.is_structure_of_integers(value_type):
        raise TypeError(
            'The `SecureSumFactory` was configured to work with integer '
            'dtypes. All values in provided `value_type` hence must be of '
            f'integer dtype. \nProvided value_type: {value_type}'
        )
    elif self._config_mode == _Config.FLOAT:
      if not type_analysis.is_structure_of_floats(value_type):
        raise TypeError(
            'The `SecureSumFactory` was configured to work with floating '
            'point dtypes. All values in provided `value_type` hence must be '
            f'of floating point dtype. \nProvided value_type: {value_type}'
        )
    else:
      raise ValueError(f'Unexpected internal config type: {self._config_mode}')


def _check_positive(value):
  if value <= 0:
    raise ValueError(
        'If only `upper_bound_threshold` is specified as a Python constant, '
        'it must be positive. Its negative will be used as a lower bound '
        'which would be larger than the upper bound. \n'
        f'Provided `upper_bound_threshold`: {value}'
    )


def _check_upper_larger_than_lower(
    upper_bound_threshold, lower_bound_threshold
):
  if upper_bound_threshold <= lower_bound_threshold:
    raise ValueError(
        'The provided `upper_bound_threshold` must be larger than the '
        'provided `lower_bound_threshold`, but received:\n'
        f'`upper_bound_threshold`: {upper_bound_threshold}\n'
        f'`lower_bound_threshold`: {lower_bound_threshold}\n'
    )


def _is_integer(value):
  return isinstance(value, (int, np.integer))


def _is_float(value):
  return isinstance(value, (float, np.floating))


@tensorflow_computation.tf_computation()
def _reduce_nest_max(value):
  max_list = tf.nest.map_structure(tf.reduce_max, tf.nest.flatten(value))
  return tf.reduce_max(tf.stack(max_list))


@tensorflow_computation.tf_computation()
def _reduce_nest_min(value):
  min_list = tf.nest.map_structure(tf.reduce_min, tf.nest.flatten(value))
  return tf.reduce_min(tf.stack(min_list))


@tensorflow_computation.tf_computation()
def _client_shift(value, upper_bound, lower_bound):
  return tf.nest.map_structure(
      lambda v: tf.clip_by_value(v, lower_bound, upper_bound) - lower_bound,
      value,
  )


@tensorflow_computation.tf_computation()
def _server_shift(value, lower_bound, num_summands):
  return tf.nest.map_structure(
      lambda v: v + (lower_bound * tf.cast(num_summands, lower_bound.dtype)),
      value,
  )


@federated_computation.federated_computation()
def _empty_state():
  return intrinsics.federated_value((), placements.SERVER)


def _client_one():
  return intrinsics.federated_eval(
      tensorflow_computation.tf_computation(lambda: tf.constant(1, tf.int32)),
      placements.CLIENTS,
  )


def _create_initial_state_two_processes(
    upper_bound_process: estimation_process.EstimationProcess,
    lower_bound_process: estimation_process.EstimationProcess,
):
  @federated_computation.federated_computation()
  def initial_state():
    return intrinsics.federated_zip(
        (upper_bound_process.initialize(), lower_bound_process.initialize())
    )

  return initial_state


def _create_get_bounds_const(upper_bound, lower_bound, bound_dtype):
  """Gets TFF value bounds when specified as constants."""

  @tensorflow_computation.tf_computation
  def bounds_fn():
    upper_bound_tf = tf.constant(upper_bound, bound_dtype)
    lower_bound_tf = tf.constant(lower_bound, bound_dtype)
    return upper_bound_tf, lower_bound_tf

  def get_bounds(state):
    del state  # Unused.
    return intrinsics.federated_eval(bounds_fn, placements.SERVER)

  return get_bounds


def _create_get_bounds_single_process(
    process: estimation_process.EstimationProcess, bound_dtype
):
  """Gets TFF value bounds when specified as single estimation process."""

  def get_bounds(state):
    cast_fn = tensorflow_computation.tf_computation(
        lambda x: tf.cast(x, bound_dtype)
    )
    upper_bound = intrinsics.federated_map(cast_fn, process.report(state))
    lower_bound = intrinsics.federated_map(
        tensorflow_computation.tf_computation(lambda x: x * -1.0), upper_bound
    )
    return upper_bound, lower_bound

  return get_bounds


def _create_get_bounds_two_processes(
    upper_bound_process: estimation_process.EstimationProcess,
    lower_bound_process: estimation_process.EstimationProcess,
    bound_dtype,
):
  """Gets TFF value bounds when specified as two estimation processes."""

  def get_bounds(state):
    cast_fn = tensorflow_computation.tf_computation(
        lambda x: tf.cast(x, bound_dtype)
    )
    upper_bound = intrinsics.federated_map(
        cast_fn, upper_bound_process.report(state[0])
    )
    lower_bound = intrinsics.federated_map(
        cast_fn, lower_bound_process.report(state[1])
    )
    return upper_bound, lower_bound

  return get_bounds


def _create_update_state_single_process(
    process: estimation_process.EstimationProcess,
):
  """Updates state when bounds specified as single estimation process."""

  expected_dtype = process.next.type_signature.parameter[1].member.dtype  # pytype: disable=unsupported-operands

  def update_state(state, value_min, value_max):
    abs_max_fn = tensorflow_computation.tf_computation(
        lambda x, y: tf.cast(tf.maximum(tf.abs(x), tf.abs(y)), expected_dtype)
    )
    abs_value_max = intrinsics.federated_map(abs_max_fn, (value_min, value_max))
    return process.next(state, abs_value_max)

  return update_state


def _create_update_state_two_processes(
    upper_bound_process: estimation_process.EstimationProcess,
    lower_bound_process: estimation_process.EstimationProcess,
):
  """Updates state when bounds specified as two estimation processes."""

  max_dtype = upper_bound_process.next.type_signature.parameter[1].member.dtype  # pytype: disable=unsupported-operands
  min_dtype = lower_bound_process.next.type_signature.parameter[1].member.dtype  # pytype: disable=unsupported-operands

  def update_state(state, value_min, value_max):
    value_min = intrinsics.federated_map(
        tensorflow_computation.tf_computation(lambda x: tf.cast(x, min_dtype)),
        value_min,
    )
    value_max = intrinsics.federated_map(
        tensorflow_computation.tf_computation(lambda x: tf.cast(x, max_dtype)),
        value_max,
    )
    return intrinsics.federated_zip((
        upper_bound_process.next(state[0], value_max),
        lower_bound_process.next(state[1], value_min),
    ))

  return update_state


def _unique_dtypes_in_structure(
    type_spec: computation_types.Type,
) -> set[tf.dtypes.DType]:
  """Returns a set of unique dtypes in `type_spec`.

  Args:
    type_spec: A `computation_types.Type`.

  Returns:
    A `set` containing unique dtypes found in `type_spec`.
  """
  py_typecheck.check_type(type_spec, computation_types.Type)
  if isinstance(type_spec, computation_types.TensorType):
    return set([tf.dtypes.as_dtype(type_spec.dtype)])
  elif isinstance(type_spec, computation_types.StructType):
    return set(
        tf.nest.flatten(
            type_conversions.structure_from_tensor_type_tree(
                lambda t: t.dtype, type_spec
            )
        )
    )
  elif isinstance(type_spec, computation_types.FederatedType):
    return _unique_dtypes_in_structure(type_spec.member)
  else:
    return set()


def _is_structure_of_single_dtype(type_spec: computation_types.Type) -> bool:
  return len(_unique_dtypes_in_structure(type_spec)) == 1
