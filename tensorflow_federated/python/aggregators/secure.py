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
"""Factory for secure summation."""

import collections
import enum
import math
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import primitives
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

NORM_TF_TYPE = tf.float32
COUNT_TF_TYPE = tf.int32

ThresholdEstType = Union[int, float, np.ndarray,
                         estimation_process.EstimationProcess]


# Enum for internal tracking of configuration of `SecureSumFactory`.
class _Config(enum.Enum):
  INT = 1
  FLOAT = 2


def _check_bound_process(bound_process: estimation_process.EstimationProcess,
                         name: str):
  """Checks type properties for estimation process for bounds.

  The process must be an `EstimationProcess` with `next` function of type
  signature (<state@SERVER, NORM_TF_TYPE@CLIENTS> -> state@SERVER), and `report`
  with type signature (state@SERVER -> NORM_TF_TYPE@SERVER).

  Args:
    bound_process: A process to check.
    name: A string name for formatting error messages.
  """
  py_typecheck.check_type(bound_process, estimation_process.EstimationProcess)

  next_parameter_type = bound_process.next.type_signature.parameter
  if not next_parameter_type.is_struct() or len(next_parameter_type) != 2:
    raise TypeError(f'`{name}.next` must take two arguments but found:\n'
                    f'{next_parameter_type}')

  float_type_at_clients = computation_types.at_clients(NORM_TF_TYPE)
  if not next_parameter_type[1].is_assignable_from(float_type_at_clients):
    raise TypeError(
        f'Second argument of `{name}.next` must be assignable from '
        f'{float_type_at_clients} but found {next_parameter_type[1]}')

  next_result_type = bound_process.next.type_signature.result
  if not bound_process.state_type.is_assignable_from(next_result_type):
    raise TypeError(f'Result type of `{name}.next` must consist of state only '
                    f'but found result type:\n{next_result_type}\n'
                    f'while the state type is:\n{bound_process.state_type}')

  report_type = bound_process.report.type_signature.result
  estimated_value_type_at_server = computation_types.at_server(
      next_parameter_type[1].member)
  if not report_type.is_assignable_from(estimated_value_type_at_server):
    raise TypeError(
        f'Report type of `{name}.report` must be assignable from '
        f'{estimated_value_type_at_server} but found {report_type}.')


class SecureSumFactory(factory.UnweightedAggregationFactory):
  """`AggregationProcess` factory for securely summing values.

  The created `tff.templates.AggregationProcess` uses the
  `tff.federated_secure_sum` operator for movement of all values from
  `tff.CLIENTS` to `tff.SERVER`.

  In order for values to be securely summed, their range needs to be known in
  advance and communicated to clients, so that clients can prepare the values in
  a form compatible with the `tff.federated_secure_sum` operator (that is,
  integers in range `[0, 2**b-1]` for some `b`), and for inverse mapping to be
  applied on the server. This will be done as specified by the
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
  """

  def __init__(self,
               upper_bound_threshold: ThresholdEstType,
               lower_bound_threshold: Optional[ThresholdEstType] = None):
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
    py_typecheck.check_type(upper_bound_threshold, ThresholdEstType.__args__)
    if lower_bound_threshold is not None:
      if not isinstance(lower_bound_threshold, type(upper_bound_threshold)):
        raise TypeError(
            f'Provided upper_bound_threshold and lower_bound_threshold '
            f'must have the same types, but found:\n'
            f'type(upper_bound_threshold): {upper_bound_threshold}\n'
            f'type(lower_bound_threshold): {lower_bound_threshold}')

    # Configuration specific for aggregating integer types.
    if _is_integer(upper_bound_threshold):
      self._config_mode = _Config.INT
      if lower_bound_threshold is None:
        _check_positive(upper_bound_threshold)
        lower_bound_threshold = -1 * upper_bound_threshold
      else:
        _check_upper_larger_than_lower(upper_bound_threshold,
                                       lower_bound_threshold)
      self._init_fn = _empty_state
      self._get_bounds_from_state = _create_get_bounds_const(
          upper_bound_threshold, lower_bound_threshold)
      self._update_state = lambda _, __, ___: _empty_state()
      self._secagg_bitwidth = math.ceil(
          math.log2(upper_bound_threshold - lower_bound_threshold))

    # Configuration specific for aggregating floating point types.
    else:
      self._config_mode = _Config.FLOAT
      if _is_float(upper_bound_threshold):
        # Bounds specified as Python constants.
        if lower_bound_threshold is None:
          _check_positive(upper_bound_threshold)
          lower_bound_threshold = -1.0 * upper_bound_threshold
        else:
          _check_upper_larger_than_lower(upper_bound_threshold,
                                         lower_bound_threshold)
        self._get_bounds_from_state = _create_get_bounds_const(
            upper_bound_threshold, lower_bound_threshold)
        self._init_fn = _empty_state
        self._update_state = lambda _, __, ___: _empty_state()
      else:
        # Bounds specified as an EstimationProcess.
        _check_bound_process(upper_bound_threshold, 'upper_bound_threshold')
        if lower_bound_threshold is None:
          self._get_bounds_from_state = _create_get_bounds_single_process(
              upper_bound_threshold)
          self._init_fn = upper_bound_threshold.initialize
          self._update_state = _create_update_state_single_process(
              upper_bound_threshold)
        else:
          _check_bound_process(lower_bound_threshold, 'lower_bound_threshold')
          self._get_bounds_from_state = _create_get_bounds_two_processes(
              upper_bound_threshold, lower_bound_threshold)
          self._init_fn = _create_initial_state_two_processes(
              upper_bound_threshold, lower_bound_threshold)
          self._update_state = _create_update_state_two_processes(
              upper_bound_threshold, lower_bound_threshold)

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    self._check_value_type_compatible_with_config_mode(value_type)

    @computations.federated_computation(self._init_fn.type_signature.result,
                                        computation_types.FederatedType(
                                            value_type, placements.CLIENTS))
    def next_fn(state, value):
      # Server-side preparation.
      upper_bound, lower_bound = self._get_bounds_from_state(state)

      # Compute min and max *before* clipping and use it to update the state.
      value_max = intrinsics.federated_map(_reduce_nest_max, value)
      value_min = intrinsics.federated_map(_reduce_nest_min, value)
      new_state = self._update_state(state, value_min, value_max)

      # Clips value to [lower_bound, upper_bound] and securely sums it.
      summed_value = self._sum_securely(value, upper_bound, lower_bound)

      # TODO(b/163880757): pass upper_bound and lower_bound through clients.
      measurements = self._compute_measurements(upper_bound, lower_bound,
                                                value_max, value_min)
      return measured_process.MeasuredProcessOutput(new_state, summed_value,
                                                    measurements)

    return aggregation_process.AggregationProcess(self._init_fn, next_fn)

  def _compute_measurements(self, upper_bound, lower_bound, value_max,
                            value_min):
    """Creates measurements to be reported. All values are summed securely."""
    is_max_clipped = intrinsics.federated_map(
        computations.tf_computation(
            lambda bound, value: tf.cast(bound < value, COUNT_TF_TYPE)),
        (intrinsics.federated_broadcast(upper_bound), value_max))
    max_clipped_count = intrinsics.federated_secure_sum(
        is_max_clipped, bitwidth=1)
    is_min_clipped = intrinsics.federated_map(
        computations.tf_computation(
            lambda bound, value: tf.cast(bound > value, COUNT_TF_TYPE)),
        (intrinsics.federated_broadcast(lower_bound), value_min))
    min_clipped_count = intrinsics.federated_secure_sum(
        is_min_clipped, bitwidth=1)
    measurements = collections.OrderedDict(
        secure_upper_clipped_count=max_clipped_count,
        secure_lower_clipped_count=min_clipped_count,
        secure_upper_threshold=upper_bound,
        secure_lower_threshold=lower_bound)
    return intrinsics.federated_zip(measurements)

  def _sum_securely(self, value, upper_bound, lower_bound):
    """Securely sums `value` placed at CLIENTS."""
    if self._config_mode == _Config.INT:
      value = intrinsics.federated_map(
          _client_shift, (value, intrinsics.federated_broadcast(upper_bound),
                          intrinsics.federated_broadcast(lower_bound)))
      value = intrinsics.federated_secure_sum(value, self._secagg_bitwidth)
      num_summands = intrinsics.federated_sum(_client_one())
      value = intrinsics.federated_map(_server_shift,
                                       (value, lower_bound, num_summands))
      return value
    elif self._config_mode == _Config.FLOAT:
      return primitives.secure_quantized_sum(value, lower_bound, upper_bound)
    else:
      raise ValueError(f'Unexpected internal config type: {self._config_mode}')

  def _check_value_type_compatible_with_config_mode(self, value_type):
    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    if self._config_mode == _Config.INT:
      if not type_analysis.is_structure_of_integers(value_type):
        raise TypeError(
            f'The `SecureSumFactory` was configured to work with integer '
            f'dtypes. All values in provided `value_type` hence must be of '
            f'integer dtype. \nProvided value_type: {value_type}')
    elif self._config_mode == _Config.FLOAT:
      if not type_analysis.is_structure_of_floats(value_type):
        raise TypeError(
            f'The `SecureSumFactory` was configured to work with floating '
            f'point dtypes. All values in provided `value_type` hence must be '
            f'of floating point dtype. \nProvided value_type: {value_type}')
    else:
      raise ValueError(f'Unexpected internal config type: {self._config_mode}')


def _check_positive(value):
  if value <= 0:
    raise ValueError(
        f'If only `upper_bound_threshold` is specified as a Python constant, '
        f'it must be positive. Its negative will be used as a lower bound '
        f'which would be larger than the upper bound. \n'
        f'Provided `upper_bound_threshold`: {value}')


def _check_upper_larger_than_lower(upper_bound_threshold,
                                   lower_bound_threshold):
  if upper_bound_threshold <= lower_bound_threshold:
    raise ValueError(
        f'The provided `upper_bound_threshold` must be larger than the '
        f'provided `lower_bound_threshold`, but received:\n'
        f'`upper_bound_threshold`: {upper_bound_threshold}\n'
        f'`lower_bound_threshold`: {lower_bound_threshold}\n')


def _is_integer(value):
  is_py_int = isinstance(value, int)
  is_np_int = isinstance(value, np.ndarray) and bool(
      np.issubdtype(value.dtype, np.integer))
  return is_py_int or is_np_int


def _is_float(value):
  is_py_float = isinstance(value, float)
  is_np_float = isinstance(value, np.ndarray) and bool(
      np.issubdtype(value.dtype, np.floating))
  return is_py_float or is_np_float


@computations.tf_computation()
def _reduce_nest_max(value):
  max_list = tf.nest.map_structure(tf.reduce_max, tf.nest.flatten(value))
  return tf.reduce_max(tf.stack(max_list))


@computations.tf_computation()
def _reduce_nest_min(value):
  min_list = tf.nest.map_structure(tf.reduce_min, tf.nest.flatten(value))
  return tf.reduce_min(tf.stack(min_list))


@computations.tf_computation()
def _client_shift(value, upper_bound, lower_bound):
  return tf.nest.map_structure(
      lambda v: tf.clip_by_value(v, lower_bound, upper_bound) - lower_bound,
      value)


@computations.tf_computation()
def _server_shift(value, lower_bound, num_summands):
  return tf.nest.map_structure(
      lambda v: v + (lower_bound * tf.cast(num_summands, lower_bound.dtype)),
      value)


@computations.federated_computation()
def _empty_state():
  return intrinsics.federated_value((), placements.SERVER)


def _client_one():
  return intrinsics.federated_eval(
      computations.tf_computation(lambda: tf.constant(1, tf.int32)),
      placements.CLIENTS)


def _create_initial_state_two_processes(upper_bound_process,
                                        lower_bound_process):

  @computations.federated_computation()
  def initial_state():
    return intrinsics.federated_zip(
        (upper_bound_process.initialize(), lower_bound_process.initialize()))

  return initial_state


def _create_get_bounds_const(upper_bound, lower_bound):

  def get_bounds(state):
    del state  # Unused.
    return (intrinsics.federated_value(upper_bound, placements.SERVER),
            intrinsics.federated_value(lower_bound, placements.SERVER))

  return get_bounds


def _create_get_bounds_single_process(process):

  def get_bounds(state):
    upper_bound = process.report(state)
    lower_bound = intrinsics.federated_map(
        computations.tf_computation(lambda x: x * -1.0), upper_bound)
    return upper_bound, lower_bound

  return get_bounds


def _create_get_bounds_two_processes(upper_bound_process, lower_bound_process):

  def get_bounds(state):
    upper_bound = upper_bound_process.report(state[0])
    lower_bound = lower_bound_process.report(state[1])
    return upper_bound, lower_bound

  return get_bounds


def _create_update_state_single_process(process):

  def update_state(state, value_min, value_max):
    abs_max_fn = computations.tf_computation(
        lambda x, y: tf.maximum(tf.abs(x), tf.abs(y)))
    abs_value_max = intrinsics.federated_map(abs_max_fn, (value_min, value_max))
    return process.next(state, abs_value_max)

  return update_state


def _create_update_state_two_processes(upper_bound_process,
                                       lower_bound_process):

  def update_state(state, value_min, value_max):
    return intrinsics.federated_zip(
        (upper_bound_process.next(state[0], value_max),
         lower_bound_process.next(state[1], value_min)))

  return update_state
