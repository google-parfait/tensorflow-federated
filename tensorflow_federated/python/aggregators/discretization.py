# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A tff.aggregator for discretizing input values to the integer grid."""

import collections
import numbers

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

OUTPUT_TF_TYPE = tf.int32


class DiscretizationFactory(factory.UnweightedAggregationFactory):
  """Aggregation factory for discretizing of floating point tensors.

  The created `tff.templates.AggregationProcess` takes an input tensor structure
  and, for each tensor, scales and rounds the values to the integer grid. The
  scaling factor is the same for all component tensors, and the rounding scheme
  can be one of:
    1. Deterministic rounding: each scaled value is deterministically rounded
        to the nearest integer.
    2. Stochastic rounding: each scaled value is stochastically rounded to
        the neighbouring integers. For example, 42.3 has 0.7 probability to be
        rounded to 42 and 0.3 probability to 43.
    3. Conditionally stochastic rounding: Like stochastic rounding, but the
        rounding procedure is resampled if the norm of the rounded vector
        exceeds a pre-computed threshold determined by a constant in [0, 1)
        that controls the concentration inequality for the probabilistic norm
        bound after stochastic rounding. For more details, see Section 4.1 of
        https://arxiv.org/pdf/2102.06387.pdf.

  The structure of the input is kept, and all values of the component tensors
  are scaled, rounded, and casted to tf.int32.

  This aggregator is intended to be used as part of a uniform quantization
  procedure, which, on top of the discretization procedure carried out by this
  aggregator, involves value clipping and (possibly) value shifting.

  This factory only accepts `value_type` of either `tff.TensorType` or
  `tff.StructWithPythonType` and expects the dtype of component tensors to be
  all real floats, and it will otherwise raise an error.
  """

  def __init__(
      self,
      inner_agg_factory,
      scale_factor=1.0,
      stochastic=False,
      beta=0.0,
      prior_norm_bound=None,
  ):
    """Initializes the DiscretizationFactory.

    Args:
      inner_agg_factory: The inner `UnweightedAggregationFactory` to aggregate
        the values after the input values are discretized to the integer grid.
      scale_factor: A positive scaling constant to be applied to the input
        record before rounding to integers. Generally, the larger the factor,
        the smaller the errors from discretization.
      stochastic: A bool constant denoting whether to round stochastically to
        the nearest integer. Defaults to False (deterministic rounding).
      beta: A float constant in [0, 1) controlling the concentration inequality
        for the probabilistic norm bound after stochastic rounding. Ignored if
        `stochastic` is False. Intuitively, this term controls the bias-variance
        trade-off of stochastic rounding: a beta of 0 means the rounding is
        unbiased, but the resulting norm could be larger (thus larger added
        noise when combined with differential privacy); a larger beta means the
        vector norm grows less but at the expense of some bias. Defaults to 0
        (unconditional stochastic rounding).
      prior_norm_bound: A float constant denoting the global L2 norm bound of
        the inputs (e.g. as a result of global L2 clipping). This is useful when
        `prior_norm_bound` is larger than the input norm, in which case we can
        allow more leeway during conditional stochastic rounding (`beta` > 0).
        If set to None, no prior L2 norm bound is used. Ignored if `stochastic`
        is False or `beta` is 0.

    Raises:
      TypeError: If `inner_agg_factory` is not an instance of
          `tff.aggregators.UnweightedAggregationFactory`
      ValueError: If `scale_factor` is not a positive number.
      ValueError: If `stochastic` is not a boolean constant.
      ValueError: If `beta` is not in the range of [0, 1).
      ValueError: If `prior_norm_bound` is given but is not a positive number.
    """
    if not isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):
      raise TypeError(
          '`inner_agg_factory` must have type '
          'UnweightedAggregationFactory. '
          f'Found {type(inner_agg_factory)}.'
      )

    if not isinstance(scale_factor, numbers.Number) or scale_factor <= 0:
      raise ValueError(
          f'`scale_factor` should be a positive number. Found {scale_factor}.'
      )
    if not isinstance(stochastic, bool):
      raise ValueError(f'`stochastic` should be a boolean. Found {stochastic}.')
    if not isinstance(beta, numbers.Number) or not 0 <= beta < 1:
      raise ValueError(f'`beta` should be a number in [0, 1). Found {beta}.')
    if prior_norm_bound is not None and (
        not isinstance(prior_norm_bound, numbers.Number)
        or prior_norm_bound <= 0
    ):
      raise ValueError(
          'If specified, `prior_norm_bound` should be a positive '
          f'number. Found {prior_norm_bound}.'
      )

    self._scale_factor = float(scale_factor)
    self._stochastic = stochastic
    self._beta = float(beta)
    # Use value 0 to denote no prior norm bounds for easier typing.
    self._prior_norm_bound = prior_norm_bound or 0.0
    self._inner_agg_factory = inner_agg_factory

  def create(self, value_type):
    # Validate input args and value_type and parse out the TF dtypes.
    if isinstance(value_type, computation_types.TensorType):
      tf_dtype = value_type.dtype
    elif isinstance(
        value_type, computation_types.StructWithPythonType
    ) and type_analysis.is_structure_of_tensors(value_type):
      if self._prior_norm_bound:
        raise TypeError(
            'If `prior_norm_bound` is specified, `value_type` must '
            f'be `TensorType`. Found type: {repr(value_type)}.'
        )
      tf_dtype = type_conversions.structure_from_tensor_type_tree(
          lambda x: x.dtype, value_type
      )
    else:
      raise TypeError(
          'Expected `value_type` to be `TensorType` or '
          '`StructWithPythonType` containing only `TensorType`. '
          f'Found type: {repr(value_type)}'
      )

    # Check that all values are floats.
    if not type_analysis.is_structure_of_floats(value_type):
      raise TypeError(
          'Component dtypes of `value_type` must all be floats. '
          f'Found {repr(value_type)}.'
      )

    discretize_fn = _build_discretize_fn(
        value_type, self._stochastic, self._beta
    )

    @tensorflow_computation.tf_computation(
        discretize_fn.type_signature.result, tf.float32
    )
    def undiscretize_fn(value, scale_factor):
      return _undiscretize_struct(value, scale_factor, tf_dtype)

    inner_value_type = discretize_fn.type_signature.result
    inner_agg_process = self._inner_agg_factory.create(inner_value_type)

    @federated_computation.federated_computation()
    def init_fn():
      state = collections.OrderedDict(
          scale_factor=intrinsics.federated_value(
              self._scale_factor, placements.SERVER
          ),
          prior_norm_bound=intrinsics.federated_value(
              self._prior_norm_bound, placements.SERVER
          ),
          inner_agg_process=inner_agg_process.initialize(),
      )
      return intrinsics.federated_zip(state)

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.FederatedType(value_type, placements.CLIENTS),
    )
    def next_fn(state, value):
      server_scale_factor = state['scale_factor']
      client_scale_factor = intrinsics.federated_broadcast(server_scale_factor)
      server_prior_norm_bound = state['prior_norm_bound']
      prior_norm_bound = intrinsics.federated_broadcast(server_prior_norm_bound)

      discretized_value = intrinsics.federated_map(
          discretize_fn, (value, client_scale_factor, prior_norm_bound)
      )

      inner_state = state['inner_agg_process']
      inner_agg_output = inner_agg_process.next(inner_state, discretized_value)

      undiscretized_agg_value = intrinsics.federated_map(
          undiscretize_fn, (inner_agg_output.result, server_scale_factor)
      )

      new_state = collections.OrderedDict(
          scale_factor=server_scale_factor,
          prior_norm_bound=server_prior_norm_bound,
          inner_agg_process=inner_agg_output.state,
      )
      measurements = collections.OrderedDict(
          discretize=inner_agg_output.measurements
      )

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(new_state),
          result=undiscretized_agg_value,
          measurements=intrinsics.federated_zip(measurements),
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _build_discretize_fn(value_type, stochastic, beta):
  """Builds a `tff.tensorflow.computation` for discretization."""

  @tensorflow_computation.tf_computation(value_type, np.float32, np.float32)
  def discretize_fn(value, scale_factor, prior_norm_bound):
    return _discretize_struct(
        value, scale_factor, stochastic, beta, prior_norm_bound
    )

  return discretize_fn


def _discretize_struct(
    struct, scale_factor, stochastic, beta, prior_norm_bound
):
  """Scales and rounds each tensor of the structure to the integer grid."""

  def discretize_tensor(x):
    x = tf.cast(x, tf.float32)
    # Scale up the values.
    scaled_x = x * scale_factor
    scaled_bound = prior_norm_bound * scale_factor  # 0 if no prior bound.
    # Round to integer grid.
    if stochastic:
      discretized_x = _stochastic_rounding(
          scaled_x, scaled_bound, scale_factor, beta
      )
    else:
      discretized_x = tf.round(scaled_x)

    return tf.cast(discretized_x, OUTPUT_TF_TYPE)

  return tf.nest.map_structure(discretize_tensor, struct)


def _undiscretize_struct(struct, scale_factor, tf_dtype_struct):
  """Unscales the discretized structure and casts back to original dtypes."""

  def undiscretize_tensor(x, original_dtype):
    unscaled_x = tf.cast(x, tf.float32) / scale_factor
    return tf.cast(unscaled_x, original_dtype)

  return tf.nest.map_structure(undiscretize_tensor, struct, tf_dtype_struct)


def inflated_l2_norm_bound(l2_norm_bound, gamma, beta, dim):
  """Computes the probabilistic L2 norm bound after stochastic quantization.

  The procedure of stochastic quantization can increase the norm of the vector.
  This function computes a probabilistic L2 norm bound after the quantization
  procedure. See Theorem 1 and Sec 4.1 of https://arxiv.org/pdf/2102.06387.pdf
  for more details.

  Args:
    l2_norm_bound: The L2 norm bound of the vector whose coordinates are to be
      stochastically rounded to the specified rounding granularity gamma.
    gamma: The rounding granularity. A value of 1 is equivalent to rounding to
      the integer grid. Equivalent to the multiplicative inverse of the scale
      factor used during the quantization procedure.
    beta: A float constant in [0, 1]. See the initializer docstring of the
      aggregator for more details.
    dim: The dimension of the vector to be rounded.

  Returns:
    The inflated L2 norm bound after stochastically (possibly conditionally
    according to beta) rounding the coordinates to grid specified by the
    rounding granularity.
  """
  l2_norm_bound = tf.convert_to_tensor(l2_norm_bound)
  norm = tf.cast(l2_norm_bound, tf.float32)
  gamma = tf.cast(gamma, tf.float32)
  gamma_f64 = tf.cast(gamma, tf.float64)
  # Use float64 for `dim` as float32 can only represent ints up to 2^24 (~16M).
  dim = tf.cast(dim, tf.float64)
  beta = tf.cast(beta, tf.float32)

  gamma_sqrt_dim = tf.cast(tf.sqrt(dim) * gamma_f64, tf.float32)
  beta_term = tf.sqrt(2.0 * tf.math.log(1.0 / beta))

  bound_1 = norm + gamma_sqrt_dim
  squared_bound_2 = tf.square(norm) + 0.25 * tf.square(gamma_sqrt_dim)
  squared_bound_2 += beta_term * gamma * (norm + 0.5 * gamma_sqrt_dim)
  bound_2 = tf.sqrt(squared_bound_2)
  # Fall back to bound_1 if beta == 0.
  bound_2 = tf.cond(tf.equal(beta, 0), lambda: bound_1, lambda: bound_2)
  return tf.cast(tf.minimum(bound_1, bound_2), l2_norm_bound.dtype)


@tf.function
def _stochastic_rounding(
    x,
    l2_norm_bound=None,
    scale=1.0,
    beta=0.0,
    seed=None,
    max_tries=1000,
):
  """Stochastically (and conditionally) round a vector's values to integers.

  Args:
    x: The input vector whose coordinates are to be rounded to the integer grid.
    l2_norm_bound: The L2 norm bound on the input vector.
    scale: The scaling that has been applied to the input vector and the
      provided L2 norm bound. This determines the equivalent rounding
      granularity in the unscaled domain and the norm inflation from rounding.
      Defaults to 1.0 (rounding granularity is the integer grid).
    beta: A float constant in [0, 1). See the initializer docstring of the
      aggregator for more details. Defaults to 0.0 (unconditional rounding).
    seed: An integer seed for the randomness.
    max_tries: When beta > 0, the number of stochastic rounding tries before
      raising an exception.

  Returns:
    The input vector with coordinates stochastically (possibly conditionally
    according to beta) rounded to the integer grid.

  Raises:
    InvalidArgumentError: If stochastic rounding fails `max_tries` times.
  """
  if l2_norm_bound is None or l2_norm_bound == 0:
    l2_norm_bound = tf.norm(x, ord=2)

  if seed is None:
    seed = tf.timestamp() * 1e6
  seed = tf.cast(seed, tf.int64)

  # Compute norm inflation in the unscaled domain to improve stability.
  gamma = 1.0 / scale  # Equivalent rounding granularity.
  unscaled_bound = l2_norm_bound / scale
  threshold = inflated_l2_norm_bound(unscaled_bound, gamma, beta, tf.size(x))
  threshold *= scale

  floored_x = tf.floor(x)
  decimal_x = x - floored_x
  shape = tf.shape(x)

  @tf.function
  def try_rounding(tries):
    uniform = tf.random.stateless_uniform(
        shape, seed=[seed, tf.cast(tries, tf.int64)], maxval=1, dtype=x.dtype
    )
    return floored_x + tf.cast(uniform < decimal_x, x.dtype)

  tries = 1
  rounded_x = try_rounding(tries)
  while beta > 0 and tf.norm(rounded_x, ord=2) > threshold:
    message = (
        f'Stochastic rounding failed for {max_tries} tries. This can happen '
        'when beta > 0.0 and low-precision arithmetic is used. Try setting '
        'beta to 0.0. See the initializer docstring of the aggregator for '
        'more details.'
    )
    tf.debugging.assert_less_equal(tries, max_tries, message)
    tries += 1
    rounded_x = try_rounding(tries)

  return rounded_x
