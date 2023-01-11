# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Factory for per-entry modular clipping before and after aggregation."""

import collections
import math
from typing import Optional
import warnings

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


# TODO(b/195870431): The below factory will be removed in the near future.
class ModularClippingSumFactory(factory.UnweightedAggregationFactory):
  """An `UnweightedAggregationFactory` for element-wise modular clipping.

  The created `tff.templates.AggregationProcess` does per-entry modular clipping
  on inputs with an exclusive upper bound: [clip_range_lower, clip_range_upper).
  For example:
      Input  = [20, 5, -15, 10], clip_range_lower=-5, clip_range_upper=10;
      Output = [5,  5,   0, -5].

  Note that the same modular clipping range is applied both before and after
  the aggregation. While the clipping logic applies to both floating and integer
  values, this factory is only intended to support integer values.

  The provided `clip_range_lower` and `clip_range_upper` should be integer
  constants within the value range of tf.int32, though we may extend them to
  `tff.templates.EstimationProcess` in the future for adaptive clipping range.

  This factory only accepts `value_type` of either `tff.TensorType` or
  `tff.StructType` and expects the dtype of component tensors to be all
  integers,
  and it will otherwise raise an error.

  This factory may optionally surface an estimated standard deviation of the
  aggregated and modular-clipped values on `tff.SERVER` as a measurement via the
  `estimate_stddev` kwarg; the estimation procedure assumes that the elements
  in the aggregate structure/tensor are approximately normally distributed
  if the modular clipping was not performed. This metric can be useful for
  downstream applications to determine the modular clip range and/or to control
  the dynamic value range of the inputs. The estimation procedure is based on
  the following paper: https://arxiv.org/pdf/1912.00131.pdf.
  """

  def __init__(
      self,
      clip_range_lower: int,
      clip_range_upper: int,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      estimate_stddev: Optional[bool] = False,
  ):
    """Initializes a `ModularClippingSumFactory` instance.

    Args:
      clip_range_lower: A Python integer specifying the inclusive lower modular
        clipping range.
      clip_range_upper: A Python integer specifying the exclusive upper modular
        clipping range.
      inner_agg_factory: (Optional) A `UnweightedAggregationFactory` specifying
        the value aggregation to be wrapped by modular clipping. Defaults to
        `tff.aggregators.SumFactory`.
      estimate_stddev: (Optional) Whether to report the estimated standard
        deviation of the aggregated and modular-clipped values in the
        measurements. The estimation procedure assumes that the input client
        values (and thus the aggregate) are (approximately) normally distributed
        and centered at the midpoint of `clip_lower` and `clip_upper`. Defaults
        to `False`.

    Raises:
      TypeError: If `clip_range_lower` or `clip_range_upper` are not integers.
      TypeError: If `inner_agg_factory` isn't an `UnweightedAggregationFactory`.
      TypeError: If `estimate_stddev` is not a bool.
      ValueError: If `clip_range_lower` or `clip_range_upper` have invalid
        values.
    """
    if inner_agg_factory is None:
      inner_agg_factory = sum_factory.SumFactory()
    elif not isinstance(
        inner_agg_factory, factory.UnweightedAggregationFactory
    ):
      raise TypeError(
          '`inner_agg_factory` must have type '
          '`UnweightedAggregationFactory`. '
          f'Found {type(inner_agg_factory)}.'
      )

    if not (
        isinstance(clip_range_lower, int) and isinstance(clip_range_upper, int)
    ):
      raise TypeError(
          '`clip_range_lower` and `clip_range_upper` must be '
          f'Python `int`; got {repr(clip_range_lower)} with type '
          f'{type(clip_range_lower)} and {repr(clip_range_upper)} '
          f'with type {type(clip_range_upper)}, respectively.'
      )

    if clip_range_lower > clip_range_upper:
      raise ValueError(
          '`clip_range_lower` should not be larger than '
          f'`clip_range_upper`, got {clip_range_lower} and '
          f'{clip_range_upper}'
      )

    if (
        clip_range_upper > tf.int32.max
        or clip_range_lower < tf.int32.min
        or clip_range_upper - clip_range_lower > tf.int32.max
    ):
      raise ValueError(
          '`clip_range_lower` and `clip_range_upper` should be '
          'set such that the range of the modulus do not overflow '
          f'tf.int32. Found clip_range_lower={clip_range_lower} '
          f'and clip_range_upper={clip_range_upper} respectively.'
      )

    if not isinstance(estimate_stddev, bool):
      raise TypeError(
          f'{estimate_stddev} must be a bool. Found {repr(estimate_stddev)}.'
      )

    self._clip_range_lower = clip_range_lower
    self._clip_range_upper = clip_range_upper
    self._inner_agg_factory = inner_agg_factory
    self._estimate_stddev = estimate_stddev

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    # Checks value_type and compute client data dimension.
    if value_type.is_struct() and type_analysis.is_structure_of_tensors(
        value_type
    ):
      num_elements_struct = type_conversions.structure_from_tensor_type_tree(
          lambda x: x.shape.num_elements(), value_type
      )
      client_dim = sum(tf.nest.flatten(num_elements_struct))
    elif value_type.is_tensor():
      client_dim = value_type.shape.num_elements()
    else:
      raise TypeError(
          'Expected `value_type` to be `TensorType` or '
          '`StructType` containing only `TensorType`. '
          f'Found type: {repr(value_type)}'
      )
    # Checks that all values are integers.
    if not type_analysis.is_structure_of_integers(value_type):
      raise TypeError(
          'Component dtypes of `value_type` must all be integers. '
          f'Found {repr(value_type)}.'
      )
    # Checks that we have enough elements to estimate standard deviation.
    if self._estimate_stddev:
      if client_dim <= 1:
        raise ValueError(
            'The stddev estimation procedure expects more than '
            '1 element from `value_type`. Found `value_type` of '
            f'{value_type} with {client_dim} elements.'
        )
      elif client_dim <= 100:
        warnings.warn(
            f'`value_type` has only {client_dim} elements. The '
            'estimated standard deviation may be noisy. Consider '
            'setting `estimate_stddev` to True only if the input '
            'tensor/structure have more than 100 elements.'
        )

    inner_agg_process = self._inner_agg_factory.create(value_type)
    init_fn = inner_agg_process.initialize
    next_fn = self._create_next_fn(
        inner_agg_process.next, init_fn.type_signature.result, value_type
    )
    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def _create_next_fn(self, inner_agg_next, state_type, value_type):
    modular_clip_by_value_fn = tensorflow_computation.tf_computation(
        modular_clip_by_value
    )
    estimator_fn = tensorflow_computation.tf_computation(
        estimate_wrapped_gaussian_stddev
    )

    @federated_computation.federated_computation(
        state_type, computation_types.at_clients(value_type)
    )
    def next_fn(state, value):
      clip_lower = intrinsics.federated_value(
          self._clip_range_lower, placements.SERVER
      )
      clip_upper = intrinsics.federated_value(
          self._clip_range_upper, placements.SERVER
      )

      # Modular clip values before aggregation.
      clipped_value = intrinsics.federated_map(
          modular_clip_by_value_fn,
          (
              value,
              intrinsics.federated_broadcast(clip_lower),
              intrinsics.federated_broadcast(clip_upper),
          ),
      )

      inner_agg_output = inner_agg_next(state, clipped_value)

      # Clip the aggregate to the same range again (not considering summands).
      clipped_agg_output_result = intrinsics.federated_map(
          modular_clip_by_value_fn,
          (inner_agg_output.result, clip_lower, clip_upper),
      )

      measurements = collections.OrderedDict(
          modclip=inner_agg_output.measurements
      )

      if self._estimate_stddev:
        estimate = intrinsics.federated_map(
            estimator_fn, (clipped_agg_output_result, clip_lower, clip_upper)
        )
        measurements['estimated_stddev'] = estimate

      return measured_process.MeasuredProcessOutput(
          state=inner_agg_output.state,
          result=clipped_agg_output_result,
          measurements=intrinsics.federated_zip(measurements),
      )

    return next_fn


def modular_clip_by_value(value, clip_range_lower, clip_range_upper):
  def mod_clip(v):
    width = clip_range_upper - clip_range_lower
    period = tf.cast(tf.floor(v / width - clip_range_lower / width), v.dtype)
    v_mod_clipped = v - period * width
    return v_mod_clipped

  return tf.nest.map_structure(mod_clip, value)


def estimate_wrapped_gaussian_stddev(values, clip_lower, clip_upper):
  """Estimate the stddev of values assuming a wrapped normal distribution.

  This function takes an input tensor `values` and estimates the sample standard
  deviation of its values with the following assumptions:
    1. The values are distributed according to a wrapped Gaussian distribution
       bounded by `clip_lower` and `clip_upper`.
       https://en.wikipedia.org/wiki/Wrapped_normal_distribution.
    2. The values are centered at the midpoint of `clip_lower` and `clip_upper`.

  The estimation procedure is based on a maximum likelihood variant of the
  method described in https://arxiv.org/pdf/1912.00131.pdf.

  Args:
    values: The input tensor whose values are assumed to be distributed
      according to a wrapped normal distribution.
    clip_lower: The lower bound of the values.
    clip_upper: The upper bound of the values.

  Returns:
    The estimated standard deviation.
  """
  # Treat any input tensor structures as a single rank-1 vector.
  vector_list = [tf.reshape(x, [-1]) for x in tf.nest.flatten(values)]
  samples = tf.concat(vector_list, axis=0)
  tau = math.pi * 2
  samples = tf.cast(samples, tf.float64)
  width = tf.cast(clip_upper - clip_lower, tf.float64)
  normalized = tf.cast(samples / width * tau, tf.float32)
  circular = tf.exp(tf.complex(real=0.0, imag=normalized))
  r_stat = tf.abs(tf.reduce_mean(circular))
  normalized_stddev = tf.sqrt(-2.0 * tf.math.log(r_stat))
  stddev = normalized_stddev / tau * tf.cast(width, tf.float32)
  return stddev
