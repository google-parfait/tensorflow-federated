# Copyright 2021, The TensorFlow Federated Authors.
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
"""The modular clipping factory for hierarchical histogram computation."""
import collections
from typing import Optional
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


# TODO(b/195870431): The below factory should be removed once secure sum with
# modular clipping is checked in in the future.
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
  constants within the value range of tf.int32.

  This factory only accepts `value_type` of `tff.TensorType` and expects the
  dtype of component tensors to be all integers.
  """

  def __init__(
      self,
      clip_range_lower: int,
      clip_range_upper: int,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None):
    """Initializes a `ModularClippingSumFactory` instance.

    Args:
      clip_range_lower: A Python integer specifying the inclusive lower modular
        clipping range.
      clip_range_upper: A Python integer specifying the exclusive upper modular
        clipping range.
      inner_agg_factory: (Optional) A `UnweightedAggregationFactory` specifying
        the value aggregation to be wrapped by modular clipping. Defaults to
        `tff.aggregators.SumFactory`.

    Raises:
      TypeError: If `clip_range_lower` or `clip_range_upper` are not integers.
      TypeError: If `inner_agg_factory` isn't an `UnweightedAggregationFactory`.
      ValueError: If `clip_range_lower` or `clip_range_upper` have invalid
        values.
    """
    _check_is_integer(clip_range_lower, 'clip_range_lower')
    _check_is_integer(clip_range_upper, 'clip_range_upper')
    _check_less_than_equal(clip_range_lower, clip_range_upper,
                           'clip_range_lower', 'clip_range_upper')
    _check_clip_range_overflow(clip_range_lower, clip_range_upper)
    if inner_agg_factory is None:
      inner_agg_factory = sum_factory.SumFactory()
    else:
      _check_is_unweighted_aggregation_factory(inner_agg_factory,
                                               'inner_agg_factory')

    if clip_range_lower > clip_range_upper:
      raise ValueError('`clip_range_lower` should not be larger than '
                       f'`clip_range_upper`, got {clip_range_lower} and '
                       f'{clip_range_upper}')

    self._clip_range_lower = clip_range_lower
    self._clip_range_upper = clip_range_upper
    self._inner_agg_factory = inner_agg_factory

  def create(self, value_type) -> aggregation_process.AggregationProcess:
    _check_is_tensor_type(value_type, 'value_type')
    _check_is_integer_struct(value_type, 'value_type')

    inner_agg_process = self._inner_agg_factory.create(value_type)
    init_fn = inner_agg_process.initialize
    next_fn = self._create_next_fn(inner_agg_process.next,
                                   init_fn.type_signature.result, value_type)
    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def _create_next_fn(self, inner_agg_next, state_type, value_type):

    modular_clip_by_value_fn = computations.tf_computation(
        _modular_clip_by_value)

    @computations.federated_computation(state_type,
                                        computation_types.at_clients(value_type)
                                       )
    def next_fn(state, value):
      clip_lower = intrinsics.federated_value(self._clip_range_lower,
                                              placements.SERVER)
      clip_upper = intrinsics.federated_value(self._clip_range_upper,
                                              placements.SERVER)

      # Modular clip values before aggregation.
      clipped_value = intrinsics.federated_map(
          modular_clip_by_value_fn,
          (value, intrinsics.federated_broadcast(clip_lower),
           intrinsics.federated_broadcast(clip_upper)))

      inner_agg_output = inner_agg_next(state, clipped_value)

      # Clip the aggregate to the same range again (not considering summands).
      clipped_agg_output_result = intrinsics.federated_map(
          modular_clip_by_value_fn,
          (inner_agg_output.result, clip_lower, clip_upper))

      measurements = collections.OrderedDict(
          modclip=inner_agg_output.measurements)

      return measured_process.MeasuredProcessOutput(
          state=inner_agg_output.state,
          result=clipped_agg_output_result,
          measurements=intrinsics.federated_zip(measurements))

    return next_fn


def _modular_clip_by_value(value, clip_range_lower, clip_range_upper):

  def mod_clip(v):
    width = clip_range_upper - clip_range_lower
    period = tf.cast(tf.floor(v / width - clip_range_lower / width), v.dtype)
    v_mod_clipped = v - period * width
    return v_mod_clipped

  return tf.nest.map_structure(mod_clip, value)


def _check_clip_range_overflow(clip_range_lower, clip_range_upper):
  if (clip_range_upper > tf.int32.max or clip_range_lower < tf.int32.min or
      clip_range_upper - clip_range_lower > tf.int32.max):
    raise ValueError('`clip_range_lower` and `clip_range_upper` should be '
                     'set such that the range of the modulus do not overflow '
                     f'tf.int32. Found clip_range_lower={clip_range_lower} '
                     f'and clip_range_upper={clip_range_upper} respectively.')


def _check_is_unweighted_aggregation_factory(value, label):
  if not isinstance(value, factory.UnweightedAggregationFactory):
    raise TypeError(f'`{label}` must have type '
                    '`UnweightedAggregationFactory`. '
                    f'Found {type(value)}.')


def _check_is_integer(value, label):
  if not isinstance(value, int):
    raise TypeError(f'`{label}` must be Python `int`. '
                    f' Found {repr(value)} with type {type(value)}.')


def _check_is_integer_struct(value_type, label):
  if not type_analysis.is_structure_of_integers(value_type):
    raise TypeError(f'Component dtypes of `{label}` must all be integers. '
                    f'Found {repr(value_type)}.')


def _check_is_tensor_type(value, label):
  if not value.is_tensor():
    raise TypeError(f'Expected `{label}` to be `TensorType`. '
                    f'Found type: {repr(value)}')


def _check_less_than_equal(lvalue, rvalue, llabel, rlabel):
  if lvalue > rvalue:
    raise ValueError(f'`{llabel}` should be no larger than '
                     f'`{rlabel}`. Found {lvalue} and '
                     f'{rvalue}.')
