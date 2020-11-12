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
"""Factory for clipping/zeroing of large values."""

import collections
from typing import Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

NORM_TF_TYPE = tf.float32
COUNT_TF_TYPE = tf.int32

_InnerFactoryType = Union[factory.UnweightedAggregationFactory,
                          factory.WeightedAggregationFactory]


def _constant_process(value):
  """Creates an `EstimationProcess` that reports a constant value."""
  init_fn = computations.federated_computation(
      lambda: intrinsics.federated_value((), placements.SERVER))
  next_fn = computations.federated_computation(
      lambda state, value: state, init_fn.type_signature.result,
      computation_types.at_clients(NORM_TF_TYPE))
  report_fn = computations.federated_computation(
      lambda state: intrinsics.federated_value(value, placements.SERVER),
      init_fn.type_signature.result)
  return estimation_process.EstimationProcess(init_fn, next_fn, report_fn)


def _contains_non_float_dtype(type_spec):
  return type_spec.is_tensor() and not type_spec.dtype.is_floating


def _check_norm_process(norm_process: estimation_process.EstimationProcess,
                        name: str):
  """Checks type properties for norm_process.

  The norm_process must be an `EstimationProcess` with `next` function of type
  signature (<state@SERVER, NORM_TF_TYPE@CLIENTS> -> state@SERVER), and `report`
  with type signature (state@SERVER -> NORM_TF_TYPE@SERVER).

  Args:
    norm_process: A process to check.
    name: A string name for formatting error messages.
  """

  py_typecheck.check_type(norm_process, estimation_process.EstimationProcess)

  next_parameter_type = norm_process.next.type_signature.parameter
  if not next_parameter_type.is_struct() or len(next_parameter_type) != 2:
    raise TypeError(f'`{name}.next` must take two arguments but found:\n'
                    f'{next_parameter_type}')

  norm_type_at_clients = computation_types.at_clients(NORM_TF_TYPE)
  if not next_parameter_type[1].is_assignable_from(norm_type_at_clients):
    raise TypeError(
        f'Second argument of `{name}.next` must be assignable from '
        f'{norm_type_at_clients} but found {next_parameter_type[1]}')

  next_result_type = norm_process.next.type_signature.result
  if not norm_process.state_type.is_assignable_from(next_result_type):
    raise TypeError(f'Result type of `{name}.next` must consist of state only '
                    f'but found result type:\n{next_result_type}\n'
                    f'while the state type is:\n{norm_process.state_type}')

  result_type = norm_process.report.type_signature.result
  norm_type_at_server = computation_types.at_server(NORM_TF_TYPE)
  if not norm_type_at_server.is_assignable_from(result_type):
    raise TypeError(f'Result type of `{name}.report` must be assignable to '
                    f'{norm_type_at_server} but found {result_type}.')


class ClippingFactory(factory.UnweightedAggregationFactory,
                      factory.WeightedAggregationFactory):
  """`AggregationProcess` factory for clipping large values.

  The created `tff.templates.AggregationProcess` projects the values onto an
  L2 ball (also referred to as "clipping") with norm determined by the provided
  `clipping_norm`, before aggregating the values as specified by
  `inner_agg_factory`.

  The provided `clipping_norm` can either be a constant (for fixed norm), or an
  instance of `tff.templates.EstimationProcess` (for adaptive norm). If it is an
  estimation process, the value returned by its `report` method will be used as
  the clipping norm. Its `next` method needs to accept a scalar float32 at
  clients, corresponding to the norm of value being aggregated. The process can
  thus adaptively determine the clipping norm based on the set of aggregated
  values. For example if a `tff.aggregators.PrivateQuantileEstimationProcess` is
  used, the clip will be an estimate of a quantile of the norms of the values
  being aggregated.
  """

  def __init__(self, clipping_norm: Union[float,
                                          estimation_process.EstimationProcess],
               inner_agg_factory: _InnerFactoryType):
    """Initializes `ClippingFactory`.

    Args:
      clipping_norm: Either a float (for fixed norm) or an `EstimationProcess`
        (for adaptive norm) that specifies the norm over which the values should
        be clipped.
      inner_agg_factory: A factory specifying the type of aggregation to be done
        after clipping.
    """
    py_typecheck.check_type(inner_agg_factory, _InnerFactoryType.__args__)
    self._inner_agg_factory = inner_agg_factory

    py_typecheck.check_type(clipping_norm,
                            (float, estimation_process.EstimationProcess))
    if isinstance(clipping_norm, float):
      clipping_norm = _constant_process(clipping_norm)
    _check_norm_process(clipping_norm, 'clipping_norm')
    self._clipping_norm_process = clipping_norm

    # The aggregation factory that will be used to count the number of clipped
    # values at each iteration. For now we are just creating it here, but soon
    # we will make this customizable to allow DP measurements.
    self._clipped_count_agg_factory = sum_factory.SumFactory()

  def create_unweighted(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    py_typecheck.check_type(value_type, factory.ValueType.__args__)
    if type_analysis.contains(value_type, predicate=_contains_non_float_dtype):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create_unweighted(value_type)
    clipped_count_agg_process = (
        self._clipped_count_agg_factory.create_unweighted(
            computation_types.to_type(COUNT_TF_TYPE)))

    init_fn = self._create_init_fn(inner_agg_process.initialize,
                                   clipped_count_agg_process.initialize)
    next_fn = self._create_next_fn(inner_agg_process.next,
                                   clipped_count_agg_process.next,
                                   init_fn.type_signature.result)

    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def create_weighted(
      self, value_type: factory.ValueType,
      weight_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    py_typecheck.check_type(value_type, factory.ValueType.__args__)
    py_typecheck.check_type(weight_type, factory.ValueType.__args__)
    if type_analysis.contains(value_type, predicate=_contains_non_float_dtype):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create_weighted(
        value_type, weight_type)
    clipped_count_agg_process = (
        self._clipped_count_agg_factory.create_unweighted(
            computation_types.to_type(COUNT_TF_TYPE)))

    init_fn = self._create_init_fn(inner_agg_process.initialize,
                                   clipped_count_agg_process.initialize)
    next_fn = self._create_next_fn(inner_agg_process.next,
                                   clipped_count_agg_process.next,
                                   init_fn.type_signature.result)

    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def _create_init_fn(self, inner_agg_initialize, clipped_count_agg_initialize):

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip(
          collections.OrderedDict(
              clipping_norm=self._clipping_norm_process.initialize(),
              inner_agg=inner_agg_initialize(),
              clipped_count_agg=clipped_count_agg_initialize()))

    return init_fn

  def _create_next_fn(self, inner_agg_next, clipped_count_agg_next, state_type):

    @computations.tf_computation(
        inner_agg_next.type_signature.parameter[1].member, NORM_TF_TYPE)
    def clip_fn(value, clipping_norm):
      clipped_value_as_list, global_norm = tf.clip_by_global_norm(
          tf.nest.flatten(value), clipping_norm)
      clipped_value = tf.nest.pack_sequence_as(value, clipped_value_as_list)
      was_clipped = tf.cast((global_norm > clipping_norm), COUNT_TF_TYPE)
      return clipped_value, global_norm, was_clipped

    def next_fn_impl(state, value, weight=None):
      clipping_norm_state, agg_state, clipped_count_state = state

      clipping_norm = self._clipping_norm_process.report(clipping_norm_state)

      clipped_value, global_norm, was_clipped = intrinsics.federated_map(
          clip_fn, (value, intrinsics.federated_broadcast(clipping_norm)))

      new_clipping_norm_state = self._clipping_norm_process.next(
          clipping_norm_state, global_norm)

      if weight is None:
        agg_output = inner_agg_next(agg_state, clipped_value)
      else:
        agg_output = inner_agg_next(agg_state, clipped_value, weight)

      clipped_count_output = clipped_count_agg_next(clipped_count_state,
                                                    was_clipped)

      new_state = collections.OrderedDict(
          clipping_norm=new_clipping_norm_state,
          inner_agg=agg_output.state,
          clipped_count_agg=clipped_count_output.state)
      measurements = collections.OrderedDict(
          agg_process=agg_output.measurements,
          clipping_norm=clipping_norm,
          clipped_count=clipped_count_output.result)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(new_state),
          result=agg_output.result,
          measurements=intrinsics.federated_zip(measurements))

    if len(inner_agg_next.type_signature.parameter) == 2:

      @computations.federated_computation(
          state_type, inner_agg_next.type_signature.parameter[1])
      def next_fn(state, value):
        return next_fn_impl(state, value)
    else:
      assert len(inner_agg_next.type_signature.parameter) == 3

      @computations.federated_computation(
          state_type, inner_agg_next.type_signature.parameter[1],
          inner_agg_next.type_signature.parameter[2])
      def next_fn(state, value, weight):
        return next_fn_impl(state, value, weight)

    return next_fn


class ZeroingFactory(factory.UnweightedAggregationFactory,
                     factory.WeightedAggregationFactory):
  """`AggregationProcess` factory for zeroing large values.

  The created `tff.templates.AggregationProcess` zeroes out any values whose
  norm is greater than that determined by the provided `zeroing_norm`, before
  aggregating the values as specified by `inner_agg_factory`.

  The provided `zeroing_norm` can either be a constant (for fixed norm), or an
  instance of `tff.templates.EstimationProcess` (for adaptive norm). If it is an
  estimation process, the value returned by its `report` method will be used as
  the zeroing norm. Its `next` method needs to accept a scalar float32 at
  clients, corresponding to the norm of value being aggregated. The process can
  thus adaptively determine the zeroing norm based on the set of aggregated
  values. For example if a `tff.aggregators.PrivateQuantileEstimationProcess` is
  used, the zeroing norm will be an estimate of a quantile of the norms of the
  values being aggregated.
  """

  def __init__(self,
               zeroing_norm: Union[float, estimation_process.EstimationProcess],
               inner_agg_factory: _InnerFactoryType,
               norm_order: float = 2.0):
    """Initializes a ZeroingFactory.

    Args:
      zeroing_norm: Either a float (for fixed norm) or an `EstimationProcess`
        (for adaptive norm) that specifies the norm over which values should be
        zeroed. If an `EstimationProcess` is passed, value norms will be passed
        to the process and its `report` function will be used as the zeroing
        norm.
      inner_agg_factory: A factory specifying the type of aggregation to be done
        after zeroing.
      norm_order: A float for the order of the norm. Must be 1, 2, or np.inf.
    """
    py_typecheck.check_type(inner_agg_factory, _InnerFactoryType.__args__)
    self._inner_agg_factory = inner_agg_factory

    py_typecheck.check_type(zeroing_norm,
                            (float, estimation_process.EstimationProcess))
    if isinstance(zeroing_norm, float):
      zeroing_norm = _constant_process(zeroing_norm)
    _check_norm_process(zeroing_norm, 'zeroing_norm')
    self._zeroing_norm_process = zeroing_norm

    py_typecheck.check_type(norm_order, float)
    if norm_order not in [1.0, 2.0, np.inf]:
      raise ValueError('norm_order must be 1.0, 2.0 or np.inf.')
    self._norm_order = norm_order

    # The aggregation factory that will be used to count the number of zeroed
    # values at each iteration. For now we are just creating it here, but soon
    # we will make this customizable to allow DP measurements.
    self._zeroed_count_agg_factory = sum_factory.SumFactory()

  def create_unweighted(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    py_typecheck.check_type(value_type, factory.ValueType.__args__)
    # This could perhaps be relaxed if we want to zero out ints for example.
    if type_analysis.contains(value_type, predicate=_contains_non_float_dtype):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create_unweighted(value_type)
    zeroed_count_agg_process = (
        self._zeroed_count_agg_factory.create_unweighted(
            computation_types.to_type(COUNT_TF_TYPE)))

    init_fn = self._create_init_fn(inner_agg_process.initialize,
                                   zeroed_count_agg_process.initialize)
    next_fn = self._create_next_fn(inner_agg_process.next,
                                   zeroed_count_agg_process.next,
                                   init_fn.type_signature.result)

    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def create_weighted(
      self, value_type: factory.ValueType,
      weight_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    py_typecheck.check_type(value_type, factory.ValueType.__args__)
    py_typecheck.check_type(weight_type, factory.ValueType.__args__)
    # This could perhaps be relaxed if we want to zero out ints for example.
    if type_analysis.contains(value_type, predicate=_contains_non_float_dtype):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create_weighted(
        value_type, weight_type)
    zeroed_count_agg_process = (
        self._zeroed_count_agg_factory.create_unweighted(
            computation_types.to_type(COUNT_TF_TYPE)))

    init_fn = self._create_init_fn(inner_agg_process.initialize,
                                   zeroed_count_agg_process.initialize)
    next_fn = self._create_next_fn(inner_agg_process.next,
                                   zeroed_count_agg_process.next,
                                   init_fn.type_signature.result)

    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def _create_init_fn(self, inner_agg_initialize, zeroed_count_agg_initialize):

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip(
          collections.OrderedDict(
              zeroing_norm=self._zeroing_norm_process.initialize(),
              inner_agg=inner_agg_initialize(),
              zeroed_count_agg=zeroed_count_agg_initialize()))

    return init_fn

  def _create_next_fn(self, inner_agg_next, zeroed_count_agg_next, state_type):

    @computations.tf_computation(
        inner_agg_next.type_signature.parameter[1].member, NORM_TF_TYPE)
    def zero_fn(value, zeroing_norm):
      if self._norm_order == 1.0:
        norm = _global_l1_norm(value)
      elif self._norm_order == 2.0:
        norm = tf.linalg.global_norm(tf.nest.flatten(value))
      else:
        assert self._norm_order is np.inf
        norm = _global_inf_norm(value)
      should_zero = (norm > zeroing_norm)
      zeroed_value = tf.cond(
          should_zero, lambda: tf.nest.map_structure(tf.zeros_like, value),
          lambda: value)
      was_zeroed = tf.cast(should_zero, COUNT_TF_TYPE)
      return zeroed_value, norm, was_zeroed

    def next_fn_impl(state, value, weight=None):
      zeroing_norm_state, agg_state, zeroed_count_state = state

      zeroing_norm = self._zeroing_norm_process.report(zeroing_norm_state)

      zeroed_value, norm, was_zeroed = intrinsics.federated_map(
          zero_fn, (value, intrinsics.federated_broadcast(zeroing_norm)))

      new_zeroing_norm_state = self._zeroing_norm_process.next(
          zeroing_norm_state, norm)

      if weight is None:
        agg_output = inner_agg_next(agg_state, zeroed_value)
      else:
        agg_output = inner_agg_next(agg_state, zeroed_value, weight)

      zeroed_count_output = zeroed_count_agg_next(zeroed_count_state,
                                                  was_zeroed)

      new_state = collections.OrderedDict(
          zeroing_norm=new_zeroing_norm_state,
          inner_agg=agg_output.state,
          zeroed_count_agg=zeroed_count_output.state)
      measurements = collections.OrderedDict(
          agg_process=agg_output.measurements,
          zeroing_norm=zeroing_norm,
          zeroed_count=zeroed_count_output.result)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(new_state),
          result=agg_output.result,
          measurements=intrinsics.federated_zip(measurements))

    if len(inner_agg_next.type_signature.parameter) == 2:

      @computations.federated_computation(
          state_type, inner_agg_next.type_signature.parameter[1])
      def next_fn(state, value):
        return next_fn_impl(state, value)

    else:
      assert len(inner_agg_next.type_signature.parameter) == 3

      @computations.federated_computation(
          state_type, inner_agg_next.type_signature.parameter[1],
          inner_agg_next.type_signature.parameter[2])
      def next_fn(state, value, weight):
        return next_fn_impl(state, value, weight)

    return next_fn


def _global_inf_norm(l):
  norms = [tf.reduce_max(tf.abs(a)) for a in tf.nest.flatten(l)]
  return tf.reduce_max(tf.stack(norms))


def _global_l1_norm(l):
  norms = [tf.reduce_sum(tf.abs(a)) for a in tf.nest.flatten(l)]
  return tf.reduce_sum(tf.stack(norms))
