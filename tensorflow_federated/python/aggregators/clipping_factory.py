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

from typing import Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

NORM_TYPE = tf.float32


def _constant_process(value):
  """Creates an `EstimationProcess` that returns a constant value."""
  init_fn = computations.federated_computation(
      lambda: intrinsics.federated_value((), placements.SERVER))
  next_fn = computations.federated_computation(
      lambda state, value: state, init_fn.type_signature.result,
      computation_types.at_clients(NORM_TYPE))
  report_fn = computations.federated_computation(
      lambda s: intrinsics.federated_value(value, placements.SERVER),
      init_fn.type_signature.result)
  return estimation_process.EstimationProcess(init_fn, next_fn, report_fn)


def _check_norm_process(norm_process: estimation_process.EstimationProcess,
                        name: str):
  """Checks type properties for norm_process.

  The norm_process must be an `EstimationProcess` with `next` function of type
  signature (<state@SERVER, NORM_TYPE@CLIENTS> -> state@SERVER), and `report`
  with type signature (state@SERVER -> NORM_TYPE@SERVER).

  Args:
    norm_process: A process to check.
    name: A string name for formatting error messages.
  """

  py_typecheck.check_type(norm_process, estimation_process.EstimationProcess)

  next_parameter_type = norm_process.next.type_signature.parameter
  if not next_parameter_type.is_struct() or len(next_parameter_type) != 2:
    raise TypeError(f'`{name}.next` must take two arguments but found:\n'
                    f'{next_parameter_type}')

  norm_type_at_clients = computation_types.at_clients(NORM_TYPE)
  if not next_parameter_type[1].is_assignable_from(norm_type_at_clients):
    raise TypeError(f'Second argument of `{name}.next` must be assignable from '
                    f'NORM_TYPE@CLIENTS but found {next_parameter_type[1]}')

  next_result_type = norm_process.next.type_signature.result
  if not norm_process.state_type.is_assignable_from(next_result_type):
    raise TypeError(f'Result type of `{name}.next` must consist of state only '
                    f'but found result type:\n{next_result_type}\n'
                    f'while the state type is:\n{norm_process.state_type}')

  result_type = norm_process.report.type_signature.result
  norm_type_at_server = computation_types.at_server(NORM_TYPE)
  if not norm_type_at_server.is_assignable_from(result_type):
    raise TypeError(f'Result type of `{name}.report` must be assignable to '
                    f'NORM_TYPE@SERVER but found {result_type}.')


class ClippingFactory(factory.AggregationProcessFactory):
  """`AggregationProcessFactory` for clipping large values.

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
               inner_agg_factory: factory.AggregationProcessFactory):
    """Initializes `ClippingFactory`.

    Args:
      clipping_norm: Either a float (for fixed norm) or an `EstimationProcess`
        (for adaptive norm) that specifies the norm over which the values should
        be clipped.
      inner_agg_factory: A factory specifying the type of aggregation to be done
        after clipping.
    """
    py_typecheck.check_type(inner_agg_factory,
                            factory.AggregationProcessFactory)
    self._inner_agg_factory = inner_agg_factory

    py_typecheck.check_type(clipping_norm,
                            (float, estimation_process.EstimationProcess))
    if isinstance(clipping_norm, float):
      clipping_norm = _constant_process(clipping_norm)
    _check_norm_process(clipping_norm, 'clipping_norm')
    self._clipping_norm_process = clipping_norm

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:

    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    if not all([t.dtype.is_floating for t in structure.flatten(value_type)]):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create(value_type)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip((self._clipping_norm_process.initialize(),
                                       inner_agg_process.initialize()))

    @computations.tf_computation(value_type, NORM_TYPE)
    def clip(value, clipping_norm):
      clipped_value_as_list, global_norm = tf.clip_by_global_norm(
          tf.nest.flatten(value), clipping_norm)
      clipped_value = tf.nest.pack_sequence_as(value, clipped_value_as_list)
      return clipped_value, global_norm

    @computations.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type),
        computation_types.at_clients(tf.float32))
    def next_fn(state, value, weight):
      clipping_norm_state, agg_state = state

      clipping_norm = self._clipping_norm_process.report(clipping_norm_state)

      clipped_value, global_norm = intrinsics.federated_map(
          clip, (value, intrinsics.federated_broadcast(clipping_norm)))

      agg_output = inner_agg_process.next(agg_state, clipped_value, weight)
      new_clipping_norm_state = self._clipping_norm_process.next(
          clipping_norm_state, global_norm)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(
              (new_clipping_norm_state, agg_output.state)),
          result=agg_output.result,
          measurements=agg_output.measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


class ZeroingFactory(factory.AggregationProcessFactory):
  """`AggregationProcessFactory` for zeroing large values.

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
               inner_agg_factory: factory.AggregationProcessFactory,
               norm_order: float = np.inf):
    """Initializes a ZeroingFactory.

    Args:
      zeroing_norm: Either a float (for fixed norm) or an `EstimationProcess`
        (for adaptive norm) that specifies the norm over which values should be
        zeroed. If an `EstimationProcess` is passed, value norms will be passed
        to the process and its `report` function will be used as the zeroing
        norm.
      inner_agg_factory: A factory specifying the type of aggregation to be done
        after zeroing.
      norm_order: A float for the order of the norm. For example, may be 1, 2,
        or np.inf.
    """
    py_typecheck.check_type(inner_agg_factory,
                            factory.AggregationProcessFactory)
    self._inner_agg_factory = inner_agg_factory

    py_typecheck.check_type(zeroing_norm,
                            (float, estimation_process.EstimationProcess))
    if isinstance(zeroing_norm, float):
      zeroing_norm = _constant_process(zeroing_norm)
    _check_norm_process(zeroing_norm, 'zeroing_norm')
    self._zeroing_norm_process = zeroing_norm

    py_typecheck.check_type(norm_order, float)
    self._norm_order = norm_order

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:

    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    # This could perhaps be relaxed if we want to zero out ints for example.
    if not all([t.dtype.is_floating for t in structure.flatten(value_type)]):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create(value_type)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip((self._zeroing_norm_process.initialize(),
                                       inner_agg_process.initialize()))

    @computations.tf_computation(value_type, NORM_TYPE)
    def zero(value, zeroing_norm):
      # Concat to take norm will introduce memory overhead. Consider optimizing.
      vectors = tf.nest.map_structure(lambda v: tf.reshape(v, [-1]), value)
      norm = tf.norm(
          tf.concat(tf.nest.flatten(vectors), axis=0), ord=self._norm_order)
      zeroed = _zero_over(value, norm, zeroing_norm)
      return zeroed, norm

    @computations.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type),
        computation_types.at_clients(tf.float32))
    def next_fn(state, value, weight):
      zeroing_norm_state, agg_state = state

      zeroing_norm = self._zeroing_norm_process.report(zeroing_norm_state)

      zeroed, norm = intrinsics.federated_map(
          zero, (value, intrinsics.federated_broadcast(zeroing_norm)))

      agg_output = inner_agg_process.next(agg_state, zeroed, weight)
      new_zeroing_norm_state = self._zeroing_norm_process.next(
          zeroing_norm_state, norm)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(
              (new_zeroing_norm_state, agg_output.state)),
          result=agg_output.result,
          measurements=agg_output.measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


class ZeroingClippingFactory(factory.AggregationProcessFactory):
  """`AggregationProcessFactory` for zeroing and clipping large values.

  The created `tff.templates.AggregationProcess` zeroes out any values whose
  norm is greater than than a given value, and further projects the values onto
  an L2 ball (also referred to as "clipping") before aggregating the values as
  specified by `inner_agg_factory`. The clipping norm is determined by the
  `clipping_norm`, and the zeroing norm is computed as a function
  (`zeroing_norm_fn`) applied to the clipping norm.

  This is intended to be used when it is preferred (for privacy reasons perhaps)
  to use only a single estimation process. If it is acceptable to use multiple
  estimation processes it would be more flexible to compose a `ZeroingFactory`
  with a `ClippingFactory`. For example, a `ZeroingFactory` allows zeroing
  values with high L-inf norm, whereas this class supports only L2 norm.

  The provided `clipping_norm` can either be a constant (for fixed norm), or an
  instance of `tff.templates.EstimationProcess` (for adaptive norm). If it is an
  estimation process, the value returned by its `report` method will be used as
  the clipping norm. Its `next` method needs to accept a scalar float32 at
  clients, corresponding to the norm of value being aggregated. The process can
  thus adaptively determine the clipping norm based on the set of aggregated
  values. For example if a `tff.aggregators.PrivateQuantileEstimationProcess` is
  used, the clipping norm will be an estimate of a quantile of the norms of the
  values being aggregated.
  """

  def __init__(self, clipping_norm: Union[float,
                                          estimation_process.EstimationProcess],
               zeroing_norm_fn: computation_base.Computation,
               inner_agg_factory: factory.AggregationProcessFactory):
    """Initializes a ZeroingClippingFactory.

    Args:
      clipping_norm: Either a float (for fixed norm) or an `EstimationProcess`
        (for adaptive norm) that specifies the norm over which values should be
        clipped. If an `EstimationProcess` is passed, value norms will be passed
        to the process and its `report` function will be used as the clipping
        norm.
      zeroing_norm_fn: A `tff.Computation` to apply to the clipping norm to
        produce the zeroing norm.
      inner_agg_factory: A factory specifying the type of aggregation to be done
        after zeroing and clipping.
    """
    py_typecheck.check_type(inner_agg_factory,
                            factory.AggregationProcessFactory)
    self._inner_agg_factory = inner_agg_factory

    py_typecheck.check_type(clipping_norm,
                            (float, estimation_process.EstimationProcess))
    if isinstance(clipping_norm, float):
      clipping_norm = _constant_process(clipping_norm)
    _check_norm_process(clipping_norm, 'clipping_norm')
    self._clipping_norm_process = clipping_norm

    py_typecheck.check_type(zeroing_norm_fn, computation_base.Computation)
    zeroing_norm_arg_type = zeroing_norm_fn.type_signature.parameter
    norm_type = clipping_norm.report.type_signature.result.member
    if not zeroing_norm_arg_type.is_assignable_from(norm_type):
      raise TypeError(
          f'Argument of `zeroing_norm_fn` must be assignable from result of '
          f'`clipping_norm`, but `clipping_norm` outputs {norm_type}\n '
          f'and the argument of `zeroing_norm_fn` is {zeroing_norm_arg_type}.')

    zeroing_norm_result_type = zeroing_norm_fn.type_signature.result
    float_type = computation_types.to_type(NORM_TYPE)
    if not float_type.is_assignable_from(zeroing_norm_result_type):
      raise TypeError(f'Result of `zeroing_norm_fn` must be assignable to '
                      f'NORM_TYPE but found {zeroing_norm_result_type}.')
    self._zeroing_norm_fn = zeroing_norm_fn

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:

    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    if not all([t.dtype.is_floating for t in structure.flatten(value_type)]):
      raise TypeError(f'All values in provided value_type must be of floating '
                      f'dtype. Provided value_type: {value_type}')

    inner_agg_process = self._inner_agg_factory.create(value_type)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip((self._clipping_norm_process.initialize(),
                                       inner_agg_process.initialize()))

    @computations.tf_computation(value_type, NORM_TYPE, NORM_TYPE)
    def clip_and_zero(value, clipping_norm, zeroing_norm):
      clipped_value_as_list, global_norm = tf.clip_by_global_norm(
          tf.nest.flatten(value), clipping_norm)
      clipped_value = tf.nest.pack_sequence_as(value, clipped_value_as_list)
      zeroed_and_clipped = _zero_over(clipped_value, global_norm, zeroing_norm)
      return zeroed_and_clipped, global_norm

    @computations.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type),
        computation_types.at_clients(tf.float32))
    def next_fn(state, value, weight):
      clipping_norm_state, agg_state = state

      clipping_norm = self._clipping_norm_process.report(clipping_norm_state)
      zeroing_norm = intrinsics.federated_map(self._zeroing_norm_fn,
                                              clipping_norm)

      zeroed_and_clipped, global_norm = intrinsics.federated_map(
          clip_and_zero, (value, intrinsics.federated_broadcast(clipping_norm),
                          intrinsics.federated_broadcast(zeroing_norm)))

      agg_output = inner_agg_process.next(agg_state, zeroed_and_clipped, weight)
      new_clipping_norm_state = self._clipping_norm_process.next(
          clipping_norm_state, global_norm)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(
              (new_clipping_norm_state, agg_output.state)),
          result=agg_output.result,
          measurements=agg_output.measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _zero_over(value, norm, zeroing_norm):
  return tf.cond((norm > zeroing_norm),
                 lambda: tf.nest.map_structure(tf.zeros_like, value),
                 lambda: value)
