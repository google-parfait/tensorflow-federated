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
from collections.abc import Callable
import math
import typing
from typing import Optional, TypeVar, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process

NORM_TF_TYPE = np.float32
COUNT_TF_TYPE = np.int32
_T = TypeVar('_T', bound=factory.AggregationFactory)


def _constant_process(value):
  """Creates an `EstimationProcess` that reports a constant value."""
  init_fn = federated_computation.federated_computation(
      lambda: intrinsics.federated_value((), placements.SERVER)
  )
  next_fn = federated_computation.federated_computation(
      lambda state, value: state,
      init_fn.type_signature.result,
      computation_types.FederatedType(NORM_TF_TYPE, placements.CLIENTS),
  )
  report_fn = federated_computation.federated_computation(
      lambda state: intrinsics.federated_value(value, placements.SERVER),
      init_fn.type_signature.result,
  )
  return estimation_process.EstimationProcess(init_fn, next_fn, report_fn)


def _check_norm_process(
    norm_process: estimation_process.EstimationProcess, name: str
):
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
  if (
      not isinstance(next_parameter_type, computation_types.StructType)
      or len(next_parameter_type) != 2
  ):
    raise TypeError(
        f'`{name}.next` must take two arguments but found:\n'
        f'{next_parameter_type}'
    )

  norm_type_at_clients = computation_types.FederatedType(
      NORM_TF_TYPE, placements.CLIENTS
  )
  if not next_parameter_type[1].is_assignable_from(norm_type_at_clients):  # pytype: disable=unsupported-operands
    raise TypeError(
        f'Second argument of `{name}.next` must be assignable from '
        f'{norm_type_at_clients} but found {next_parameter_type[1]}'  # pytype: disable=unsupported-operands
    )

  next_result_type = norm_process.next.type_signature.result
  if not norm_process.state_type.is_assignable_from(next_result_type):
    raise TypeError(
        f'Result type of `{name}.next` must consist of state only '
        f'but found result type:\n{next_result_type}\n'
        f'while the state type is:\n{norm_process.state_type}'
    )

  result_type = norm_process.report.type_signature.result
  norm_type_at_server = computation_types.FederatedType(
      NORM_TF_TYPE, placements.SERVER
  )
  if not norm_type_at_server.is_assignable_from(result_type):
    raise TypeError(
        f'Result type of `{name}.report` must be assignable to '
        f'{norm_type_at_server} but found {result_type}.'
    )


def clipping_factory(
    clipping_norm: Union[float, estimation_process.EstimationProcess],
    inner_agg_factory: factory.AggregationFactory,
    clipped_count_sum_factory: Optional[
        factory.UnweightedAggregationFactory
    ] = None,
) -> factory.AggregationFactory:
  """Creates an aggregation factory to perform L2 clipping.

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

  The `value_type` provided to the `create` method must be a structure of
  floats, but they do not all need to be the same, e.g. a mix of `np.float32`
  and `np.float16` dtypes is allowed.

  The created process will report measurements
  * `clipped_count`: The number of aggregands clipped.
  * `clipping_norm`: The norm used to determine whether to clip an aggregand.

  The returned `AggregationFactory` takes its weightedness
  (`UnweightedAggregationFactory` vs. `WeightedAggregationFactory`) from
  `inner_agg_factory`.

  Args:
    clipping_norm: Either a float (for fixed norm) or an `EstimationProcess`
      (for adaptive norm) that specifies the norm over which the values should
      be clipped.
    inner_agg_factory: A factory specifying the type of aggregation to be done
      after clipping.
    clipped_count_sum_factory: A factory specifying the type of aggregation done
      for `clipped_count` measurement. If `None`, `tff.aggregators.SumFactory`
      will be used.

  Returns:
    An aggregation factory to perform L2 clipping.
  """

  if clipped_count_sum_factory is None:
    clipped_count_sum_factory = sum_factory.SumFactory()
  py_typecheck.check_type(
      clipped_count_sum_factory, factory.UnweightedAggregationFactory
  )

  def make_clip_fn(value_type):
    @tensorflow_computation.tf_computation(value_type, NORM_TF_TYPE)
    def clip_fn(value, clipping_norm):
      clipped_value, global_norm = _clip_by_global_l2_norm(value, clipping_norm)
      was_clipped = tf.cast((global_norm > clipping_norm), COUNT_TF_TYPE)
      return clipped_value, global_norm, was_clipped

    return clip_fn

  return _make_wrapper(
      clipping_norm,
      inner_agg_factory,
      clipped_count_sum_factory,
      make_clip_fn,
      'clipp',
  )


def zeroing_factory(
    zeroing_norm: Union[float, estimation_process.EstimationProcess],
    inner_agg_factory: _T,
    norm_order: float = math.inf,
    zeroed_count_sum_factory: Optional[
        factory.UnweightedAggregationFactory
    ] = None,
) -> _T:
  """Creates an aggregation factory to perform zeroing.

  The created `tff.templates.AggregationProcess` zeroes out any values whose
  norm is greater than that determined by the provided `zeroing_norm`, before
  aggregating the values as specified by `inner_agg_factory`. Note that for
  weighted aggregation if some value is zeroed, the weight is unchanged. So for
  example if you have a zeroed weighted mean and a lot of zeroing occurs, the
  average will tend to be pulled toward zero. This is for consistency between
  weighted and unweighted aggregation

  The provided `zeroing_norm` can either be a constant (for fixed norm), or an
  instance of `tff.templates.EstimationProcess` (for adaptive norm). If it is an
  estimation process, the value returned by its `report` method will be used as
  the zeroing norm. Its `next` method needs to accept a scalar float32 at
  clients, corresponding to the norm of value being aggregated. The process can
  thus adaptively determine the zeroing norm based on the set of aggregated
  values. For example if a `tff.aggregators.PrivateQuantileEstimationProcess` is
  used, the zeroing norm will be an estimate of a quantile of the norms of the
  values being aggregated.

  The `value_type` provided to the `create` method must be a structure of
  floats, but they do not all need to be the same, e.g. a mix of `np.float32`
  and `np.float16` dtypes is allowed.

  The created process will report measurements
  * `zeroed_count`: The number of aggregands zeroed out.
  * `zeroing_norm`: The norm used to determine whether to zero out an aggregand.

  The returned `AggregationFactory` takes its weightedness
  (`UnweightedAggregationFactory` vs. `WeightedAggregationFactory`) from
  `inner_agg_factory`.

  Args:
    zeroing_norm: Either a float (for fixed norm) or an `EstimationProcess` (for
      adaptive norm) that specifies the norm over which the values should be
      zeroed.
    inner_agg_factory: A factory specifying the type of aggregation to be done
      after zeroing.
    norm_order: A float for the order of the norm. Must be 1., 2., or infinity.
    zeroed_count_sum_factory: A factory specifying the type of aggregation done
      for `zeroed_count` measurement. If `None`, `tff.aggregators.SumFactory`
      will be used.

  Returns:
    An aggregation factory to perform zeroing.
  """

  if zeroed_count_sum_factory is None:
    zeroed_count_sum_factory = sum_factory.SumFactory()
  py_typecheck.check_type(
      zeroed_count_sum_factory, factory.UnweightedAggregationFactory
  )
  py_typecheck.check_type(norm_order, float)
  if not (norm_order in [1.0, 2.0] or math.isinf(norm_order)):
    raise ValueError('norm_order must be 1.0, 2.0 or infinity')

  def make_zero_fn(value_type):
    """Creates a zeroing function for the value_type."""

    @tensorflow_computation.tf_computation(value_type, NORM_TF_TYPE)
    def zero_fn(value, zeroing_norm):
      if norm_order == 1.0:
        global_norm = _global_l1_norm(value)
      elif norm_order == 2.0:
        global_norm = _global_l2_norm(value)
      else:
        assert math.isinf(norm_order)
        global_norm = _global_inf_norm(value)
      should_zero = global_norm > zeroing_norm
      zeroed_value = tf.cond(
          should_zero,
          lambda: tf.nest.map_structure(tf.zeros_like, value),
          lambda: value,
      )
      was_zeroed = tf.cast(should_zero, COUNT_TF_TYPE)
      return zeroed_value, global_norm, was_zeroed

    return zero_fn

  return _make_wrapper(
      zeroing_norm,
      inner_agg_factory,
      zeroed_count_sum_factory,
      make_zero_fn,
      'zero',
  )


def _make_wrapper(
    clipping_norm: Union[float, estimation_process.EstimationProcess],
    inner_agg_factory: _T,
    clipped_count_sum_factory: factory.UnweightedAggregationFactory,
    make_clip_fn: Callable[[factory.ValueType], computation_base.Computation],
    attribute_prefix: str,
) -> _T:
  """Constructs an aggregation factory that applies clip_fn before aggregation.

  Args:
    clipping_norm: Either a float (for fixed norm) or an `EstimationProcess`
      (for adaptive norm) that specifies the norm over which the values should
      be clipped.
    inner_agg_factory: A factory specifying the type of aggregation to be done
      after zeroing.
    clipped_count_sum_factory: A factory specifying the type of aggregation done
      for the `clipped_count` measurement.
    make_clip_fn: A callable that takes a value type and returns a
      tff.computation specifying the clip operation to apply before aggregation.
    attribute_prefix: A str for prefixing state and measurement names.

  Returns:
    An aggregation factory that applies clip_fn before aggregation.
  """
  py_typecheck.check_type(
      inner_agg_factory,
      (
          factory.UnweightedAggregationFactory,
          factory.WeightedAggregationFactory,
      ),
  )
  py_typecheck.check_type(
      clipped_count_sum_factory, factory.UnweightedAggregationFactory
  )
  py_typecheck.check_type(
      clipping_norm, (float, estimation_process.EstimationProcess)
  )
  if isinstance(clipping_norm, float):
    clipping_norm_process = _constant_process(clipping_norm)
  else:
    clipping_norm_process = clipping_norm
  _check_norm_process(clipping_norm_process, 'clipping_norm_process')

  clipped_count_agg_process = clipped_count_sum_factory.create(
      computation_types.to_type(COUNT_TF_TYPE)  # pytype: disable=wrong-arg-types
  )

  prefix = lambda s: attribute_prefix + s

  def init_fn_impl(inner_agg_process):
    state = collections.OrderedDict([
        (prefix('ing_norm'), clipping_norm_process.initialize()),
        ('inner_agg', inner_agg_process.initialize()),
        (prefix('ed_count_agg'), clipped_count_agg_process.initialize()),
    ])
    return intrinsics.federated_zip(state)

  def next_fn_impl(state, value, clip_fn, inner_agg_process, weight=None):
    clipping_norm_state, agg_state, clipped_count_state = state

    clipping_norm = clipping_norm_process.report(clipping_norm_state)

    clients_clipping_norm = intrinsics.federated_broadcast(clipping_norm)

    clipped_value, global_norm, was_clipped = intrinsics.federated_map(
        clip_fn, (value, clients_clipping_norm)
    )

    new_clipping_norm_state = clipping_norm_process.next(
        clipping_norm_state, global_norm
    )

    if weight is None:
      agg_output = inner_agg_process.next(agg_state, clipped_value)
    else:
      agg_output = inner_agg_process.next(agg_state, clipped_value, weight)

    clipped_count_output = clipped_count_agg_process.next(
        clipped_count_state, was_clipped
    )

    new_state = collections.OrderedDict([
        (prefix('ing_norm'), new_clipping_norm_state),
        ('inner_agg', agg_output.state),
        (prefix('ed_count_agg'), clipped_count_output.state),
    ])
    measurements = collections.OrderedDict([
        (prefix('ing'), agg_output.measurements),
        (prefix('ing_norm'), clipping_norm),
        (prefix('ed_count'), clipped_count_output.result),
    ])

    return measured_process.MeasuredProcessOutput(
        state=intrinsics.federated_zip(new_state),
        result=agg_output.result,
        measurements=intrinsics.federated_zip(measurements),
    )

  if isinstance(inner_agg_factory, factory.WeightedAggregationFactory):

    class WeightedRobustFactory(factory.WeightedAggregationFactory):
      """`WeightedAggregationFactory` factory for clipping large values."""

      def create(
          self, value_type: factory.ValueType, weight_type: factory.ValueType
      ) -> aggregation_process.AggregationProcess:
        _check_value_type(value_type)
        type_args = typing.get_args(factory.ValueType)
        py_typecheck.check_type(weight_type, type_args)

        inner_agg_process = inner_agg_factory.create(value_type, weight_type)
        clip_fn = make_clip_fn(value_type)

        @federated_computation.federated_computation()
        def init_fn():
          return init_fn_impl(inner_agg_process)

        @federated_computation.federated_computation(
            init_fn.type_signature.result,
            computation_types.FederatedType(value_type, placements.CLIENTS),
            computation_types.FederatedType(weight_type, placements.CLIENTS),
        )
        def next_fn(state, value, weight):
          return next_fn_impl(state, value, clip_fn, inner_agg_process, weight)

        return aggregation_process.AggregationProcess(init_fn, next_fn)

    return WeightedRobustFactory()
  elif isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):

    class UnweightedRobustFactory(factory.UnweightedAggregationFactory):
      """`UnweightedAggregationFactory` factory for clipping large values."""

      def create(
          self, value_type: factory.ValueType
      ) -> aggregation_process.AggregationProcess:
        _check_value_type(value_type)

        inner_agg_process = inner_agg_factory.create(value_type)
        clip_fn = make_clip_fn(value_type)

        @federated_computation.federated_computation()
        def init_fn():
          return init_fn_impl(inner_agg_process)

        @federated_computation.federated_computation(
            init_fn.type_signature.result,
            computation_types.FederatedType(value_type, placements.CLIENTS),
        )
        def next_fn(state, value):
          return next_fn_impl(state, value, clip_fn, inner_agg_process)

        return aggregation_process.AggregationProcess(init_fn, next_fn)

    return UnweightedRobustFactory()
  else:
    raise NotImplementedError(f'Unexpected factory found: {inner_agg_factory}.')


def _check_value_type(value_type):
  type_args = typing.get_args(factory.ValueType)
  py_typecheck.check_type(value_type, type_args)
  if not type_analysis.is_structure_of_floats(value_type):
    raise TypeError(
        'All values in provided value_type must be of floating '
        f'dtype. Provided value_type: {value_type}'
    )


def _global_inf_norm(l):
  norms = [
      tf.cast(tf.norm(a, ord=np.inf), tf.float32) for a in tf.nest.flatten(l)
  ]
  return tf.reduce_max(tf.stack(norms))


def _global_l2_norm(l):
  norms_squared = [
      tf.cast(tf.norm(a, ord=2) ** 2, tf.float32) for a in tf.nest.flatten(l)
  ]
  return tf.math.sqrt(tf.reduce_sum(tf.stack(norms_squared)))


def _global_l1_norm(l):
  norms = [tf.cast(tf.norm(a, ord=1), tf.float32) for a in tf.nest.flatten(l)]
  return tf.reduce_sum(tf.stack(norms))


def _clip_by_global_l2_norm(value, clip_norm):
  """Same as `tf.clip_by_global_norm`, but supports mixed float dtypes."""
  global_norm = _global_l2_norm(value)
  clipped_value = tf.cond(
      tf.math.greater(global_norm, clip_norm),
      lambda: _do_clip_by_global_l2_norm(value, clip_norm, global_norm),
      lambda: value,
  )
  return clipped_value, global_norm


def _do_clip_by_global_l2_norm(value, clip_norm, global_norm):
  divisor = global_norm / clip_norm
  return tf.nest.map_structure(lambda x: x / tf.cast(divisor, x.dtype), value)
