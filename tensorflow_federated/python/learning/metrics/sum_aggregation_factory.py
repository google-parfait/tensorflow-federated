# Copyright 2022, The TensorFlow Federated Authors.
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
"""AggregationFactory that sums and then finalizes federated learning metrics."""

import collections
from collections.abc import Mapping
import dataclasses
import math
import typing
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.aggregators import sum_factory as sum_factory_lib
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.metrics import aggregation_utils
from tensorflow_federated.python.learning.metrics import types


def _initialize_unfinalized_metrics_accumulators(
    local_unfinalized_metrics_type, initial_unfinalized_metrics
):
  """Initializes the unfinalized metrics accumulators."""
  if initial_unfinalized_metrics is not None:
    return intrinsics.federated_value(
        initial_unfinalized_metrics, placements.SERVER
    )

  @tensorflow_computation.tf_computation
  def create_all_zero_state():
    return type_conversions.structure_from_tensor_type_tree(
        lambda t: tf.zeros(shape=t.shape, dtype=t.dtype),
        local_unfinalized_metrics_type,
    )

  return intrinsics.federated_eval(create_all_zero_state, placements.SERVER)


# TODO: b/227811468 - Support other inner aggregators for SecAgg and DP.
class SumThenFinalizeFactory(factory.UnweightedAggregationFactory):
  """Aggregation Factory that sums and then finalizes the metrics.

  The created `tff.templates.AggregationProcess` uses the inner summation
  process created by the inner summation factory to sum unfinalized metrics
  from `tff.CLIENTS` to `tff.SERVER`, accumulates the summed unfinalized metrics
  in the `state`, and then finalize the metrics for both current round and total
  rounds. If the inner summation factory is not specified,
  `tff.aggregators.SumFactory` is used by default. The inner summation factory
  can also use SecAgg.

  The accumulated unfinalized metrics across rounds are initialized to be the
  intial value of the unfinalized metrics, if the inital value is not specified,
  zero is used.

  The `next` function of the created `tff.templates.AggregationProcess` takes
  the `state` and local unfinalized metrics reported from `tff.CLIENTS`, and
  returns a `tff.templates.MeasuredProcessOutput` object with the following
  properties:
    - `state`: a tuple of the `state` of the inner summation process and the
      accumulated unfinalized metrics across rounds.
    - `result`: a tuple of the finalized metrics of the current round and total
      rounds.
    - `measurements`: the measurements of the inner summation process.
  """

  def __init__(
      self,
      metric_finalizers: types.MetricFinalizersType,
      initial_unfinalized_metrics: Optional[
          collections.OrderedDict[str, Any]
      ] = None,
      inner_summation_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ):
    """Initialize the `SumThenFinalizeFactory`.

    Args:
      metric_finalizers: An `collections.OrderedDict` of metric names to
        finalizers, should have same keys as the unfinalized metrics. A
        finalizer is a function (typically a `tf.function` decorated callable or
        a `tff.tensorflow.computation` decorated TFF Computation) that takes in
        a metric's unfinalized values, and returns the finalized metric values.
        This can be obtained from
        `tff.learning.models.VariableModel.metric_finalizers()`.
      initial_unfinalized_metrics: Optional. An `collections.OrderedDict` of
        metric names to the initial values of local unfinalized metrics, its
        structure should match that of `local_unfinalized_metrics_type`. If not
        specified, defaults to zero.
      inner_summation_factory: Optional. A
        `tff.aggregators.UnweightedAggregationFactory` that creates a
        `tff.templates.AggregationProcess` to sum the metrics from clients to
        server. If not specified, `tff.aggregators.SumFactory` is used. If the
        metrics aggregation needs SecAgg, `aggregation_factory.SecureSumFactory`
        can be used as the inner summation factory.

    Raises:
      TypeError: If any argument type mismatches.
    """
    aggregation_utils.check_metric_finalizers(metric_finalizers)
    self._metric_finalizers = metric_finalizers

    if initial_unfinalized_metrics is not None:
      py_typecheck.check_type(
          initial_unfinalized_metrics, collections.OrderedDict
      )
    self._initial_unfinalized_metrics = initial_unfinalized_metrics

    if inner_summation_factory is None:
      inner_summation_factory = sum_factory_lib.SumFactory()
    else:
      py_typecheck.check_type(
          inner_summation_factory, factory.UnweightedAggregationFactory
      )
    self._inner_summation_factory = inner_summation_factory

  def create(
      self,
      local_unfinalized_metrics_type: computation_types.StructWithPythonType,
  ) -> aggregation_process.AggregationProcess:
    """Creates a `tff.templates.AggregationProcess` for metrics aggregation.

    Args:
      local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
        `collections.OrderedDict` as the Python container) of a client's local
        unfinalized metrics. For example, `local_unfinalized_metrics` could
        represent the output type of
        `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`.

    Returns:
      An instance of `tff.templates.AggregationProcess`.

    Raises:
      TypeError: If any argument type mismatches; if the metric finalizers
        mismatch the type of local unfinalized metrics; if the initial
        unfinalized metrics mismatch the type of local unfinalized metrics.
    """
    aggregation_utils.check_local_unfinalized_metrics_type(
        local_unfinalized_metrics_type
    )
    if not callable(self._metric_finalizers):
      # If we have a FunctionalMetricsFinalizerType its a function that can only
      # be checked when we call it, as users may have used *args/**kwargs
      # arguments or otherwise making it hard to deduce the type.
      aggregation_utils.check_finalizers_matches_unfinalized_metrics(
          self._metric_finalizers, local_unfinalized_metrics_type
      )

    inner_summation_process = self._inner_summation_factory.create(
        local_unfinalized_metrics_type
    )

    @federated_computation.federated_computation
    def init_fn():
      unfinalized_metrics_accumulators = (
          _initialize_unfinalized_metrics_accumulators(
              local_unfinalized_metrics_type, self._initial_unfinalized_metrics
          )
      )
      return intrinsics.federated_zip((
          inner_summation_process.initialize(),
          unfinalized_metrics_accumulators,
      ))

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.FederatedType(
            local_unfinalized_metrics_type, placements.CLIENTS
        ),
    )
    def next_fn(
        state, unfinalized_metrics
    ) -> measured_process.MeasuredProcessOutput:
      inner_summation_state, unfinalized_metrics_accumulators = state

      inner_summation_output = inner_summation_process.next(
          inner_summation_state, unfinalized_metrics
      )
      summed_unfinalized_metrics = inner_summation_output.result
      inner_summation_state = inner_summation_output.state

      @tensorflow_computation.tf_computation(
          local_unfinalized_metrics_type, local_unfinalized_metrics_type
      )
      def add_unfinalized_metrics(
          unfinalized_metrics, summed_unfinalized_metrics
      ):
        return tf.nest.map_structure(
            tf.add, unfinalized_metrics, summed_unfinalized_metrics
        )

      unfinalized_metrics_accumulators = intrinsics.federated_map(
          add_unfinalized_metrics,
          (unfinalized_metrics_accumulators, summed_unfinalized_metrics),
      )

      finalizer_computation = aggregation_utils.build_finalizer_computation(
          self._metric_finalizers, local_unfinalized_metrics_type
      )

      current_round_metrics = intrinsics.federated_map(
          finalizer_computation, summed_unfinalized_metrics
      )
      total_rounds_metrics = intrinsics.federated_map(
          finalizer_computation, unfinalized_metrics_accumulators
      )

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(
              (inner_summation_state, unfinalized_metrics_accumulators)
          ),
          result=intrinsics.federated_zip(
              (current_round_metrics, total_rounds_metrics)
          ),
          measurements=inner_summation_output.measurements,
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)


MetricValueLowerBoundType = Union[int, float, None]
MetricValueUpperBoundType = Union[
    int, float, estimation_process.EstimationProcess
]


@dataclasses.dataclass(frozen=True)
class _MetricRange:
  """An opaque structure defining a closed range.

  This is used as an opaque object in a nested structure to prevent
  `tree.map_structure` from traversing to the numeric leaves.
  """

  lower: MetricValueLowerBoundType
  upper: MetricValueUpperBoundType

  def __eq__(self, other):
    """A type-aware equality that prevents int/float conversion."""
    if isinstance(self.upper, estimation_process.EstimationProcess):
      return False
    return (
        type(self.upper) is type(other.upper)
        and self.upper == other.upper
        and type(self.lower) is type(other.lower)
        and self.lower == other.lower
    )


UserMetricValueRange = Union[tuple[float, float], tuple[int, int], _MetricRange]
UserMetricValueRangeDict = collections.OrderedDict[
    str, Union[UserMetricValueRange, 'UserMetricValueRangeDict']
]
MetricValueRange = Union[
    tuple[float, float],
    tuple[int, int],
    tuple[None, estimation_process.EstimationProcess],
]
MetricValueRangeDict = collections.OrderedDict[
    str, Union[MetricValueRange, 'MetricValueRangeDict']
]

DEFAULT_FIXED_SECURE_LOWER_BOUND = 0
# Use a power of 2 minus one to more accurately encode floating dtypes that
# actually contain integer values. 2 ^ 20 gives us approximately a range of
# [0, 1 million].
DEFAULT_FIXED_SECURE_UPPER_BOUND = 2**20 - 1


class UnquantizableDTypeError(Exception):
  """An error raised when a tensor dtype is not quantizable."""


def _check_user_metric_value_range(value_range: UserMetricValueRange):
  """Validates a value range inputted by users is a valid metric value range.

  A user speficied metric range should be defined as a two-element `tuple`:
  (`lower_bound_threshold`, `upper_bound_threshold`), and its elements should be
  one of the allowed types of `UserMetricValueRange`.

  Args:
    value_range: The metric value range to validate.

  Raises:
    TypeError: If the value is not a `tuple` or its elements are not allowed
      types of `MetricValueRange`.
    ValueError: If the value has length other than two.
  """
  py_typecheck.check_type(value_range, tuple, 'range')
  value_range = typing.cast(
      Union[tuple[float, float], tuple[int, int]], value_range
  )
  if len(value_range) != 2:
    raise ValueError(
        'Ranges must be defined as a 2-tuple, got a tuple of '
        f'length {len(value_range)}.'
    )

  lower, upper = value_range
  py_typecheck.check_type(lower, (int, float), 'lower bound')
  if type(upper) is not type(lower):
    raise TypeError(
        'The lower bound threshold should have the same type as '
        'the upper bound threshold, but found the lower bound '
        f'threshold type {type(lower)} and the upper bound '
        f'threshold type {type(upper)}.'
    )
