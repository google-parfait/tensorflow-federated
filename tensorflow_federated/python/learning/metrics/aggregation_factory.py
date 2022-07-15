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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""AggregationFactory for metrics."""

import collections
from typing import Any, Optional, OrderedDict

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory as sum_factory_lib
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning.metrics import aggregator


def _build_finalizer_computation(
    metric_finalizers: model_lib.MetricFinalizersType,
    local_unfinalized_metrics_type: computation_types.StructWithPythonType
) -> computation_base.Computation:
  """Builds computation for finalizing metrics."""

  @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
  def finazlier_computation(unfinalized_metrics):
    finalized_metrics = collections.OrderedDict()
    for metric_name, metric_finalizer in metric_finalizers.items():
      finalized_metrics[metric_name] = metric_finalizer(
          unfinalized_metrics[metric_name])
    return finalized_metrics

  return finazlier_computation


def _intialize_unfinalized_metrics_accumulators(local_unfinalized_metrics_type,
                                                initial_unfinalized_metrics):
  """Initalizes the unfinalized metrics accumulators."""
  if initial_unfinalized_metrics is not None:
    initial_unfinalized_metrics_type = type_conversions.type_from_tensors(
        initial_unfinalized_metrics)
    if initial_unfinalized_metrics_type != local_unfinalized_metrics_type:
      raise TypeError('The initial unfinalized metrics type doesn\'t match '
                      'with the `local_unfinalized_metrics_type`, expect: '
                      f'{local_unfinalized_metrics_type}, found: '
                      f'{initial_unfinalized_metrics_type}.')

  @tensorflow_computation.tf_computation
  def create_all_zero_state():
    return type_conversions.structure_from_tensor_type_tree(
        lambda t: tf.zeros(shape=t.shape, dtype=t.dtype),
        local_unfinalized_metrics_type)

  if initial_unfinalized_metrics is None:
    return intrinsics.federated_eval(create_all_zero_state, placements.SERVER)

  return intrinsics.federated_value(initial_unfinalized_metrics,
                                    placements.SERVER)


# TODO(b/227811468): Support other inner aggregators for SecAgg and DP.
class SumThenFinalizeFactory(factory.UnweightedAggregationFactory):
  """Aggregation Factory that sums and then finalizes the metrics.

  The created `tff.templates.AggregationProcess` uses the inner summation
  process created by the `tff.aggregators.SumFactory` to sum unfinalized metrics
  from `tff.CLIENTS` to `tff.SERVER`, accumulates the summed unfinalized metrics
  in the `state`, and then finalize the metrics for both current round and total
  rounds.

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

  def create(
      self,
      metric_finalizers: model_lib.MetricFinalizersType,
      local_unfinalized_metrics_type: computation_types.StructWithPythonType,
      initial_unfinalized_metrics: Optional[OrderedDict[str, Any]] = None
  ) -> aggregation_process.AggregationProcess:
    """Creates a `tff.templates.AggregationProcess` for metrics aggregation.

    Args:
      metric_finalizers: An `OrderedDict` of metric names to finalizers, should
        have same keys as the unfinalized metrics. A finalizer is a function
        (typically a `tf.function` decorated callable or a `tff.tf_computation`
        decoreated TFF Computation) that takes in a metric's unfinalized values,
        and returns the finalized metric values. This can be obtained from
        `tff.learning.Model.metric_finalizers()`.
      local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
        `OrderedDict` as the Python container) of a client's local unfinalized
        metrics. Let `local_unfinalized_metrics` be the output of
        `tff.learning.Model.report_local_unfinalized_metrics()`, its type can be
        obtained by
        `tff.framework.type_from_tensors(local_unfinalized_metrics)`.
      initial_unfinalized_metrics: Optional. An `OrderedDict` of metric names to
        the initial values of local unfinalized metrics, its structure should
        match that of `local_unfinalized_metrics_type`. If not specified,
        defaults to zero.

    Returns:
      An instance of `tff.templates.AggregationProcess`.

    Raises:
      TypeError: If any argument type mismatches; if the metric finalizers
        mismatch the type of local unfinalized metrics; if the initial
        unfinalized metrics mismatch the type of local unfinalized metrics.
    """
    aggregator.check_metric_finalizers(metric_finalizers)
    aggregator.check_local_unfinalzied_metrics_type(
        local_unfinalized_metrics_type)
    aggregator.check_finalizers_matches_unfinalized_metrics(
        metric_finalizers, local_unfinalized_metrics_type)

    inner_summation_process = sum_factory_lib.SumFactory().create(
        local_unfinalized_metrics_type)

    @federated_computation.federated_computation
    def init_fn():
      unfinalized_metrics_accumulators = (
          _intialize_unfinalized_metrics_accumulators(
              local_unfinalized_metrics_type, initial_unfinalized_metrics))
      return intrinsics.federated_zip((inner_summation_process.initialize(),
                                       unfinalized_metrics_accumulators))

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.at_clients(local_unfinalized_metrics_type))
    def next_fn(state,
                unfinalized_metrics) -> measured_process.MeasuredProcessOutput:
      inner_summation_state, unfinalized_metrics_accumulators = state

      inner_summation_output = inner_summation_process.next(
          inner_summation_state, unfinalized_metrics)
      summed_unfinalized_metrics = inner_summation_output.result
      inner_summation_state = inner_summation_output.state

      @tensorflow_computation.tf_computation(local_unfinalized_metrics_type,
                                             local_unfinalized_metrics_type)
      def add_unfinalized_metrics(unfinalized_metrics,
                                  summed_unfinalized_metrics):
        return tf.nest.map_structure(tf.add, unfinalized_metrics,
                                     summed_unfinalized_metrics)

      unfinalized_metrics_accumulators = intrinsics.federated_map(
          add_unfinalized_metrics,
          (unfinalized_metrics_accumulators, summed_unfinalized_metrics))

      finalizer_computation = _build_finalizer_computation(
          metric_finalizers, local_unfinalized_metrics_type)

      current_round_metrics = intrinsics.federated_map(
          finalizer_computation, summed_unfinalized_metrics)
      total_rounds_metrics = intrinsics.federated_map(
          finalizer_computation, unfinalized_metrics_accumulators)

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(
              (inner_summation_state, unfinalized_metrics_accumulators)),
          result=intrinsics.federated_zip(
              (current_round_metrics, total_rounds_metrics)),
          measurements=inner_summation_output.measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)
