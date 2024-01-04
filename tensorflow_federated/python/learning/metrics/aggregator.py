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
"""Library of common single-round metric aggregators."""

import collections
from typing import Optional, Union

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.learning.metrics import aggregation_utils
from tensorflow_federated.python.learning.metrics import sampling_aggregation_factory
from tensorflow_federated.python.learning.metrics import sum_aggregation_factory
from tensorflow_federated.python.learning.metrics import types


class InternalError(Exception):
  """An error internal to TFF. File a bug report."""


def sum_then_finalize(
    metric_finalizers: Union[
        types.MetricFinalizersType,
        types.FunctionalMetricFinalizersType,
    ],
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
) -> computation_base.Computation:
  """Creates a TFF computation that aggregates metrics via `sum_then_finalize`.

  The returned federated TFF computation has the following type signature:
  `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`, where the
  input is given by
  `tff.learning.models.VariableModel.report_local_unfinalized_metrics()` at
  `CLIENTS`, and the output is computed by first summing the unfinalized metrics
  from `CLIENTS`, followed by applying the finalizers at `SERVER`.

  Args:
    metric_finalizers: Either the result of
      `tff.learning.models.VariableModel.metric_finalizers` (an `OrderedDict` of
      callables) or the `tff.learning.models.FunctionalModel.finalize_metrics`
      method (a callable that takes an `OrderedDict` argument). If the former,
      the keys must be the same as the `OrderedDict` returned by
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics`. If
      the later, the callable must compute over the same keyspace of the result
      returned by `tff.learning.models.FunctionalModel.update_metrics_state`.
    local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
      `OrderedDict` as the Python container) of a client's local unfinalized
      metrics. Let `local_unfinalized_metrics` be the output of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`.
      Its type can be obtained by
      `tff.types.infer_unplaced_type(local_unfinalized_metrics)`.

  Returns:
    A federated TFF computation that sums the unfinalized metrics from
    `CLIENTS`, and applies the correponding finalizers at `SERVER`.

  Raises:
    TypeError: If the inputs are of the wrong types.
    ValueError: If the keys (i.e., metric names) in `metric_finalizers` are not
      the same as those expected by `local_unfinalized_metrics_type`.
  """
  aggregation_utils.check_metric_finalizers(metric_finalizers)
  aggregation_utils.check_local_unfinalzied_metrics_type(
      local_unfinalized_metrics_type
  )
  if not callable(metric_finalizers):
    # If we have a FunctionalMetricsFinalizerType it's a function that can only
    # we checked when we call it, as users may have used *args/**kwargs
    # arguments or otherwise making it hard to deduce the type.
    aggregation_utils.check_finalizers_matches_unfinalized_metrics(
        metric_finalizers, local_unfinalized_metrics_type
    )

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          local_unfinalized_metrics_type, placements.CLIENTS
      )
  )
  def aggregator_computation(client_local_unfinalized_metrics):
    unfinalized_metrics_sum = intrinsics.federated_sum(
        client_local_unfinalized_metrics
    )

    if callable(metric_finalizers):
      finalizer_computation = tensorflow_computation.tf_computation(
          metric_finalizers, local_unfinalized_metrics_type
      )
    else:

      @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
      def finalizer_computation(unfinalized_metrics):
        finalized_metrics = collections.OrderedDict()
        for metric_name, metric_finalizer in metric_finalizers.items():
          finalized_metrics[metric_name] = metric_finalizer(
              unfinalized_metrics[metric_name]
          )
        return finalized_metrics

    return intrinsics.federated_map(
        finalizer_computation, unfinalized_metrics_sum
    )

  return aggregator_computation


DEFAULT_SECURE_LOWER_BOUND = 0
# Use a power of 2 minus one to more accurately encode floating dtypes that
# actually contain integer values. 2 ^ 20 gives us approximately a range of
# [0, 1 million].
DEFAULT_SECURE_UPPER_BOUND = 2**20 - 1


def secure_sum_then_finalize(
    metric_finalizers: Union[
        types.MetricFinalizersType,
        types.FunctionalMetricFinalizersType,
    ],
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
    metric_value_ranges: Optional[
        sum_aggregation_factory.UserMetricValueRangeDict
    ] = None,
) -> computation_base.Computation:
  """Creates a TFF computation that aggregates metrics using secure summation.

  The returned federated TFF computation has the following type signature:

  ```
  (local_unfinalized_metrics@CLIENTS ->
   <aggregated_metrics@SERVER, secure_sum_measurements@SERVER)
  ```

  where the input is given by
  `tff.learning.models.VariableModel.report_local_unfinalized_metrics()` at
  `CLIENTS`, and the
  first output (`aggregated_metrics`) is computed by first securely summing the
  unfinalized metrics from `CLIENTS`, followed by applying the finalizers at
  `SERVER`. The second output (`secure_sum_measurements`) is an `OrderedDict`
  that maps from `factory_key`s to the secure summation measurements (e.g. the
  number of clients gets clipped. See `tff.aggregators.SecureSumFactory` for
  details). A `factory_key` is uniquely defined by three scalars: lower bound,
  upper bound, and tensor dtype (denoted as datatype enum). Metric values of the
  same `factory_key` are grouped and aggegrated together (and hence, the
  `secure_sum_measurements` are also computed at a group level).

  Since secure summation works in fixed-point arithmetic space, floating point
  numbers must be encoding using integer quantization. By default, each tensor
  in `local_unfinalized_metrics_type` will be clipped to `[0, 2**20 - 1]` and
  encoded to integers inside `tff.aggregators.SecureSumFactory`. Callers can
  change this range by setting `metric_value_ranges`, which may be a partial
  tree matching the structure of `local_unfinalized_metrics_type`.

  Example partial value range specification:

  >>> finalizers = ...
  >>> metrics_type = tff.to_type(collections.OrderedDict(
      a=tff.types.TensorType(np.int32),
      b=tff.types.TensorType(np.float32),
      c=[tff.types.TensorType(np.float32), tff.types.TensorType(np.float32)])
  >>> value_ranges = collections.OrderedDict(
      b=(0.0, 1.0),
      c=[None, (0.0, 1.0)])
  >>> aggregator = tff.learning.metrics.secure_sum_then_finalize(
      finalizers, metrics_type, value_ranges)

  This sets the range of the *second* tensor of `b` in the dictionary, using the
  range for the first tensor, and the `a` tensor.

  Args:
    metric_finalizers: Either the result of
      `tff.learning.models.VariableModel.metric_finalizers` (an `OrderedDict` of
      callables) or the `tff.learning.models.FunctionalModel.finalize_metrics`
      method (a callable that takes an `OrderedDict` argument). If the former,
      the keys must be the same as the `OrderedDict` returned by
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics`. If
      the later, the callable must compute over the same keyspace of the result
      returned by `tff.learning.models.FunctionalModel.update_metrics_state`.
    local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
      `OrderedDict` as the Python container) of a client's local unfinalized
      metrics. Let `local_unfinalized_metrics` be the output of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`.
      Its type can be obtained by
      `tff.types.infer_unplaced_type(local_unfinalized_metrics)`.
    metric_value_ranges: A `collections.OrderedDict` that matches the structure
      of `local_unfinalized_metrics_type` (a value for each
      `tff.types.TensorType` in the type tree). Each leaf in the tree should
      have a 2-tuple that defines the range of expected values for that variable
      in the metric. If the entire structure is `None`, a default range of
      `[0.0, 2.0**20 - 1]` will be applied to all variables. Each leaf may also
      be `None`, which will also get the default range; allowing partial user
      sepcialization. At runtime, values that fall outside the ranges specified
      at the leaves, those values will be clipped to within the range.

  Returns:
    A federated TFF computation that securely sums the unfinalized metrics from
    `CLIENTS`, and applies the correponding finalizers at `SERVER`.

  Raises:
    TypeError: If the inputs are of the wrong types.
    ValueError: If the keys (i.e., metric names) in `metric_finalizers` are not
      the same as those expected by `local_unfinalized_metrics_type`.
  """
  aggregation_utils.check_metric_finalizers(metric_finalizers)
  aggregation_utils.check_local_unfinalzied_metrics_type(
      local_unfinalized_metrics_type
  )
  if not callable(metric_finalizers):
    # If we have a FunctionalMetricsFinalizerType it's a function that can only
    # we checked when we call it, as users may have used *args/**kwargs
    # arguments or otherwise making it hard to deduce the type.
    aggregation_utils.check_finalizers_matches_unfinalized_metrics(
        metric_finalizers, local_unfinalized_metrics_type
    )

  default_metric_value_ranges = (
      sum_aggregation_factory.create_default_secure_sum_quantization_ranges(
          local_unfinalized_metrics_type,
          lower_bound=DEFAULT_SECURE_LOWER_BOUND,
          upper_bound=DEFAULT_SECURE_UPPER_BOUND,
          use_auto_tuned_bounds_for_float_values=False,
      )
  )
  try:
    metric_value_ranges = (
        sum_aggregation_factory.fill_missing_values_with_defaults(
            default_metric_value_ranges, metric_value_ranges
        )
    )
  except TypeError as e:
    raise TypeError(
        f'Failed to create encoding value range from: {metric_value_ranges}'
    ) from e

  # only one inner secure aggregation process will be created for each group.
  # This is an optimization for computation tracing and compiling, which can be
  # slow when there are a large number of independent aggregations.
  secure_sum_factory = sum_aggregation_factory.SecureSumFactory(
      metric_value_ranges
  )
  secure_sum_process = secure_sum_factory.create(local_unfinalized_metrics_type)
  # Check the secure sum process is stateless.
  assert not iterative_process.is_stateful(secure_sum_process)

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          local_unfinalized_metrics_type, placements.CLIENTS
      )
  )
  def aggregator_computation(client_local_unfinalized_metrics):
    unused_state = secure_sum_process.initialize()
    output = secure_sum_process.next(
        unused_state, client_local_unfinalized_metrics
    )

    unfinalized_metrics = output.result
    unfinalized_metrics_type = (
        secure_sum_process.next.type_signature.result.result.member  # pytype: disable=attribute-error
    )
    # One minor downside of grouping the inner aggregation processes is that the
    # SecAgg measurements (e.g., clipped_count) are computed at a group level
    # (a group means all metric values belonging to the same `factory_key`).
    secure_sum_measurements = output.measurements
    secure_sum_measurements_type = (
        secure_sum_process.next.type_signature.result.measurements.member  # pytype: disable=attribute-error
    )

    @tensorflow_computation.tf_computation(
        unfinalized_metrics_type, secure_sum_measurements_type
    )
    def finalizer_computation(unfinalized_metrics, secure_sum_measurements):
      finalized_metrics = collections.OrderedDict(
          secure_sum_measurements=secure_sum_measurements
      )
      if callable(metric_finalizers):
        finalized_metrics.update(metric_finalizers(unfinalized_metrics))
      else:
        for metric_name, metric_finalizer in metric_finalizers.items():
          finalized_metrics[metric_name] = metric_finalizer(
              unfinalized_metrics[metric_name]
          )
      return finalized_metrics

    return intrinsics.federated_map(
        finalizer_computation, (unfinalized_metrics, secure_sum_measurements)
    )

  return aggregator_computation


def finalize_then_sample(
    metric_finalizers: Union[
        types.MetricFinalizersType,
        types.FunctionalMetricFinalizersType,
    ],
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
    sample_size: int = 100,
) -> computation_base.Computation:
  """Creates a TFF computation to aggregate metrics via `finalize_then_sample`.

  The returned federated TFF computation has the following type signature:
  `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`. The input is
  given by
  `tff.learning.models.VariableModel.report_local_unfinalized_metrics()` at
  `CLIENTS`. The output is computed by first finalizing each client's metrics
  locally, and then collecting metrics from at most `sample_size` clients at the
  `SERVER`. If more than `sample_size` clients participating, then `sample_size`
  clients are sampled (by reservoir sampling algorithm); otherwise, all clients'
  metrics are collected. Sampling is done in a "per-client" manner, i.e.,
  a client, once sampled, will contribute all its metrics to the final result.

  The collected metrics samples at `SERVER` has the same structure (i.e., same
  keys in a dictionary) as the client's local metrics, except that each leaf
  node contains a list of scalar metric values, where each value comes from a
  sampled client, e.g.,
    ```
    sampled_metrics_at_server = {
        'metric_a': [a1, a2, ...],
        'metric_b': [b1, b2, ...],
        ...
    }
    ```
  where "a1" and "b1" are from the same client (similary for "a2" and "b2" etc).

  Example usage:
    ```
    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=..., ..., metrics_aggregator=finalize_then_sample,
    )
    state = training_process.initialize()
    for i in range(num_rounds):
      output = training_process.next(state, client_data_at_round_i)
      state = output.state
      sampled_client_metrics = output.metrics['client_work']
    ```
  Args:
    metric_finalizers: Either the result of
      `tff.learning.models.VariableModel.metric_finalizers` (an `OrderedDict` of
      callables) or the `tff.learning.models.FunctionalModel.finalize_metrics`
      method (a callable that takes an `OrderedDict` argument). If the former,
      the keys must be the same as the `OrderedDict` returned by
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics`. If
      the later, the callable must compute over the same keyspace of the result
      returned by `tff.learning.models.FunctionalModel.update_metrics_state`.
    local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
      `OrderedDict` as the Python container) of a client's local unfinalized
      metrics. Let `local_unfinalized_metrics` be the output of
      `tff.learning.models.VariableModel.report_local_unfinalized_metrics()`.
      Its type can be obtained by
      `tff.types.infer_unplaced_type(local_unfinalized_metrics)`.
    sample_size: An integer specifying the number of clients sampled by the
      reservoir sampling algorithm. Metrics from the sampled clients are
      collected at the server. If the total number of participating clients are
      smaller than this value, then all clients' metrics are collected. Default
      value is 100.

  Returns:
    A federated TFF computation that finalizes the unfinalized metrics from
    `CLIENTS`, samples the clients, and returns the sampled metrics at `SERVER`.

  Raises:
    TypeError: If the inputs are of the wrong types.
    ValueError: If the keys (i.e., metric names) in `metric_finalizers` are not
      the same as those expected by `local_unfinalized_metrics_type`.
    ValueError: If `sample_size` is not positive.
  """
  aggregation_utils.check_metric_finalizers(metric_finalizers)
  aggregation_utils.check_local_unfinalzied_metrics_type(
      local_unfinalized_metrics_type
  )
  if not callable(metric_finalizers):
    # If we have a FunctionalMetricsFinalizerType it's a function that can only
    # we checked when we call it, as users may have used *args/**kwargs
    # arguments or otherwise making it hard to deduce the type.
    aggregation_utils.check_finalizers_matches_unfinalized_metrics(
        metric_finalizers, local_unfinalized_metrics_type
    )
  py_typecheck.check_type(sample_size, int, 'sample_size')
  if sample_size <= 0:
    raise ValueError('sample_size must be positive.')
  sample_process = sampling_aggregation_factory.FinalizeThenSampleFactory(
      sample_size=sample_size
  ).create(
      metric_finalizers=metric_finalizers,
      local_unfinalized_metrics_type=local_unfinalized_metrics_type,
  )

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          local_unfinalized_metrics_type, placements.CLIENTS
      )
  )
  def aggregator_computation(client_local_unfinalized_metrics):
    unused_state = sample_process.initialize()
    output = sample_process.next(unused_state, client_local_unfinalized_metrics)
    # `output.result` is a tuple (current_round_samples, total_rounds_samples).
    current_round_samples, _ = output.result
    return current_round_samples

  return aggregator_computation
