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
"""AggregationFactory for Federated Learning metrics."""

import collections
from collections.abc import Mapping
import dataclasses
import math
import typing
from typing import Any, Optional, Union

import tensorflow as tf
import tree

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.aggregators import sum_factory as sum_factory_lib
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.metrics import aggregation_utils
from tensorflow_federated.python.learning.metrics import types


def _build_finalizer_computation(
    metric_finalizers: Union[
        types.MetricFinalizersType,
        types.FunctionalMetricFinalizersType,
    ],
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
) -> computation_base.Computation:
  """Builds computation for finalizing metrics."""
  if callable(metric_finalizers):
    return tensorflow_computation.tf_computation(
        local_unfinalized_metrics_type
    )(metric_finalizers)
  metric_finalizers = typing.cast(types.MetricFinalizersType, metric_finalizers)

  @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
  def finalizer_computation(unfinalized_metrics):
    finalized_metrics = collections.OrderedDict()
    for metric_name, metric_finalizer in metric_finalizers.items():
      finalized_metrics[metric_name] = metric_finalizer(
          unfinalized_metrics[metric_name]
      )
    return finalized_metrics

  return finalizer_computation


def _intialize_unfinalized_metrics_accumulators(
    local_unfinalized_metrics_type, initial_unfinalized_metrics
):
  """Initalizes the unfinalized metrics accumulators."""
  if initial_unfinalized_metrics is not None:
    initial_unfinalized_metrics_type = type_conversions.type_from_tensors(
        initial_unfinalized_metrics
    )
    if initial_unfinalized_metrics_type != local_unfinalized_metrics_type:
      raise TypeError(
          "The initial unfinalized metrics type doesn't match "
          'with the `local_unfinalized_metrics_type`, expect: '
          f'{local_unfinalized_metrics_type}, found: '
          f'{initial_unfinalized_metrics_type}.'
      )

  @tensorflow_computation.tf_computation
  def create_all_zero_state():
    return type_conversions.structure_from_tensor_type_tree(
        lambda t: tf.zeros(shape=t.shape, dtype=t.dtype),
        local_unfinalized_metrics_type,
    )

  if initial_unfinalized_metrics is None:
    return intrinsics.federated_eval(create_all_zero_state, placements.SERVER)

  return intrinsics.federated_value(
      initial_unfinalized_metrics, placements.SERVER
  )


# TODO(b/227811468): Support other inner aggregators for SecAgg and DP.
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
        a `tff.tf_computation` decoreated TFF Computation) that takes in a
        metric's unfinalized values, and returns the finalized metric values.
        This can be obtained from `tff.learning.Model.metric_finalizers()`.
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
        unfinalized metrics. Let `local_unfinalized_metrics` be the output of
        `tff.learning.Model.report_local_unfinalized_metrics()`, its type can be
        obtained by
        `tff.types.type_from_tensors(local_unfinalized_metrics)`.

    Returns:
      An instance of `tff.templates.AggregationProcess`.

    Raises:
      TypeError: If any argument type mismatches; if the metric finalizers
        mismatch the type of local unfinalized metrics; if the initial
        unfinalized metrics mismatch the type of local unfinalized metrics.
    """
    aggregation_utils.check_local_unfinalzied_metrics_type(
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
          _intialize_unfinalized_metrics_accumulators(
              local_unfinalized_metrics_type, self._initial_unfinalized_metrics
          )
      )
      return intrinsics.federated_zip((
          inner_summation_process.initialize(),
          unfinalized_metrics_accumulators,
      ))

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.at_clients(local_unfinalized_metrics_type),
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

      finalizer_computation = _build_finalizer_computation(
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


# TODO(b/233054212): re-enable lint
def create_default_secure_sum_quantization_ranges(
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
    lower_bound: Union[int, float] = DEFAULT_FIXED_SECURE_LOWER_BOUND,
    upper_bound: Union[int, float] = DEFAULT_FIXED_SECURE_UPPER_BOUND,
    use_auto_tuned_bounds_for_float_values: Optional[bool] = True,
) -> MetricValueRangeDict:  # pylint: disable=g-bare-generic
  """Create a nested structure of quantization ranges for secure sum encoding.

  Args:
    local_unfinalized_metrics_type: The `tff.Type` structure to generate default
      secure sum quantization ranges form. Must be a `tff.Type` tree containing
      only `tff.TensorType` and `tff.StructType`. Each `tff.TensorType` must be
      of floating point or integer dtype.
    lower_bound: An optional integer or floating point lower bound for the
      secure sum quantization range. Values smaller than this will be clipped to
      this value. By default is `0`. If a `float`, any `tff.TensorType` in
      `local_unfinalized_metrics_type` with an integer dtype will use
      `math.ceil(lower_bound)` as a bound.
    upper_bound: An optional integer or floating point upper bound for the
      secure sum quantization range. Values larger than this will be clipped to
      this value. By default is `2^20 - 1` (~1 million). If a `float`, any
      `tff.TensorType` in `local_unfinalized_metrics_type` with an integer dtype
      will use `math.floor(lower_bound)` as a bound.
    use_auto_tuned_bounds_for_float_values: An optional boolean for specifying
      whether to use auto-tuned bounds for float values. If True, a default
      `tff.templates.EstimationProcess` is used for `upper_bound`, and the
      `lower_bound` is None to allow `tff.aggregators.SecureSumFactory` to
      determine the `lower_bound`.

  Returns:
    A nested structure matching the structure of
    `local_unfinalized_metrics_type` where each `tf.TensorType` has been
    replaced with a 2-tuple of lower bound and upper bound, where the tuple
    can be (`float`, `float`) or (None, `tff.templates.EstimationProcess`) for
    floating dtypes, and (`int`, `int`) for integer dtypes.

  Raises:
    UnquantizableDTypeError: If A `tff.TensorType` in
      `local_unfinalized_metrics_type` has a non-float or non-integer dtype.
    ValueError: If an integer dtype in `local_unfinalized_metrics_type` will
      have a zero range (e.g. `math.ceil(lower_bound) - math.floor(upper_bound)
      < 1`).
  """
  py_typecheck.check_type(upper_bound, (int, float))
  py_typecheck.check_type(lower_bound, (int, float))
  if lower_bound >= upper_bound:
    raise ValueError('`upper_bound` must be greater than `lower_bound`.')
  integer_range_width = math.floor(upper_bound) - math.ceil(lower_bound)

  auto_tuned_float_upper_bound = (
      quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
          initial_estimate=50.0,
          target_quantile=0.95,
          learning_rate=1.0,
          multiplier=2.0,
          secure_estimation=True,
      )
  )

  def create_default_range(
      type_spec: computation_types.TensorType,
  ) -> MetricValueRange:
    if type_spec.dtype.is_floating:
      if use_auto_tuned_bounds_for_float_values:
        return None, auto_tuned_float_upper_bound
      return float(lower_bound), float(upper_bound)
    elif type_spec.dtype.is_integer:
      if integer_range_width < 1:
        raise ValueError(
            'Encounter an integer tensor in the type, but quantization range '
            f'[{lower_bound}, {upper_bound}] is not wide enough to quantize '
            f'any integers (becomes [{int(lower_bound)}, {int(upper_bound)}]).'
        )
      return math.ceil(lower_bound), math.floor(upper_bound)
    else:
      raise UnquantizableDTypeError(
          'Do not know how to create a default range for dtype '
          f'{type_spec.dtype}. Only floating or integer types are supported.'
      )

  return type_conversions.structure_from_tensor_type_tree(
      create_default_range, local_unfinalized_metrics_type
  )


# TODO(b/233054212): re-enable lint
# pylint: disable=g-bare-generic
def fill_missing_values_with_defaults(
    default_values: MetricValueRangeDict, user_values: UserMetricValueRangeDict
) -> MetricValueRangeDict:
  # pylint: enable=g-bare-generic
  """Fill missing user provided metric value ranges with default ranges.

  Args:
    default_values: Default metric value ranges.
    user_values: User provided metric value ranges.

  Returns:
    A `MetricValueRangeDict` with all metric value ranges filled.

  Raises:
    TypeError: If the user value is not a `_MetricRange` or a `tuple` or its
      elements are not allowed types of `MetricValueRange`.
    ValueError: If the value has length other than two.
  """
  if isinstance(default_values, Mapping):
    if user_values is None:
      user_values = {}
    filled_with_defaults_values = []
    for key, default_value in default_values.items():
      filled_with_defaults_values.append((
          key,
          fill_missing_values_with_defaults(
              default_value, user_values.get(key)
          ),
      ))
    return type(default_values)(filled_with_defaults_values)
  elif isinstance(default_values, list):
    if user_values is None:
      user_values = [None] * len(default_values)
    return [
        fill_missing_values_with_defaults(default_value, user_values[idx])
        for idx, default_value in enumerate(default_values)
    ]
  elif user_values is None:
    return _MetricRange(*default_values)
  else:
    if isinstance(user_values, _MetricRange):
      return user_values
    _check_user_metric_value_range(user_values)
    return _MetricRange(*user_values)


# Define a delimiter that is used to generate a string key of a inner secure
# summation factory.
_DELIMITER = '/'


# TODO(b/222112465): Avoid converting floats to strings as it may cause problem.
# Helper function for factory keys used in secure summation.
# A factory key is uniquely defined by three values: lower bound, upper bound,
# and tensor dtype. In secure summation, we will create a aggregation process
# for each factory key. Metric values sharing the same factory key will be
# aggregated together.
def create_factory_key(
    lower: MetricValueLowerBoundType,
    upper: MetricValueUpperBoundType,
    tensor_dtype: tf.dtypes.DType,
) -> str:
  """Creates a string key for a `tff.aggregators.SecureSumFactory`."""
  # The `tff.templates.EstimationProcess` are only used as the default upper
  # bound for float values, so replace it as a fixed string.
  if isinstance(upper, estimation_process.EstimationProcess):
    upper = 'default_estimation_process'
  return _DELIMITER.join(
      str(item) for item in [lower, upper, tensor_dtype.as_datatype_enum]
  )


class SecureSumFactory(factory.UnweightedAggregationFactory):
  """Aggregation Factory that performs secure summation over metrics.

  The created `tff.templates.AggregationProcess` uses the inner summation
  processes created by the `tff.aggregators.SecureSumFactory` to sum unfinalized
  metrics from `tff.CLIENTS` to `tff.SERVER`.

  Internally metrics are grouped by their value range and dtype, and only one
  secure aggregation process will be created for each group. This is an
  optimization for computation tracing and compiling, which can be slow
  when there are a large number of independent aggregations.

  The `initialize` function initializes the `state` for each inner secure
  aggregation progress. The `next` function takes the `state` and local
  unfinalized metrics reported from `tff.CLIENTS`, and returns a
  `tff.templates.MeasuredProcessOutput` object with the following properties:
    - `state`: an `collections.OrderedDict` of the `state`s of the inner secure
      aggregation processes.
    - `result`: an `collections.OrderedDict` of secure summed unfinalized
      metrics.
    - `measurements`: an `collections.OrderedDict` of the measurements of inner
      secure aggregation processes.
  """

  # TODO(b/233054212): re-enable lint
  # pylint: disable=g-bare-generic
  def __init__(
      self, metric_value_ranges: Optional[UserMetricValueRangeDict] = None
  ):
    # pylint: enable=g-bare-generic
    """Initializes `SecureSumFactory`.

    Since secure summation works in fixed-point arithmetic space, floating point
    numbers must be encoding using integer quantization. By default, each
    integer tensor in `local_unfinalized_metrics_type` will be clipped to
    `[0, 2**20 - 1]`, and each float tensor will be clipped using an auto-tuned
    range. Callers can change this range by setting `metric_value_ranges`, which
    may be a partial tree matching the structure of
    `local_unfinalized_metrics_type`.

    Example partial value range specification:

    >>> metrics_type = tff.to_type(collections.OrderedDict(
        a=tff.types.TensorType(tf.int32),
        b=tff.types.TensorType(tf.float32),
        c=[tff.types.TensorType(tf.float32), tff.types.TensorType(tf.float32)])
    >>> value_ranges = collections.OrderedDict(
        b=(0.0, 1.0),
        c=[None, (0.0, 1.0)])

    This sets the range of `b` and the *second* tensor of `c` in the dictionary,
    using the default range for `a` and the *first* tensor of `c`.

    Args:
      metric_value_ranges: An optional `collections.OrderedDict` that matches
        the structure of `local_unfinalized_metrics_type` (a value for each
        `tff.types.TensorType` in the type tree). Each leaf in the tree should
        have a 2-tuple that defines the range of expected values for that
        variable in the metric. If the entire structure is `None`, a default
        range of `[0.0, 2.0**20 - 1]` will be applied to integer variables and
        auto-tuned bounds will be applied to float variable. Each leaf may also
        be `None`, which will also get the default range according to the
        variable value type; allowing partial user sepcialization. At runtime,
        values that fall outside the ranges specified at the leaves will be
        clipped to within the range.

    Raises:
      TypeError: If `metric_value_ranges` type mismatches.
    """
    if metric_value_ranges is not None:
      py_typecheck.check_type(metric_value_ranges, collections.OrderedDict)
    self._metric_value_ranges = metric_value_ranges

  def create(
      self,
      local_unfinalized_metrics_type: computation_types.StructWithPythonType,
  ) -> aggregation_process.AggregationProcess:
    """Creates an `AggregationProcess` for secure summation over metrics.

    Args:
      local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
        `collections.OrderedDict` as the Python container) of a client's local
        unfinalized metrics. Let `local_unfinalized_metrics` be the output of
        `tff.learning.Model.report_local_unfinalized_metrics()`, its type can be
        obtained by
        `tff.types.type_from_tensors(local_unfinalized_metrics)`.

    Returns:
      An instance of `tff.templates.AggregationProcess`.

    Raises:
      TypeError: If any argument type mismatches.
    """
    aggregation_utils.check_local_unfinalzied_metrics_type(
        local_unfinalized_metrics_type
    )

    default_metric_value_ranges = create_default_secure_sum_quantization_ranges(
        local_unfinalized_metrics_type
    )

    # Walk the incoming `metric_value_ranges` and `default_metric_value_ranges`
    # and fill in any missing ranges using the defaults.
    try:
      metric_value_ranges = fill_missing_values_with_defaults(
          default_metric_value_ranges, self._metric_value_ranges
      )
    except TypeError as e:
      raise TypeError(
          'Failed to create encoding value range from: '
          f'{self._metric_value_ranges}'
      ) from e

    # Create an aggregator factory for each unique value range, rather than each
    # leaf tensor (which could introduce a lot of duplication).
    aggregator_factories = {
        value_range: secure.SecureSumFactory(
            value_range.upper, value_range.lower
        )
        for value_range in set(tree.flatten(metric_value_ranges))
    }
    # Construct a python container of `tff.TensorType` so we can traverse it in
    # parallel with the value ranges during AggregationProcess construction.
    # Otherwise we have a `tff.Type` but `metric_value_ranges` is a Python
    # container which are difficult to traverse in parallel.
    structure_of_tensor_types = (
        type_conversions.structure_from_tensor_type_tree(
            lambda t: t, local_unfinalized_metrics_type
        )
    )

    # We will construct groups of tensors with the same dtype and quantization
    # value range so that we can construct fewer aggregations-of-structures,
    # rather than a large structure-of-aggregations. Without this, the TFF
    # compiler pipeline results in large slow downs (see b/218312198).
    factory_key_by_path = collections.OrderedDict()
    value_range_by_factory_key = collections.OrderedDict()
    path_list_by_factory_key = collections.defaultdict(list)
    # Maintain a flattened list of paths. This is useful to flatten the
    # aggregated values, which will then be used by `tf.nest.pack_sequence_as`.
    flattened_path_list = []
    for (path, tensor_spec), (_, value_range) in zip(
        tree.flatten_with_path(structure_of_tensor_types),
        tree.flatten_with_path(metric_value_ranges),
    ):
      factory_key = create_factory_key(
          value_range.lower, value_range.upper, tensor_spec.dtype
      )
      factory_key_by_path[path] = factory_key
      value_range_by_factory_key[factory_key] = value_range
      path_list_by_factory_key[factory_key].append(path)
      flattened_path_list.append(path)

    @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
    def group_value_by_factory_key(local_unfinalized_metrics):
      """Groups client local metrics into a map of `factory_key` to value list."""
      # We cannot use `collections.defaultdict(list)` here because its result is
      # incompatible with `structure_from_tensor_type_tree`.
      value_list_by_factory_key = collections.OrderedDict()
      for path, value in tree.flatten_with_path(local_unfinalized_metrics):
        factory_key = factory_key_by_path[path]
        if factory_key in value_list_by_factory_key:
          value_list_by_factory_key[factory_key].append(value)
        else:
          value_list_by_factory_key[factory_key] = [value]
      return value_list_by_factory_key

    def flatten_grouped_values(value_list_by_factory_key):
      """Flatten the values in the same order as in `flattened_path_list`."""
      value_by_path = collections.OrderedDict()
      for factory_key in value_list_by_factory_key:
        path_list = path_list_by_factory_key[factory_key]
        value_list = value_list_by_factory_key[factory_key]
        for path, value in zip(path_list, value_list):
          value_by_path[path] = value
      flattened_value_list = [
          value_by_path[path] for path in flattened_path_list
      ]
      return flattened_value_list

    # Create an aggregation process for each factory key.
    aggregation_process_by_factory_key = collections.OrderedDict()
    # Construct a python container of `tff.TensorType` so we can traverse it and
    # create aggregation processes from the factories.
    tensor_type_list_by_factory_key = (
        type_conversions.structure_from_tensor_type_tree(
            lambda t: t, group_value_by_factory_key.type_signature.result
        )
    )
    for (
        factory_key,
        tensor_type_list,
    ) in tensor_type_list_by_factory_key.items():
      value_range = value_range_by_factory_key[factory_key]
      aggregation_process_by_factory_key[
          factory_key
      ] = aggregator_factories.get(value_range).create(
          computation_types.to_type(tensor_type_list)
      )  # pytype: disable=attribute-error

    @federated_computation.federated_computation
    def init_fn():
      factory_init_states = collections.OrderedDict()
      for factory_key, process in aggregation_process_by_factory_key.items():
        factory_init_states[factory_key] = process.initialize()
      return intrinsics.federated_zip(factory_init_states)

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.at_clients(local_unfinalized_metrics_type),
    )
    def next_fn(state, client_local_unfinalized_metrics):
      client_local_grouped_unfinalized_metrics = intrinsics.federated_map(
          group_value_by_factory_key, client_local_unfinalized_metrics
      )
      metrics_aggregation_output = collections.OrderedDict()
      new_state = collections.OrderedDict()
      for factory_key, process in aggregation_process_by_factory_key.items():
        metrics_aggregation_output[factory_key] = process.next(
            state[factory_key],
            client_local_grouped_unfinalized_metrics[factory_key],
        )
        new_state[factory_key] = metrics_aggregation_output[factory_key].state

      metrics_aggregation_output = intrinsics.federated_zip(
          metrics_aggregation_output
      )

      @tensorflow_computation.tf_computation(
          metrics_aggregation_output.type_signature.member
      )
      def flatten_aggregation_output(grouped_aggregation_output):
        secure_sum_measurements = collections.OrderedDict(
            (factory_key, output.measurements)
            for factory_key, output in grouped_aggregation_output.items()
        )
        grouped_unfinalized_metrics = collections.OrderedDict(
            (factory_key, output.result)
            for factory_key, output in grouped_aggregation_output.items()
        )
        flattened_unfinalized_metrics_list = flatten_grouped_values(
            grouped_unfinalized_metrics
        )
        unfinalized_metrics = tf.nest.pack_sequence_as(
            structure_of_tensor_types, flattened_unfinalized_metrics_list
        )
        return unfinalized_metrics, secure_sum_measurements

      unfinalized_metrics, secure_sum_measurements = intrinsics.federated_map(
          flatten_aggregation_output, metrics_aggregation_output
      )

      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_zip(new_state),
          result=unfinalized_metrics,
          measurements=secure_sum_measurements,
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)
