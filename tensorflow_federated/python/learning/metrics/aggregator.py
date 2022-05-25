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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Library of common metric aggregators."""

import collections
import dataclasses
import math
from typing import Optional, OrderedDict, Tuple, Union

import tensorflow as tf
import tree

from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning import model as model_lib


class InternalError(Exception):
  """An error internal to TFF. File a bug report."""


def _check_finalizers_matches_unfinalized_metrics(
    metric_finalizers: model_lib.MetricFinalizersType,
    local_unfinalized_metrics_type: computation_types.StructWithPythonType):
  """Verifies that compatibility of variables and finalizers.

  Args:
    metric_finalizers: The finalizers to validate.
    local_unfinalized_metrics_type: The unfinalized metrics type to validate.

  Raises:
    ValueError: If `metric_finalizers` cannot finalize a variable structure
      with type `local_unfinalized_metrics_type`.
  """
  metric_names_in_metric_finalizers = set(metric_finalizers.keys())
  metric_names_in_local_unfinalized_metrics = set(
      structure.name_list(local_unfinalized_metrics_type))
  if (metric_names_in_metric_finalizers !=
      metric_names_in_local_unfinalized_metrics):
    difference_1 = (
        metric_names_in_metric_finalizers -
        metric_names_in_local_unfinalized_metrics)
    difference_2 = (
        metric_names_in_local_unfinalized_metrics -
        metric_names_in_metric_finalizers)
    raise ValueError(
        'The metric names in `metric_finalizers` do not match those in the '
        '`local_unfinalized_metrics`. Metric names in the `metric_finalizers`'
        f'but not the `local_unfinalized_metrics`: {difference_1}. '
        'Metric names in the `local_unfinalized_metrics` but not the '
        f'`metric_finalizers`: {difference_2}.\n'
        'Metrics names in the `metric_finalizers`: '
        f'{metric_names_in_metric_finalizers}. Metric names in the '
        '`local_unfinalized_metrics`: '
        f'{metric_names_in_local_unfinalized_metrics}.')


def _check_metric_finalizers(metric_finalizers: model_lib.MetricFinalizersType):
  """Validates `metric_finalizers` raising error on failure.

  Args:
    metric_finalizers: The finalizers to validate.

  Raises:
    TypeError: If `metric_finalizers` is not a `collections.OrderedDict` or
      any key is not a `str` type, or value is not a `callable`.
  """
  py_typecheck.check_type(metric_finalizers, collections.OrderedDict,
                          'metric_finalizers')
  for key, value in metric_finalizers.items():
    py_typecheck.check_type(key, str, f'metric_finalizers key {key}')
    py_typecheck.check_callable(value, f'metric_finalizers value {value}')


def _check_local_unfinalzied_metrics_type(
    local_unfinalized_metrics_type: computation_types.StructWithPythonType):
  """Validates `local_unfinalized_metrics_type` raising error on failure.

  Args:
    local_unfinalized_metrics_type: The unfinalized metrics type to validate.

  Raises:
    TypeError: If `local_unfinalized_metrics_type` is not a
      `tff.types.StructWithPythonType` or has a `.container` attribute that is
      not the `collections.OrderedDict` type.
  """
  # Directly check the type (instead of using `py_typecheck`) here so that the
  # the error message has a better format (specifically, the expected type is
  # shown as `tff.types.StructWithPythonType` in the error message).
  if not isinstance(local_unfinalized_metrics_type,
                    computation_types.StructWithPythonType):
    raise TypeError(
        'Expected the input `local_unfinalized_metrics_type` to be a '
        '`tff.types.StructWithPythonType`, found '
        f'{py_typecheck.type_string(type(local_unfinalized_metrics_type))}.')
  local_metrics_container = local_unfinalized_metrics_type.python_container
  if local_metrics_container is not collections.OrderedDict:
    raise TypeError(
        'Expected the input `local_unfinalized_metrics_type` to be a '
        '`tff.types.StructWithPythonType` with `collections.OrderedDict` as '
        'the Python container, found a `tff.types.StructWithPythonType` with '
        f'Python container {py_typecheck.type_string(local_metrics_container)}.'
    )


def sum_then_finalize(
    metric_finalizers: model_lib.MetricFinalizersType,
    local_unfinalized_metrics_type: computation_types.StructWithPythonType
) -> computation_base.Computation:
  """Creates a TFF computation that aggregates metrics via `sum_then_finalize`.

  The returned federated TFF computation has the following type signature:
  `local_unfinalized_metrics@CLIENTS -> aggregated_metrics@SERVER`, where the
  input is given by `tff.learning.Model.report_local_unfinalized_metrics()` at
  `CLIENTS`, and the output is computed by first summing the unfinalized metrics
  from `CLIENTS`, followed by applying the finalizers at `SERVER`.

  Args:
    metric_finalizers: An `OrderedDict` of `string` metric names to finalizer
      functions returned by `tff.learning.Model.metric_finalizers()`. It should
      have the same keys (i.e., metric names) as the `OrderedDict` returned by
      `tff.learning.Model.report_local_unfinalized_metrics()`. A finalizer is a
      callable (typically `tf.function` or `tff.tf_computation` decoreated
      function) that takes in a metric's unfinalized values, and returns the
      finalized values.
    local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
      `OrderedDict` as the Python container) of a client's local unfinalized
      metrics. Let `local_unfinalized_metrics` be the output of
      `tff.learning.Model.report_local_unfinalized_metrics()`. Its type can be
      obtained by `tff.framework.type_from_tensors(local_unfinalized_metrics)`.

  Returns:
    A federated TFF computation that sums the unfinalized metrics from
    `CLIENTS`, and applies the correponding finalizers at `SERVER`.

  Raises:
    TypeError: If the inputs are of the wrong types.
    ValueError: If the keys (i.e., metric names) in `metric_finalizers` are not
      the same as those expected by `local_unfinalized_metrics_type`.
  """
  _check_metric_finalizers(metric_finalizers)
  _check_local_unfinalzied_metrics_type(local_unfinalized_metrics_type)
  _check_finalizers_matches_unfinalized_metrics(metric_finalizers,
                                                local_unfinalized_metrics_type)

  @federated_computation.federated_computation(
      computation_types.at_clients(local_unfinalized_metrics_type))
  def aggregator_computation(client_local_unfinalized_metrics):
    unfinalized_metrics_sum = intrinsics.federated_sum(
        client_local_unfinalized_metrics)

    @tensorflow_computation.tf_computation(local_unfinalized_metrics_type)
    def finalizer_computation(unfinalized_metrics):
      finalized_metrics = collections.OrderedDict()
      for metric_name, metric_finalizer in metric_finalizers.items():
        finalized_metrics[metric_name] = metric_finalizer(
            unfinalized_metrics[metric_name])
      return finalized_metrics

    return intrinsics.federated_map(finalizer_computation,
                                    unfinalized_metrics_sum)

  return aggregator_computation


MetricValueRange = Union[Tuple[float, float], Tuple[int, int]]
MetricValueRangeDict = OrderedDict[str, Union[MetricValueRange,
                                              'MetricValueRangeDict']]


def _check_range(user_value: MetricValueRange):
  """Validates a value is a valid range.

  Args:
    user_value: The value to validate.

  Raises:
    TypeError: If `user_value` is not a `tuple` or its elements are not `int` or
      `float` type.
    ValueError: If `user_value` has length other than two.
  """
  py_typecheck.check_type(user_value, tuple)
  if len(user_value) != 2:
    raise ValueError('Ranges must be defined as a 2-tuple, got a tuple of '
                     f'length {len(user_value)}.')
  for element in user_value:
    py_typecheck.check_type(element, (int, float))


@dataclasses.dataclass(frozen=True)
class _MetricRange:
  """An opaque structure defining a closed range.

  This is used as an opaque object in a nested structure to prevent
  `tree.map_structure` from traversing to the numeric leaves.
  """
  lower: Union[int, float]
  upper: Union[int, float]

  def __eq__(self, other):
    """A type-aware equality that prevents int/float conversion."""
    return (type(self.upper) is type(other.upper) and
            self.upper == other.upper and
            type(self.lower) is type(other.lower) and self.lower == other.lower)


class UnquantizableDTypeError(Exception):
  """An error raised when a tensor dtype is not quantizable."""


DEFAULT_SECURE_LOWER_BOUND = 0
# Use a power of 2 minus one to more accurately encode floating dtypes that
# actually contain integer values. 2 ^ 20 gives us approximately a range of
# [0, 1 million].
DEFAULT_SECURE_UPPER_BOUND = 2**20 - 1


def create_default_secure_sum_quantization_ranges(
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
    lower_bound: Union[int, float] = DEFAULT_SECURE_LOWER_BOUND,
    upper_bound: Union[int, float] = DEFAULT_SECURE_UPPER_BOUND
) -> MetricValueRangeDict:
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

  Returns:
    A nested structure matching the structure of
    `local_unfinalized_metrics_type` where each `tf.TensorType` has been
    replaced with a 2-tuple of lower bound and upper bound, where the tupel
    elements are `float` for floating dtypes, and `int` for integer dtypes.

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

  def create_default_range(
      type_spec: computation_types.TensorType) -> MetricValueRange:
    if type_spec.dtype.is_floating:
      return float(lower_bound), float(upper_bound)
    elif type_spec.dtype.is_integer:
      if integer_range_width < 1:
        raise ValueError(
            'Encounter an integer tensor in the type, but quantization range '
            f'[{lower_bound}, {upper_bound}] is not wide enough to quantize '
            f'any integers (becomes [{int(lower_bound)}, {int(upper_bound)}]).')
      return math.ceil(lower_bound), math.floor(upper_bound)
    else:
      raise UnquantizableDTypeError(
          'Do not know how to create a default range for dtype '
          f'{type_spec.dtype}. Only floating or integer types are supported.')

  return type_conversions.structure_from_tensor_type_tree(
      create_default_range, local_unfinalized_metrics_type)


# Helper functions for factory keys used in `secure_sum_then_finalize`.
# A factory key is uniquely defined by three values: lower bound, upper bound,
# and tensor dtype. In `secure_sum_then_finalize`, we will create a aggregation
# process for each factory key. Metric values sharing the same factory key will
# be aggregated together.
_DELIMITER = '/'


# TODO(b/222112465): Avoid converting floats to strings as it may cause problem.
def _create_factory_key(lower: Union[int, float], upper: Union[int, float],
                        tensor_dtype: tf.dtypes.DType) -> str:
  return _DELIMITER.join(
      str(item) for item in [lower, upper, tensor_dtype.as_datatype_enum])


def secure_sum_then_finalize(
    metric_finalizers: model_lib.MetricFinalizersType,
    local_unfinalized_metrics_type: computation_types.StructWithPythonType,
    metric_value_ranges: Optional[MetricValueRangeDict] = None
) -> computation_base.Computation:
  """Creates a TFF computation that aggregates metrics using secure summation.

  The returned federated TFF computation has the following type signature:

  ```
  (local_unfinalized_metrics@CLIENTS ->
   <aggregated_metrics@SERVER, secure_sum_measurements@SERVER)
  ```

  where the input is given by
  `tff.learning.Model.report_local_unfinalized_metrics()` at `CLIENTS`, and the
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
      a=tff.types.TensorType(tf.int32),
      b=tff.types.TensorType(tf.float32),
      c=[tff.types.TensorType(tf.float32), tff.types.TensorType(tf.float32)])
  >>> value_ranges = collections.OrderedDict(
      b=(0.0, 1.0),
      c=[None, (0.0, 1.0)])
  >>> aggregator = tff.learning.metrics.secure_sum_then_finalize(
      finalizers, metrics_type, value_ranges)

  This sets the range of the *second* tensor of `b` in the dictionary, using the
  range for the first tensor, and the `a` tensor.

  Args:
    metric_finalizers: An `OrderedDict` of `string` metric names to finalizer
      functions returned by `tff.learning.Model.metric_finalizers()`. It should
      have the same keys (i.e., metric names) as the `OrderedDict` returned by
      `tff.learning.Model.report_local_unfinalized_metrics()`. A finalizer is a
      callable (typically `tf.function` or `tff.tf_computation` decoreated
      function) that takes in a metric's unfinalized values, and returns the
      finalized values.
    local_unfinalized_metrics_type: A `tff.types.StructWithPythonType` (with
      `OrderedDict` as the Python container) of a client's local unfinalized
      metrics. Let `local_unfinalized_metrics` be the output of
      `tff.learning.Model.report_local_unfinalized_metrics()`. Its type can be
      obtained by `tff.framework.type_from_tensors(local_unfinalized_metrics)`.
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
  _check_metric_finalizers(metric_finalizers)
  _check_local_unfinalzied_metrics_type(local_unfinalized_metrics_type)
  _check_finalizers_matches_unfinalized_metrics(metric_finalizers,
                                                local_unfinalized_metrics_type)

  default_metric_value_ranges = create_default_secure_sum_quantization_ranges(
      local_unfinalized_metrics_type)
  if metric_value_ranges is None:
    metric_value_ranges = default_metric_value_ranges

  # Walk the incoming `metric_value_ranges` and `default_metric_value_ranges`
  # and fill in any missing ranges using the defaults.
  def fill_missing_values_with_defaults(default_values, user_values):
    if isinstance(default_values, collections.abc.Mapping):
      if user_values is None:
        user_values = {}
      return type(default_values)(
          (key,
           fill_missing_values_with_defaults(default_value, user_values.get(
               key))) for key, default_value in default_values.items())
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
      _check_range(user_values)
      return _MetricRange(*user_values)

  try:
    metric_value_ranges = fill_missing_values_with_defaults(
        default_metric_value_ranges, metric_value_ranges)
  except TypeError as e:
    raise TypeError('Failed to create encoding value range from: '
                    f'{metric_value_ranges}') from e

  # Create an aggregator factory for each unique value range, rather than each
  # leaf tensor (which could introduce a lot of duplication).
  aggregator_factories = {
      value_range: secure.SecureSumFactory(value_range.upper, value_range.lower)
      for value_range in set(tree.flatten(metric_value_ranges))
  }
  # Construct a python container of `tff.TensorType` so we can traverse it in
  # parallel with the value ranges during AggregationProcess construction.
  # Otherwise we have a `tff.Type` but `metric_value_ranges` is a Python
  # container which are difficult to traverse in parallel.
  structure_of_tensor_types = type_conversions.structure_from_tensor_type_tree(
      lambda t: t, local_unfinalized_metrics_type)

  # We will construct groups of tensors with the same dtype and quantization
  # value range so that we can construct fewer aggregations-of-structures,
  # rather than a large structure-of-aggregations. Without this, the TFF
  # compiler pipeline results in large slow downs (see b/218312198).
  factory_key_by_path = collections.OrderedDict()
  value_range_by_factory_key = collections.OrderedDict()
  path_list_by_factory_key = collections.defaultdict(list)
  # Maintain a flattened list of paths. This is useful to flatten the aggregated
  # values, which will then be used by `tf.nest.pack_sequence_as`.
  flattened_path_list = []
  for (path, tensor_spec), (_, value_range) in zip(
      tree.flatten_with_path(structure_of_tensor_types),
      tree.flatten_with_path(metric_value_ranges)):
    factory_key = _create_factory_key(value_range.lower, value_range.upper,
                                      tensor_spec.dtype)
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
    flattened_value_list = [value_by_path[path] for path in flattened_path_list]
    return flattened_value_list

  # Create a aggregation process for each factory key.
  aggregation_process_by_factory_key = collections.OrderedDict()
  # Construct a python container of `tff.TensorType` so we can traverse it and
  # create aggregation processes from the factories.
  tensor_type_list_by_factory_key = (
      type_conversions.structure_from_tensor_type_tree(
          lambda t: t, group_value_by_factory_key.type_signature.result))
  for factory_key, tensor_type_list in tensor_type_list_by_factory_key.items():
    value_range = value_range_by_factory_key[factory_key]
    aggregation_process_by_factory_key[factory_key] = aggregator_factories.get(
        value_range).create(computation_types.to_type(tensor_type_list))

  @federated_computation.federated_computation(
      computation_types.at_clients(local_unfinalized_metrics_type))
  def aggregator_computation(client_local_unfinalized_metrics):
    unused_state = intrinsics.federated_value((), placements.SERVER)

    client_local_grouped_unfinalized_metrics = intrinsics.federated_map(
        group_value_by_factory_key, client_local_unfinalized_metrics)
    metrics_aggregation_output = collections.OrderedDict()
    for factory_key, process in aggregation_process_by_factory_key.items():
      metrics_aggregation_output[factory_key] = process.next(
          unused_state, client_local_grouped_unfinalized_metrics[factory_key])

    metrics_aggregation_output = intrinsics.federated_zip(
        metrics_aggregation_output)

    @tensorflow_computation.tf_computation(
        metrics_aggregation_output.type_signature.member)
    def finalizer_computation(grouped_aggregation_output):

      # One minor downside of grouping the aggregation processes is that the
      # SecAgg measurements (e.g., clipped_count) are computed at a group level
      # (a group means all metric values belonging to the same `factory_key`).
      secure_sum_measurements = collections.OrderedDict(
          (factory_key, output.measurements)
          for factory_key, output in grouped_aggregation_output.items())
      finalized_metrics = collections.OrderedDict(
          secure_sum_measurements=secure_sum_measurements)
      grouped_unfinalized_metrics = collections.OrderedDict(
          (factory_key, output.result)
          for factory_key, output in grouped_aggregation_output.items())
      flattened_unfinalized_metrics_list = flatten_grouped_values(
          grouped_unfinalized_metrics)
      unfinalized_metrics = tf.nest.pack_sequence_as(
          structure_of_tensor_types, flattened_unfinalized_metrics_list)
      for metric_name, metric_finalizer in metric_finalizers.items():
        finalized_metrics[metric_name] = metric_finalizer(
            unfinalized_metrics[metric_name])
      return finalized_metrics

    return intrinsics.federated_map(finalizer_computation,
                                    metrics_aggregation_output)

  return aggregator_computation
