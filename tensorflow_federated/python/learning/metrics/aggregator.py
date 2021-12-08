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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import model as model_lib


# TODO(b/199278536): expand the API so that users can specifiy metrics
# aggregation using SecAgg.
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
    A federated TFF computation that sums the unfinalized metrics from CLIENTS,
    and applies the correponding finalizers at SERVER.

  Raises:
    TypeError: If the inputs are of the wrong types.
    ValueError: If the keys (i.e., metric names) in `metric_finalizers` are not
      the same as those expected by `local_unfinalized_metrics_type`.
  """
  py_typecheck.check_type(metric_finalizers, collections.OrderedDict,
                          'metric_finalizers')
  for key, value in metric_finalizers.items():
    py_typecheck.check_type(key, str, f'metric_finalizers key {key}')
    py_typecheck.check_callable(value, f'metric_finalizers value {value}')
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

  @computations.federated_computation(
      computation_types.at_clients(local_unfinalized_metrics_type))
  def aggregator_computation(client_local_unfinalized_metrics):
    unfinalized_metrics_sum = intrinsics.federated_sum(
        client_local_unfinalized_metrics)

    @computations.tf_computation(local_unfinalized_metrics_type)
    def finalizer_computation(unfinalized_metrics):
      finalized_metrics = collections.OrderedDict()
      for metric_name, metric_finalizer in metric_finalizers.items():
        finalized_metrics[metric_name] = metric_finalizer(
            unfinalized_metrics[metric_name])
      return finalized_metrics

    return intrinsics.federated_map(finalizer_computation,
                                    unfinalized_metrics_sum)

  return aggregator_computation
