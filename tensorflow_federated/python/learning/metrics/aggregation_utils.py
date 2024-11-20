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
"""Utility methods for metrics aggregation."""

import collections
import typing
from typing import Union

import federated_language

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.learning.metrics import types


def check_finalizers_matches_unfinalized_metrics(
    metric_finalizers: types.MetricFinalizersType,
    local_unfinalized_metrics_type: federated_language.StructWithPythonType,
):
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
      structure.name_list(local_unfinalized_metrics_type)
  )
  if (
      metric_names_in_metric_finalizers
      != metric_names_in_local_unfinalized_metrics
  ):
    difference_1 = (
        metric_names_in_metric_finalizers
        - metric_names_in_local_unfinalized_metrics
    )
    difference_2 = (
        metric_names_in_local_unfinalized_metrics
        - metric_names_in_metric_finalizers
    )
    raise ValueError(
        'The metric names in `metric_finalizers` do not match those in the '
        '`local_unfinalized_metrics`. Metric names in the `metric_finalizers`'
        f'but not the `local_unfinalized_metrics`: {difference_1}. '
        'Metric names in the `local_unfinalized_metrics` but not the '
        f'`metric_finalizers`: {difference_2}.\n'
        'Metrics names in the `metric_finalizers`: '
        f'{metric_names_in_metric_finalizers}. Metric names in the '
        '`local_unfinalized_metrics`: '
        f'{metric_names_in_local_unfinalized_metrics}.'
    )


def check_metric_finalizers(
    metric_finalizers: Union[
        types.MetricFinalizersType,
        types.FunctionalMetricFinalizersType,
    ]
):
  """Validates `metric_finalizers` raising error on failure.

  Args:
    metric_finalizers: The finalizers to validate.

  Raises:
    TypeError: If `metric_finalizers` is not a `collections.OrderedDict` or
      any key is not a `str` type, or value is not a `callable`.
  """
  if not callable(metric_finalizers):
    py_typecheck.check_type(
        metric_finalizers, collections.OrderedDict, 'metric_finalizers'
    )
    for key, _ in metric_finalizers.items():
      py_typecheck.check_type(key, str, f'metric_finalizers key {key}')


def check_local_unfinalized_metrics_type(
    local_unfinalized_metrics_type: federated_language.StructWithPythonType,
):
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
  if not isinstance(
      local_unfinalized_metrics_type, federated_language.StructWithPythonType
  ):
    raise TypeError(
        'Expected the input `local_unfinalized_metrics_type` to be a '
        '`tff.types.StructWithPythonType`, found '
        f'{py_typecheck.type_string(type(local_unfinalized_metrics_type))}.'
    )
  local_metrics_container = local_unfinalized_metrics_type.python_container
  if local_metrics_container is not collections.OrderedDict:
    raise TypeError(
        'Expected the input `local_unfinalized_metrics_type` to be a '
        '`tff.types.StructWithPythonType` with `collections.OrderedDict` as '
        'the Python container, found a `tff.types.StructWithPythonType` with '
        f'Python container {py_typecheck.type_string(local_metrics_container)}.'
    )


def build_finalizer_computation(
    metric_finalizers: Union[
        types.MetricFinalizersType,
        types.FunctionalMetricFinalizersType,
    ],
    local_unfinalized_metrics_type: federated_language.StructWithPythonType,
) -> federated_language.framework.Computation:
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
