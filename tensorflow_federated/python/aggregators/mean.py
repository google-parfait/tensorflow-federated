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
"""Factory for mean."""

import collections
from typing import Optional
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class MeanFactory(factory.WeightedAggregationFactory):
  """Aggregation factory for weighted mean.

  The created `tff.templates.AggregationProcess` computes the weighted mean of
  values placed at `CLIENTS`, and outputs the mean placed at `SERVER`.

  The input arguments of the `next` attribute of the process returned by
  `create` are `<state, value, weight>`, where `weight` is a scalar broadcasted
  to the structure of `value`, and the weighted mean refers to the expression
  `sum(value * weight) / sum(weight)`.

  The implementation is parameterized by two inner aggregation factories
  responsible for the summations above, with the following high-level steps.
    - Multiplication of `value` and `weight` at `CLIENTS`.
    - Delegation to inner `value_sum_factory` and `weight_sum_factory` to
      realize the sum of weighted values and weights.
    - Division of summed weighted values and summed weights at `SERVER`.

  Note that the the division at `SERVER` can protect against division by 0, as
  specified by `no_nan_division` constructor argument.

  The `state` is the composed `state` of the aggregation processes created by
  the two inner aggregation factories. The same holds for `measurements`.
  """

  def __init__(
      self,
      value_sum_factory: Optional[factory.UnweightedAggregationFactory] = None,
      weight_sum_factory: Optional[factory.UnweightedAggregationFactory] = None,
      no_nan_division: bool = False):
    """Initializes `MeanFactory`.

    Args:
      value_sum_factory: An optional
        `tff.aggregators.UnweightedAggregationFactory` responsible for summation
        of weighted values. If not specified, `tff.aggregators.SumFactory` is
        used.
      weight_sum_factory: An optional
        `tff.aggregators.UnweightedAggregationFactory` responsible for summation
        of weights. If not specified, `tff.aggregators.SumFactory` is used.
      no_nan_division: A bool. If True, the computed mean is 0 if sum of weights
        is equal to 0.

    Raises:
      TypeError: If provided `value_sum_factory` or `weight_sum_factory` is not
        an instance of `tff.aggregators.UnweightedAggregationFactory`.
    """
    if value_sum_factory is None:
      value_sum_factory = sum_factory.SumFactory()
    py_typecheck.check_type(value_sum_factory,
                            factory.UnweightedAggregationFactory)
    self._value_sum_factory = value_sum_factory

    if weight_sum_factory is None:
      weight_sum_factory = sum_factory.SumFactory()
    py_typecheck.check_type(weight_sum_factory,
                            factory.UnweightedAggregationFactory)
    self._weight_sum_factory = weight_sum_factory

    py_typecheck.check_type(no_nan_division, bool)
    self._no_nan_division = no_nan_division

  def create(
      self, value_type: factory.ValueType,
      weight_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    _check_value_type(value_type)
    py_typecheck.check_type(weight_type, factory.ValueType.__args__)

    value_sum_process = self._value_sum_factory.create(value_type)
    weight_sum_process = self._weight_sum_factory.create(weight_type)

    @computations.federated_computation()
    def init_fn():
      state = collections.OrderedDict(
          value_sum_process=value_sum_process.initialize(),
          weight_sum_process=weight_sum_process.initialize())
      return intrinsics.federated_zip(state)

    @computations.federated_computation(
        init_fn.type_signature.result,
        computation_types.FederatedType(value_type, placements.CLIENTS),
        computation_types.FederatedType(weight_type, placements.CLIENTS))
    def next_fn(state, value, weight):
      # Client computation.
      weighted_value = intrinsics.federated_map(_mul, (value, weight))

      # Inner aggregations.
      value_output = value_sum_process.next(state['value_sum_process'],
                                            weighted_value)
      weight_output = weight_sum_process.next(state['weight_sum_process'],
                                              weight)

      # Server computation.
      weighted_mean_value = intrinsics.federated_map(
          _div_no_nan if self._no_nan_division else _div,
          (value_output.result, weight_output.result))

      # Output preparation.
      state = collections.OrderedDict(
          value_sum_process=value_output.state,
          weight_sum_process=weight_output.state)
      measurements = collections.OrderedDict(
          mean_value=value_output.measurements,
          mean_weight=weight_output.measurements)
      return measured_process.MeasuredProcessOutput(
          intrinsics.federated_zip(state), weighted_mean_value,
          intrinsics.federated_zip(measurements))

    return aggregation_process.AggregationProcess(init_fn, next_fn)


class UnweightedMeanFactory(factory.UnweightedAggregationFactory):
  """Aggregation factory for unweighted mean.

  The created `tff.templates.AggregationProcess` computes the unweighted mean of
  values placed at `CLIENTS`, and outputs the mean placed at `SERVER`.

  The input arguments of the `next` attribute of the process returned by
  `create` are `<state, value>`, and the unweighted mean refers to the
  expression `sum(value * weight) / count(value)` where `count(value)` is the
  cardinality of the `CLIENTS` placement.

  The implementation is parameterized by an inner aggregation factory
  responsible for the summation of values.
  """

  def __init__(
      self,
      value_sum_factory: Optional[factory.UnweightedAggregationFactory] = None):
    """Initializes `UnweightedMeanFactory`.

    Args:
      value_sum_factory: An optional
        `tff.aggregators.UnweightedAggregationFactory` responsible for summation
        of values. If not specified, `tff.aggregators.SumFactory` is used.

    Raises:
      TypeError: If provided `value_sum_factory` is not an instance of
        `tff.aggregators.UnweightedAggregationFactory`.
    """
    if value_sum_factory is None:
      value_sum_factory = sum_factory.SumFactory()
    py_typecheck.check_type(value_sum_factory,
                            factory.UnweightedAggregationFactory)
    self._value_sum_factory = value_sum_factory

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    _check_value_type(value_type)
    value_sum_process = self._value_sum_factory.create(value_type)

    @computations.federated_computation()
    def init_fn():
      return value_sum_process.initialize()

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.FederatedType(
                                            value_type, placements.CLIENTS))
    def next_fn(state, value):
      value_sum_output = value_sum_process.next(state, value)
      count = intrinsics.federated_sum(
          intrinsics.federated_value(1, placements.CLIENTS))

      mean_value = intrinsics.federated_map(_div,
                                            (value_sum_output.result, count))
      state = value_sum_output.state
      measurements = intrinsics.federated_zip(
          collections.OrderedDict(mean_value=value_sum_output.measurements))

      return measured_process.MeasuredProcessOutput(state, mean_value,
                                                    measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _check_value_type(value_type):
  py_typecheck.check_type(value_type, factory.ValueType.__args__)
  if not type_analysis.is_structure_of_floats(value_type):
    raise TypeError(f'All values in provided value_type must be of floating '
                    f'dtype. Provided value_type: {value_type}')


@computations.tf_computation()
def _mul(value, weight):
  return tf.nest.map_structure(lambda x: x * tf.cast(weight, x.dtype), value)


@computations.tf_computation()
def _div(weighted_value_sum, weight_sum):
  return tf.nest.map_structure(
      lambda x: tf.math.divide(x, tf.cast(weight_sum, x.dtype)),
      weighted_value_sum)


@computations.tf_computation()
def _div_no_nan(weighted_value_sum, weight_sum):
  return tf.nest.map_structure(
      lambda x: tf.math.divide_no_nan(x, tf.cast(weight_sum, x.dtype)),
      weighted_value_sum)
