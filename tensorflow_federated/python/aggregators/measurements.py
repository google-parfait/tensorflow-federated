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
"""Aggregation factory for adding custom measurements."""

import inspect

from typing import Any, Dict, Callable

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


def add_measurements(
    inner_agg_factory: factory.AggregationFactory,
    measurement_fn: Callable[..., Dict[str, Any]],
) -> factory.AggregationFactory:
  """Wraps `AggregationFactory` to report additional measurements.

  The function `measurement_fn` is a python callable that will be called on
  `value` (if `inner_agg_factory` is an `UnweightedAggregationFactory`) or
  `(value, weight)` (if `inner_agg_factory` is a `WeightedAggregationFactory`)
  in the `next` function of the `AggregationProcess` produced by the returned
  factory to generate additional measurements. It must be traceable by TFF and
  expect `tff.Value` objects placed at `CLIENTS` as inputs, and return
  `collections.OrderedDicts` mapping string names to tensor values placed at
  `SERVER`, which will be added to the measurement dict produced by the
  `inner_agg_factory`.

  Args:
    inner_agg_factory: The factory to wrap and add measurements.
    measurement_fn: A python callable that will be called on `value` (and/or
      `weight`) provided to the `next` function to compute additional
      measurements.

  Returns:
    An `AggregationFactory` that reports additional measurements.
  """
  py_typecheck.check_callable(measurement_fn)

  if isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):
    if len(inspect.signature(measurement_fn).parameters) != 1:
      raise ValueError('`measurement_fn` must take a single parameter if '
                       '`inner_agg_factory` is unweighted.')
  elif isinstance(inner_agg_factory, factory.WeightedAggregationFactory):
    if len(inspect.signature(measurement_fn).parameters) != 2:
      raise ValueError('`measurement_fn` must take a two parameters if '
                       '`inner_agg_factory` is weighted.')
  else:
    raise TypeError(
        f'`inner_agg_factory` must be of type `UnweightedAggregationFactory` or'
        f'`WeightedAggregationFactory`. Found {type(inner_agg_factory)}.')

  @computations.tf_computation()
  def dict_update(orig_dict, new_values):
    if not orig_dict:
      return new_values
    orig_dict.update(new_values)
    return orig_dict

  if isinstance(inner_agg_factory, factory.WeightedAggregationFactory):

    class WeightedWrappedFactory(factory.WeightedAggregationFactory):
      """Wrapper for `WeightedAggregationFactory` that adds new measurements."""

      def create(
          self, value_type: factory.ValueType, weight_type: factory.ValueType
      ) -> aggregation_process.AggregationProcess:
        py_typecheck.check_type(value_type, factory.ValueType.__args__)
        py_typecheck.check_type(weight_type, factory.ValueType.__args__)

        inner_agg_process = inner_agg_factory.create(value_type, weight_type)
        init_fn = inner_agg_process.initialize

        @computations.federated_computation(
            init_fn.type_signature.result,
            computation_types.at_clients(value_type),
            computation_types.at_clients(weight_type))
        def next_fn(state, value, weight):
          inner_agg_output = inner_agg_process.next(state, value, weight)
          extra_measurements = measurement_fn(value, weight)
          measurements = intrinsics.federated_map(
              dict_update, (inner_agg_output.measurements, extra_measurements))
          return measured_process.MeasuredProcessOutput(
              state=inner_agg_output.state,
              result=inner_agg_output.result,
              measurements=measurements)

        return aggregation_process.AggregationProcess(init_fn, next_fn)

    return WeightedWrappedFactory()
  else:

    class UnweightedWrappedFactory(factory.UnweightedAggregationFactory):
      """Wrapper for `UnweightedAggregationFactory` that adds new measurements."""

      def create(
          self, value_type: factory.ValueType
      ) -> aggregation_process.AggregationProcess:
        py_typecheck.check_type(value_type, factory.ValueType.__args__)

        inner_agg_process = inner_agg_factory.create(value_type)
        init_fn = inner_agg_process.initialize

        @computations.federated_computation(
            init_fn.type_signature.result,
            computation_types.at_clients(value_type))
        def next_fn(state, value):
          inner_agg_output = inner_agg_process.next(state, value)
          extra_measurements = measurement_fn(value)
          measurements = intrinsics.federated_map(
              dict_update, (inner_agg_output.measurements, extra_measurements))
          return measured_process.MeasuredProcessOutput(
              state=inner_agg_output.state,
              result=inner_agg_output.result,
              measurements=measurements)

        return aggregation_process.AggregationProcess(init_fn, next_fn)

    return UnweightedWrappedFactory()
