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
"""Aggregation factory for adding custom measurements."""

import inspect

from typing import Any, Callable, Dict, Optional

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


def add_measurements(
    inner_agg_factory: factory.AggregationFactory,
    *,
    client_measurement_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    server_measurement_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> factory.AggregationFactory:
  """Wraps `AggregationFactory` to report additional measurements.

  The function `client_measurement_fn` should be a Python callable that will be
  called as `client_measurement_fn(value)` or `client_measurement_fn(value,
  weight)` depending on whether `inner_agg_factory` is weighted or unweighted.
  It must be traceable by TFF and expect `tff.Value` objects placed at `CLIENTS`
  as inputs, and return a `collections.OrderedDict` mapping string names to
  tensor values placed at `SERVER`, which will be added to the measurement dict
  produced by the `inner_agg_factory`.

  Similarly, `server_measurement_fn` should be a Python callable that will be
  called as `server_measurement_fn(result)` where `result` is the result (on
  server) of the inner aggregation.

  One or both of `client_measurement_fn` and `server_measurement_fn` must be
  specified.

  Args:
    inner_agg_factory: The factory to wrap and add measurements.
    client_measurement_fn: A Python callable that will be called on `value`
      (and/or `weight`) provided to the `next` function to compute additional
      measurements of the client values/weights.
    server_measurement_fn: A Python callable that will be called on the `result`
      of aggregation at server to compute additional measurements of the result.

  Returns:
    An `AggregationFactory` that reports additional measurements.
  """
  py_typecheck.check_type(inner_agg_factory,
                          factory.AggregationFactory.__args__)

  if not (client_measurement_fn or server_measurement_fn):
    raise ValueError('Must specify one or both of `client_measurement_fn` or '
                     '`server_measurement_fn`.')

  if client_measurement_fn:
    py_typecheck.check_callable(client_measurement_fn)
    if isinstance(inner_agg_factory, factory.UnweightedAggregationFactory):
      if len(inspect.signature(client_measurement_fn).parameters) != 1:
        raise ValueError(
            '`client_measurement_fn` must take a single parameter if '
            '`inner_agg_factory` is unweighted.')
    elif isinstance(inner_agg_factory, factory.WeightedAggregationFactory):
      if len(inspect.signature(client_measurement_fn).parameters) != 2:
        raise ValueError(
            '`client_measurement_fn` must take a two parameters if '
            '`inner_agg_factory` is weighted.')

  if server_measurement_fn:
    py_typecheck.check_callable(server_measurement_fn)
    if len(inspect.signature(server_measurement_fn).parameters) != 1:
      raise ValueError('`server_measurement_fn` must take a single parameter.')

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
          measurements = inner_agg_output.measurements
          if client_measurement_fn:
            client_measurements = client_measurement_fn(value, weight)
            measurements = intrinsics.federated_map(
                dict_update, (measurements, client_measurements))
          if server_measurement_fn:
            server_measurements = server_measurement_fn(inner_agg_output.result)
            measurements = intrinsics.federated_map(
                dict_update, (measurements, server_measurements))
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
          measurements = inner_agg_output.measurements
          if client_measurement_fn:
            client_measurements = client_measurement_fn(value)
            measurements = intrinsics.federated_map(
                dict_update, (measurements, client_measurements))
          if server_measurement_fn:
            server_measurements = server_measurement_fn(inner_agg_output.result)
            measurements = intrinsics.federated_map(
                dict_update, (measurements, server_measurements))
          return measured_process.MeasuredProcessOutput(
              state=inner_agg_output.state,
              result=inner_agg_output.result,
              measurements=measurements)

        return aggregation_process.AggregationProcess(init_fn, next_fn)

    return UnweightedWrappedFactory()
