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
"""Utilities for building aggregation factories."""

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process


def as_weighted_aggregator(
    unweighted_aggregator: factory.UnweightedAggregationFactory,
) -> factory.WeightedAggregationFactory:
  """Constructs a weighted wrapper for an unweighted aggregation factory.

  The returned `tff.aggregators.WeightedAggregationFactory` has the same
  functionality as the provided `unweighted_aggregator`, ignoring the provided
  weights. This is useful when converting unweighted aggregator to a weighted
  aggregator which is "always uniformly weighted".

  Args:
    unweighted_aggregator: A `tff.aggregators.UnweightedAggregationFactory`.

  Returns:
    A `tff.aggregators.WeightedAggregationFactory`.
  """
  return _UnweightedAsWeightedFactory(unweighted_aggregator)


class _UnweightedAsWeightedFactory(factory.WeightedAggregationFactory):
  """Weighted wrapper for an unweighted aggregation factory."""

  def __init__(self, unweighted_factory: factory.UnweightedAggregationFactory):
    py_typecheck.check_type(
        unweighted_factory, factory.UnweightedAggregationFactory
    )
    self._factory = unweighted_factory

  def create(self, value_type, weight_type):
    aggregator = self._factory.create(value_type)

    @federated_computation.federated_computation(
        aggregator.state_type,
        computation_types.at_clients(value_type),
        computation_types.at_clients(weight_type),
    )
    def next_fn(state, value, weight):
      del weight  # Unused.
      return aggregator.next(state, value)

    return aggregation_process.AggregationProcess(
        aggregator.initialize, next_fn
    )
