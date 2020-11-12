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
"""Factory for aggregations parameterized by tensorflow_privacy DPQueries."""

import collections
from typing import Optional
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class DifferentiallyPrivateFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` for tensorflow_privacy DPQueries.

  The created `tff.templates.AggregationProcess` aggregates values placed at
  `CLIENTS` according to the provided DPQuery, and outputs the result placed at
  `SERVER`.

  A DPQuery defines preprocessing to perform on each value, and postprocessing
  to perform on the aggregated, preprocessed values. Provided the preprocessed
  values ("records") are aggregated in a way that is consistent with the
  DPQuery, formal (epsilon, delta) privacy guarantees can be derived. This
  aggregation is controlled by `record_aggregation_factory`.

  A simple summation (using the default `tff.templates.SumFactory`) is usually
  acceptable. Aggregations that change the records (such as compression or
  secure aggregation) may be allowed so long as they do not increase the
  sensitivity of the query. It is the users' responsibility to ensure that the
  mode of aggregation is consistent with the DPQuery. Note that the DPQuery's
  built-in aggregation functions (accumulate_preprocessed_record and
  merge_sample_states) are ignored in favor of the provided aggregator.

  To obtain concrete (epsilon, delta) guarantees, one could use the analysis
  tools provided in tensorflow_privacy by using QueryWithLedger.
  """

  def __init__(self,
               query: tfp.DPQuery,
               record_aggregation_factory: Optional[
                   factory.UnweightedAggregationFactory] = None):
    """Initializes `DifferentiallyPrivateFactory`.

    Args:
      query: A `tfp.SumAggregationDPQuery` to perform private estimation.
      record_aggregation_factory: A
        `tff.aggregators.UnweightedAggregationFactory` to aggregate values
        after preprocessing by the `query`. If `None`, defaults to
        `tff.aggregators.SumFactory`. The provided factory is assumed to
        implement a sum, and to have the property that it does not increase
        the sensitivity of the query - typically this means that it should not
        increase the l2 norm of the records when aggregating.

    Raises:
      TypeError: If `query` is not an instance of `tfp.SumAggregationDPQuery` or
        `record_aggregation_factory` is not an instance of
        `tff.aggregators.UnweightedAggregationFactory`.
    """
    py_typecheck.check_type(query, tfp.SumAggregationDPQuery)
    self._query = query

    if record_aggregation_factory is None:
      record_aggregation_factory = sum_factory.SumFactory()

    py_typecheck.check_type(record_aggregation_factory,
                            factory.UnweightedAggregationFactory)
    self._record_aggregation_factory = record_aggregation_factory

  def create_unweighted(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    py_typecheck.check_type(value_type, factory.ValueType.__args__)

    query_initial_state_fn = computations.tf_computation(
        self._query.initial_global_state)

    query_state_type = query_initial_state_fn.type_signature.result
    derive_sample_params = computations.tf_computation(
        self._query.derive_sample_params, query_state_type)
    get_query_record = computations.tf_computation(
        self._query.preprocess_record,
        derive_sample_params.type_signature.result, value_type)
    query_record_type = get_query_record.type_signature.result
    get_noised_result = computations.tf_computation(
        self._query.get_noised_result, query_record_type, query_state_type)
    derive_metrics = computations.tf_computation(self._query.derive_metrics,
                                                 query_state_type)

    record_agg_process = self._record_aggregation_factory.create_unweighted(
        query_record_type)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip(
          (intrinsics.federated_eval(query_initial_state_fn, placements.SERVER),
           record_agg_process.initialize()))

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.FederatedType(
                                            value_type, placements.CLIENTS))
    def next_fn(state, value):
      query_state, agg_state = state

      params = intrinsics.federated_broadcast(
          intrinsics.federated_map(derive_sample_params, query_state))
      record = intrinsics.federated_map(get_query_record, (params, value))

      (new_agg_state, agg_result,
       agg_measurements) = record_agg_process.next(agg_state, record)

      result, new_query_state = intrinsics.federated_map(
          get_noised_result, (agg_result, query_state))

      query_metrics = intrinsics.federated_map(derive_metrics, new_query_state)

      new_state = (new_query_state, new_agg_state)
      measurements = collections.OrderedDict(
          query_metrics=query_metrics, record_agg_process=agg_measurements)
      return measured_process.MeasuredProcessOutput(
          intrinsics.federated_zip(new_state), result,
          intrinsics.federated_zip(measurements))

    return aggregation_process.AggregationProcess(init_fn, next_fn)
