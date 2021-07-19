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
from typing import Optional, Tuple
import warnings

import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


def adaptive_clip_noise_params(
    noise_multiplier: float,
    expected_clients_per_round: float,
    clipped_count_stddev: Optional[float] = None) -> Tuple[float, float]:
  """Computes noising parameters for the adaptive L2 clipping procedure.

  The adaptive clipping method (described in https://arxiv.org/abs/1905.03871)
  runs a private quantile estimation procedure which may require the number of
  clipped clients in a round to be also noised for privacy. Thus, to maintain
  the same level of privacy as intended by the total noise multiplier, the
  effective noise multiplier to be applied on the client records may need to be
  (slightly) higher to account for the private quantile estimation.

  Args:
    noise_multiplier: The total noise multiplier for the mechanism.
    expected_clients_per_round: A float specifying the expected number of
      clients per round.
    clipped_count_stddev: The stddev of the noise added to the clipped counts in
      the adaptive clipping algorithm.

  Returns:
    A tuple with the `value_noise_multiplier` (to be applied to client records)
    and the `clipped_count_stddev` (a default value if not specified).
  """
  if noise_multiplier > 0.0:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.05 * expected_clients_per_round

    if noise_multiplier >= 2 * clipped_count_stddev:
      raise ValueError(
          f'clipped_count_stddev = {clipped_count_stddev} (defaults to '
          f'0.05 * `expected_clients_per_round` if not specified) is too low '
          f'to achieve the desired effective `noise_multiplier` '
          f'({noise_multiplier}). You must either increase '
          f'`clipped_count_stddev` or decrease `noise_multiplier`.')

    value_noise_multiplier = (noise_multiplier**-2 -
                              (2 * clipped_count_stddev)**-2)**-0.5

    added_noise_factor = value_noise_multiplier / noise_multiplier
    if added_noise_factor >= 2:
      warnings.warn(
          f'A significant amount of noise ({added_noise_factor:.2f}x) has to '
          f'be added for record aggregation to achieve the desired effective '
          f'`noise_multiplier` ({noise_multiplier}). If you are manually '
          f'specifying `clipped_count_stddev` you may want to increase it. Or '
          f'you may need more `expected_clients_per_round`.')
  else:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.0
    value_noise_multiplier = 0.0

  return value_noise_multiplier, clipped_count_stddev


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

  A simple summation (using the default `tff.aggregators.SumFactory`) is usually
  acceptable. Aggregations that change the records (such as compression or
  secure aggregation) may be allowed so long as they do not increase the
  sensitivity of the query. It is the users' responsibility to ensure that the
  mode of aggregation is consistent with the DPQuery. Note that the DPQuery's
  built-in aggregation functions (accumulate_preprocessed_record and
  merge_sample_states) are ignored in favor of the provided aggregator.

  To obtain concrete (epsilon, delta) guarantees, one could use the analysis
  tools provided in tensorflow_privacy by using QueryWithLedger.
  """

  @classmethod
  def gaussian_adaptive(
      cls,
      noise_multiplier: float,
      clients_per_round: float,
      initial_l2_norm_clip: float = 0.1,
      target_unclipped_quantile: float = 0.5,
      learning_rate: float = 0.2,
      clipped_count_stddev: Optional[float] = None
  ) -> factory.UnweightedAggregationFactory:
    """`DifferentiallyPrivateFactory` with adaptive clipping and Gaussian noise.

    Performs adaptive clipping and addition of Gaussian noise for differentially
    private learning. For details of the DP algorithm see McMahan et. al (2017)
    https://arxiv.org/abs/1710.06963. The adaptive clipping uses the geometric
    method described in Thakkar et al. (2019) https://arxiv.org/abs/1905.03871.

    The adaptive clipping parameters have been chosen to yield a process that
    starts small and adapts relatively quickly to the median, without using
    much of the privacy budget. This works well on most problems.

    Args:
      noise_multiplier: A float specifying the noise multiplier for the Gaussian
        mechanism for model updates. A value of 1.0 or higher may be needed for
        strong privacy. See above mentioned papers to compute (epsilon, delta)
        privacy guarantee. Note that this is the effective total noise
        multiplier, accounting for the privacy loss due to adaptive clipping.
        The noise actually added to the aggregated values will be slightly
        higher.
      clients_per_round: A float specifying the expected number of clients per
        round. Must be positive.
      initial_l2_norm_clip: The initial value of the adaptive clipping norm.
      target_unclipped_quantile: The quantile to which the clipping norm should
        adapt.
      learning_rate: The learning rate for the adaptive clipping process.
      clipped_count_stddev: The stddev of the noise added to the clipped counts
        in the adaptive clipping algorithm. If None, defaults to `0.05 *
        clients_per_round` (unless `noise_multiplier` is 0, in which case it is
        also 0).

    Returns:
      A `DifferentiallyPrivateFactory` with adaptive clipping and Gaussian
        noise.
    """

    if isinstance(clients_per_round, int):
      clients_per_round = float(clients_per_round)

    _check_float_nonnegative(noise_multiplier, 'noise_multiplier')
    _check_float_positive(clients_per_round, 'clients_per_round')
    _check_float_positive(initial_l2_norm_clip, 'initial_l2_norm_clip')
    _check_float_probability(target_unclipped_quantile,
                             'target_unclipped_quantile')
    _check_float_nonnegative(learning_rate, 'learning_rate')
    if clipped_count_stddev is not None:
      _check_float_nonnegative(clipped_count_stddev, 'clipped_count_stddev')

    value_noise_multiplier, clipped_count_stddev = adaptive_clip_noise_params(
        noise_multiplier, clients_per_round, clipped_count_stddev)

    query = tfp.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=initial_l2_norm_clip,
        noise_multiplier=value_noise_multiplier,
        target_unclipped_quantile=target_unclipped_quantile,
        learning_rate=learning_rate,
        clipped_count_stddev=clipped_count_stddev,
        expected_num_records=clients_per_round,
        geometric_update=True)
    query = tfp.NormalizedQuery(query, denominator=clients_per_round)

    return cls(query)

  @classmethod
  def gaussian_fixed(cls, noise_multiplier: float, clients_per_round: float,
                     clip: float) -> factory.UnweightedAggregationFactory:
    """`DifferentiallyPrivateFactory` with fixed clipping and Gaussian noise.

    Performs fixed clipping and addition of Gaussian noise for differentially
    private learning. For details of the DP algorithm see McMahan et. al (2017)
    https://arxiv.org/abs/1710.06963.

    Args:
      noise_multiplier: A float specifying the noise multiplier for the Gaussian
        mechanism for model updates. A value of 1.0 or higher may be needed for
        strong privacy. See above mentioned paper to compute (epsilon, delta)
        privacy guarantee.
      clients_per_round: A float specifying the expected number of clients per
        round. Must be positive.
      clip: The value of the clipping norm.

    Returns:
      A `DifferentiallyPrivateFactory` with fixed clipping and Gaussian noise.
    """

    if isinstance(clients_per_round, int):
      clients_per_round = float(clients_per_round)

    _check_float_nonnegative(noise_multiplier, 'noise_multiplier')
    _check_float_positive(clients_per_round, 'clients_per_round')
    _check_float_positive(clip, 'clip')

    query = tfp.NormalizedQuery(
        tfp.GaussianSumQuery(l2_norm_clip=clip, stddev=clip * noise_multiplier),
        denominator=clients_per_round)

    return cls(query)

  def __init__(self,
               query: tfp.DPQuery,
               record_aggregation_factory: Optional[
                   factory.UnweightedAggregationFactory] = None):
    """Initializes `DifferentiallyPrivateFactory`.

    Args:
      query: A `tfp.SumAggregationDPQuery` to perform private estimation.
      record_aggregation_factory: A
        `tff.aggregators.UnweightedAggregationFactory` to aggregate values after
        preprocessing by the `query`. If `None`, defaults to
        `tff.aggregators.SumFactory`. The provided factory is assumed to
        implement a sum, and to have the property that it does not increase the
        sensitivity of the query - typically this means that it should not
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

  def create(
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
    record_agg_process = self._record_aggregation_factory.create(
        query_record_type)

    agg_output_type = record_agg_process.next.type_signature.result.result.member
    get_noised_result = computations.tf_computation(
        self._query.get_noised_result, agg_output_type, query_state_type)
    derive_metrics = computations.tf_computation(self._query.derive_metrics,
                                                 query_state_type)

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
          dp_query_metrics=query_metrics, dp=agg_measurements)
      return measured_process.MeasuredProcessOutput(
          intrinsics.federated_zip(new_state), result,
          intrinsics.federated_zip(measurements))

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _check_float_positive(value, label):
  py_typecheck.check_type(value, float, label)
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {value}.')


def _check_float_nonnegative(value, label):
  py_typecheck.check_type(value, float, label)
  if value < 0:
    raise ValueError(f'{label} must be nonnegative. Found {value}.')


def _check_float_probability(value, label):
  py_typecheck.check_type(value, float, label)
  if not 0 <= value <= 1:
    raise ValueError(f'{label} must be between 0 and 1 (inclusive). '
                     f'Found {value}.')
