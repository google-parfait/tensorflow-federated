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
"""Factory for aggregations parameterized by TensorFlow Privacy DPQueries."""

import collections
from collections.abc import Collection
import functools
import time
import typing
from typing import Any, NamedTuple, Optional
import warnings

from absl import logging
import dp_accounting
import federated_language as flang
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax_privacy.stream_privatization import gradient_privatizer as gradient_privatizer_lib
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class ExtractingDpEventFromInitialStateError(ValueError):
  pass


class DPAggregatorState(NamedTuple):
  query_state: Any
  agg_state: Any
  dp_event: Any
  is_init_state: Any


def adaptive_clip_noise_params(
    noise_multiplier: float,
    expected_clients_per_round: float,
    clipped_count_stddev: Optional[float] = None,
) -> tuple[float, float]:
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
          '0.05 * `expected_clients_per_round` if not specified) is too low '
          'to achieve the desired effective `noise_multiplier` '
          f'({noise_multiplier}). You must either increase '
          '`clipped_count_stddev` or decrease `noise_multiplier`.'
      )

    value_noise_multiplier = (
        noise_multiplier**-2 - (2 * clipped_count_stddev) ** -2
    ) ** -0.5

    added_noise_factor = value_noise_multiplier / noise_multiplier
    if added_noise_factor >= 2:
      warnings.warn(
          f'A significant amount of noise ({added_noise_factor:.2f}x) has to '
          'be added for record aggregation to achieve the desired effective '
          f'`noise_multiplier` ({noise_multiplier}). If you are manually '
          'specifying `clipped_count_stddev` you may want to increase it. Or '
          'you may need more `expected_clients_per_round`.'
      )
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

  The state of the created `AggregationProcess` contains a DPEvent released by
  the DPQuery that can be extracted using `differential_privacy.
  extract_dp_event_from_state`.
  """

  @classmethod
  def gaussian_adaptive(
      cls,
      noise_multiplier: float,
      clients_per_round: float,
      initial_l2_norm_clip: float = 0.1,
      target_unclipped_quantile: float = 0.5,
      learning_rate: float = 0.2,
      clipped_count_stddev: Optional[float] = None,
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
    _check_float_probability(
        target_unclipped_quantile, 'target_unclipped_quantile'
    )
    _check_float_nonnegative(learning_rate, 'learning_rate')
    if clipped_count_stddev is not None:
      _check_float_nonnegative(clipped_count_stddev, 'clipped_count_stddev')

    value_noise_multiplier, clipped_count_stddev = adaptive_clip_noise_params(
        noise_multiplier, clients_per_round, clipped_count_stddev
    )
    logging.info(
        (
            'Adaptive clipping, value noise multiplier: %s -> %s,'
            'clipped_count_stddev: %s'
        ),
        noise_multiplier,
        value_noise_multiplier,
        clipped_count_stddev,
    )

    query = tfp.QuantileAdaptiveClipSumQuery(
        initial_l2_norm_clip=initial_l2_norm_clip,
        noise_multiplier=value_noise_multiplier,
        target_unclipped_quantile=target_unclipped_quantile,
        learning_rate=learning_rate,
        clipped_count_stddev=clipped_count_stddev,
        expected_num_records=clients_per_round,
        geometric_update=True,
    )
    query = tfp.NormalizedQuery(query, denominator=clients_per_round)

    return cls(query)

  @classmethod
  def gaussian_fixed(
      cls, noise_multiplier: float, clients_per_round: float, clip: float
  ) -> factory.UnweightedAggregationFactory:
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
        denominator=clients_per_round,
    )

    return cls(query)

  @classmethod
  def tree_aggregation(
      cls,
      noise_multiplier: float,
      clients_per_round: float,
      l2_norm_clip: float,
      record_specs: Collection[tf.TensorSpec],
      noise_seed: Optional[int] = None,
      use_efficient: bool = True,
      record_aggregation_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ) -> factory.UnweightedAggregationFactory:
    """`DifferentiallyPrivateFactory` with tree aggregation noise.

    Performs clipping on client, averages clients records, and adds noise for
    differential privacy. The noise is estimated based on tree aggregation for
    the cumulative summation over rounds, and then take the residual between the
    current round and the previous round. Combining this aggregator with a SGD
    optimizer on server can be used to implement the DP-FTRL algorithm in
    "Practical and Private (Deep) Learning without Sampling or Shuffling"
    (https://arxiv.org/abs/2103.00039).

    The standard deviation of the Gaussian noise added at each tree node is
    `l2_norm_clip * noise_multiplier`. Note that noise is added during summation
    of client model updates per round, *before* normalization (the noise will be
    scaled down when dividing by `clients_per_round`). Thus `noise_multiplier`
    can be used to compute the (epsilon, delta) privacy guarantee as described
    in the paper.

    Args:
      noise_multiplier: Noise multiplier for the Gaussian noise in tree
        aggregation. Must be non-negative, zero means no noise is applied.
      clients_per_round: A positive number specifying the expected number of
        clients per round.
      l2_norm_clip: The value of the clipping norm. Must be positive.
      record_specs: The specs of client results to be aggregated.
      noise_seed: Random seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated.
      use_efficient: If true, use the efficient tree aggregation algorithm based
        on the paper "Efficient Use of Differentially Private Binary Trees".
      record_aggregation_factory: An optional
        `tff.aggregators.UnweightedAggregationFactory` to aggregate values after
        preprocessing by the `query`. See the __init__ method for more details.

    Returns:
      A `DifferentiallyPrivateFactory` with Gaussian noise by tree aggregation.
    """
    if isinstance(clients_per_round, int):
      clients_per_round = float(clients_per_round)

    _check_float_nonnegative(noise_multiplier, 'noise_multiplier')
    _check_float_positive(clients_per_round, 'clients_per_round')
    _check_float_positive(l2_norm_clip, 'l2_norm_clip')

    sum_query = tfp.TreeResidualSumQuery.build_l2_gaussian_query(
        l2_norm_clip,
        noise_multiplier,
        record_specs,
        noise_seed=noise_seed,
        use_efficient=use_efficient,
    )
    mean_query = tfp.NormalizedQuery(sum_query, denominator=clients_per_round)
    return cls(
        mean_query, record_aggregation_factory=record_aggregation_factory
    )

  @classmethod
  def tree_adaptive(
      cls,
      noise_multiplier: float,
      clients_per_round: float,
      record_specs: Collection[tf.TensorSpec],
      initial_l2_norm_clip: float = 0.1,
      restart_warmup: int = 128,
      restart_frequency: int = 1024,
      target_unclipped_quantile: float = 0.5,
      clip_learning_rate: float = 0.2,
      clipped_count_stddev: Optional[float] = None,
      noise_seed: Optional[int] = None,
  ) -> factory.UnweightedAggregationFactory:
    """`DifferentiallyPrivateFactory` with adaptive clipping and tree aggregation.

    Performs clipping on client, averages clients records, and adds noise for
    differential privacy. The noise is estimated based on tree aggregation for
    the cumulative summation over rounds, and then take the residual between the
    current round and the previous round. Combining this aggregator with a SGD
    optimizer on server can be used to implement the DP-FTRL algorithm in
    "Practical and Private (Deep) Learning without Sampling or Shuffling"
    (https://arxiv.org/abs/2103.00039).

    The standard deviation of the Gaussian noise added at each tree node is
    `l2_norm_clip * noise_multiplier`. Note that noise is added during summation
    of client model updates per round, *before* normalization (the noise will be
    scaled down when dividing by `clients_per_round`). Thus `noise_multiplier`
    can be used to compute the (epsilon, delta) privacy guarantee as described
    in the paper.

    The `l2_norm_clip` is estimated and periodically reset for tree aggregation
    based on "Differentially Private Learning with Adaptive Clipping"
    (https://arxiv.org/abs/1905.03871).

    Args:
      noise_multiplier: Noise multiplier for the Gaussian noise in tree
        aggregation. Must be non-negative, zero means no noise is applied.
      clients_per_round: A positive number specifying the expected number of
        clients per round.
      record_specs: The specs of client results to be aggregated.
      initial_l2_norm_clip: The value of the initial clipping norm. Must be
        positive.
      restart_warmup: Restart the tree and adopt the estimated clip norm at the
        end of `restart_warmup` times of calling `next`.
      restart_frequency: Restart the tree and adopt the estimated clip norm
        every `restart_frequency` times of calling `next`.
      target_unclipped_quantile: The desired quantile of updates which should be
        unclipped.
      clip_learning_rate: The learning rate for the clipping norm adaptation.
        With geometric updating, a rate of r means that the clipping norm will
        change by a maximum factor of exp(r) at each round.
      clipped_count_stddev: The stddev of the noise added to the clipped_count.
        If `None`, set to `clients_per_round / 20`.
      noise_seed: Random seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated when
        `initialize`.

    Returns:
      A `DifferentiallyPrivateFactory` with Gaussian noise by tree aggregation.
    """
    if isinstance(clients_per_round, int):
      clients_per_round = float(clients_per_round)

    _check_float_nonnegative(noise_multiplier, 'noise_multiplier')
    _check_float_positive(clients_per_round, 'clients_per_round')
    _check_float_positive(initial_l2_norm_clip, 'initial_l2_norm_clip')
    _check_float_nonnegative(clip_learning_rate, 'clip_learning_rate')
    _check_float_probability(
        target_unclipped_quantile, 'target_unclipped_quantile'
    )
    if clipped_count_stddev is None:
      clipped_count_stddev = clients_per_round / 20.0
    else:
      _check_float_nonnegative(clipped_count_stddev, 'clipped_count_stddev')

    value_noise_multiplier, clipped_count_stddev = adaptive_clip_noise_params(
        noise_multiplier, clients_per_round, clipped_count_stddev
    )
    logging.info(
        (
            'Adaptive clipping, value noise multiplier: %s -> %s,'
            'clipped_count_stddev: %s.'
        ),
        noise_multiplier,
        value_noise_multiplier,
        clipped_count_stddev,
    )

    sum_query = tfp.QAdaClipTreeResSumQuery(
        initial_l2_norm_clip,
        value_noise_multiplier,
        record_specs,
        target_unclipped_quantile,
        clip_learning_rate,
        clipped_count_stddev,
        clients_per_round,
        geometric_update=True,
        noise_seed=noise_seed,
    )
    restart_indicator = tfp.restart_query.PeriodicRoundRestartIndicator(
        period=restart_frequency, warmup=restart_warmup
    )
    sum_query = tfp.RestartQuery(sum_query, restart_indicator)
    mean_query = tfp.NormalizedQuery(sum_query, denominator=clients_per_round)
    return cls(mean_query)

  def __init__(
      self,
      query: tfp.DPQuery,
      record_aggregation_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ):
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

    py_typecheck.check_type(
        record_aggregation_factory, factory.UnweightedAggregationFactory
    )
    self._record_aggregation_factory = record_aggregation_factory

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    type_args = typing.get_args(factory.ValueType)
    py_typecheck.check_type(value_type, type_args)

    query_initial_state_fn = tensorflow_computation.tf_computation(
        self._query.initial_global_state
    )
    tensor_spec = type_conversions.type_to_tf_tensor_specs(value_type)
    query_sample_state_fn = tensorflow_computation.tf_computation(
        lambda: self._query.initial_sample_state(tensor_spec)
    )
    query_state_type = query_initial_state_fn.type_signature.result
    derive_sample_params = tensorflow_computation.tf_computation(
        self._query.derive_sample_params, query_state_type
    )

    get_query_record = tensorflow_computation.tf_computation(
        self._query.preprocess_record,
        derive_sample_params.type_signature.result,
        value_type,
    )

    query_record_type = get_query_record.type_signature.result
    record_agg_process = self._record_aggregation_factory.create(
        query_record_type
    )

    agg_output_type = (
        record_agg_process.next.type_signature.result.result.member  # pytype: disable=attribute-error
    )

    @tensorflow_computation.tf_computation(agg_output_type, query_state_type)
    def get_noised_result(sample_state, global_state):
      result, new_global_state, event = self._query.get_noised_result(
          sample_state, global_state
      )
      if isinstance(result, tf.RaggedTensor):
        result = {
            'flat_values': result.flat_values,
            'nested_row_splits': result.nested_row_splits,
        }
      return result, new_global_state, event

    derive_metrics = tensorflow_computation.tf_computation(
        self._query.derive_metrics, query_state_type
    )

    dp_event_type = get_noised_result.type_signature.result[2]
    convert_dp_event = tensorflow_computation.tf_computation(
        lambda event: event.to_named_tuple(), dp_event_type
    )

    @flang.federated_computation()
    def init_fn():
      query_initial_state = flang.federated_eval(
          query_initial_state_fn, flang.SERVER
      )
      query_sample_state = flang.federated_eval(
          query_sample_state_fn, flang.SERVER
      )
      _, _, dp_event = flang.federated_map(
          get_noised_result, (query_sample_state, query_initial_state)
      )
      dp_event = flang.federated_map(convert_dp_event, dp_event)
      is_init_state = flang.federated_value(True, flang.SERVER)
      init_state = DPAggregatorState(
          query_initial_state,
          record_agg_process.initialize(),
          dp_event,
          is_init_state,
      )
      return flang.federated_zip(init_state)

    @flang.federated_computation(
        init_fn.type_signature.result,
        flang.FederatedType(value_type, flang.CLIENTS),
    )
    def next_fn(state, value):
      query_state, agg_state, _, _ = state

      params = flang.federated_broadcast(
          flang.federated_map(derive_sample_params, query_state)
      )
      record = flang.federated_map(get_query_record, (params, value))

      record_agg_output = record_agg_process.next(agg_state, record)

      result, new_query_state, dp_event = flang.federated_map(
          get_noised_result, (record_agg_output.result, query_state)
      )
      dp_event = flang.federated_map(convert_dp_event, dp_event)

      is_init_state = flang.federated_value(False, flang.SERVER)

      query_metrics = flang.federated_map(derive_metrics, new_query_state)

      new_state = DPAggregatorState(
          new_query_state,
          record_agg_output.state,
          dp_event,
          is_init_state,
      )
      measurements = collections.OrderedDict(
          dp_query_metrics=query_metrics, dp=record_agg_output.measurements
      )
      return measured_process.MeasuredProcessOutput(
          flang.federated_zip(new_state),
          result,
          flang.federated_zip(measurements),
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _shapes_dtypes_from_typespec(typespec: flang.Type):
  """Converts a spec to a tree of shapes and dtypes."""
  return type_conversions.structure_from_tensor_type_tree(
      lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
      typespec,
  )


_jax2tf_convert_cpu_native = functools.partial(
    jax2tf.convert,
    native_serialization=True,
    native_serialization_platforms=['cpu'],
)


class DPMFAggregatorFactory(factory.UnweightedAggregationFactory):
  """Aggregation factory for Differentially Private mean across clients.

  This factory expecs a GradientPrivatizer from the `jax_privacy` to be created,
  supporting methods such as those from [Scaling up the Banded Matrix
  Factorization Mechanism for Differentially Private
  ML](https://arxiv.org/abs/2405.15913).
  """

  def __init__(
      self,
      *,
      gradient_privatizer: gradient_privatizer_lib.GradientPrivatizer,
      clients_per_round: int,
      l2_clip_norm: float = 1.0,
      initial_rng_key: int | None = None,
  ):
    """Initializes `DPMFAggregatorFactory`."""
    if clients_per_round <= 0.0:
      raise ValueError('clients_per_round must be a positive float.')
    if l2_clip_norm <= 0.0:
      raise ValueError('l2_clip_norm must be a positive float.')
    if initial_rng_key is None:
      # Generate a noise seed based on the current time in milliseconds.
      self._initial_rng_key = int(time.time() * 1_000)
    else:
      self._initial_rng_key = initial_rng_key

    self._grad_privatizer = gradient_privatizer
    self._l2_clip_norm = l2_clip_norm
    self._clients_per_round = clients_per_round

  def create(
      self, value_type: flang.TensorType | flang.StructType
  ) -> aggregation_process.AggregationProcess:
    """Creates a `tff.templates.AggregationProcess` for DP MF mean."""

    @flang.federated_computation()
    def init_fn():

      @tensorflow_computation.tf_computation
      def init_privatizer():

        def jax_init_privatizer():
          return (
              jax.random.key_data(jax.random.key(self._initial_rng_key)),
              self._grad_privatizer.init(
                  _shapes_dtypes_from_typespec(value_type)
              ),
          )

        return _jax2tf_convert_cpu_native(jax_init_privatizer)()

      return flang.federated_eval(init_privatizer, flang.SERVER)

    @flang.federated_computation(
        init_fn.type_signature.result,
        flang.FederatedType(value_type, flang.CLIENTS),
    )
    def next_fn(state, value):

      @tensorflow_computation.tf_computation
      def build_zero():

        def jax_zero():
          accumulator_zero = type_conversions.structure_from_tensor_type_tree(
              lambda arr: jnp.zeros(shape=arr.shape, dtype=arr.dtype),
              value_type,
          )
          metrics_zero = {'num_clipped_updates': jnp.zeros([], dtype=jnp.int32)}
          return (accumulator_zero, metrics_zero)

        return _jax2tf_convert_cpu_native(jax_zero)()

      @tensorflow_computation.tf_computation(
          build_zero.type_signature.result, value_type
      )
      def clipped_sum(state, client_updates):
        """Clips the updates by the global l2 norm and adds them to the sum."""

        def jax_clipped_sum(state, client_updates):

          def _clip_by_global_l2_norm(global_l2_norm, updates):
            clip_rescale_factor = self._l2_clip_norm / global_l2_norm
            clipped_updates = jax.tree.map(
                lambda update: update * clip_rescale_factor, updates
            )
            metrics = {'num_clipped_updates': 1}
            return clipped_updates, metrics

          global_l2_norm = jnp.sqrt(
              sum(
                  jnp.sum(jnp.square(x))
                  for x in jax.tree.leaves(client_updates)
              )
          )
          # Conditionally clip the updates if updates' norm is above the
          # clipnorm.
          # NOTE: jax_privacy will have an improved method for this, migrate to
          # that when ready.
          clipped_updates, new_metrics = jax.lax.cond(
              global_l2_norm > self._l2_clip_norm,
              _clip_by_global_l2_norm,
              lambda norm, updates: (updates, {'num_clipped_updates': 0}),
              global_l2_norm,
              client_updates,
          )
          accumulator_pytree, metrics_pytree = state
          new_accumulators = jax.tree_util.tree_map(
              jnp.add, accumulator_pytree, clipped_updates
          )
          new_metrics = jax.tree_util.tree_map(
              jnp.add, new_metrics, metrics_pytree
          )
          return new_accumulators, new_metrics

        return _jax2tf_convert_cpu_native(jax_clipped_sum)(
            state, client_updates
        )

      @tensorflow_computation.tf_computation
      def add(a, b):
        """Merge two partial clipped sums."""
        return tf.nest.map_structure(tf.add, a, b)

      @tensorflow_computation.tf_computation
      def identity(a):
        return a

      unnoised_sum, metrics = flang.federated_aggregate(
          value,
          build_zero(),
          accumulate=clipped_sum,
          merge=add,
          report=identity,
      )

      @tensorflow_computation.tf_computation
      def finalize_noise(state, unnoised_aggregate):
        """Compute the noised mean.

        Adds noise to the unnoised sum and divides by number of clients that
        participated in the round.

        Args:
          state: The 2-tuple of (rng_key, noise state). The state used to create
            a noise slice to add to the unnoised sum this round.
          unnoised_aggregate: The unnoised sum to add noise to.

        Returns:
          The noised mean.
        """

        def jax_finalize_noise(state, unnoised_aggregate):
          rng_key, noise_state = state
          rng_key = jax.random.wrap_key_data(rng_key)
          noised_aggregate, new_noise_state = self._grad_privatizer.privatize(
              sum_of_clipped_grads=unnoised_aggregate,
              noise_state=noise_state,
              prng_key=rng_key,
          )
          # Advance the RNG key for the next round.
          rng_key, _ = jax.random.split(rng_key)
          return (jax.random.key_data(rng_key), new_noise_state), jax.tree.map(
              lambda arr: arr / self._clients_per_round, noised_aggregate
          )

        return _jax2tf_convert_cpu_native(jax_finalize_noise)(
            state, unnoised_aggregate
        )

      new_state, noised_aggregate = flang.federated_map(
          finalize_noise, (state, unnoised_sum)
      )
      return measured_process.MeasuredProcessOutput(
          new_state, noised_aggregate, metrics
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def extract_dp_event_from_state(
    state: DPAggregatorState,
) -> dp_accounting.DpEvent:
  """Extracts a DPEvent from a DP AggregationProcess' state.

  The intended use of this method is to call it on each state generated by a
  call to process.next(), and then keep a ledger of all the DPEvents extracted
  in this manner. For aggregation processes created by
  DifferentiallyPrivateFactory, initialize() returns a state with a placeholder
  DPEvent. That event is not meant to be used for accounting purposes, so this
  method raises a ValueError if it is called on a state returned from
  initialize().

  Args:
    state: The state output by process.next(), where process is a
      `tff.templates.AggregationProcess` created by
      `DifferentiallyPrivateFactory`.

  Returns:
    A DPEvent corresponding to the DP aggregation that produced this state.

  Raises:
    ExtractingDpEventFromInitialStateError: If the state is the one returned by
      initialize.
  """

  if state.is_init_state:
    raise ExtractingDpEventFromInitialStateError(
        'State was generated by a call to process.initialize(), whose DPEvent '
        'is a placeholder. extract_dp_event_from_state only accepts states '
        'from calls to process.next().'
    )
  return dp_accounting.DpEvent.from_named_tuple(state.dp_event)


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
    raise ValueError(
        f'{label} must be between 0 and 1 (inclusive). Found {value}.'
    )
