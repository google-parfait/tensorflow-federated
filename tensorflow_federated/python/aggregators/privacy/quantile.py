# Copyright 2020, The TensorFlow Authors.
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
"""Implements DPQuery interface for quantile estimator."""

import collections

import dp_accounting
import tensorflow as tf

from tensorflow_federated.python.aggregators.privacy import query
from tensorflow_federated.python.aggregators.privacy import tree


class QuantileEstimatorQuery(query.SumAggregationDPQuery):
  """DPQuery to estimate target quantile of a univariate distribution.

  Uses the algorithm of Andrew et al. (https://arxiv.org/abs/1905.03871). See
  the paper for details and suggested hyperparameter settings.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState',
      [
          'current_estimate',
          'target_quantile',
          'learning_rate',
          'below_estimate_state',
      ],
  )

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['current_estimate', 'below_estimate_params']
  )

  # No separate SampleState-- sample state is just below_estimate_query's
  # SampleState.

  def __init__(
      self,
      initial_estimate,
      target_quantile,
      learning_rate,
      below_estimate_stddev,
      expected_num_records,
      geometric_update=False,
  ):
    """Initializes the QuantileEstimatorQuery.

    Args:
      initial_estimate: The initial estimate of the quantile.
      target_quantile: The target quantile. I.e., a value of 0.8 means a value
        should be found for which approximately 80% of updates are less than the
        estimate each round.
      learning_rate: The learning rate. A rate of r means that the estimate will
        change by a maximum of r at each step (for arithmetic updating) or by a
        maximum factor of exp(r) (for geometric updating). Andrew et al.
        recommends that this be set to 0.2 for geometric updating.
      below_estimate_stddev: The stddev of the noise added to the count of
        records currently below the estimate. Andrew et al. recommends that this
        be set to `expected_num_records / 20` for reasonably fast adaptation and
        high privacy.
      expected_num_records: The expected number of records per round.
      geometric_update: If True, use geometric updating of estimate. Geometric
        updating is preferred for non-negative records like vector norms that
        could potentially be very large or very close to zero.
    """

    if target_quantile < 0 or target_quantile > 1:
      raise ValueError(
          f'`target_quantile` must be between 0 and 1, got {target_quantile}.'
      )

    if learning_rate < 0:
      raise ValueError(
          f'`learning_rate` must be non-negative, got {learning_rate}'
      )

    self._initial_estimate = initial_estimate
    self._target_quantile = target_quantile
    self._learning_rate = learning_rate

    self._below_estimate_query = self._construct_below_estimate_query(
        below_estimate_stddev, expected_num_records
    )
    assert isinstance(self._below_estimate_query, query.SumAggregationDPQuery)

    self._geometric_update = geometric_update

  def _construct_below_estimate_query(
      self, below_estimate_stddev, expected_num_records
  ):
    # A DPQuery used to estimate the fraction of records that are less than the
    # current quantile estimate. It accumulates an indicator 0/1 of whether each
    # record is below the estimate, and normalizes by the expected number of
    # records. In practice, we accumulate counts shifted by -0.5 so they are
    # centered at zero. This makes the sensitivity of the below_estimate count
    # query 0.5 instead of 1.0, since the maximum that a single record could
    # affect the count is 0.5. Note that although the l2_norm_clip of the
    # below_estimate query is 0.5, no clipping will ever actually occur
    # because the value of each record is always +/-0.5.
    return query.NormalizedQuery(
        query.GaussianSumQuery(l2_norm_clip=0.5, stddev=below_estimate_stddev),
        denominator=expected_num_records,
    )

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self._GlobalState(
        tf.cast(self._initial_estimate, tf.float32),
        tf.cast(self._target_quantile, tf.float32),
        tf.cast(self._learning_rate, tf.float32),
        self._below_estimate_query.initial_global_state(),
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    below_estimate_params = self._below_estimate_query.derive_sample_params(
        global_state.below_estimate_state
    )
    return self._SampleParams(
        global_state.current_estimate, below_estimate_params
    )

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    # Template is ignored because records are required to be scalars.
    del template

    return self._below_estimate_query.initial_sample_state(0.0)

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    tf.debugging.assert_scalar(record)

    # Shift counts by 0.5 so they are centered at zero. (See comment in
    # `_construct_below_estimate_query`.)
    below = tf.cast(record <= params.current_estimate, tf.float32) - 0.5
    return self._below_estimate_query.preprocess_record(
        params.below_estimate_params, below
    )

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    below_estimate_result, new_below_estimate_state, below_estimate_event = (
        self._below_estimate_query.get_noised_result(
            sample_state, global_state.below_estimate_state
        )
    )

    # Unshift below_estimate percentile by 0.5. (See comment in
    # `_construct_below_estimate_query`.)
    below_estimate = below_estimate_result + 0.5

    # Protect against out-of-range estimates.
    below_estimate = tf.minimum(1.0, tf.maximum(0.0, below_estimate))

    loss_grad = below_estimate - global_state.target_quantile

    update = global_state.learning_rate * loss_grad

    if self._geometric_update:
      new_estimate = global_state.current_estimate * tf.math.exp(-update)
    else:
      new_estimate = global_state.current_estimate - update

    new_global_state = global_state._replace(
        current_estimate=new_estimate,
        below_estimate_state=new_below_estimate_state,
    )

    return new_estimate, new_global_state, below_estimate_event

  def derive_metrics(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_metrics`."""
    return collections.OrderedDict(estimate=global_state.current_estimate)


class NoPrivacyQuantileEstimatorQuery(QuantileEstimatorQuery):
  """Iterative process to estimate target quantile of a univariate distribution.

  Unlike the base class, this uses a NoPrivacyQuery to estimate the fraction
  below estimate with an exact denominator, so there are no privacy guarantees.
  """

  def __init__(
      self,
      initial_estimate,
      target_quantile,
      learning_rate,
      geometric_update=False,
  ):
    """Initializes the NoPrivacyQuantileEstimatorQuery.

    Args:
      initial_estimate: The initial estimate of the quantile.
      target_quantile: The target quantile. I.e., a value of 0.8 means a value
        should be found for which approximately 80% of updates are less than the
        estimate each round.
      learning_rate: The learning rate. A rate of r means that the estimate will
        change by a maximum of r at each step (for arithmetic updating) or by a
        maximum factor of exp(r) (for geometric updating). Andrew et al.
        recommends that this be set to 0.2 for geometric updating.
      geometric_update: If True, use geometric updating of estimate. Geometric
        updating is preferred for non-negative records like vector norms that
        could potentially be very large or very close to zero.
    """
    super().__init__(
        initial_estimate,
        target_quantile,
        learning_rate,
        below_estimate_stddev=None,
        expected_num_records=None,
        geometric_update=geometric_update,
    )

  def _construct_below_estimate_query(
      self, below_estimate_stddev, expected_num_records
  ):
    del below_estimate_stddev
    del expected_num_records
    return query.NoPrivacyAverageQuery()


class TreeQuantileEstimatorQuery(QuantileEstimatorQuery):
  """Iterative process to estimate target quantile of a univariate distribution.

  Unlike the base class, this uses a `TreeResidualSumQuery` to estimate the
  fraction below estimate with an exact denominator. This assumes that below
  estimate value is used in a SGD-like update and we want to privatize the
  cumsum of the below estimate.

  See "Practical and Private (Deep) Learning without Sampling or Shuffling"
  (https://arxiv.org/abs/2103.00039) for tree aggregation and privacy
  accounting, and "Differentially Private Learning with Adaptive Clipping"
  (https://arxiv.org/abs/1905.03871) for how below estimate is used in a
  SGD-like algorithm.
  """

  def _construct_below_estimate_query(
      self, below_estimate_stddev, expected_num_records
  ):
    # See comments in `QuantileEstimatorQuery._construct_below_estimate_query`
    # for why clip norm 0.5 is used for the query.
    sum_query = tree.TreeResidualSumQuery.build_l2_gaussian_query(
        clip_norm=0.5,
        noise_multiplier=2 * below_estimate_stddev,
        record_specs=tf.TensorSpec([]),
    )
    return query.NormalizedQuery(sum_query, denominator=expected_num_records)

  def reset_state(self, noised_results, global_state):
    new_numerator_state = self._below_estimate_query._numerator.reset_state(  # pylint: disable=protected-access,line-too-long
        noised_results, global_state.below_estimate_state.numerator_state
    )
    new_below_estimate_state = global_state.below_estimate_state._replace(
        numerator_state=new_numerator_state
    )
    return global_state._replace(below_estimate_state=new_below_estimate_state)


class QuantileAdaptiveClipSumQuery(query.SumAggregationDPQuery):
  """`DPQuery` for Gaussian sum queries with adaptive clipping.

  Clipping norm is tuned adaptively to converge to a value such that a specified
  quantile of updates are clipped, using the algorithm of Andrew et al. (
  https://arxiv.org/abs/1905.03871). See the paper for details and suggested
  hyperparameter settings.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState',
      ['noise_multiplier', 'sum_state', 'quantile_estimator_state'],
  )

  # pylint: disable=invalid-name
  _SampleState = collections.namedtuple(
      '_SampleState', ['sum_state', 'quantile_estimator_state']
  )

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['sum_params', 'quantile_estimator_params']
  )

  def __init__(
      self,
      initial_l2_norm_clip,
      noise_multiplier,
      target_unclipped_quantile,
      learning_rate,
      clipped_count_stddev,
      expected_num_records,
      geometric_update=True,
  ):
    """Initializes the QuantileAdaptiveClipSumQuery.

    Args:
      initial_l2_norm_clip: The initial value of clipping norm.
      noise_multiplier: The stddev of the noise added to the output will be this
        times the current value of the clipping norm.
      target_unclipped_quantile: The desired quantile of updates which should be
        unclipped. I.e., a value of 0.8 means a value of l2_norm_clip should be
        found for which approximately 20% of updates are clipped each round.
        Andrew et al. recommends that this be set to 0.5 to clip to the median.
      learning_rate: The learning rate for the clipping norm adaptation. With
        geometric updating, a rate of r means that the clipping norm will change
        by a maximum factor of exp(r) at each round. This maximum is attained
        when |actual_unclipped_fraction - target_unclipped_quantile| is 1.0.
        Andrew et al. recommends that this be set to 0.2 for geometric updating.
      clipped_count_stddev: The stddev of the noise added to the clipped_count.
        Andrew et al. recommends that this be set to `expected_num_records / 20`
        for reasonably fast adaptation and high privacy.
      expected_num_records: The expected number of records per round, used to
        estimate the clipped count quantile.
      geometric_update: If `True`, use geometric updating of clip (recommended).
    """
    self._noise_multiplier = noise_multiplier

    self._quantile_estimator_query = QuantileEstimatorQuery(
        initial_l2_norm_clip,
        target_unclipped_quantile,
        learning_rate,
        clipped_count_stddev,
        expected_num_records,
        geometric_update,
    )

    self._sum_query = query.GaussianSumQuery(
        initial_l2_norm_clip, noise_multiplier * initial_l2_norm_clip
    )

    assert isinstance(self._sum_query, query.SumAggregationDPQuery)
    assert isinstance(
        self._quantile_estimator_query, query.SumAggregationDPQuery
    )

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self._GlobalState(
        tf.cast(self._noise_multiplier, tf.float32),
        self._sum_query.initial_global_state(),
        self._quantile_estimator_query.initial_global_state(),
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._SampleParams(
        self._sum_query.derive_sample_params(global_state.sum_state),
        self._quantile_estimator_query.derive_sample_params(
            global_state.quantile_estimator_state
        ),
    )

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self._SampleState(
        self._sum_query.initial_sample_state(template),
        self._quantile_estimator_query.initial_sample_state(),
    )

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    clipped_record, global_norm = self._sum_query.preprocess_record_impl(
        params.sum_params, record
    )

    was_unclipped = self._quantile_estimator_query.preprocess_record(
        params.quantile_estimator_params, global_norm
    )

    return self._SampleState(clipped_record, was_unclipped)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_vectors, sum_state, sum_event = self._sum_query.get_noised_result(
        sample_state.sum_state, global_state.sum_state
    )
    del sum_state  # To be set explicitly later when we know the new clip.

    new_l2_norm_clip, new_quantile_estimator_state, quantile_event = (
        self._quantile_estimator_query.get_noised_result(
            sample_state.quantile_estimator_state,
            global_state.quantile_estimator_state,
        )
    )

    new_l2_norm_clip = tf.maximum(new_l2_norm_clip, 0.0)
    new_sum_stddev = new_l2_norm_clip * global_state.noise_multiplier
    new_sum_query_state = self._sum_query.make_global_state(
        l2_norm_clip=new_l2_norm_clip, stddev=new_sum_stddev
    )

    new_global_state = self._GlobalState(
        global_state.noise_multiplier,
        new_sum_query_state,
        new_quantile_estimator_state,
    )

    event = dp_accounting.ComposedDpEvent(events=[sum_event, quantile_event])
    return noised_vectors, new_global_state, event

  def derive_metrics(self, global_state):
    """Returns the current clipping norm as a metric."""
    return collections.OrderedDict(clip=global_state.sum_state.l2_norm_clip)


class QAdaClipTreeResSumQuery(query.SumAggregationDPQuery):
  """`DPQuery` for tree aggregation queries with adaptive clipping.

  The implementation is based on tree aggregation noise for cumulative sum in
  "Practical and Private (Deep) Learning without Sampling or Shuffling"
  (https://arxiv.org/abs/2103.00039) and quantile-based adaptive clipping in
  "Differentially Private Learning with Adaptive Clipping"
  (https://arxiv.org/abs/1905.03871).

  The quantile value will be continuously estimated, but the clip norm is only
  updated when `reset_state` is called, and the tree state will be reset. This
  will force the clip norm (and corresponding stddev) in a tree unchanged.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState',
      ['noise_multiplier', 'sum_state', 'quantile_estimator_state'],
  )

  # pylint: disable=invalid-name
  _SampleState = collections.namedtuple(
      '_SampleState', ['sum_state', 'quantile_estimator_state']
  )

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple(
      '_SampleParams', ['sum_params', 'quantile_estimator_params']
  )

  def __init__(
      self,
      initial_l2_norm_clip,
      noise_multiplier,
      record_specs,
      target_unclipped_quantile,
      learning_rate,
      clipped_count_stddev,
      expected_num_records,
      geometric_update=True,
      noise_seed=None,
  ):
    """Initializes the `QAdaClipTreeResSumQuery`.

    Args:
      initial_l2_norm_clip: The initial value of clipping norm.
      noise_multiplier: The stddev of the noise added to the output will be this
        times the current value of the clipping norm.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      target_unclipped_quantile: The desired quantile of updates which should be
        unclipped. I.e., a value of 0.8 means a value of l2_norm_clip should be
        found for which approximately 20% of updates are clipped each round.
        Andrew et al. recommends that this be set to 0.5 to clip to the median.
      learning_rate: The learning rate for the clipping norm adaptation. With
        geometric updating, a rate of r means that the clipping norm will change
        by a maximum factor of exp(r) at each round. This maximum is attained
        when |actual_unclipped_fraction - target_unclipped_quantile| is 1.0.
        Andrew et al. recommends that this be set to 0.2 for geometric updating.
      clipped_count_stddev: The stddev of the noise added to the clipped_count.
        Andrew et al. recommends that this be set to `expected_num_records / 20`
        for reasonably fast adaptation and high privacy.
      expected_num_records: The expected number of records per round, used to
        estimate the clipped count quantile.
      geometric_update: If `True`, use geometric updating of clip (recommended).
      noise_seed: Integer seed for the Gaussian noise generator of
        `TreeResidualSumQuery`. If `None`, a nondeterministic seed based on
        system time will be generated.
    """
    self._noise_multiplier = noise_multiplier

    self._quantile_estimator_query = TreeQuantileEstimatorQuery(
        initial_l2_norm_clip,
        target_unclipped_quantile,
        learning_rate,
        clipped_count_stddev,
        expected_num_records,
        geometric_update,
    )

    self._sum_query = tree.TreeResidualSumQuery.build_l2_gaussian_query(
        initial_l2_norm_clip,
        noise_multiplier,
        record_specs,
        noise_seed=noise_seed,
        use_efficient=True,
    )

    assert isinstance(self._sum_query, query.SumAggregationDPQuery)
    assert isinstance(
        self._quantile_estimator_query, query.SumAggregationDPQuery
    )

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self._GlobalState(
        tf.cast(self._noise_multiplier, tf.float32),
        self._sum_query.initial_global_state(),
        self._quantile_estimator_query.initial_global_state(),
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._SampleParams(
        self._sum_query.derive_sample_params(global_state.sum_state),
        self._quantile_estimator_query.derive_sample_params(
            global_state.quantile_estimator_state
        ),
    )

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self._SampleState(
        self._sum_query.initial_sample_state(template),
        self._quantile_estimator_query.initial_sample_state(),
    )

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    clipped_record, global_norm = self._sum_query.preprocess_record_l2_impl(
        params.sum_params, record
    )

    below_estimate = self._quantile_estimator_query.preprocess_record(
        params.quantile_estimator_params, global_norm
    )

    return self._SampleState(clipped_record, below_estimate)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_vectors, sum_state, sum_event = self._sum_query.get_noised_result(
        sample_state.sum_state, global_state.sum_state
    )

    _, quantile_estimator_state, quantile_event = (
        self._quantile_estimator_query.get_noised_result(
            sample_state.quantile_estimator_state,
            global_state.quantile_estimator_state,
        )
    )

    new_global_state = self._GlobalState(
        global_state.noise_multiplier, sum_state, quantile_estimator_state
    )
    event = dp_accounting.ComposedDpEvent(events=[sum_event, quantile_event])
    return noised_vectors, new_global_state, event

  def reset_state(self, noised_results, global_state):
    """Returns state after resetting the tree and updating the clip norm.

    This function will be used in `restart_query.RestartQuery` after calling
    `get_noised_result` when the restarting condition is met. The clip norm (
    and corresponding noise stddev) for the tree aggregated sum query is only
    updated from the quantile-based estimation when `reset_state` is called.

    Args:
      noised_results: Noised cumulative sum returned by `get_noised_result`.
      global_state: Updated global state returned by `get_noised_result`, which
        records noise for the conceptual cumulative sum of the current leaf
        node, and tree state for the next conceptual cumulative sum.

    Returns:
      New global state with restarted tree state, and new clip norm.
    """
    new_l2_norm_clip = tf.math.maximum(
        global_state.quantile_estimator_state.current_estimate, 0.0
    )
    new_sum_stddev = new_l2_norm_clip * global_state.noise_multiplier
    sum_state = self._sum_query.reset_l2_clip_gaussian_noise(
        global_state.sum_state,
        clip_norm=new_l2_norm_clip,
        stddev=new_sum_stddev,
    )
    sum_state = self._sum_query.reset_state(noised_results, sum_state)
    quantile_estimator_state = self._quantile_estimator_query.reset_state(
        noised_results, global_state.quantile_estimator_state
    )

    return global_state._replace(
        sum_state=sum_state, quantile_estimator_state=quantile_estimator_state
    )

  def derive_metrics(self, global_state):
    """Returns the clipping norm and estimated quantile value as a metric."""
    return collections.OrderedDict(
        current_clip=global_state.sum_state.clip_value,
        estimate_clip=global_state.quantile_estimator_state.current_estimate,
    )
