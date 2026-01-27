# Copyright 2021, The TensorFlow Authors.
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
"""`DPQuery`s for differentially private tree aggregation protocols."""

import abc
import collections
from collections.abc import Callable, Collection
import math
from typing import Any, NamedTuple, Optional, Union

import dp_accounting
import tensorflow as tf

from tensorflow_federated.python.aggregators.privacy import query


class RestartIndicator(metaclass=abc.ABCMeta):
  """Base class establishing interface for restarting the tree state.

  A `RestartIndicator` maintains a state, and each time `next` is called, a bool
  value is generated to indicate whether to restart, and the indicator state is
  advanced.
  """

  @abc.abstractmethod
  def initialize(self):
    """Makes an initialized state for `RestartIndicator`.

    Returns:
      An initial state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def next(self, state):
    """Gets next bool indicator and advances the `RestartIndicator` state.

    Args:
      state: The current state.

    Returns:
      A pair (value, new_state) where value is bool indicator and new_state
        is the advanced state.
    """
    raise NotImplementedError


class PeriodicRoundRestartIndicator(RestartIndicator):
  """Indicator for resetting the tree state after every a few number of queries.

  The indicator will maintain an internal counter as state.
  """

  def __init__(self, period: int, warmup: Optional[int] = None):
    """Construct the `PeriodicRoundRestartIndicator`.

    Args:
      period: The `next` function will return `True` every `period` number of
        `next` calls.
      warmup: The first `True` will be returned at the `warmup` times call of
        `next`.
    """
    if period < 1:
      raise ValueError(
          f'Restart period should be equal or larger than 1, got {period}'
      )
    if warmup is None:
      warmup = 0
    elif warmup <= 0 or warmup >= period:
      raise ValueError(
          f'Warmup must be between 1 and `period`-1={period-1}, got {warmup}'
      )
    self.period = period
    self.warmup = warmup

  def initialize(self):
    """Returns initialized state of 0 for `PeriodicRoundRestartIndicator`."""
    return tf.constant(0, tf.int32)

  def next(self, state):
    """Gets next bool indicator and advances the state.

    Args:
      state: The current state.

    Returns:
      A pair (value, new_state) where value is the bool indicator and new_state
        of `state+1`.
    """
    period = tf.constant(self.period, tf.int32)
    warmup = tf.constant(self.warmup, tf.int32)
    state = state + tf.constant(1, tf.int32)
    flag = tf.math.equal(tf.math.floormod(state, period), warmup)
    return flag, state


class _RestartQueryGlobalState(NamedTuple):
  inner_query_state: Any
  indicator_state: Any


class RestartQuery(query.SumAggregationDPQuery):
  """`DPQuery` for `SumAggregationDPQuery` with a `reset_state` function."""

  _inner_query: Any
  _restart_indicator: RestartIndicator

  def __init__(
      self,
      inner_query: Any,
      restart_indicator: RestartIndicator,
  ):
    """Initializes `RestartQuery`.

    Args:
      inner_query: A `SumAggregationDPQuery` has `reset_state` attribute.
      restart_indicator: A `RestartIndicator` to generate the boolean indicator
        for resetting the state.
    """
    if not hasattr(inner_query, 'reset_state'):
      raise ValueError(
          f'{type(inner_query)} must define `reset_state` to be '
          'composed with `RestartQuery`.'
      )
    self._inner_query = inner_query
    self._restart_indicator = restart_indicator

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return _RestartQueryGlobalState(
        inner_query_state=self._inner_query.initial_global_state(),
        indicator_state=self._restart_indicator.initialize(),
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._inner_query.derive_sample_params(
        global_state.inner_query_state
    )

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self._inner_query.initial_sample_state(template)

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    return self._inner_query.preprocess_record(params, record)

  @tf.function
  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_results, inner_state, event = self._inner_query.get_noised_result(
        sample_state, global_state.inner_query_state
    )
    restart_flag, indicator_state = self._restart_indicator.next(
        global_state.indicator_state
    )
    if restart_flag:
      inner_state = self._inner_query.reset_state(noised_results, inner_state)
    return (
        noised_results,
        _RestartQueryGlobalState(inner_state, indicator_state),
        event,
    )

  def derive_metrics(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_metrics`."""
    return self._inner_query.derive_metrics(global_state.inner_query_state)


class TreeCumulativeSumQuery(query.SumAggregationDPQuery):
  """Returns private cumulative sums by clipping and adding correlated noise.

  Consider calling `get_noised_result` T times, and each (x_i, i=0,2,...,T-1) is
  the private value returned by `accumulate_record`, i.e. x_i = sum_{j=0}^{n-1}
  x_{i,j} where each x_{i,j} is a private record in the database. This class is
  intended to make multiple queries, which release privatized values of the
  cumulative sums s_i = sum_{k=0}^{i} x_k, for i=0,...,T-1.
  Each call to `get_noised_result` releases the next cumulative sum s_i, which
  is in contrast to the GaussianSumQuery that releases x_i. Noise for the
  cumulative sums is accomplished using the tree aggregation logic in
  `tree_aggregation`, which is proportional to log(T).

  Example usage:
    query = TreeCumulativeSumQuery(...)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i, samples in enumerate(streaming_samples):
      sample_state = query.initial_sample_state(samples[0])
      # Compute  x_i = sum_{j=0}^{n-1} x_{i,j}
      for j,sample in enumerate(samples):
        sample_state = query.accumulate_record(params, sample_state, sample)
      # noised_cumsum is privatized estimate of s_i
      noised_cumsum, global_state, event = query.get_noised_result(
        sample_state, global_state)

  Attributes:
    clip_fn: Callable that specifies clipping function. `clip_fn` receives two
      arguments: a flat list of vars in a record and a `clip_value` to clip the
        corresponding record, e.g. clip_fn(flat_record, clip_value).
    clip_value: float indicating the value at which to clip the record.
    record_specs: `Collection[tf.TensorSpec]` specifying shapes of records.
    tree_aggregator: `tree_aggregation.TreeAggregator` initialized with user
      defined `noise_generator`. `noise_generator` is a
      `tree_aggregation.ValueGenerator` to generate the noise value for a tree
      node. Noise stdandard deviation is specified outside the `dp_query` by the
      user when defining `noise_fn` and should have order
      O(clip_norm*log(T)/eps) to guarantee eps-DP.
  """

  class GlobalState(NamedTuple):
    """Class defining global state for Tree sum queries.

    Attributes:
      tree_state: Current state of noise tree keeping track of current leaf and
        each level state.
      clip_value: The clipping value to be passed to clip_fn.
      samples_cumulative_sum: Noiseless cumulative sum of samples over time.
    """

    tree_state: Any
    clip_value: Any
    samples_cumulative_sum: Any

  def __init__(
      self,
      record_specs,
      noise_generator,
      clip_fn,
      clip_value,
      use_efficient=True,
  ):
    """Initializes the `TreeCumulativeSumQuery`.

    Consider using `build_l2_gaussian_query` for the construction of a
    `TreeCumulativeSumQuery` with L2 norm clipping and Gaussian noise.

    Args:
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_generator: `tree_aggregation.ValueGenerator` to generate the noise
        value for a tree node. Should be coupled with clipping norm to guarantee
        privacy.
      clip_fn: Callable that specifies clipping function. Input to clip is a
        flat list of vars in a record.
      clip_value: Float indicating the value at which to clip the record.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    self._clip_fn = clip_fn
    self._clip_value = clip_value
    self._record_specs = record_specs
    if use_efficient:
      self._tree_aggregator = EfficientTreeAggregator(noise_generator)
    else:
      self._tree_aggregator = TreeAggregator(noise_generator)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    initial_tree_state = self._tree_aggregator.init_state()
    initial_samples_cumulative_sum = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape), self._record_specs
    )
    return TreeCumulativeSumQuery.GlobalState(
        tree_state=initial_tree_state,
        clip_value=tf.constant(self._clip_value, tf.float32),
        samples_cumulative_sum=initial_samples_cumulative_sum,
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.clip_value

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    Args:
      params: `clip_value` for the record.
      record: The record to be processed.

    Returns:
      Structure of clipped tensors.
    """
    clip_value = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list = self._clip_fn(record_as_list, clip_value)
    return tf.nest.pack_sequence_as(record, clipped_as_list)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`.

    Updates tree state, and returns noised cumulative sum and updated state.

    Computes new cumulative sum, and returns its noised value. Grows tree state
    by one new leaf, and returns the new state.

    Args:
      sample_state: Sum of clipped records for this round.
      global_state: Global state with current sample's cumulative sum and tree
        state.

    Returns:
      A tuple of (noised_cumulative_sum, new_global_state).
    """
    new_cumulative_sum = tf.nest.map_structure(
        tf.add, global_state.samples_cumulative_sum, sample_state
    )
    cumulative_sum_noise, new_tree_state = (
        self._tree_aggregator.get_cumsum_and_update(global_state.tree_state)
    )
    noised_cumulative_sum = tf.nest.map_structure(
        tf.add, new_cumulative_sum, cumulative_sum_noise
    )
    new_global_state = TreeCumulativeSumQuery.GlobalState(
        tree_state=new_tree_state,
        clip_value=global_state.clip_value,
        samples_cumulative_sum=new_cumulative_sum,
    )
    event = dp_accounting.UnsupportedDpEvent()
    return noised_cumulative_sum, new_global_state, event

  def reset_state(self, noised_results, global_state):
    """Returns state after resetting the tree.

    This function will be used in `restart_query.RestartQuery` after calling
    `get_noised_result` when the restarting condition is met.

    Args:
      noised_results: Noised cumulative sum returned by `get_noised_result`.
      global_state: Updated global state returned by `get_noised_result`, which
        has current sample's cumulative sum and tree state for the next
        cumulative sum.

    Returns:
      New global state with current noised cumulative sum and restarted tree
        state for the next cumulative sum.
    """
    new_tree_state = self._tree_aggregator.reset_state(global_state.tree_state)
    return TreeCumulativeSumQuery.GlobalState(
        tree_state=new_tree_state,
        clip_value=global_state.clip_value,
        samples_cumulative_sum=noised_results,
    )

  @classmethod
  def build_l2_gaussian_query(
      cls,
      clip_norm,
      noise_multiplier,
      record_specs,
      noise_seed=None,
      use_efficient=True,
  ):
    """Returns a query instance with L2 norm clipping and Gaussian noise.

    Args:
      clip_norm: Each record will be clipped so that it has L2 norm at most
        `clip_norm`.
      noise_multiplier: The effective noise multiplier for the sum of records.
        Noise standard deviation is `clip_norm*noise_multiplier`. The value can
        be used as the input of the privacy accounting functions in
        `analysis.tree_aggregation_accountant`.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_seed: Integer seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    if clip_norm <= 0:
      raise ValueError(f'`clip_norm` must be positive, got {clip_norm}.')

    if noise_multiplier < 0:
      raise ValueError(
          f'`noise_multiplier` must be non-negative, got {noise_multiplier}.'
      )

    gaussian_noise_generator = GaussianNoiseGenerator(
        noise_std=clip_norm * noise_multiplier,
        specs=record_specs,
        seed=noise_seed,
    )

    def l2_clip_fn(record_as_list, clip_norm):
      clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_norm)
      return clipped_record

    return cls(
        clip_fn=l2_clip_fn,
        clip_value=clip_norm,
        record_specs=record_specs,
        noise_generator=gaussian_noise_generator,
        use_efficient=use_efficient,
    )


class TreeResidualSumQuery(query.SumAggregationDPQuery):
  """Implements DPQuery for adding correlated noise through tree structure.

  Clips and sums records in current sample x_i = sum_{j=0}^{n-1} x_{i,j};
  returns the current sample adding the noise residual from tree aggregation.
  The returned value is conceptually equivalent to the following: calculates
  cumulative sum of samples over time s_i = sum_{k=0}^i x_i (instead of only
  current sample) with added noise by tree aggregation protocol that is
  proportional to log(T), T being the number of times the query is called; r
  eturns the residual between the current noised cumsum noised(s_i) and the
  previous one noised(s_{i-1}) when the query is called.

  This can be used as a drop-in replacement for `GaussianSumQuery`, and can
  offer stronger utility/privacy tradeoffs when aplification-via-sampling is not
  possible, or when privacy epsilon is relativly large.  This may result in
  more noise by a log(T) factor in each individual estimate of x_i, but if the
  x_i are used in the underlying code to compute cumulative sums, the noise in
  those sums can be less. That is, this allows us to adapt code that was written
  to use a regular `SumQuery` to benefit from the tree aggregation protocol.

  Combining this query with a SGD optimizer can be used to implement the
  DP-FTRL algorithm in
  "Practical and Private (Deep) Learning without Sampling or Shuffling".

  Example usage:
    query = TreeResidualSumQuery(...)
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)
    for i, samples in enumerate(streaming_samples):
      sample_state = query.initial_sample_state(samples[0])
      # Compute  x_i = sum_{j=0}^{n-1} x_{i,j}
      for j,sample in enumerate(samples):
        sample_state = query.accumulate_record(params, sample_state, sample)
      # noised_sum is privatized estimate of x_i by conceptually postprocessing
      # noised cumulative sum s_i
      noised_sum, global_state, event = query.get_noised_result(
        sample_state, global_state)

  Attributes:
    clip_fn: Callable that specifies clipping function. `clip_fn` receives two
      arguments: a flat list of vars in a record and a `clip_value` to clip the
        corresponding record, e.g. clip_fn(flat_record, clip_value).
    clip_value: float indicating the value at which to clip the record.
    record_specs: A nested structure of `tf.TensorSpec`s specifying structure
      and shapes of records.
    tree_aggregator: `TreeAggregator` initialized with user defined
      `noise_generator`. `noise_generator` is a `tree.ValueGenerator` to
      generate the noise value for a tree node. Noise stdandard deviation is
      specified outside the `query` by the user when defining `noise_fn` and
      should have order O(clip_norm*log(T)/eps) to guarantee eps-DP.
  """

  class GlobalState(NamedTuple):
    """Class defining global state for Tree sum queries.

    Attributes:
      tree_state: Current state of noise tree keeping track of current leaf and
        each level state.
      clip_value: The clipping value to be passed to clip_fn.
      previous_tree_noise: Cumulative noise by tree aggregation from the
        previous time the query is called on a sample.
    """

    tree_state: Any
    clip_value: Any
    previous_tree_noise: Any

  def __init__(
      self,
      record_specs,
      noise_generator,
      clip_fn,
      clip_value,
      use_efficient=True,
  ):
    """Initializes the `TreeCumulativeSumQuery`.

    Consider using `build_l2_gaussian_query` for the construction of a
    `TreeCumulativeSumQuery` with L2 norm clipping and Gaussian noise.

    Args:
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_generator: `tree.ValueGenerator` to generate the noise value for a
        tree node. Should be coupled with clipping norm to guarantee privacy.
      clip_fn: Callable that specifies clipping function. Input to clip is a
        flat list of vars in a record.
      clip_value: Float indicating the value at which to clip the record.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """

    self._clip_fn = clip_fn
    self._clip_value = clip_value
    self._record_specs = record_specs
    if use_efficient:
      self._tree_aggregator = EfficientTreeAggregator(noise_generator)
    else:
      self._tree_aggregator = TreeAggregator(noise_generator)

  def _zero_initial_noise(self):
    return tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape), self._record_specs
    )

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    initial_tree_state = self._tree_aggregator.init_state()
    return TreeResidualSumQuery.GlobalState(
        tree_state=initial_tree_state,
        clip_value=tf.constant(self._clip_value, tf.float32),
        previous_tree_noise=self._zero_initial_noise(),
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.clip_value

  def preprocess_record_l2_impl(self, params, record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return tf.nest.pack_sequence_as(record, clipped_as_list), norm

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    Args:
      params: `clip_value` for the record.
      record: The record to be processed.

    Returns:
      Structure of clipped tensors.
    """
    clip_value = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list = self._clip_fn(record_as_list, clip_value)
    return tf.nest.pack_sequence_as(record, clipped_as_list)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`.

    Updates tree state, and returns residual of noised cumulative sum.

    Args:
      sample_state: Sum of clipped records for this round.
      global_state: Global state with current samples cumulative sum and tree
        state.

    Returns:
      A tuple of (noised_cumulative_sum, new_global_state).
    """
    tree_noise, new_tree_state = self._tree_aggregator.get_cumsum_and_update(
        global_state.tree_state
    )
    noised_sample = tf.nest.map_structure(
        lambda a, b, c: a + b - c,
        sample_state,
        tree_noise,
        global_state.previous_tree_noise,
    )
    new_global_state = TreeResidualSumQuery.GlobalState(
        tree_state=new_tree_state,
        clip_value=global_state.clip_value,
        previous_tree_noise=tree_noise,
    )
    event = dp_accounting.UnsupportedDpEvent()
    return noised_sample, new_global_state, event

  def reset_state(self, noised_results, global_state):
    """Returns state after resetting the tree.

    This function will be used in `restart_query.RestartQuery` after calling
    `get_noised_result` when the restarting condition is met.

    Args:
      noised_results: Noised results returned by `get_noised_result`.
      global_state: Updated global state returned by `get_noised_result`, which
        records noise for the conceptual cumulative sum of the current leaf
        node, and tree state for the next conceptual cumulative sum.

    Returns:
      New global state with zero noise and restarted tree state.
    """
    del noised_results
    new_tree_state = self._tree_aggregator.reset_state(global_state.tree_state)
    return TreeResidualSumQuery.GlobalState(
        tree_state=new_tree_state,
        clip_value=global_state.clip_value,
        previous_tree_noise=self._zero_initial_noise(),
    )

  def reset_l2_clip_gaussian_noise(self, global_state, clip_norm, stddev):
    noise_generator_state = global_state.tree_state.value_generator_state
    assert isinstance(
        self._tree_aggregator.value_generator, GaussianNoiseGenerator
    )
    noise_generator_state = self._tree_aggregator.value_generator.make_state(
        noise_generator_state.seeds, stddev
    )
    new_tree_state = TreeState(
        level_buffer=global_state.tree_state.level_buffer,
        level_buffer_idx=global_state.tree_state.level_buffer_idx,
        value_generator_state=noise_generator_state,
    )
    return TreeResidualSumQuery.GlobalState(
        tree_state=new_tree_state,
        clip_value=clip_norm,
        previous_tree_noise=global_state.previous_tree_noise,
    )

  def derive_metrics(self, global_state):
    """Returns the clip norm as a metric."""
    return collections.OrderedDict(tree_agg_dpftrl_clip=global_state.clip_value)

  @classmethod
  def build_l2_gaussian_query(
      cls,
      clip_norm,
      noise_multiplier,
      record_specs,
      noise_seed=None,
      use_efficient=True,
  ):
    """Returns `TreeResidualSumQuery` with L2 norm clipping and Gaussian noise.

    Args:
      clip_norm: Each record will be clipped so that it has L2 norm at most
        `clip_norm`.
      noise_multiplier: The effective noise multiplier for the sum of records.
        Noise standard deviation is `clip_norm*noise_multiplier`. The value can
        be used as the input of the privacy accounting functions in
        `analysis.tree_aggregation_accountant`.
      record_specs: A nested structure of `tf.TensorSpec`s specifying structure
        and shapes of records.
      noise_seed: Integer seed for the Gaussian noise generator. If `None`, a
        nondeterministic seed based on system time will be generated.
      use_efficient: Boolean indicating the usage of the efficient tree
        aggregation algorithm based on the paper "Efficient Use of
        Differentially Private Binary Trees".
    """
    if clip_norm < 0:
      raise ValueError(f'`clip_norm` must be non-negative, got {clip_norm}.')

    if noise_multiplier < 0:
      raise ValueError(
          f'`noise_multiplier` must be non-negative, got {noise_multiplier}.'
      )

    gaussian_noise_generator = GaussianNoiseGenerator(
        noise_std=clip_norm * noise_multiplier,
        specs=record_specs,
        seed=noise_seed,
    )

    def l2_clip_fn(record_as_list, clip_norm):
      clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_norm)
      return clipped_record

    return cls(
        clip_fn=l2_clip_fn,
        clip_value=clip_norm,
        record_specs=record_specs,
        noise_generator=gaussian_noise_generator,
        use_efficient=use_efficient,
    )


def _build_tree_from_leaf(leaf_nodes: tf.Tensor, arity: int) -> tf.RaggedTensor:
  """A function constructs a complete tree given all the leaf nodes.

  The function takes a 1-D array representing the leaf nodes of a tree and the
  tree's arity, and constructs a complete tree by recursively summing the
  adjacent children to get the parent until reaching the root node. Because we
  assume a complete tree, if the number of leaf nodes does not divide arity, the
  leaf nodes will be padded with zeros.

  Args:
    leaf_nodes: A 1-D array storing the leaf nodes of the tree.
    arity: A `int` for the branching factor of the tree, i.e. the number of
      children for each internal node.

  Returns:
    `tf.RaggedTensor` representing the tree. For example, if
    `leaf_nodes=tf.Tensor([1, 2, 3, 4])` and `arity=2`, then the returned value
    should be `tree=tf.RaggedTensor([[10],[3,7],[1,2,3,4]])`. In this way,
    `tree[layer][index]` can be used to access the node indexed by (layer,
    index) in the tree,
  """

  def pad_zero(leaf_nodes, size):
    paddings = tf.zeros(
        shape=(size - leaf_nodes.shape[0],), dtype=leaf_nodes.dtype
    )
    return tf.concat((leaf_nodes, paddings), axis=0)

  leaf_nodes_size = tf.constant(leaf_nodes.shape[0], dtype=tf.float32)
  num_layers = (
      tf.math.ceil(
          tf.math.log(leaf_nodes_size)
          / tf.math.log(tf.cast(arity, dtype=tf.float32))
      )
      + 1
  )
  leaf_nodes = pad_zero(
      leaf_nodes, tf.math.pow(tf.cast(arity, dtype=tf.float32), num_layers - 1)
  )

  def _shrink_layer(layer: tf.Tensor, arity: int) -> tf.Tensor:
    return tf.reduce_sum((tf.reshape(layer, (-1, arity))), 1)

  # The following `tf.while_loop` constructs the tree from bottom up by
  # iteratively applying `_shrink_layer` to each layer of the tree. The reason
  # for the choice of TF1.0-style `tf.while_loop` is that @tf.function does not
  # support auto-translation from python loop to tf loop when loop variables
  # contain a `RaggedTensor` whose shape changes across iterations.

  idx = tf.identity(num_layers)
  loop_cond = lambda i, h: tf.less_equal(2.0, i)

  def _loop_body(i, h):
    return [
        tf.add(i, -1.0),
        tf.concat(([_shrink_layer(h[0], arity)], h), axis=0),
    ]

  _, tree = tf.while_loop(
      loop_cond,
      _loop_body,
      [idx, tf.RaggedTensor.from_tensor([leaf_nodes])],
      shape_invariants=[
          idx.get_shape(),
          tf.RaggedTensorSpec(dtype=leaf_nodes.dtype, ragged_rank=1),
      ],
  )

  return tree


class TreeRangeSumQuery(query.SumAggregationDPQuery):
  """Implements dp_query for accurate range queries using tree aggregation.

  Implements a variant of the tree aggregation protocol from. "Is interaction
  necessary for distributed private learning?. Adam Smith, Abhradeep Thakurta,
  Jalaj Upadhyay." Builds a tree on top of the input record and adds noise to
  the tree for differential privacy. Any range query can be decomposed into the
  sum of O(log(n)) nodes in the tree compared to O(n) when using a histogram.
  Improves efficiency and reduces noise scale.
  """

  class GlobalState(NamedTuple):
    """Class defining global state for TreeRangeSumQuery.

    Attributes:
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has).
      inner_query_state: The global state of the inner query.
    """

    arity: Any
    inner_query_state: Any

  def __init__(self, inner_query: query.SumAggregationDPQuery, arity: int = 2):
    """Initializes the `TreeRangeSumQuery`.

    Args:
      inner_query: The inner `DPQuery` that adds noise to the tree.
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has). Defaults to 2.
    """
    self._inner_query = inner_query
    self._arity = arity

    if self._arity < 1:
      raise ValueError(f'Invalid arity={arity} smaller than 2.')

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return TreeRangeSumQuery.GlobalState(
        arity=self._arity,
        inner_query_state=self._inner_query.initial_global_state(),
    )

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return self.preprocess_record(
        self.derive_sample_params(self.initial_global_state()),
        super().initial_sample_state(template),
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return (
        global_state.arity,
        self._inner_query.derive_sample_params(global_state.inner_query_state),
    )

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    This method builds the tree, flattens it and applies
    `inner_query.preprocess_record` to the flattened tree.

    Args:
      params: Hyper-parameters for preprocessing record.
      record: A histogram representing the leaf nodes of the tree.

    Returns:
      A `tf.Tensor` representing the flattened version of the preprocessed tree.
    """
    arity, inner_query_params = params
    preprocessed_record = _build_tree_from_leaf(record, arity).flat_values
    # The following codes reshape the output vector so the output shape of can
    # be statically inferred. This is useful when used with
    # `tff.aggregators.DifferentiallyPrivateFactory` because it needs to know
    # the output shape of this function statically and explicitly.
    preprocessed_record_shape = [
        (
            self._arity
            ** (math.ceil(math.log(record.shape[0], self._arity)) + 1)
            - 1
        )
        // (self._arity - 1)
    ]
    preprocessed_record = tf.reshape(
        preprocessed_record, preprocessed_record_shape
    )
    preprocessed_record = self._inner_query.preprocess_record(
        inner_query_params, preprocessed_record
    )

    return preprocessed_record

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`.

    This function re-constructs the `tf.RaggedTensor` from the flattened tree
    output by `preprocess_records.`

    Args:
      sample_state: A `tf.Tensor` for the flattened tree.
      global_state: The global state of the protocol.

    Returns:
      A `tf.RaggedTensor` representing the tree.
    """
    # The [0] is needed because of how tf.RaggedTensor.from_two_splits works.
    # print(tf.RaggedTensor.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6],
    #                                       row_splits=[0, 4, 4, 7, 8, 8]))
    # <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
    # This part is not written in tensorflow and will be executed on the server
    # side instead of the client side if used with
    # tff.aggregators.DifferentiallyPrivateFactory for federated learning.
    sample_state, inner_query_state, _ = self._inner_query.get_noised_result(
        sample_state, global_state.inner_query_state
    )
    new_global_state = TreeRangeSumQuery.GlobalState(
        arity=global_state.arity, inner_query_state=inner_query_state
    )

    row_splits = [0] + [
        (self._arity ** (x + 1) - 1) // (self._arity - 1)
        for x in range(
            math.floor(math.log(sample_state.shape[0], self._arity)) + 1
        )
    ]
    tree = tf.RaggedTensor.from_row_splits(
        values=sample_state, row_splits=row_splits
    )
    event = dp_accounting.UnsupportedDpEvent()
    return tree, new_global_state, event

  @classmethod
  def build_central_gaussian_query(
      cls, l2_norm_clip: float, stddev: float, arity: int = 2
  ):
    """Returns `TreeRangeSumQuery` with central Gaussian noise.

    Args:
      l2_norm_clip: Each record should be clipped so that it has L2 norm at most
        `l2_norm_clip`.
      stddev: Stddev of the central Gaussian noise.
      arity: The branching factor of the tree (i.e. the number of children each
        internal node has). Defaults to 2.
    """
    if l2_norm_clip <= 0:
      raise ValueError(f'`l2_norm_clip` must be positive, got {l2_norm_clip}.')

    if stddev < 0:
      raise ValueError(f'`stddev` must be non-negative, got {stddev}.')

    if arity < 2:
      raise ValueError(f'`arity` must be at least 2, got {arity}.')

    inner_query = query.GaussianSumQuery(l2_norm_clip, stddev)

    return cls(arity=arity, inner_query=inner_query)


class ValueGenerator(metaclass=abc.ABCMeta):
  """Base class establishing interface for stateful value generation.

  A `ValueGenerator` maintains a state, and each time `next` is called, a new
  value is generated and the state is advanced.
  """

  @abc.abstractmethod
  def initialize(self):
    """Makes an initialized state for the ValueGenerator.

    Returns:
      An initial state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def next(self, state):
    """Gets next value and advances the ValueGenerator.

    Args:
      state: The current state.

    Returns:
      A pair (value, new_state) where value is the next value and new_state
        is the advanced state.
    """
    raise NotImplementedError


class GaussianNoiseGenerator(ValueGenerator):
  """Gaussian noise generator with counter as pseudo state.

  Produces i.i.d. spherical Gaussian noise at each step shaped according to a
  nested structure of `tf.TensorSpec`s.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState', ['seeds', 'stddev'])

  def __init__(
      self,
      noise_std: float,
      specs: Collection[tf.TensorSpec],
      seed: Optional[int] = None,
  ):
    """Initializes the GaussianNoiseGenerator.

    Args:
      noise_std: The standard deviation of the noise.
      specs: A nested structure of `tf.TensorSpec`s specifying the shape of the
        noise to generate.
      seed: An optional integer seed. If None, generator is seeded from the
        clock.
    """
    self._noise_std = noise_std
    self._specs = specs
    self._seed = seed

  def initialize(self):
    """Makes an initial state for the GaussianNoiseGenerator.

    Returns:
      A named tuple of (seeds, stddev).
    """
    if self._seed is None:
      time_now = tf.timestamp()
      residual = time_now - tf.math.floor(time_now)
      return self._GlobalState(
          tf.cast(
              tf.stack([
                  tf.math.floor(tf.timestamp() * 1e6),
                  tf.math.floor(residual * 1e9),
              ]),
              dtype=tf.int64,
          ),
          tf.constant(self._noise_std, dtype=tf.float32),
      )
    else:
      return self._GlobalState(
          tf.constant(self._seed, dtype=tf.int64, shape=(2,)),
          tf.constant(self._noise_std, dtype=tf.float32),
      )

  def next(self, state):
    """Gets next value and advances the GaussianNoiseGenerator.

    Args:
      state: The current state (seed, noise_std).

    Returns:
      A tuple of (sample, new_state) where sample is a new sample and new_state
        is the advanced state (seed+1, noise_std).
    """
    flat_structure = tf.nest.flatten(self._specs)
    flat_seeds = [state.seeds + i for i in range(len(flat_structure))]
    nest_seeds = tf.nest.pack_sequence_as(self._specs, flat_seeds)

    def _get_noise(spec, seed):
      return tf.random.stateless_normal(
          shape=spec.shape, seed=seed, stddev=state.stddev
      )

    nest_noise = tf.nest.map_structure(_get_noise, self._specs, nest_seeds)
    return nest_noise, self._GlobalState(flat_seeds[-1] + 1, state.stddev)

  def make_state(self, seeds: tf.Tensor, stddev: tf.Tensor):
    """Returns a new named tuple of (seeds, stddev)."""
    seeds = tf.ensure_shape(seeds, shape=(2,))
    return self._GlobalState(
        tf.cast(seeds, dtype=tf.int64), tf.cast(stddev, dtype=tf.float32)
    )


class StatelessValueGenerator(ValueGenerator):
  """A wrapper for stateless value generator that calls a no-arg function."""

  def __init__(self, value_fn):
    """Initializes the StatelessValueGenerator.

    Args:
      value_fn: The function to call to generate values.
    """
    self.value_fn = value_fn

  def initialize(self):
    """Makes an initialized state for the StatelessValueGenerator.

    Returns:
      An initial state (empty, because stateless).
    """
    return ()

  def next(self, state):
    """Gets next value.

    Args:
      state: The current state (simply passed through).

    Returns:
      A pair (value, new_state) where value is the next value and new_state
        is the advanced state.
    """
    return self.value_fn(), state


class TreeState(NamedTuple):
  """Class defining state of the tree.

  Attributes:
    level_buffer: A `tf.Tensor` saves the last node value of the left child
      entered for the tree levels recorded in `level_buffer_idx`.
    level_buffer_idx: A `tf.Tensor` for the tree level index of the
      `level_buffer`.  The tree level index starts from 0, i.e.,
      `level_buffer[0]` when `level_buffer_idx[0]==0` recorded the noise value
      for the most recent leaf node.
   value_generator_state: State of a stateful `ValueGenerator` for tree node.
  """

  level_buffer: tf.Tensor
  level_buffer_idx: tf.Tensor
  value_generator_state: Any


class TreeAggregator:
  """Tree aggregator to compute accumulated noise in private algorithms.

  This class implements the tree aggregation algorithm for noise values to
  efficiently privatize streaming algorithms based on Dwork et al. (2010)
  https://dl.acm.org/doi/pdf/10.1145/1806689.1806787. A buffer at the scale of
  tree depth is maintained and updated when a new conceptual leaf node arrives.

  Example usage:
    random_generator = GaussianNoiseGenerator(...)
    tree_aggregator = TreeAggregator(random_generator)
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      assert leaf_node_idx == get_step_idx(state))
      noise, state = tree_aggregator.get_cumsum_and_update(state)

  Attributes:
    value_generator: A `ValueGenerator` or a no-arg function to generate a noise
      value for each tree node.
  """

  def __init__(self, value_generator: Union[ValueGenerator, Callable[[], Any]]):
    """Initialize the aggregator with a noise generator.

    Args:
      value_generator: A `ValueGenerator` or a no-arg function to generate a
        noise value for each tree node.
    """
    if isinstance(value_generator, ValueGenerator):
      self.value_generator = value_generator
    else:
      self.value_generator = StatelessValueGenerator(value_generator)

  def _get_init_state(self, value_generator_state) -> TreeState:
    """Returns initial `TreeState` given `value_generator_state`."""
    level_buffer_idx = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    level_buffer_idx = level_buffer_idx.write(
        0, tf.constant(0, dtype=tf.int32)
    ).stack()

    new_val, value_generator_state = self.value_generator.next(
        value_generator_state
    )
    level_buffer_structure = tf.nest.map_structure(
        lambda x: tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True),
        new_val,
    )
    level_buffer = tf.nest.map_structure(
        lambda x, y: x.write(0, y).stack(), level_buffer_structure, new_val
    )
    return TreeState(
        level_buffer=level_buffer,
        level_buffer_idx=level_buffer_idx,
        value_generator_state=value_generator_state,
    )

  def init_state(self) -> TreeState:
    """Returns initial `TreeState`.

    Initializes `TreeState` for a tree of a single leaf node: the respective
    initial node value in `TreeState.level_buffer` is generated by the value
    generator function, and the node index is 0.

    Returns:
      An initialized `TreeState`.
    """
    value_generator_state = self.value_generator.initialize()
    return self._get_init_state(value_generator_state)

  def reset_state(self, state: TreeState) -> TreeState:
    """Returns reset `TreeState` after restarting a new tree."""
    return self._get_init_state(state.value_generator_state)

  @tf.function
  def _get_cumsum(self, level_buffer: Collection[tf.Tensor]) -> tf.Tensor:
    return tf.nest.map_structure(
        lambda x: tf.reduce_sum(x, axis=0), level_buffer
    )

  @tf.function
  def get_cumsum_and_update(
      self, state: TreeState
  ) -> tuple[tf.Tensor, TreeState]:
    """Returns tree aggregated noise and updates `TreeState` for the next step.

    `TreeState` is updated to prepare for accepting the *next* leaf node. Note
    that `get_step_idx` can be called to get the current index of the leaf node
    before calling this function. This function accept state for the current
    leaf node and prepare for the next leaf node because TFF prefers to know
    the types of state at initialization.

    Args:
      state: `TreeState` for the current leaf node, index can be queried by
        `get_step_idx(state.level_buffer_idx)`.

    Returns:
      Tuple of (noise, state) where `noise` is generated by tree aggregated
      protocol for the cumulative sum of streaming data, and `state` is the
      updated `TreeState`.
    """

    level_buffer_idx, level_buffer, value_generator_state = (
        state.level_buffer_idx,
        state.level_buffer,
        state.value_generator_state,
    )
    # We only publicize a combined function for updating state and returning
    # noised results because this DPQuery is designed for the streaming data,
    # and we only maintain a dynamic memory buffer of max size logT. Only the
    # the most recent noised results can be queried, and the queries are
    # expected to happen for every step in the streaming setting.
    cumsum = self._get_cumsum(level_buffer)

    new_level_buffer = tf.nest.map_structure(
        lambda x: tf.TensorArray(  # pylint: disable=g-long-lambda
            dtype=tf.float32, size=0, dynamic_size=True
        ),
        level_buffer,
    )
    new_level_buffer_idx = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True
    )
    # `TreeState` stores the left child node necessary for computing the cumsum
    # noise. To update the buffer, let us find the lowest level that will switch
    # from a right child (not in the buffer) to a left child.
    level_idx = 0  # new leaf node starts from level 0
    while tf.less(level_idx, len(level_buffer_idx)) and tf.equal(
        level_idx, level_buffer_idx[level_idx]
    ):
      level_idx += 1
    # Left child nodes for the level lower than `level_idx` will be removed
    # and a new node will be created at `level_idx`.
    write_buffer_idx = 0
    new_level_buffer_idx = new_level_buffer_idx.write(
        write_buffer_idx, level_idx
    )
    new_value, value_generator_state = self.value_generator.next(
        value_generator_state
    )
    new_level_buffer = tf.nest.map_structure(
        lambda x, y: x.write(write_buffer_idx, y), new_level_buffer, new_value
    )
    write_buffer_idx += 1
    # Buffer index will now different from level index for the old `TreeState`
    # i.e., `level_buffer_idx[level_idx] != level_idx`. Rename parameter to
    # buffer index for clarity.
    buffer_idx = level_idx
    while tf.less(buffer_idx, len(level_buffer_idx)):
      new_level_buffer_idx = new_level_buffer_idx.write(
          write_buffer_idx, level_buffer_idx[buffer_idx]
      )
      new_level_buffer = tf.nest.map_structure(
          lambda nb, b: nb.write(write_buffer_idx, b[buffer_idx]),
          new_level_buffer,
          level_buffer,
      )
      buffer_idx += 1
      write_buffer_idx += 1
    new_level_buffer_idx = new_level_buffer_idx.stack()
    new_level_buffer = tf.nest.map_structure(
        lambda x: x.stack(), new_level_buffer
    )
    new_state = TreeState(
        level_buffer=new_level_buffer,
        level_buffer_idx=new_level_buffer_idx,
        value_generator_state=value_generator_state,
    )
    return cumsum, new_state


class EfficientTreeAggregator:
  """Efficient tree aggregator to compute accumulated noise.

  This class implements the efficient tree aggregation algorithm based on
  Honaker 2015 "Efficient Use of Differentially Private Binary Trees".
  The noise standard deviation for a node at depth d is roughly
  `sigma * sqrt(2^{d-1}/(2^d-1))`. which becomes `sigma / sqrt(2)` when
  the tree is very tall.

  Example usage:
    random_generator = GaussianNoiseGenerator(...)
    tree_aggregator = EfficientTreeAggregator(random_generator)
    state = tree_aggregator.init_state()
    for leaf_node_idx in range(total_steps):
      assert leaf_node_idx == get_step_idx(state))
      noise, state = tree_aggregator.get_cumsum_and_update(state)

  Attributes:
    value_generator: A `ValueGenerator` or a no-arg function to generate a noise
      value for each tree node.
  """

  def __init__(self, value_generator: Union[ValueGenerator, Callable[[], Any]]):
    """Initialize the aggregator with a noise generator.

    Args:
      value_generator: A `ValueGenerator` or a no-arg function to generate a
        noise value for each tree node.
    """
    if isinstance(value_generator, ValueGenerator):
      self.value_generator = value_generator
    else:
      self.value_generator = StatelessValueGenerator(value_generator)

  def _get_init_state(self, value_generator_state):
    """Returns initial buffer for `TreeState`."""
    level_buffer_idx = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    level_buffer_idx = level_buffer_idx.write(
        0, tf.constant(0, dtype=tf.int32)
    ).stack()

    new_val, value_generator_state = self.value_generator.next(
        value_generator_state
    )
    level_buffer_structure = tf.nest.map_structure(
        lambda x: tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True),
        new_val,
    )
    level_buffer = tf.nest.map_structure(
        lambda x, y: x.write(0, y).stack(), level_buffer_structure, new_val
    )
    return TreeState(
        level_buffer=level_buffer,
        level_buffer_idx=level_buffer_idx,
        value_generator_state=value_generator_state,
    )

  def init_state(self) -> TreeState:
    """Returns initial `TreeState`.

    Initializes `TreeState` for a tree of a single leaf node: the respective
    initial node value in `TreeState.level_buffer` is generated by the value
    generator function, and the node index is 0.

    Returns:
      An initialized `TreeState`.
    """
    value_generator_state = self.value_generator.initialize()
    return self._get_init_state(value_generator_state)

  def reset_state(self, state: TreeState) -> TreeState:
    """Returns reset `TreeState` after restarting a new tree."""
    return self._get_init_state(state.value_generator_state)

  @tf.function
  def _get_cumsum(self, state: TreeState) -> tf.Tensor:
    """Returns weighted cumulative sum of noise based on `TreeState`."""
    # Note that the buffer saved recursive results of the weighted average of
    # the node value (v) and its two children (l, r), i.e., node = v + (l+r)/2.
    # To get unbiased estimation with reduced variance for each node, we have to
    # reweight it by 1/(2-2^{-d}) where d is the depth of the node.
    level_weights = tf.math.divide(
        1.0, 2.0 - tf.math.pow(0.5, tf.cast(state.level_buffer_idx, tf.float32))
    )

    def _weighted_sum(buffer):
      expand_shape = [len(level_weights)] + [1] * (len(tf.shape(buffer)) - 1)
      weighted_buffer = tf.math.multiply(
          buffer, tf.reshape(level_weights, expand_shape)
      )
      return tf.reduce_sum(weighted_buffer, axis=0)

    return tf.nest.map_structure(_weighted_sum, state.level_buffer)

  @tf.function
  def get_cumsum_and_update(
      self, state: TreeState
  ) -> tuple[tf.Tensor, TreeState]:
    """Returns tree aggregated noise and updates `TreeState` for the next step.

    `TreeState` is updated to prepare for accepting the *next* leaf node. Note
    that `get_step_idx` can be called to get the current index of the leaf node
    before calling this function. This function accept state for the current
    leaf node and prepare for the next leaf node because TFF prefers to know
    the types of state at initialization. Note that the value of new node in
    `TreeState.level_buffer` will depend on its two children, and is updated
    from bottom up for the right child.

    Args:
      state: `TreeState` for the current leaf node, index can be queried by
        `tree.get_step_idx(state.level_buffer_idx)`.

    Returns:
      Tuple of (noise, state) where `noise` is generated by tree aggregated
      protocol for the cumulative sum of streaming data, and `state` is the
      updated `TreeState`..
    """
    # We only publicize a combined function for updating state and returning
    # noised results because this DPQuery is designed for the streaming data,
    # and we only maintain a dynamic memory buffer of max size logT. Only the
    # the most recent noised results can be queried, and the queries are
    # expected to happen for every step in the streaming setting.
    cumsum = self._get_cumsum(state)

    level_buffer_idx, level_buffer, value_generator_state = (
        state.level_buffer_idx,
        state.level_buffer,
        state.value_generator_state,
    )
    new_level_buffer = tf.nest.map_structure(
        lambda x: tf.TensorArray(  # pylint: disable=g-long-lambda
            dtype=tf.float32, size=0, dynamic_size=True
        ),
        level_buffer,
    )
    new_level_buffer_idx = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True
    )
    # `TreeState` stores the left child node necessary for computing the cumsum
    # noise. To update the buffer, let us find the lowest level that will switch
    # from a right child (not in the buffer) to a left child.
    level_idx = 0  # new leaf node starts from level 0
    new_value, value_generator_state = self.value_generator.next(
        value_generator_state
    )
    while tf.less(level_idx, len(level_buffer_idx)) and tf.equal(
        level_idx, level_buffer_idx[level_idx]
    ):
      # Recursively update if the current node is a right child.
      node_value, value_generator_state = self.value_generator.next(
          value_generator_state
      )
      new_value = tf.nest.map_structure(
          lambda l, r, n: 0.5 * (l[level_idx] + r) + n,
          level_buffer,
          new_value,
          node_value,
      )
      level_idx += 1
    # A new (left) node will be created at `level_idx`.
    write_buffer_idx = 0
    new_level_buffer_idx = new_level_buffer_idx.write(
        write_buffer_idx, level_idx
    )
    new_level_buffer = tf.nest.map_structure(
        lambda x, y: x.write(write_buffer_idx, y), new_level_buffer, new_value
    )
    write_buffer_idx += 1
    # Buffer index will now different from level index for the old `TreeState`
    # i.e., `level_buffer_idx[level_idx] != level_idx`. Rename parameter to
    # buffer index for clarity.
    buffer_idx = level_idx
    while tf.less(buffer_idx, len(level_buffer_idx)):
      new_level_buffer_idx = new_level_buffer_idx.write(
          write_buffer_idx, level_buffer_idx[buffer_idx]
      )
      new_level_buffer = tf.nest.map_structure(
          lambda nb, b: nb.write(write_buffer_idx, b[buffer_idx]),
          new_level_buffer,
          level_buffer,
      )
      buffer_idx += 1
      write_buffer_idx += 1
    new_level_buffer_idx = new_level_buffer_idx.stack()
    new_level_buffer = tf.nest.map_structure(
        lambda x: x.stack(), new_level_buffer
    )
    new_state = TreeState(
        level_buffer=new_level_buffer,
        level_buffer_idx=new_level_buffer_idx,
        value_generator_state=value_generator_state,
    )
    return cumsum, new_state
