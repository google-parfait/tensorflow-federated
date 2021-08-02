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
"""Differentially private tree aggregation factory."""
import math
from typing import Optional

import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process

# Supported clip mechanisms.
CLIP_MECHANISMS = [
    'sub-sampling',
    'distinct',
]

# Supported DP mechanisms.
CENTRAL_DP_MECHANISMS = [
    'gaussian',  # Central Gaussian mechanism.
    'no-noise',  # Without noise.
]


def create_central_hierarchical_histogram_aggregation_factory(
    num_bins: int,
    arity: int = 2,
    clip_mechanism: str = 'sub-sampling',
    max_records_per_user: int = 10,
    dp_mechanism: str = 'gaussian',
    noise_multiplier: float = 0.0):
  """Creates hierarchical histogram aggregation factory with central DP.

  Args:
    num_bins: An `int` representing the input histogram size.
    arity: An `int` representing the branching factor of the tree. Defaults to
      2.
   clip_mechanism: A `str` representing the clipping mechanism. Currently
     supported mechanisms are
      - 'sub-sampling': (Default) Uniformly sample up to `max_records_per_user`
        records without replacement from the client dataset.
      - 'distinct': Uniquify client dataset and uniformly sample up to
        `max_records_per_user` records without replacement from it.
    max_records_per_user: An `int` representing the maximum of records each user
      can include in their local histogram. Defaults to 10.
    dp_mechanism: A `str` representing the differentially private mechanism to
      use. Currently supported mechanisms are
      - 'gaussian': (Default) Tree aggregation with Gaussian mechanism.
      - 'no-noise': Tree aggregation mechanism without noise.
    noise_multiplier: A `float` specifying the noise multiplier (central noise
      stddev / L2 clip norm) for model updates.

  Returns:
    `tff.aggregators.UnweightedAggregationFactory`.

  Raises:
    TypeError: If arguments have the wrong type(s).
    ValueError: If arguments have invalid value(s).
  """
  _check_positive(num_bins, 'num_bins')
  _check_greater_equal(arity, 2, 'arity')
  _check_membership(clip_mechanism, CLIP_MECHANISMS, 'clip_mechanism')
  _check_positive(max_records_per_user, 'max_records_per_user')
  _check_membership(dp_mechanism, CENTRAL_DP_MECHANISMS, 'dp_mechanism')
  _check_non_negative(noise_multiplier, 'noise_multiplier')

  # Build nested aggregtion factory from innermost to outermost.
  # 1. Sum factory.
  nested_factory = sum_factory.SumFactory()

  # 2. DP operations. Converts `max_records_per_user` to the corresponding norm
  # bound and constructs `DifferentiallyPrivateFactory`.
  if dp_mechanism == 'gaussian':
    if clip_mechanism == 'sub-sampling':
      l2_norm_bound = max_records_per_user * math.sqrt(
          _tree_depth(num_bins, arity))
    elif clip_mechanism == 'distinct':
      square_l2_norm_bound = 0.
      square_layer_l2_norm_bound = max_records_per_user
      for _ in range(_tree_depth(num_bins, arity)):
        square_l2_norm_bound += min(max_records_per_user**2,
                                    square_layer_l2_norm_bound)
        square_layer_l2_norm_bound *= arity
      l2_norm_bound = math.sqrt(square_l2_norm_bound)
    query = tfp.privacy.dp_query.tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
        l2_norm_bound, noise_multiplier * l2_norm_bound, arity)

  elif dp_mechanism == 'no-noise':
    inner_query = tfp.privacy.dp_query.no_privacy_query.NoPrivacySumQuery()
    query = tfp.privacy.dp_query.tree_aggregation_query.TreeRangeSumQuery(
        arity=arity, inner_query=inner_query)

  nested_factory = differential_privacy.DifferentiallyPrivateFactory(
      query, nested_factory)

  # 3. Clip as specified by `clip_mechanism`.
  nested_factory = _ClipFactory(
      clip_mechanism=clip_mechanism,
      max_records_per_user=max_records_per_user,
      inner_agg_factory=nested_factory,
      inner_agg_factory_dtype=tf.float32)

  return nested_factory


class _ClipFactory(factory.UnweightedAggregationFactory):
  """An `UnweightedAggregationFactory` for bounding client-side contribution.

  Supports two types of clipping on local histograms to bound client-side
  contribution.
  (1) Sub-sampling: Uniformly samples up to `max_records_per_user` records
  without replacement from the client dataset.

  For example:
      Input  = [1, 2, 3, 5], max_records_per_user=5;
      Example Outputs = [1, 2, 1, 1], [0, 0, 0, 5], [0, 2, 3, 0]

  (2) Distinct Sub-sampling: Uniquifies client dataset and uniformly samples up
  to `max_records_per_user` records without replacement from it.

  For example:
      Input  = [0, 2, 3, 5], max_records_per_user=2;
      Example Outputs = [0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]
  """

  def __init__(
      self,
      clip_mechanism: str = 'sub-sampling',
      max_records_per_user: int = 10,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      inner_agg_factory_dtype: Optional[tf.dtypes.DType] = None):
    """Initializes a `_ClipFactory` instance.

    Args:
      clip_mechanism: A `str` representing the clipping mechanism. Currently
        supported mechanisms are
      - 'sub-sampling': (Default) Uniformly sample up to `max_records_per_user`
        records without replacement from the client dataset.
      - 'distinct': Uniquify client dataset and uniformly sample up to
        `max_records_per_user` records without replacement from it.
      max_records_per_user: An `int` representing the maximum of records each
        user can include in their local histogram. Defaults to 10.
      inner_agg_factory: (Optional) A `UnweightedAggregationFactory` specifying
        the value aggregation to be wrapped by `_ClipFactory`. Defaults to
        `tff.aggregators.SumFactory`.
      inner_agg_factory_dtype: (Optional) The input value type for
        `inner_agg_factory`. Defaults to the `tf.int32`.

    Raises:
      TypeError: If arguments have the wrong type(s).
      ValueError: If arguments have invalid value(s).
    """
    _check_membership(clip_mechanism, CLIP_MECHANISMS, 'clip_mechanism')
    _check_positive(max_records_per_user, 'max_records_per_user')

    self._clip_mechanism = clip_mechanism
    self._max_records_per_user = max_records_per_user
    self._inner_agg_factory = inner_agg_factory
    if inner_agg_factory_dtype is None:
      self._inner_agg_factory_dtype = tf.int32
    else:
      _check_membership(inner_agg_factory_dtype, [tf.int32, tf.float32],
                        'inner_agg_factory_dtype')
      self._inner_agg_factory_dtype = inner_agg_factory_dtype

  def create(self, value_type):
    _check_is_tensor_type(value_type, 'value_type')
    _check_is_integer(value_type, 'value_type')

    if self._clip_mechanism == 'sub-sampling':
      clip_fn = _sub_sample_clip
    elif self._clip_mechanism == 'distinct':
      clip_fn = _distinct_clip

    if self._inner_agg_factory is None:
      self._inner_agg_factory = sum_factory.SumFactory()

    inner_value_type = computation_types.to_type(
        (self._inner_agg_factory_dtype, value_type.shape))
    inner_agg_process = self._inner_agg_factory.create(inner_value_type)
    init_fn = inner_agg_process.initialize

    tff_clip_fn = computations.tf_computation(clip_fn)
    tff_cast_fn = computations.tf_computation(
        lambda x: tf.cast(x, self._inner_agg_factory_dtype))

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.at_clients(value_type)
                                       )
    def next_fn(state, value):
      # Clip values before aggregation.
      clipped_value = intrinsics.federated_map(
          tff_clip_fn, (value,
                        intrinsics.federated_value(self._max_records_per_user,
                                                   placements.CLIENTS)))
      # `clipped_value` has dtype `tf.int32` but `inner_agg_process.next` might
      # require inputs with other dtypes. For example,
      # (1) `DifferentiallyPrivateFactory(query=GaussianSumQuery())`
      #     expects inputs of dtype `tf.float32`.
      # (2) `DifferentiallyPrivateFactory(query=
      #                                  DistributedDiscreteGaussianSumQuery())`
      #     expects inputs of dtype `tf.int32`.
      # To deal with various dtype requirements, `clip_value` is casted to
      # different dtypes below.
      clipped_value = intrinsics.federated_map(tff_cast_fn, clipped_value)

      return inner_agg_process.next(state, clipped_value)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


@tf.function
def _sub_sample_clip(histogram, sample_num):
  """Clips `value` by sub-sampling.

  Uniformly samples up to `max_records_per_user` records without replacement
  from the client dataset.

  Args:
    histogram: A `tf.Tensor` with `dtype=tf.int32` representing a histogram.
    sample_num: The number of samples to draw from the histogram. If histogram
      contains fewer elements than `sample_num`, returns the original histogram.

  Returns:
    A `tf.Tensor` with `dtype=tf.int32` representing the sub-sampled histogram.
  """

  def sub_sample():
    indices = tf.repeat(tf.range(tf.shape(histogram)[0]), histogram)
    seed = tf.cast(
        tf.stack([
            tf.math.floor(tf.timestamp() * 1e6),
            tf.math.floor(tf.math.log(tf.timestamp() * 1e6))
        ]),
        dtype=tf.int64)
    samples = tf.random.stateless_uniform(tf.shape(indices), seed)
    _, sampled_idx = tf.math.top_k(samples, k=sample_num, sorted=False)
    ind = tf.expand_dims(tf.gather(indices, sampled_idx), axis=1)
    upd = tf.ones(tf.shape(sampled_idx), dtype=tf.int32)
    return tf.scatter_nd(indices=ind, updates=upd, shape=tf.shape(histogram))

  l1_norm = tf.norm(histogram, ord=1)
  result = tf.cond(
      tf.greater(l1_norm, sample_num), sub_sample, lambda: histogram)
  return tf.reshape(result, tf.shape(histogram))


@tf.function
def _distinct_clip(histogram, sample_num):
  """Clips `value` by distinct sub-sampling.

  Uniquifies client dataset and uniformly samples up to `max_records_per_user`
  records without replacement from it.

  Args:
    histogram: A `tf.Tensor` with `dtype=tf.int32` representing a histogram.
    sample_num: The number of samples to draw from the histogram. If histogram
      contains fewer elements than `sample_num`, returns the uniquified
      histogram.

  Returns:
    A `tf.Tensor` with `dtype=tf.int32` representing the uniquified and
    sub-sampled histogram.
  """

  def distinct():
    indices = tf.cast(
        tf.squeeze(tf.where(tf.not_equal(histogram, 0))), tf.int32)
    seed = tf.cast(
        tf.stack([
            tf.math.floor(tf.timestamp() * 1e6),
            tf.math.floor(tf.math.log(tf.timestamp() * 1e6))
        ]),
        dtype=tf.int64)
    samples = tf.random.stateless_uniform(tf.shape(indices), seed)
    _, sampled_idx = tf.math.top_k(samples, k=sample_num, sorted=False)
    ind = tf.expand_dims(tf.gather(indices, sampled_idx), axis=1)
    upd = tf.ones(tf.shape(sampled_idx), dtype=tf.int32)
    return tf.scatter_nd(indices=ind, updates=upd, shape=tf.shape(histogram))

  l0_norm = tf.math.count_nonzero(histogram, dtype=tf.int32)
  result = tf.cond(
      tf.greater(l0_norm, sample_num), distinct,
      lambda: tf.minimum(histogram, 1))
  return tf.reshape(result, tf.shape(histogram))


def _check_greater_equal(value, threshold, label):
  if value < threshold:
    raise ValueError(f'`{label}` must be at least {threshold}, got {value}.')


def _check_positive(value, label):
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {value}.')


def _check_non_negative(value, label):
  if value < 0:
    raise ValueError(f'{label} must be non-negative. Found {value}.')


def _check_membership(value, valid_set, label):
  if value not in valid_set:
    raise ValueError(f'`{label}` must be one of {valid_set}. '
                     f'Found {value}.')


def _check_is_tensor_type(value, label):
  if not value.is_tensor():
    raise TypeError(f'Expected `{label}` to be `TensorType`. '
                    f'Found type: {repr(value)}')


def _check_is_integer(value_type, label):
  if not type_analysis.is_structure_of_integers(value_type):
    raise TypeError(f'Component dtypes of `{label}` must all be integers. '
                    f'Found {repr(value_type)}.')


def _check_is_integer_or_float(value_type, label):
  if (not type_analysis.is_structure_of_integers(value_type) and
      not type_analysis.is_structure_of_floats(value_type)):
    raise TypeError(
        f'Component dtypes of `{label}` must all be integers or floats. '
        f'Found {repr(value_type)}.')


def _tree_depth(num_leaves: int, arity: int):
  """Returns the depth of the tree given the number of leaf nodes and arity."""
  return math.ceil(math.log(num_leaves) / math.log(arity)) + 1
