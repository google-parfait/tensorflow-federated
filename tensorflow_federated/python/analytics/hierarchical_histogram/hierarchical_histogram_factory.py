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

import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.analytics.hierarchical_histogram import clipping_factory

# Supported no-noise mechanisms.
NO_NOISE_MECHANISMS = ['no-noise']

# Supported central DP mechanisms.
CENTRAL_DP_MECHANISMS = [
    'central-gaussian',  # Central Gaussian mechanism.
]

DP_MECHANISMS = CENTRAL_DP_MECHANISMS + NO_NOISE_MECHANISMS


def create_hierarchical_histogram_aggregation_factory(
    num_bins: int,
    arity: int = 2,
    clip_mechanism: str = 'sub-sampling',
    max_records_per_user: int = 10,
    dp_mechanism: str = 'central-gaussian',
    noise_multiplier: float = 0.0):
  """Creates hierarchical histogram aggregation factory.

  Hierarchical histogram factory is constructed by composing 3 aggregation
  factories.
  (1) The inner-most factory is `SumFactory`.
  (2) The middle factory is `DifferentiallyPrivateFactory` whose inner query is
      `TreeRangeSumQuery`. This factory 1) takes in a clipped histogram,
      constructs the hierarchical histogram and checks the norm bound of the
      hierarchical histogram at clients, 2) adds noise either at clients or at
      server according to `dp_mechanism`.
  (3) The outer-most factory is `HistogramClippingSumFactory` which clips the
      input histogram to bound each user's contribution.

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
      - 'central-gaussian': (Default) Tree aggregation with central Gaussian
        mechanism.
      - 'no-noise': Tree aggregation mechanism without noise.
    noise_multiplier: A `float` specifying the noise multiplier (central noise
      stddev / L2 clip norm) for model updates. Only needed when `dp_mechanism`
      is not 'no-noise'. Defaults to 0.0.

  Returns:
    `tff.aggregators.UnweightedAggregationFactory`.

  Raises:
    TypeError: If arguments have the wrong type(s).
    ValueError: If arguments have invalid value(s).
  """
  _check_positive(num_bins, 'num_bins')
  _check_greater_equal(arity, 2, 'arity')
  _check_membership(clip_mechanism, clipping_factory.CLIP_MECHANISMS,
                    'clip_mechanism')
  _check_positive(max_records_per_user, 'max_records_per_user')
  _check_membership(dp_mechanism, DP_MECHANISMS, 'dp_mechanism')
  _check_non_negative(noise_multiplier, 'noise_multiplier')

  # Build nested aggregtion factory from innermost to outermost.
  # 1. Sum factory. The most inner factory that sums the preprocessed records.
  nested_factory = sum_factory.SumFactory()

  # 2. DP operations.
  # (1) Converts `max_records_per_user` to the corresponding norm bound
  #     according to the chosen `clip_mechanism` and `dp_mechanism`.
  if dp_mechanism in ['central-gaussian']:
    if clip_mechanism == 'sub-sampling':
      l2_norm_bound = max_records_per_user * math.sqrt(
          _tree_depth(num_bins, arity))
    elif clip_mechanism == 'distinct':
      # The following code block converts `max_records_per_user` to L2 norm
      # bound of the hierarchical histogram layer by layer. For the bottom
      # layer with only 0s and at most `max_records_per_user` 1s, the L2 norm
      # bound is `sqrt(max_records_per_user)`. For the second layer from bottom,
      # the worst case is only 0s and `max_records_per_user/2` 2s. And so on
      # until the root node. Another natural L2 norm bound on each layer is
      # `max_records_per_user` so we take the minimum between the two bounds.
      square_l2_norm_bound = 0.
      square_layer_l2_norm_bound = max_records_per_user
      for _ in range(_tree_depth(num_bins, arity)):
        square_l2_norm_bound += min(max_records_per_user**2,
                                    square_layer_l2_norm_bound)
        square_layer_l2_norm_bound *= arity
      l2_norm_bound = math.sqrt(square_l2_norm_bound)

  # (2) Constructs `DifferentiallyPrivateFactory` according to the chosen
  #     `dp_mechanism`.
  if dp_mechanism == 'central-gaussian':
    query = tfp.privacy.dp_query.tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
        l2_norm_bound, noise_multiplier * l2_norm_bound, arity)
    # If the inner `DifferentiallyPrivateFactory` uses `GaussianSumQuery`, then
    # the record is casted to `tf.float32` before feeding to the DP factory.
    cast_to_float = True
  elif dp_mechanism == 'no-noise':
    inner_query = tfp.privacy.dp_query.no_privacy_query.NoPrivacySumQuery()
    query = tfp.privacy.dp_query.tree_aggregation_query.TreeRangeSumQuery(
        arity=arity, inner_query=inner_query)
    # If the inner `DifferentiallyPrivateFactory` uses `NoPrivacyQuery`, then
    # the record is kept as `tf.int32` before feeding to the DP factory.
    cast_to_float = False
  else:
    raise ValueError('Unexpected dp_mechanism.')
  nested_factory = differential_privacy.DifferentiallyPrivateFactory(
      query, nested_factory)

  # 3. Clip as specified by `clip_mechanism`.
  nested_factory = clipping_factory.HistogramClippingSumFactory(
      clip_mechanism=clip_mechanism,
      max_records_per_user=max_records_per_user,
      inner_agg_factory=nested_factory,
      cast_to_float=cast_to_float)

  return nested_factory


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


def _tree_depth(num_leaves: int, arity: int):
  """Returns the depth of the tree given the number of leaf nodes and arity."""
  return math.ceil(math.log(num_leaves) / math.log(arity)) + 1
