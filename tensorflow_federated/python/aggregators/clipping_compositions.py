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
"""Functions providing recommended aggregator compositions."""

import tensorflow as tf
import tensorflow_privacy

from tensorflow_federated.python.aggregators import clipping_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics


def _affine_transform(multiplier, increment):
  transform_tf_comp = computations.tf_computation(
      lambda value: multiplier * value + increment, tf.float32)
  return computations.federated_computation(
      lambda value: intrinsics.federated_map(transform_tf_comp, value),
      computation_types.at_server(tf.float32))


def _make_quantile_estimation_process(initial_estimate: float,
                                      target_quantile: float,
                                      learning_rate: float):
  return quantile_estimation.PrivateQuantileEstimationProcess(
      tensorflow_privacy.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=initial_estimate,
          target_quantile=target_quantile,
          learning_rate=learning_rate,
          geometric_update=True))


def adaptive_zeroing_mean(
    initial_quantile_estimate: float,
    target_quantile: float,
    multiplier: float,
    increment: float,
    learning_rate: float,
    norm_order: bool,
    no_nan_mean: bool = False) -> factory.AggregationProcessFactory:
  """Creates a factory for mean with adaptive zeroing.

  Estimates value at quantile `Z` of value norm distribution and zeroes out
  values whose norm is greater than `rZ + i` for multiplier `r` and increment
  `i`. The quantile `Z` is estimated using the geometric method described in
  Thakkar et al. 2019, "Differentially Private Learning with Adaptive Clipping"
  (https://arxiv.org/abs/1905.03871) without noise added (so not differentially
  private).

  Args:
    initial_quantile_estimate: The initial estimate of the target quantile `Z`.
    target_quantile: Which quantile to match, as a float in [0, 1]. For example,
      0.5 for median, or 0.98 to zero out only the largest 2% of updates (if
      multiplier=1 and increment=0).
    multiplier: Factor `r` in zeroing norm formula `rZ + i`.
    increment: Increment `i` in zeroing norm formula `rZ + i`.
    learning_rate: Learning rate for quantile matching algorithm.
    norm_order: A float for the order of the norm. Must be 1, 2, or np.inf.
    no_nan_mean: A bool. If True, the computed mean is 0 if sum of weights is
      equal to 0.

  Returns:
    A factory that performs mean after adaptive clipping.
  """

  zeroing_quantile = _make_quantile_estimation_process(
      initial_estimate=initial_quantile_estimate,
      target_quantile=target_quantile,
      learning_rate=learning_rate)
  zeroing_norm = zeroing_quantile.map(_affine_transform(multiplier, increment))
  mean = mean_factory.MeanFactory(no_nan_division=no_nan_mean)
  return clipping_factory.ZeroingFactory(zeroing_norm, mean, norm_order)


def adaptive_zeroing_clipping_mean(
    initial_zeroing_quantile_estimate: float,
    target_zeroing_quantile: float,
    zeroing_multiplier: float,
    zeroing_increment: float,
    zeroing_learning_rate: float,
    zeroing_norm_order: bool,
    initial_clipping_quantile_estimate: float,
    target_clipping_quantile: float,
    clipping_learning_rate: float,
    no_nan_mean: bool = False) -> factory.AggregationProcessFactory:
  """Makes a factory for mean with adaptive zeroing and clipping.

  Estimates value at quantile `Z` of value norm distribution and zeroes out
  values whose norm is greater than `rZ + i` for multiplier `r` and increment
  `i`. Also estimates value at quantile `C` and clips values whose L2 norm is
  greater than `C` (without any multiplier or increment). The quantiles are
  estimated using the geometric method described in Thakkar et al. 2019,
  "Differentially Private Learning with Adaptive Clipping"
  (https://arxiv.org/abs/1905.03871) without noise added (so not differentially
  private). Zeroing occurs before clipping, so the estimation process for `C`
  uses already zeroed values.

  Note while the zeroing_norm_order may be 1.0 or np.inf, only L2 norm is used
  for clipping.

  Args:
    initial_zeroing_quantile_estimate: The initial estimate of the target
      quantile `Z` for zeroing.
    target_zeroing_quantile: Which quantile to match for zeroing, as a float in
      [0, 1]. For example, 0.5 for median, or 0.98 to zero out only the largest
      2% of updates (if multiplier=1 and increment=0).
    zeroing_multiplier: Factor `r` in zeroing norm formula `rZ + i`.
    zeroing_increment: Increment `i` in zeroing norm formula `rZ + i`.
    zeroing_learning_rate: Learning rate for zeroing quantile estimate.
    zeroing_norm_order: A float for the order of the norm for zeroing. Must be
      1, 2, or np.inf.
    initial_clipping_quantile_estimate: The initial estimate of the target
      quantile `C` for clipping. (Multiplier and increment are not used for
      clipping.)
    target_clipping_quantile: Which quantile to match for clipping, as a float
      in [0, 1].
    clipping_learning_rate: Learning rate for clipping quantile estimate.
    no_nan_mean: A bool. If True, the computed mean is 0 if sum of weights is
      equal to 0.

  Returns:
    A factory that performs mean after adaptive zeroing and clipping.
  """

  zeroing_quantile = _make_quantile_estimation_process(
      initial_estimate=initial_zeroing_quantile_estimate,
      target_quantile=target_zeroing_quantile,
      learning_rate=zeroing_learning_rate)

  zeroing_norm = zeroing_quantile.map(
      _affine_transform(zeroing_multiplier, zeroing_increment))

  clipping_norm = _make_quantile_estimation_process(
      initial_estimate=initial_clipping_quantile_estimate,
      target_quantile=target_clipping_quantile,
      learning_rate=clipping_learning_rate)

  mean = mean_factory.MeanFactory(no_nan_division=no_nan_mean)
  clip = clipping_factory.ClippingFactory(clipping_norm, mean)
  return clipping_factory.ZeroingFactory(zeroing_norm, clip, zeroing_norm_order)
