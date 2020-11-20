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
"""Generic aggregator for model updates in federated averaging."""

import math
from typing import Optional, Union

import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import clipping_factory
from tensorflow_federated.python.aggregators import dp_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics


def _check_positive(value, label):
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {value}.')


def _check_nonnegative(value, label):
  if value < 0:
    raise ValueError(f'{label} must be nonnegative. Found {value}.')


def _affine_transform(multiplier, increment):
  transform_tf_comp = computations.tf_computation(
      lambda value: multiplier * value + increment, tf.float32)
  return computations.federated_computation(
      lambda value: intrinsics.federated_map(transform_tf_comp, value),
      computation_types.at_server(tf.float32))


class QuantileEstimationConfig:
  """A config class for quantile estimation."""

  def __init__(self, initial_estimate: float, target_quantile: float,
               learning_rate: float):
    """Initializes a QuantileEstimationConfig.

    Args:
      initial_estimate: A float representing the initial quantile estimate.
      target_quantile: A float in [0, 1] representing the quantile to match.
      learning_rate: A float determining the learning rate for the process.
    """
    py_typecheck.check_type(initial_estimate, float, 'initial_estimate')
    _check_positive(initial_estimate, 'initial_estimate')
    self._initial_estimate = initial_estimate

    py_typecheck.check_type(target_quantile, float, 'target_quantile')
    if not 0 <= target_quantile <= 1:
      raise ValueError('target_quantile must be in the range [0, 1].')
    self._target_quantile = target_quantile

    py_typecheck.check_type(learning_rate, float, 'learning_rate')
    _check_positive(learning_rate, 'learning_rate')
    self._learning_rate = learning_rate

  @property
  def initial_estimate(self) -> float:
    return self._initial_estimate

  @property
  def target_quantile(self) -> float:
    return self._target_quantile

  @property
  def learning_rate(self) -> float:
    return self._learning_rate

  def to_quantile_estimation_process(
      self) -> quantile_estimation.PrivateQuantileEstimationProcess:
    return quantile_estimation.PrivateQuantileEstimationProcess(
        tfp.NoPrivacyQuantileEstimatorQuery(
            initial_estimate=self._initial_estimate,
            target_quantile=self._target_quantile,
            learning_rate=self._learning_rate,
            geometric_update=True))


class ZeroingConfig:
  """Config for adaptive zeroing based on a quantile estimate."""

  def __init__(self,
               quantile: Optional[QuantileEstimationConfig] = None,
               multiplier: float = 2.0,
               increment: float = 1.0):
    """Initializes a ZeroingConfig.

    Estimates value at quantile `Z` of value norm distribution and zeroes out
    values whose norm is greater than `rZ + i` for multiplier `r` and increment
    `i`. The quantile `Z` is estimated using the geometric method described in
    Thakkar et al. 2019, "Differentially Private Learning with Adaptive
    Clipping" (https://arxiv.org/abs/1905.03871) without noise added (so not
    differentially private).

    Args:
      quantile: A `QuantileEstimationConfig` specifying the quantile estimation
        process for zeroing. If None, defaults to a fast-adapting process that
        zeroes only very high values.
      multiplier: A float for factor `r` in zeroing norm formula `rZ + i`.
      increment: A float for increment `i` in zeroing norm formula `rZ + i`.
    """
    if quantile is None:
      quantile = QuantileEstimationConfig(10.0, 0.98, math.log(10))
    else:
      py_typecheck.check_type(quantile, QuantileEstimationConfig, 'quantile')
    self._quantile = quantile
    py_typecheck.check_type(multiplier, float, 'multiplier')
    _check_positive(multiplier, 'multiplier')
    self._multiplier = multiplier
    py_typecheck.check_type(increment, float, 'increment')
    _check_nonnegative(increment, 'increment')
    self._increment = increment

  def to_factory(self, inner_factory) -> clipping_factory.ZeroingFactory:
    zeroing_quantile = self._quantile.to_quantile_estimation_process()
    zeroing_norm = zeroing_quantile.map(
        _affine_transform(self._multiplier, self._increment))
    return clipping_factory.ZeroingFactory(zeroing_norm, inner_factory,
                                           float('inf'))


class ClippingConfig:
  """Config for fixed or adaptive clipping with recommended defaults."""

  def __init__(self,
               clip: Optional[Union[float, QuantileEstimationConfig]] = None):
    """Initializes a ClippingConfig.

    Args:
      clip: Either a float representing the fixed clip norm, or a
        QuantileEstimationConfig specifying the quantile estimation process for
        adaptive clipping. If None, defaults to a quantile estimation process
        that adapts reasonably fast and clips to a moderately high norm.
    """
    if clip is None:
      clip = QuantileEstimationConfig(1.0, 0.8, 0.2)
    elif isinstance(clip, float):
      _check_positive(clip, 'clip')
    else:
      py_typecheck.check_type(clip, QuantileEstimationConfig, 'clip')

    self._clip = clip

  @property
  def clip(self) -> Union[float, QuantileEstimationConfig]:
    return self._clip

  @property
  def is_fixed(self) -> bool:
    return isinstance(self._clip, float)

  def to_factory(self, inner_factory) -> clipping_factory.ClippingFactory:
    if self.is_fixed:
      return clipping_factory.ClippingFactory(self._clip, inner_factory)
    else:
      return clipping_factory.ClippingFactory(
          self._clip.to_quantile_estimation_process(), inner_factory)


class DPConfig:
  """A config class for differential privacy with recommended defaults."""

  def __init__(self,
               noise_multiplier: float,
               clients_per_round: float,
               clipping: Optional[ClippingConfig] = None,
               clipped_count_stddev: Optional[float] = None):
    """Initializes a DPConfig.

    Args:
      noise_multiplier: A float specifying the noise multiplier for the Gaussian
        mechanism for model updates.
      clients_per_round: A float specifying the expected number of clients per
        round.
      clipping: A ClippingConfig specifying the clipping to use. If None,
        adaptive clipping with default parameters will be used.
      clipped_count_stddev: A float specifying the stddev for clipped counts. If
        None, defaults to 0.05 times `clients_per_round`.
    """
    py_typecheck.check_type(noise_multiplier, float, 'noise_multiplier')
    _check_nonnegative(noise_multiplier, 'noise_multiplier')
    self._noise_multiplier = noise_multiplier

    py_typecheck.check_type(clients_per_round, float, 'clients_per_round')
    _check_positive(clients_per_round, 'clients_per_round')
    self._clients_per_round = clients_per_round

    if clipping is None:
      clipping = ClippingConfig(QuantileEstimationConfig(1e-1, 0.5, 0.2))
    else:
      py_typecheck.check_type(clipping, ClippingConfig, 'clipping')
    self._clipping = clipping

    if clipped_count_stddev is None:
      # Default to 0.05 * clients_per_round. This way the noised fraction
      # of unclipped updates will be within 0.1 of the true fraction with
      # 95.4% probability, and will be within 0.15 of the true fraction with
      # 99.7% probability. Even in this unlikely case, the error on the update
      # would be a factor of exp(0.15) = 1.16, not a huge deviation. So this
      # default gives maximal privacy for acceptable probability of deviation.
      clipped_count_stddev = 0.05 * clients_per_round
    py_typecheck.check_type(clipped_count_stddev, float, 'clipped_count_stddev')
    _check_nonnegative(clipped_count_stddev, 'clipped_count_stddev')
    self._clipped_count_stddev = clipped_count_stddev

  def to_factory(self) -> dp_factory.DifferentiallyPrivateFactory:
    """Creates factory based on config settings."""
    if self._clipping.is_fixed:
      stddev = self._clipping.clip * self._noise_multiplier
      query = tfp.GaussianAverageQuery(
          l2_norm_clip=self._clipping.clip,
          sum_stddev=stddev,
          denominator=self._clients_per_round)
    else:
      query = tfp.QuantileAdaptiveClipAverageQuery(
          initial_l2_norm_clip=self._clipping.clip.initial_estimate,
          noise_multiplier=self._noise_multiplier,
          denominator=self._clients_per_round,
          target_unclipped_quantile=self._clipping.clip.target_quantile,
          learning_rate=self._clipping.clip.learning_rate,
          clipped_count_stddev=self._clipped_count_stddev,
          expected_num_records=self._clients_per_round,
          geometric_update=True)

    return dp_factory.DifferentiallyPrivateFactory(query)


def model_update_aggregator(
    zeroing: Optional[ZeroingConfig] = ZeroingConfig(),
    clipping_and_noise: Optional[Union[ClippingConfig, DPConfig]] = None
) -> Union[factory.WeightedAggregationFactory,
           factory.UnweightedAggregationFactory]:
  """Builds model update aggregator.

  Args:
    zeroing: A ZeroingConfig. If None, no zeroing will be performed.
    clipping_and_noise: An optional ClippingConfig or DPConfig. If unspecified,
      no clipping or noising will be performed.

  Returns:
    A `factory.WeightedAggregationFactory` intended for model update aggregation
      in federated averaging with zeroing and clipping for robustness.
  """
  if not clipping_and_noise:
    factory_ = mean_factory.MeanFactory()
  elif isinstance(clipping_and_noise, ClippingConfig):
    factory_ = clipping_and_noise.to_factory(mean_factory.MeanFactory())
  else:
    py_typecheck.check_type(clipping_and_noise, DPConfig, 'clipping_and_noise')
    factory_ = clipping_and_noise.to_factory()
  if zeroing:
    factory_ = zeroing.to_factory(factory_)
  return factory_
