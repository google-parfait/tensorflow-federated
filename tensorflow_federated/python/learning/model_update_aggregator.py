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

import attr
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import clipping_factory
from tensorflow_federated.python.aggregators import dp_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics


AggregationFactory = Union[factory.WeightedAggregationFactory,
                           factory.UnweightedAggregationFactory]


def _check_positive(instance, attribute, value):
  if value <= 0:
    raise ValueError(f'{attribute.name} must be positive. Found {value}.')


def _check_nonnegative(instance, attribute, value):
  if value < 0:
    raise ValueError(f'{attribute.name} must be nonnegative. Found {value}.')


def _check_probability(instance, attribute, value):
  if not 0 <= value <= 1:
    raise ValueError(f'{attribute.name} must be between 0 and 1 (inclusive). '
                     f'Found {value}.')


def _affine_transform(multiplier, increment):
  transform_tf_comp = computations.tf_computation(
      lambda value: multiplier * value + increment, tf.float32)
  return computations.federated_computation(
      lambda value: intrinsics.federated_map(transform_tf_comp, value),
      computation_types.at_server(tf.float32))


@attr.s(frozen=True, kw_only=True)
class AdaptiveZeroingConfig:
  """Config for adaptive zeroing based on a quantile estimate.

  Estimates value at quantile `Z` of value norm distribution and zeroes out
  values whose norm is greater than `rZ + i` for multiplier `r` and increment
  `i`. The quantile `Z` is estimated using the geometric method described in
  Thakkar et al. 2019, "Differentially Private Learning with Adaptive
  Clipping" (https://arxiv.org/abs/1905.03871) without noise added (so not
  differentially private).

  Default values are recommended for adaptive zeroing for data corruption
  mitigation.

  Attributes:
    initial_quantile_estimate: The initial estimate of `Z`.
    target_quantile: The quantile to which `Z` will be adapted.
    learning_rate: The learning rate for the adaptive algorithm.
    multiplier: The multiplier `r` to determine the zeroing norm.
    increment: The increment `i` to determine the zeroing norm.
  """

  initial_quantile_estimate: float = attr.ib(
      default=10.0,
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)

  target_quantile: float = attr.ib(
      default=0.98,
      validator=[attr.validators.instance_of(float), _check_probability],
      converter=float)

  learning_rate: float = attr.ib(
      default=math.log(10),
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)

  multiplier: float = attr.ib(
      default=2.0,
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)

  increment: float = attr.ib(
      default=1.0,
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)


def _build_quantile_estimation_process(initial_estimate, target_quantile,
                                       learning_rate):
  return quantile_estimation.PrivateQuantileEstimationProcess(
      tfp.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=initial_estimate,
          target_quantile=target_quantile,
          learning_rate=learning_rate,
          geometric_update=True))


def _apply_zeroing(config: AdaptiveZeroingConfig,
                   inner_factory: AggregationFactory) -> AggregationFactory:
  """Applies zeroing to `inner_factory` according to `config`."""
  zeroing_quantile = _build_quantile_estimation_process(
      config.initial_quantile_estimate, config.target_quantile,
      config.learning_rate)
  zeroing_norm = zeroing_quantile.map(
      _affine_transform(config.multiplier, config.increment))
  return clipping_factory.ZeroingFactory(
      zeroing_norm, inner_factory, norm_order=float('inf'))


@attr.s(frozen=True)
class FixedClippingConfig:
  """Config for clipping to a fixed value.

  Attributes:
    clip: The fixed clipping norm.
  """

  clip: float = attr.ib(
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)


@attr.s(frozen=True, kw_only=True)
class AdaptiveClippingConfig:
  """Config for adaptive clipping based on a quantile estimate.

  Estimates value at quantile `C` of value norm distribution and clips
  values whose norm is greater than `C`. The quantile is estimated using the
  geometric method described in Thakkar et al. 2019, "Differentially Private
  Learning with Adaptive Clipping" (https://arxiv.org/abs/1905.03871) without
  noise added (so not differentially private).

  Default values are recommended for adaptive clipping for robustness.

  Attributes:
    initial_clip: The initial estimate of `C`.
    target_quantile: The quantile to which `C` will be adapted.
    learning_rate: The learning rate for the adaptive algorithm.
  """

  initial_clip: float = attr.ib(
      default=1.0,
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)

  target_quantile: float = attr.ib(
      default=0.8,
      validator=[attr.validators.instance_of(float), _check_probability],
      converter=float)

  learning_rate: float = attr.ib(
      default=0.2,
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)


ClippingConfig = Union[FixedClippingConfig, AdaptiveClippingConfig]


def _apply_clipping(config: ClippingConfig,
                    inner_factory: AggregationFactory) -> AggregationFactory:
  """Applies clipping to `inner_factory` according to `config`."""
  if isinstance(config, FixedClippingConfig):
    return clipping_factory.ClippingFactory(config.clip, inner_factory)
  elif isinstance(config, AdaptiveClippingConfig):
    clipping_quantile = _build_quantile_estimation_process(
        config.initial_clip, config.target_quantile, config.learning_rate)
    return clipping_factory.ClippingFactory(clipping_quantile, inner_factory)
  else:
    raise TypeError(f'config is not a supported type of ClippingConfig. Found '
                    f'type {type(config)}.')


@attr.s(frozen=True, kw_only=True)
class DifferentialPrivacyConfig:
  """A config class for differential privacy with recommended defaults.

  Attributes:
    noise_multiplier: The ratio of the noise standard deviation to the clip
      norm.
    clients_per_round: The number of clients per round.
    clipping: A FixedClippingConfig or AdaptiveClippingConfig specifying the
      type of clipping. Defaults to an adaptive clip process that starts small
      and adapts moderately quickly to the median.
    clipped_count_stddev: The standard deviation of the clipped count estimate,
      for private adaptation of the clipping norm. If unspecified, defaults to a
      value that gives maximal privacy without disrupting the adaptive clipping
      norm process too greatly.
  """

  noise_multiplier: float = attr.ib(
      validator=[attr.validators.instance_of(float), _check_nonnegative],
      converter=float)

  clients_per_round: float = attr.ib(
      validator=[attr.validators.instance_of(float), _check_positive],
      converter=float)

  clipping: ClippingConfig = attr.ib(
      default=AdaptiveClippingConfig(
          initial_clip=1e-1, target_quantile=0.5, learning_rate=0.2),
      validator=attr.validators.instance_of(
          (FixedClippingConfig, AdaptiveClippingConfig)))

  clipped_count_stddev: float = attr.ib(
      validator=[attr.validators.instance_of(float), _check_nonnegative],
      converter=float)

  @clipped_count_stddev.default
  def _set_default_clipped_count_stddev(self):
    # Default to 0.05 * clients_per_round. This way the noised fraction
    # of unclipped updates will be within 0.1 of the true fraction with
    # 95.4% probability, and will be within 0.15 of the true fraction with
    # 99.7% probability. Even in this unlikely case, the error on the update
    # would be a factor of exp(0.15) = 1.16, not a huge deviation. So this
    # default gives maximal privacy for acceptable probability of deviation.
    return 0.05 * self.clients_per_round


def _dp_factory(
    config: DifferentialPrivacyConfig
) -> dp_factory.DifferentiallyPrivateFactory:
  """Creates DifferentiallyPrivateFactory based on config settings."""
  if isinstance(config.clipping, FixedClippingConfig):
    stddev = config.clipping.clip * config.noise_multiplier
    query = tfp.GaussianAverageQuery(
        l2_norm_clip=config.clipping.clip,
        sum_stddev=stddev,
        denominator=config.clients_per_round)
  elif isinstance(config.clipping, AdaptiveClippingConfig):
    query = tfp.QuantileAdaptiveClipAverageQuery(
        initial_l2_norm_clip=config.clipping.initial_clip,
        noise_multiplier=config.noise_multiplier,
        denominator=config.clients_per_round,
        target_unclipped_quantile=config.clipping.target_quantile,
        learning_rate=config.clipping.learning_rate,
        clipped_count_stddev=config.clipped_count_stddev,
        expected_num_records=config.clients_per_round,
        geometric_update=True)
  else:
    raise TypeError(
        f'config.clipping is not a supported type of ClippingConfig. Found '
        f'type {type(config.clipping)}.')

  return dp_factory.DifferentiallyPrivateFactory(query)


def model_update_aggregator(
    zeroing: Optional[AdaptiveZeroingConfig] = AdaptiveZeroingConfig(),
    clipping_and_noise: Optional[Union[ClippingConfig,
                                       DifferentialPrivacyConfig]] = None
) -> AggregationFactory:
  """Builds aggregator for model updates in FL according to configs.

  The default aggregator (produced if no arguments are overridden) performs
  mean with adaptive zeroing for robustness. To turn off adaptive zeroing set
  `zeroing=None`. (Adaptive) clipping and/or differential privacy can
  optionally be enabled by setting `clipping_and_noise`.

  Args:
    zeroing: A ZeroingConfig. If None, no zeroing will be performed.
    clipping_and_noise: An optional ClippingConfig or DifferentialPrivacyConfig.
      If unspecified, no clipping or noising will be performed.

  Returns:
    A `factory.WeightedAggregationFactory` intended for model update aggregation
      in federated averaging with zeroing and clipping for robustness.
  """
  if not clipping_and_noise:
    factory_ = mean_factory.MeanFactory()
  elif isinstance(clipping_and_noise, ClippingConfig):
    factory_ = _apply_clipping(clipping_and_noise, mean_factory.MeanFactory())
  elif isinstance(clipping_and_noise, DifferentialPrivacyConfig):
    factory_ = _dp_factory(clipping_and_noise)
  else:
    raise TypeError(f'clipping_and_noise must be a supported type of clipping '
                    f'or noise config. Found type {type(clipping_and_noise)}.')
  if zeroing:
    factory_ = _apply_zeroing(zeroing, factory_)
  return factory_
