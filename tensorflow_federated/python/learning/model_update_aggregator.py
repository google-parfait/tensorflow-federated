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
from typing import Union

from tensorflow_federated.python.aggregators import clipping_factory
from tensorflow_federated.python.aggregators import dp_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import quantile_estimation

_AggregationFactory = Union[factory.UnweightedAggregationFactory,
                            factory.WeightedAggregationFactory]


def _default_zeroing(inner_factory: _AggregationFactory) -> _AggregationFactory:
  # Adapts very quickly to a value somewhat higher than the highest values so
  # far seen.
  zeroing_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=10.0,
      target_quantile=0.98,
      learning_rate=math.log(10.0),
      multiplier=2.0,
      increment=1.0)
  return clipping_factory.ZeroingFactory(zeroing_norm, inner_factory)


def robust_aggregator(
    zeroing: bool = True,
    clipping: bool = True) -> factory.WeightedAggregationFactory:
  """Creates aggregator for mean with adaptive zeroing and clipping.

  Zeroes out extremely large values for robustness to data corruption on
  clients, and clips to moderately high norm for robustness to outliers.

  For details on clipping and zeroing see `tff.aggregators.ClippingFactory` and
  `tff.aggregators.ZeroingFactory`. For details on the quantile-based adaptive
  algorithm see `tff.aggregators.PrivateQuantileEstimationProcess`.

  Args:
    zeroing: Whether to enable adaptive zeroing.
    clipping: Whether to enable adaptive clipping.

  Returns:
    A `tff.aggregators.WeightedAggregationFactory` with zeroing and clipping.
  """
  factory_ = mean_factory.MeanFactory()

  if clipping:
    # Adapts relatively quickly to a moderately high norm.
    clipping_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=1.0, target_quantile=0.8, learning_rate=0.2)
    factory_ = clipping_factory.ClippingFactory(clipping_norm, factory_)

  if zeroing:
    factory_ = _default_zeroing(factory_)

  return factory_


def dp_aggregator(noise_multiplier: float,
                  clients_per_round: float,
                  zeroing: bool = True) -> factory.UnweightedAggregationFactory:
  """Creates aggregator with adaptive zeroing and differential privacy.

  Zeroes out extremely large values for robustness to data corruption on
  clients, and performs adaptive clipping and addition of Gaussian noise for
  differentially private learning. For details of the DP algorithm see McMahan
  et. al (2017) https://arxiv.org/abs/1710.06963. The adaptive clipping uses the
  geometric method described in Thakkar et al. (2019)
  https://arxiv.org/abs/1905.03871.

  Args:
    noise_multiplier: A float specifying the noise multiplier for the Gaussian
      mechanism for model updates. A value of 1.0 or higher may be needed for
      meaningful privacy. See above mentioned papers to compute (epsilon, delta)
      privacy guarantee.
    clients_per_round: A float specifying the expected number of clients per
      round. Must be positive.
    zeroing: Whether to enable adaptive zeroing.

  Returns:
    A `tff.aggregators.WeightedAggregationFactory` with zeroing and differential
      privacy.
  """

  factory_ = dp_factory.DifferentiallyPrivateFactory.gaussian_adaptive(
      noise_multiplier, clients_per_round)

  if zeroing:
    factory_ = _default_zeroing(factory_)

  return factory_
