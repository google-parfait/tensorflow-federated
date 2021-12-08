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
#
# pytype: skip-file
# This modules disables the Pytype analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Generic aggregator for model updates in federated averaging."""

import math

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import encoded
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.learning import debug_measurements


def _default_zeroing(
    inner_factory: factory.AggregationFactory) -> factory.AggregationFactory:
  """The default adaptive zeroing wrapper."""

  # Adapts very quickly to a value somewhat higher than the highest values so
  # far seen.
  zeroing_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=10.0,
      target_quantile=0.98,
      learning_rate=math.log(10.0),
      multiplier=2.0,
      increment=1.0)
  return robust.zeroing_factory(zeroing_norm, inner_factory)


def _default_clipping(
    inner_factory: factory.AggregationFactory) -> factory.AggregationFactory:
  """The default adaptive clipping wrapper."""

  # Adapts relatively quickly to a moderately high norm.
  clipping_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=1.0, target_quantile=0.8, learning_rate=0.2)
  return robust.clipping_factory(clipping_norm, inner_factory)


def robust_aggregator(
    *,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
    add_debug_measurements: bool = False,
) -> factory.AggregationFactory:
  """Creates aggregator for mean with adaptive zeroing and clipping.

  Zeroes out extremely large values for robustness to data corruption on
  clients, and clips in the L2 norm to moderately high norm for robustness to
  outliers.

  For details on clipping and zeroing see `tff.aggregators.clipping_factory`
  and `tff.aggregators.zeroing_factory`. For details on the quantile-based
  adaptive algorithm see `tff.aggregators.PrivateQuantileEstimationProcess`.

  Args:
    zeroing: Whether to enable adaptive zeroing for data corruption mitigation.
    clipping: Whether to enable adaptive clipping in the L2 norm for robustness.
    weighted: Whether the mean is weighted (vs. unweighted).
    add_debug_measurements: Whether to add measurements suitable for debugging
      learning algorithms. For more detail on these measurements, see
      `tff.learning.add_debug_measurements`.

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory_ = mean.MeanFactory() if weighted else mean.UnweightedMeanFactory()

  if add_debug_measurements:
    factory_ = debug_measurements.add_debug_measurements(factory_)

  if clipping:
    factory_ = _default_clipping(factory_)

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
    zeroing: Whether to enable adaptive zeroing for data corruption mitigation.

  Returns:
    A `tff.aggregators.UnweightedAggregationFactory`.
  """

  factory_ = differential_privacy.DifferentiallyPrivateFactory.gaussian_adaptive(
      noise_multiplier, clients_per_round)

  if zeroing:
    factory_ = _default_zeroing(factory_)

  return factory_


def compression_aggregator(
    *,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
    add_debug_measurements: bool = False,
) -> factory.AggregationFactory:
  """Creates aggregator with compression and adaptive zeroing and clipping.

  Zeroes out extremely large values for robustness to data corruption on
  clients and clips in the L2 norm to moderately high norm for robustness to
  outliers. After weighting in mean, the weighted values are uniformly quantized
  to reduce the size of the model update communicated from clients to the
  server. For details, see Suresh et al. (2017)
  http://proceedings.mlr.press/v70/suresh17a/suresh17a.pdf. The default
  configuration is chosen such that compression does not have adverse effect on
  trained model quality in typical tasks.

  Args:
    zeroing: Whether to enable adaptive zeroing for data corruption mitigation.
    clipping: Whether to enable adaptive clipping in the L2 norm for robustness.
      Note this clipping is performed prior to the per-coordinate clipping
      required for quantization.
    weighted: Whether the mean is weighted (vs. unweighted).
    add_debug_measurements: Whether to add measurements suitable for debugging
      learning algorithms. For more detail on these measurements, see
      `tff.learning.add_debug_measurements`.

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory_ = encoded.EncodedSumFactory.quantize_above_threshold(
      quantization_bits=8, threshold=20000)

  factory_ = (
      mean.MeanFactory(factory_)
      if weighted else mean.UnweightedMeanFactory(factory_))

  if add_debug_measurements:
    factory_ = debug_measurements.add_debug_measurements(factory_)

  if clipping:
    factory_ = _default_clipping(factory_)

  if zeroing:
    factory_ = _default_zeroing(factory_)

  return factory_


def secure_aggregator(
    *,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
) -> factory.AggregationFactory:
  """Creates secure aggregator with adaptive zeroing and clipping.

  Zeroes out extremely large values for robustness to data corruption on
  clients, clips to moderately high norm for robustness to outliers. After
  weighting in mean, the weighted values are summed using cryptographic protocol
  ensuring that the server cannot see individual updates until sufficient number
  of updates have been added together. For details, see Bonawitz et al. (2017)
  https://dl.acm.org/doi/abs/10.1145/3133956.3133982. In TFF, this is realized
  using the `tff.federated_secure_sum_bitwidth` operator.

  Args:
    zeroing: Whether to enable adaptive zeroing for data corruption mitigation.
    clipping: Whether to enable adaptive clipping in the L2 norm for robustness.
      Note this clipping is performed prior to the per-coordinate clipping
      required for secure aggregation.
    weighted: Whether the mean is weighted (vs. unweighted).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  secure_clip_bound = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=50.0,
      target_quantile=0.95,
      learning_rate=1.0,
      multiplier=2.0)

  factory_ = secure.SecureSumFactory(secure_clip_bound)

  factory_ = (
      mean.MeanFactory(factory_)
      if weighted else mean.UnweightedMeanFactory(factory_))

  if clipping:
    factory_ = _default_clipping(factory_)

  if zeroing:
    factory_ = _default_zeroing(factory_)

  return factory_
