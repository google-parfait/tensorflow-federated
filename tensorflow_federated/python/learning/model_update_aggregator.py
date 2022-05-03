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
from typing import Callable, Optional

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import distributed_dp
from tensorflow_federated.python.aggregators import encoded
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import secure


def _default_zeroing(
    inner_factory: factory.AggregationFactory,
    secure_estimation: bool = False) -> factory.AggregationFactory:
  """The default adaptive zeroing wrapper."""

  # Adapts very quickly to a value somewhat higher than the highest values so
  # far seen.
  zeroing_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=10.0,
      target_quantile=0.98,
      learning_rate=math.log(10.0),
      multiplier=2.0,
      increment=1.0,
      secure_estimation=secure_estimation)
  if secure_estimation:
    secure_count_factory = secure.SecureSumFactory(
        upper_bound_threshold=1, lower_bound_threshold=0)
    return robust.zeroing_factory(
        zeroing_norm,
        inner_factory,
        zeroed_count_sum_factory=secure_count_factory)
  else:
    return robust.zeroing_factory(zeroing_norm, inner_factory)


def _default_clipping(
    inner_factory: factory.AggregationFactory,
    secure_estimation: bool = False) -> factory.AggregationFactory:
  """The default adaptive clipping wrapper."""

  # Adapts relatively quickly to a moderately high norm.
  clipping_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=1.0,
      target_quantile=0.8,
      learning_rate=0.2,
      secure_estimation=secure_estimation)
  if secure_estimation:
    secure_count_factory = secure.SecureSumFactory(
        upper_bound_threshold=1, lower_bound_threshold=0)
    return robust.clipping_factory(
        clipping_norm,
        inner_factory,
        clipped_count_sum_factory=secure_count_factory)
  else:
    return robust.clipping_factory(clipping_norm, inner_factory)


def robust_aggregator(
    *,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
    debug_measurements_fn: Optional[Callable[
        [factory.AggregationFactory], factory.AggregationFactory]] = None,
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
    debug_measurements_fn: A callable to add measurements suitable for debugging
      learning algorithms. Often useful values include None,
      `tff.learning.add_debug_measurements` or
      `tff.learning.add_debug_measurements_with_mixed_dtype`.

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory_ = mean.MeanFactory() if weighted else mean.UnweightedMeanFactory()

  if debug_measurements_fn:
    factory_ = debug_measurements_fn(factory_)

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
    debug_measurements_fn: Optional[Callable[
        factory.AggregationFactory, factory.AggregationFactory]] = None,
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
    debug_measurements_fn: A callable to add measurements suitable for debugging
      learning algorithms, with possible values as None,
      `tff.learning.add_debug_measurements` or
      `tff.learning.add_debug_measurements_with_mixed_dtype`.

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory_ = encoded.EncodedSumFactory.quantize_above_threshold(
      quantization_bits=8, threshold=20000)

  factory_ = (
      mean.MeanFactory(factory_)
      if weighted else mean.UnweightedMeanFactory(factory_))

  if debug_measurements_fn:
    factory_ = debug_measurements_fn(factory_)
    if (weighted and
        not isinstance(factory_, factory.WeightedAggregationFactory)) or (
            (not weighted) and
            (not isinstance(factory_, factory.UnweightedAggregationFactory))):
      raise TypeError('debug_measurements_fn should return the same type.')

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
      multiplier=2.0,
      secure_estimation=True)

  factory_ = secure.SecureSumFactory(secure_clip_bound)

  if weighted:
    factory_ = mean.MeanFactory(
        value_sum_factory=factory_,
        # Use a power of 2 minus one to more accurately encode floating dtypes
        # that actually contain integer values. 2 ^ 20 gives us approximately a
        # range of [0, 1 million]. Existing use cases have the weights either
        # all ones, or a variant of number of examples processed locally.
        weight_sum_factory=secure.SecureSumFactory(
            upper_bound_threshold=float(2**20 - 1), lower_bound_threshold=0.0))
  else:
    factory_ = mean.UnweightedMeanFactory(
        value_sum_factory=factory_,
        count_sum_factory=secure.SecureSumFactory(
            upper_bound_threshold=1, lower_bound_threshold=0))

  if clipping:
    factory_ = _default_clipping(factory_, secure_estimation=True)

  if zeroing:
    factory_ = _default_zeroing(factory_, secure_estimation=True)

  return factory_


def ddp_secure_aggregator(
    noise_multiplier: float,
    expected_clients_per_round: int,
    bits: int = 20,
    zeroing: bool = True,
    rotation_type: str = 'hd') -> factory.UnweightedAggregationFactory:
  """Creates aggregator with adaptive zeroing and distributed DP.

  Zeroes out extremely large values for robustness to data corruption on
  clients, and performs distributed DP (compression, discrete noising, and
  SecAgg) with adaptive clipping for differentially private learning. For
  details of the two main distributed DP algorithms see
  https://arxiv.org/pdf/2102.06387
  or https://arxiv.org/pdf/2110.04995.pdf. The adaptive clipping uses the
  geometric method described in https://arxiv.org/abs/1905.03871.

  Args:
    noise_multiplier: A float specifying the noise multiplier (with respect to
      the initial L2 cipping) for the distributed DP mechanism for model
      updates. A value of 1.0 or higher may be needed for meaningful privacy.
    expected_clients_per_round: An integer specifying the expected number of
      clients per round. Must be positive.
    bits: An integer specifying the bit-width for the aggregation. Note that
      this is for the noisy, quantized aggregate at the server and thus should
      account for the `expected_clients_per_round`. Must be in the inclusive
      range of [1, 22]. This is set to 20 bits by default, and it dictates the
      computational and communication efficiency of Secure Aggregation. Setting
      it to less than 20 bits should work fine for most cases. For instance, for
      an expected number of securely aggregated client updates of 100, 12 bits
      should be enough, and for an expected number of securely aggregated client
      updates of 1000, 16 bits should be enough.
    zeroing: A bool indicating whether to enable adaptive zeroing for data
      corruption mitigation. Defaults to `True`.
    rotation_type: A string indicating what rotation to use for distributed DP.
      Valid options are 'hd' (Hadamard transform) and 'dft' (discrete Fourier
      transform). Defaults to `hd`.

  Returns:
    A `tff.aggregators.UnweightedAggregationFactory`.
  """
  agg_factory = distributed_dp.DistributedDpSumFactory(
      noise_multiplier=noise_multiplier,
      expected_clients_per_round=expected_clients_per_round,
      bits=bits,
      l2_clip=0.1,
      mechanism='distributed_skellam',
      rotation_type=rotation_type,
      auto_l2_clip=True)
  agg_factory = mean.UnweightedMeanFactory(
      value_sum_factory=agg_factory,
      count_sum_factory=secure.SecureSumFactory(
          upper_bound_threshold=1, lower_bound_threshold=0))

  if zeroing:
    agg_factory = _default_zeroing(agg_factory, secure_estimation=True)

  return agg_factory
