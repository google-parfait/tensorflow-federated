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

from collections.abc import Callable
import math
from typing import Optional, TypeVar

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import secure


_AggregationFactory = TypeVar(
    '_AggregationFactory', bound=factory.AggregationFactory
)


def _default_zeroing(
    inner_factory: _AggregationFactory, secure_estimation: bool = False
) -> _AggregationFactory:
  """The default adaptive zeroing wrapper."""

  # Adapts very quickly to a value somewhat higher than the highest values so
  # far seen.
  zeroing_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=10.0,
      target_quantile=0.98,
      learning_rate=math.log(10.0),
      multiplier=2.0,
      increment=1.0,
      secure_estimation=secure_estimation,
  )
  if secure_estimation:
    secure_count_factory = secure.SecureSumFactory(
        upper_bound_threshold=1, lower_bound_threshold=0
    )
    return robust.zeroing_factory(
        zeroing_norm,
        inner_factory,
        zeroed_count_sum_factory=secure_count_factory,
    )
  else:
    return robust.zeroing_factory(zeroing_norm, inner_factory)


def _default_clipping(
    inner_factory: factory.AggregationFactory, secure_estimation: bool = False
) -> factory.AggregationFactory:
  """The default adaptive clipping wrapper."""

  # Adapts relatively quickly to a moderately high norm.
  clipping_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=1.0,
      target_quantile=0.8,
      learning_rate=0.2,
      secure_estimation=secure_estimation,
  )
  if secure_estimation:
    secure_count_factory = secure.SecureSumFactory(
        upper_bound_threshold=1, lower_bound_threshold=0
    )
    return robust.clipping_factory(
        clipping_norm,
        inner_factory,
        clipped_count_sum_factory=secure_count_factory,
    )
  else:
    return robust.clipping_factory(clipping_norm, inner_factory)


def robust_aggregator(
    *,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
    debug_measurements_fn: Optional[
        Callable[[factory.AggregationFactory], factory.AggregationFactory]
    ] = None,
) -> factory.AggregationFactory:
  """Creates aggregator for mean with adaptive zeroing and clipping.

  Zeroes out extremely large values for robustness to data corruption on
  clients, and clips in the L2 norm to moderately high norm for robustness to
  outliers.

  Adaptive clipping approach is as described in Andrew, Thakkar et al. (2021)
  https://arxiv.org/abs/1905.03871, which the robust_aggregator applies without
  the addition of noise.

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

  Raises:
    TypeError: if debug_measurement_fn yields an aggregation factory whose
      weight type does not match `weighted`.
  """
  aggregation_factory = (
      mean.MeanFactory() if weighted else mean.UnweightedMeanFactory()
  )

  if debug_measurements_fn:
    aggregation_factory = debug_measurements_fn(aggregation_factory)
    if (
        weighted
        and not isinstance(
            aggregation_factory, factory.WeightedAggregationFactory
        )
    ) or (
        (not weighted)
        and (
            not isinstance(
                aggregation_factory, factory.UnweightedAggregationFactory
            )
        )
    ):
      raise TypeError('debug_measurements_fn should return the same type.')

  if clipping:
    aggregation_factory = _default_clipping(aggregation_factory)

  if zeroing:
    aggregation_factory = _default_zeroing(aggregation_factory)

  return aggregation_factory
