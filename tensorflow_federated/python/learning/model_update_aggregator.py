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

from tensorflow_federated.python.aggregators import encoded
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import secure


_AggregationFactory = TypeVar(
    '_AggregationFactory', bound=factory.AggregationFactory
)


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
    raise ValueError("Clipping unimplemented.")

  if zeroing:
    raise ValueError("Zeroing unimplemented.")

  return aggregation_factory


def compression_aggregator(
    *,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
    debug_measurements_fn: Optional[
        Callable[[factory.AggregationFactory], factory.AggregationFactory]
    ] = None,
    **kwargs,
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
    **kwargs: Keyword arguments.

  Returns:
    A `tff.aggregators.AggregationFactory`.

  Raises:
    TypeError: if debug_measurement_fn yields an aggregation factory whose
      weight type does not match `weighted`.
  """
  aggregation_factory = encoded.EncodedSumFactory.quantize_above_threshold(
      quantization_bits=8, threshold=20000, **kwargs
  )

  aggregation_factory = (
      mean.MeanFactory(aggregation_factory)
      if weighted
      else mean.UnweightedMeanFactory(aggregation_factory)
  )

  if debug_measurements_fn is not None:
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
