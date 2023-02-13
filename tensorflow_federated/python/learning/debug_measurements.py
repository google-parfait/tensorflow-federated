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
"""Library of aggregator measurements useful for debugging learning processes."""

import collections
from collections.abc import Callable
from typing import Any, TypeVar

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import measurements
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import placements


_AggregationFactory = TypeVar(
    '_AggregationFactory', bound=factory.AggregationFactory
)


@tensorflow_computation.tf_computation
def _calculate_global_norm(tensor_struct):
  """Calculate the Euclidean norm of a nested structure of tensors."""
  return tf.linalg.global_norm(tf.nest.flatten(tensor_struct))


@tensorflow_computation.tf_computation
def _calculate_global_norm_mixed_dtype(tensor_struct):
  """Calculate the Euclidean norm of a nested structure of tensors."""
  norms_squared = [
      tf.cast(tf.norm(a, ord=2) ** 2, tf.float32)
      for a in tf.nest.flatten(tensor_struct)
  ]
  return tf.math.sqrt(tf.reduce_sum(tf.stack(norms_squared)))


@tensorflow_computation.tf_computation
def _square_value(tensor_value):
  """Computes the square of a tensor."""
  return tensor_value**2


@tensorflow_computation.tf_computation
def _calculate_server_update_statistics_mixed_dtype(server_update):
  """Calculate server update statistics of mixed data types."""
  flattened_struct_max = [
      tf.cast(tf.math.reduce_max(d), tf.float32)
      for d in tf.nest.flatten(server_update)
  ]
  flattened_struct_min = [
      tf.cast(tf.math.reduce_min(d), tf.float32)
      for d in tf.nest.flatten(server_update)
  ]
  max_value = tf.math.reduce_max(flattened_struct_max)
  min_value = tf.math.reduce_min(flattened_struct_min)
  global_norm = _calculate_global_norm_mixed_dtype(server_update)
  return collections.OrderedDict(
      server_update_max=max_value,
      server_update_norm=global_norm,
      server_update_min=min_value,
  )


@tensorflow_computation.tf_computation
def _calculate_server_update_statistics(server_update):
  """Calculate the L2 norm, and the max and min values of a server update."""
  flattened_struct = tf.nest.flatten(server_update)
  max_value = tf.math.reduce_max(
      tf.nest.map_structure(tf.math.reduce_max, flattened_struct)
  )
  min_value = tf.math.reduce_min(
      tf.nest.map_structure(tf.math.reduce_min, flattened_struct)
  )
  global_norm = _calculate_global_norm(server_update)
  return collections.OrderedDict(
      server_update_max=max_value,
      server_update_norm=global_norm,
      server_update_min=min_value,
  )


@tensorflow_computation.tf_computation
def _calculate_unbiased_std_dev(
    expected_value,
    expected_squared_value,
    sum_of_weights,
    sum_of_squared_weights,
):
  """Calculate the standard_deviation of a discrete distribution.

  Here, we assume that we have some distribution that takes on values `x_1` up
  through `x_n` with probabilities `w_1, ..., w_n`. We compute the standard
  deviation of this distribution, relative to the unbiased variance.

  This involves multipying the biased variance by a correction factor involving
  sums of weights and weights squared. If `a` is the sum of the `w_i` and `b` is
  the sum of the `w_i**2`, then the correction factor for the variance is
  `a**2/(a**2-b)`. Note that when the weights are all equal, this reduces to the
  standard Bessel correction factor of `n/(n-1)`. We then take a square root to
  get the standard deviation.

  Args:
    expected_value: A float representing the weighted mean of the distribution.
    expected_squared_value: A float representing the expected square value of
      the distribution.
    sum_of_weights: A float representing the sum of weights in the distribution.
    sum_of_squared_weights: A float representing the sum of the squared weights
      in the distribution.

  Returns:
    A float representing the standard deviation with respect to the unbiased
      variance.
  """
  biased_variance = expected_squared_value - expected_value**2
  correction_factor = tf.math.divide_no_nan(
      sum_of_weights**2, sum_of_weights**2 - sum_of_squared_weights
  )
  return tf.math.sqrt(correction_factor * biased_variance)


def _calculate_client_update_statistics_with_norm(client_norms, client_weights):
  """Calculate client updates with client norms."""
  client_norms_squared = intrinsics.federated_map(_square_value, client_norms)

  average_client_norm = intrinsics.federated_mean(client_norms, client_weights)
  average_client_norm_squared = intrinsics.federated_mean(
      client_norms_squared, client_weights
  )

  # TODO(b/197972289): Add SecAgg compatibility to these measurements
  sum_of_client_weights = intrinsics.federated_sum(client_weights)
  client_weights_squared = intrinsics.federated_map(
      _square_value, client_weights
  )
  sum_of_client_weights_squared = intrinsics.federated_sum(
      client_weights_squared
  )

  unbiased_std_dev = intrinsics.federated_map(
      _calculate_unbiased_std_dev,
      (
          average_client_norm,
          average_client_norm_squared,
          sum_of_client_weights,
          sum_of_client_weights_squared,
      ),
  )

  return intrinsics.federated_zip(
      collections.OrderedDict(
          average_client_norm=average_client_norm,
          std_dev_client_norm=unbiased_std_dev,
      )
  )


def _calculate_client_update_statistics(client_updates, client_weights):
  """Calculate the average and standard deviation of client updates."""
  client_norms = intrinsics.federated_map(
      _calculate_global_norm, client_updates
  )
  return _calculate_client_update_statistics_with_norm(
      client_norms, client_weights
  )


def _calculate_client_update_statistics_mixed_dtype(
    client_updates, client_weights
):
  """Calculate client update statistics of mixed data types."""
  client_norms = intrinsics.federated_map(
      _calculate_global_norm_mixed_dtype, client_updates
  )
  return _calculate_client_update_statistics_with_norm(
      client_norms, client_weights
  )


def _build_aggregator_measurement_fns(
    client_measurement_fn: Callable[
        ..., dict[str, Any]
    ] = _calculate_client_update_statistics,
    server_measurement_fn: Callable[
        ..., dict[str, Any]
    ] = _calculate_server_update_statistics,
    weighted_aggregator: bool = True,
) -> ...:
  """Create measurement functions suitable for debugging learning processes.

  These functions are intended for use with `tff.aggregators.add_measurements`.
  This function creates client and server measurements functions. The client
  measurement function computes:

  *   The (weighted) average Euclidean norm of client updates.
  *   The (weighted) standard deviation of these norms.

  The standard deviation we report is the square root of the **unbiased**
  variance. The server measurement function computes:

  *   The maximum entry of the aggregate client update.
  *   The Euclidean norm of the aggregate client update.
  *   The minimum entry of the aggregate client update.

  Note that the `client_measurement_fn` will either have input arguments
  `(client_value, client_weight)` or `client_value`, depending on whether
  `weighted_aggregator = True` or `False`, respectively. The
  `server_measurement_fn` will have input argument `server_value`.

  Args:
    client_measurement_fn: Python callable matching the docstring above.
    server_measurement_fn: Python callable matching the docstring above.
    weighted_aggregator: A boolean indicating whether the client measurement
      function is intended for use with weighted aggregators (`True`) or not
      (`False`).

  Returns:
    A tuple `(client_measurement_fn, server_measurement_fn)` of Python callables
      matching the docstring above.
  """
  if weighted_aggregator:
    federated_client_measurement_fn = client_measurement_fn
  else:

    def federated_client_measurement_fn(value):
      client_weights = intrinsics.federated_value(1.0, placements.CLIENTS)
      return client_measurement_fn(value, client_weights)

  def federated_server_measurement_fn(value):
    server_measurements = intrinsics.federated_map(server_measurement_fn, value)
    return server_measurements

  return federated_client_measurement_fn, federated_server_measurement_fn


def add_debug_measurements(
    aggregation_factory: _AggregationFactory,
) -> _AggregationFactory:
  """Adds measurements suitable for debugging learning processes.

  This will wrap a `tff.aggregator.AggregationFactory` as a new factory that
  will produce additional measurements useful for debugging learning processes.
  The underlying aggregation of client values will remain unchanged.

  These measurements generally concern the norm of the client updates, and the
  norm of the aggregated server update. The implicit weighting will be
  determined by `aggregation_factory`: If this is weighted, then the debugging
  measurements will use this weighting when computing averages. If it is
  unweighted, the debugging measurements will use uniform weighting.

  The client measurements are:

  *   The average Euclidean norm of client updates.
  *   The standard deviation of these norms.

  The standard deviation we report is the square root of the **unbiased**
  variance. The server measurements are:

  *   The maximum entry of the aggregate client update.
  *   The Euclidean norm of the aggregate client update.
  *   The minimum entry of the aggregate client update.

  In the above, an "entry" means any coordinate across all tensors in the
  structure. For example, suppose that we have client structures before
  aggregation:

  *   Client A: `[[-1, -3, -5], [2]]`
  *   Client B: `[[-1, -3, 1], [0]]`

  If we use unweighted averaging, then the aggregate client update will be the
  structure `[[-1, -3, -2], [1]]`. The maximum entry is `1`, the minimum entry
  is `-3`, and the euclidean norm is `sqrt(15)`.

  Args:
    aggregation_factory: A `tff.aggregators.AggregationFactory`. Can be weighted
      or unweighted.

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  is_weighted_aggregator = isinstance(
      aggregation_factory, factory.WeightedAggregationFactory
  )
  client_measurement_fn, server_measurement_fn = (
      _build_aggregator_measurement_fns(
          client_measurement_fn=_calculate_client_update_statistics,
          server_measurement_fn=_calculate_server_update_statistics,
          weighted_aggregator=is_weighted_aggregator,
      )
  )

  return measurements.add_measurements(
      aggregation_factory,
      client_measurement_fn=client_measurement_fn,
      server_measurement_fn=server_measurement_fn,
  )


def add_debug_measurements_with_mixed_dtype(
    aggregation_factory: _AggregationFactory,
) -> _AggregationFactory:
  """Adds measurements suitable for debugging learning processes.

  WARNING: This method works for model updates with mixed, non-`tf.float32`
  dtypes by casting all tensors to `tf.float32`. This has important numerical
  considerations, for example if the updates are quantized to `tf.int32`
  precision can be lost for values creating than approximately 2^24, and
  `tf.float64` dtypes will be narrowed to `tf.float32`. Most users
  should prefer `tff.learning.add_debug_measurements`.

  This will wrap a `tff.aggregator.AggregationFactory` as a new factory that
  will produce additional measurements useful for debugging learning processes.
  The underlying aggregation of client values will remain unchanged.

  These measurements generally concern the norm of the client updates, and the
  norm of the aggregated server update. The implicit weighting will be
  determined by `aggregation_factory`: If this is weighted, then the debugging
  measurements will use this weighting when computing averages. If it is
  unweighted, the debugging measurements will use uniform weighting.

  The client measurements are:

  *   The average Euclidean norm of client updates.
  *   The standard deviation of these norms.

  The standard deviation we report is the square root of the **unbiased**
  variance. The server measurements are:

  *   The maximum entry of the aggregate client update.
  *   The Euclidean norm of the aggregate client update.
  *   The minimum entry of the aggregate client update.

  In the above, an "entry" means any coordinate across all tensors in the
  structure. For example, suppose that we have client structures before
  aggregation:

  *   Client A: `[[-1, -3, -5], [2]]`
  *   Client B: `[[-1, -3, 1], [0]]`

  If we use unweighted averaging, then the aggregate client update will be the
  structure `[[-1, -3, -2], [1]]`. The maximum entry is `1`, the minimum entry
  is `-3`, and the euclidean norm is `sqrt(15)`.

  Args:
    aggregation_factory: A `tff.aggregators.AggregationFactory`. Can be weighted
      or unweighted.

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  is_weighted_aggregator = isinstance(
      aggregation_factory, factory.WeightedAggregationFactory
  )
  client_measurement_fn, server_measurement_fn = (
      _build_aggregator_measurement_fns(
          client_measurement_fn=_calculate_client_update_statistics_mixed_dtype,
          server_measurement_fn=_calculate_server_update_statistics_mixed_dtype,
          weighted_aggregator=is_weighted_aggregator,
      )
  )

  return measurements.add_measurements(
      aggregation_factory,
      client_measurement_fn=client_measurement_fn,
      server_measurement_fn=server_measurement_fn,
  )
