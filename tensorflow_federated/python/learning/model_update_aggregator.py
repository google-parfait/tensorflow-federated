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


def _affine_transform(multiplier, increment):
  transform_tf_comp = computations.tf_computation(
      lambda value: multiplier * value + increment, tf.float32)
  return computations.federated_computation(
      lambda value: intrinsics.federated_map(transform_tf_comp, value),
      computation_types.at_server(tf.float32))


def zeroing_aggregator() -> factory.WeightedAggregationFactory:
  """Creates aggregator for mean with adaptive zeroing.

  Returns:
    A `factory.WeightedAggregationFactory` with zeroing for data corruption
    mitigation.
  """
  zeroing_quantile = quantile_estimation.PrivateQuantileEstimationProcess(
      tfp.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=10.0,
          target_quantile=0.98,
          learning_rate=math.log(10.0),
          geometric_update=True))
  zeroing_norm = zeroing_quantile.map(
      _affine_transform(multiplier=2.0, increment=1.0))
  return clipping_factory.ZeroingFactory(
      zeroing_norm, mean_factory.MeanFactory(), norm_order=float('inf'))


def zeroing_clipping_aggregator(
    clip: float = None) -> factory.WeightedAggregationFactory:
  """Creates aggregator for mean with adaptive zeroing and clipping.

  Args:
    clip: Fixed clipping norm if fixed clipping is desired. If unspecified,
      adaptive clipping will be used.

  Returns:
    A `factory.WeightedAggregationFactory` with zeroing for data corruption
    mitigation and clipping for robustness to outliers.
  """
  if clip is None:
    clip = quantile_estimation.PrivateQuantileEstimationProcess(
        tfp.NoPrivacyQuantileEstimatorQuery(
            initial_estimate=1.0,
            target_quantile=0.8,
            learning_rate=0.2,
            geometric_update=True))
  else:
    py_typecheck.check_type(clip, float, 'clip')
    _check_positive(clip, 'clip')

  factory_ = clipping_factory.ClippingFactory(clip, mean_factory.MeanFactory())

  zeroing_quantile = quantile_estimation.PrivateQuantileEstimationProcess(
      tfp.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=10.0,
          target_quantile=0.98,
          learning_rate=math.log(10.0),
          geometric_update=True))
  zeroing_norm = zeroing_quantile.map(
      _affine_transform(multiplier=2.0, increment=1.0))
  return clipping_factory.ZeroingFactory(
      zeroing_norm, factory_, norm_order=float('inf'))


def zeroing_dp_aggregator(
    noise_multiplier: float,
    clients_per_round: float) -> factory.UnweightedAggregationFactory:
  """Creates aggregator with adaptive zeroing and differential privacy.

  Args:
    noise_multiplier: A float specifying the noise multiplier for the Gaussian
      mechanism for model updates.
    clients_per_round: A float specifying the expected number of clients per
      round.

  Returns:
    A `factory.WeightedAggregationFactory` with zeroing for data corruption
    mitigation and differential privacy.
  """

  # clipped_count_stddev defaults to 0.05 * clients_per_round. The noised
  # fraction of unclipped updates will be within 0.1 of the true fraction with
  # of unclipped updates will be within 0.1 of the true fraction with
  # 95.4% probability, and will be within 0.15 of the true fraction with
  # 99.7% probability. Even in this unlikely case, the error on the update
  # would be a factor of exp(0.2 * 0.15) = 1.03, a small deviation. So this
  # default gives maximal privacy for acceptable probability of deviation.
  clipped_count_stddev = 0.05 * clients_per_round

  query = tfp.QuantileAdaptiveClipAverageQuery(
      initial_l2_norm_clip=0.1,
      noise_multiplier=noise_multiplier,
      denominator=clients_per_round,
      target_unclipped_quantile=0.5,
      learning_rate=0.2,
      clipped_count_stddev=clipped_count_stddev,
      expected_num_records=clients_per_round,
      geometric_update=True)

  factory_ = dp_factory.DifferentiallyPrivateFactory(query)

  zeroing_quantile = quantile_estimation.PrivateQuantileEstimationProcess(
      tfp.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=10.0,
          target_quantile=0.98,
          learning_rate=math.log(10.0),
          geometric_update=True))
  zeroing_norm = zeroing_quantile.map(
      _affine_transform(multiplier=2.0, increment=1.0))
  return clipping_factory.ZeroingFactory(
      zeroing_norm, factory_, norm_order=float('inf'))
