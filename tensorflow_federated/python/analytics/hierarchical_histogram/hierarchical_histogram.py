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
"""The functions for creating the federated computation for hierarchical histogram aggregation."""

from typing import Optional
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types


@tf.function
def _discretized_histogram_counts(client_data: tf.data.Dataset,
                                  lower_bound: float, upper_bound: float,
                                  num_bins: int) -> tf.Tensor:
  """Disretizes `client_data` and creates a histogram on the discretized data.

  Discretizes `client_data` by allocating records into `num_bins` bins between
  `lower_bound` and `upper_bound`. Data outside the range will be ignored.

  Args:
    client_data: A `tf.data.Dataset` containing the client-side records.
    lower_bound: A `float` specifying the lower bound of the data range.
    upper_bound: A `float` specifying the upper bound of the data range.
    num_bins: An `int`. The integer number of bins to compute.

  Returns:
    A `tf.Tensor` of shape `(num_bins,)` representing the histogram on
    discretized data.
  """

  if upper_bound < lower_bound:
    raise ValueError(f'upper_bound: {upper_bound} is smaller than '
                     f'lower_bound: {lower_bound}.')

  if num_bins <= 0:
    raise ValueError(f'num_bins: {num_bins} smaller or equal to zero.')

  data_type = client_data.element_spec.dtype

  if data_type != tf.float32:
    raise ValueError(f'`client_data` contains {data_type} values.'
                     f'`tf.float32` is expected.')

  precision = (upper_bound - lower_bound) / num_bins

  def insert_record(histogram, record):
    """Inserts a record to the histogram.

    If the record is outside the valid range, it will be dropped.

    Args:
      histogram: A `tf.Tensor` representing the histogram.
      record: A `float` representing the incoming record.

    Returns:
      A `tf.Tensor` representing updated histgoram with the input record
      inserted.
    """

    if histogram.shape != (num_bins,):
      raise ValueError(f'Expected shape ({num_bins}, ). '
                       f'Get {histogram.shape}.')

    if record < lower_bound or record >= upper_bound:
      return histogram
    else:
      bin_index = tf.cast(
          tf.math.floor((record - lower_bound) / precision), tf.int32)
    return tf.tensor_scatter_nd_add(
        tensor=histogram, indices=[[bin_index]], updates=[1])

  histogram = client_data.reduce(
      tf.zeros([num_bins], dtype=tf.int32), insert_record)

  return histogram


def _build_hierarchical_histogram_computation(
    lower_bound: float, upper_bound: float, num_bins: int,
    aggregation_factory: Optional[factory.UnweightedAggregationFactory]):
  """Utility function creating tff computation given the parameters and factory.

  Args:
    lower_bound: A `float` specifying the lower bound of the data range.
    upper_bound: A `float` specifying the upper bound of the data range.
    num_bins: The integer number of bins to compute.
    aggregation_factory: The aggregation factory used to construct the federated
      computation.

  Returns:
    A tff federated computation function.
  """

  @computations.tf_computation(computation_types.SequenceType(tf.float32))
  def client_work(client_data):
    return _discretized_histogram_counts(client_data, lower_bound, upper_bound,
                                         num_bins)

  aggregator = aggregation_factory.create(client_work.type_signature.result)

  @computations.federated_computation(
      computation_types.at_clients(client_work.type_signature.parameter))
  def hierarchical_histogram_computation(federated_client_data):
    # Work done at clients.
    client_histogram = intrinsics.federated_map(client_work,
                                                federated_client_data)
    # Aggregation to server.
    return aggregator.next(aggregator.initialize(), client_histogram).result

  return hierarchical_histogram_computation


def build_central_hierarchical_histogram_computation(
    lower_bound: float,
    upper_bound: float,
    num_bins: int,
    arity: int = 2,
    clip_mechanism: str = 'sub-sampling',
    max_records_per_user: int = 10,
    dp_mechanism: str = 'gaussian',
    noise_multiplier: float = 0.0):
  """Create the tff federated computation for central hierarchical histogram aggregation.

  Args:
    lower_bound: A `float` specifying the lower bound of the data range.
    upper_bound: A `float` specifying the upper bound of the data range.
    num_bins: The integer number of bins to compute.
    arity: The branching factor of the tree. Defaults to 2.
    clip_mechanism: A `str` representing the clipping mechanism. Currently
      supported mechanisms are
      - 'sub-sampling': (Default) Uniformly sample up to `max_records_per_user`
        records without replacement from the client dataset.
      - 'distinct': Uniquify client dataset and uniformly sample up to
        `max_records_per_user` records without replacement from it.
    max_records_per_user: An `int` representing the maximum of records each user
      can include in their local histogram. Defaults to 10.
    dp_mechanism: A `str` representing the differentially private mechanism to
      use. Currently supported mechanisms are
      - 'gaussian': (Default) Tree aggregation with Gaussian mechanism.
      - 'no-noise': Tree aggregation mechanism without noise.
     noise_multiplier: A `float` specifying the noise multiplier (central noise
       stddev / L2 clip norm) for model updates. Defaults to 0.0.

  Returns:
    A tff.federated_computation function to perform central tree aggregation.
  """
  if upper_bound < lower_bound:
    raise ValueError(f'upper_bound: {upper_bound} is smaller than '
                     f'lower_bound: {lower_bound}.')
  if num_bins <= 0:
    raise ValueError(f'`num_bins` should be positive. Found {num_bins}.')
  if arity < 2:
    raise ValueError(f'`arity` should be at least 2. Found {arity}.')
  if clip_mechanism not in ['sub-sampling', 'distinct']:
    raise ValueError('`clip_mechanism` should be one of '
                     f'[sub-sampling, distinct]. Found {clip_mechanism}.')
  if max_records_per_user < 1:
    raise ValueError(f'`max_records_per_user` should be at least 1. '
                     f'Found {max_records_per_user}.')
  if dp_mechanism not in ['gaussian', 'no-noise']:
    raise ValueError('`dp_mechanism` should be one of '
                     f'[gaussian, no-noise]. Found {dp_mechanism}.')
  if noise_multiplier < 0.:
    raise ValueError(f'`noise_multiplier` should be positive. '
                     f'Found {noise_multiplier}.')

  agg_factory = hierarchical_histogram_factory.create_central_hierarchical_histogram_aggregation_factory(
      num_bins, arity, clip_mechanism, max_records_per_user, dp_mechanism,
      noise_multiplier)

  return _build_hierarchical_histogram_computation(lower_bound, upper_bound,
                                                   num_bins, agg_factory)
