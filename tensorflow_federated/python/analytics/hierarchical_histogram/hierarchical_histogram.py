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

import math

from scipy import optimize
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types

# TODO(b/193903764): Replace these privacy utility functions with TensorFlow
# Privacy version once they are upstreamed.


# Standard privacy accounting mechanism
def _get_eps(sampling_rate=1.0,
             noise_multiplier=1.0,
             steps=1,
             target_delta=1e-10):
  """Simple wrapper on rdp_accountant."""
  rdp = tfp.privacy.analysis.rdp_accountant.compute_rdp(
      q=sampling_rate,
      noise_multiplier=noise_multiplier,
      steps=steps,
      orders=list(range(1, 2000)))
  eps, _, _ = tfp.privacy.analysis.rdp_accountant.get_privacy_spent(
      list(range(1, 2000)), rdp, target_delta=target_delta)
  return eps


def _find_noise_multiplier(eps=1.0, delta=0.001, steps=1):
  """Given eps and delta, find the noise_multiplier."""

  def get_eps_for_noise_multiplier(z):
    eps = _get_eps(noise_multiplier=z, steps=steps, target_delta=delta)
    return eps

  opt_noise_multiplier, r = optimize.brentq(
      lambda z: get_eps_for_noise_multiplier(z) - eps,
      0.1,
      10000,
      full_output=True)
  if r.converged:
    return opt_noise_multiplier
  else:
    return -1


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
    num_bins: A `int`. The integer number of bins to compute.

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

  return tf.cast(histogram, tf.float32)


def _build_hierarchical_histogram_computation(
    lower_bound: float, upper_bound: float, num_bins: int,
    aggregation_factory: factory.UnweightedAggregationFactory):
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
    max_records_per_user: int = 100,
    epsilon: float = 1,
    delta: float = 1e-5,
    secure_sum: bool = False):
  """Create the tff federated computation for central hierarchical histogram aggregation.

  Args:
    lower_bound: A `float` specifying the lower bound of the data range.
    upper_bound: A `float` specifying the upper bound of the data range.
    num_bins: The integer number of bins to compute.
    arity: The branching factor of the tree. Defaults to 2.
    max_records_per_user: The maximum number of records each user is allowed to
      contribute. Defaults to 100.
    epsilon: Differential privacy parameter. Defaults to 1.
    delta: Differential privacy parameter. Defaults to 1e-5.
    secure_sum: A boolean deciding whether to use secure aggregation. Defaults
      to `False`.

  Returns:
    A tff.federated_computation function to perform central tree aggregation.
  """

  if upper_bound < lower_bound:
    raise ValueError(f'upper_bound: {upper_bound} is smaller than '
                     f'lower_bound: {lower_bound}.')

  if num_bins <= 0:
    raise ValueError(f'num_bins: {num_bins} smaller or equal to zero.')

  if arity < 2:
    raise ValueError(f'Arity should be at least 2.' f'arity={arity} is given.')

  if max_records_per_user < 1:
    raise ValueError(f'Maximum records per user should be at least 1. '
                     f'max_records_per_user={max_records_per_user} is given.')

  if epsilon < 0 or delta < 0 or delta > 1:
    raise ValueError(f'Privacy parameters in wrong range: '
                     f'(epsilon, delta): ({epsilon}, {delta})')

  if epsilon == 0.:
    stddev = 0.
  else:
    stddev = _find_noise_multiplier(
        epsilon, delta, steps=math.ceil(math.log(num_bins, arity)))

  central_tree_aggregation_factory = hierarchical_histogram_factory.create_central_hierarchical_histogram_factory(
      stddev, arity, max_records_per_user, secure_sum=secure_sum)

  return _build_hierarchical_histogram_computation(
      lower_bound, upper_bound, num_bins, central_tree_aggregation_factory)
