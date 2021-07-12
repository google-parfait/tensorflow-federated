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
"""The functions for creating the federated computation for hierarchical histogram aggregation."""

import math
from typing import Tuple

from scipy import optimize
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_factory
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


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
def discretized_histogram_counts(client_data, data_range: Tuple[float, float],
                                 num_bins: int):
  """Calculate counts of elements in the domain specified by data_range and precision in the client data.

  Args:
    client_data: A `tf.data.Dataset`, in which each element is within the domain
      specified by data_range and precision.
    data_range: a tuple of two floats specifying the start and the end of the
      domain.
    num_bins: The integer number of bins to compute.

  Returns:
    A tensor containing a list of tf.float32 counts for the domain elements in
    the client_data.
  """

  if data_range[1] < data_range[0]:
    raise ValueError(
        f'data_range[1]: {data_range[1]} is smaller than data_range[0]: {data_range[0]}.'
    )
  if num_bins <= 0:
    raise ValueError(f'num_bins: {num_bins} smaller or equal to zero.')

  bucket_boundaries = [
      tf.cast(
          tf.linspace(data_range[0], data_range[1], num_bins + 1), tf.float32)
  ]
  if isinstance(client_data, tf.data.Dataset):
    tensor_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for v in client_data:
      tensor_list = tensor_list.write(tensor_list.size(), v)
    client_data = tensor_list.stack()

  bucket_indexes = tf.raw_ops.BoostedTreesBucketize(
      float_values=[tf.cast(client_data, tf.float32)],
      bucket_boundaries=bucket_boundaries)[0]

  def accumulate_counts(totals, bin_index):
    """Accumulate counts with one example.

    Args:
      totals: A tensor contains the histogram counts of processed example.
      bin_index: The bin index of an item in the input examples.

    Returns:
      Updated histgoram counts with the input `bin_index`.
    """
    one_hot_histogram_counts = tf.math.bincount(
        bin_index, minlength=num_bins, maxlength=num_bins)
    return totals + one_hot_histogram_counts

  histogram = tf.foldl(
      accumulate_counts,
      bucket_indexes,
      initializer=tf.zeros([num_bins], dtype=tf.int32))

  histogram = tf.cast(histogram, tf.float32)
  return histogram


class HistogramClientWork():
  """Wrapper class for the client-side update function."""

  def __init__(self, data_range: Tuple[float, float], num_bins: int):

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def client_work(client_data):
      return discretized_histogram_counts(client_data, data_range, num_bins)

    self._work = client_work

  @property
  def work(self) -> computation_base.Computation:
    return self._work


def build_hierarchical_histogram_computation(client_work, aggregation_factory):
  """A function wrapping client_work function and aggregation_factory into a tff computation.

  Args:
    client_work: The client-side update function used to construct the federated
      computation.
    aggregation_factory: The aggregation factory used to construct the federated
      computation.

  Returns:
    A tff federated computation function.
  """
  assert isinstance(aggregation_factory, factory.UnweightedAggregationFactory)
  aggregator = aggregation_factory.create(
      client_work.work.type_signature.result)

  @computations.federated_computation(
      computation_types.FederatedType(client_work.work.type_signature.parameter,
                                      placements.CLIENTS))
  def hierarchical_histogram_computation(federated_client_data):
    # Work done at clients.
    client_histogram = intrinsics.federated_map(client_work.work,
                                                federated_client_data)
    # Aggregation to server.
    return aggregator.next(aggregator.initialize(), client_histogram).result

  return hierarchical_histogram_computation


def build_central_hierarchical_histogram_computation(
    data_range: Tuple[float, float],
    num_bins: int,
    arity: int = 2,
    max_records_per_user: int = 100,
    epsilon: float = 1,
    delta: float = 1e-5):
  """Create the tff federated computation for central hierarchical histogram aggregation.

  Args:
    data_range: a list of two floats specifying the start and the end of the
      domain.
    num_bins: The integer number of bins to compute.
    arity: the branching factor of the tree, default 2.
    max_records_per_user: the maximum number of records each user is allowed to
      contribute.
    epsilon: differential privacy parameter.
    delta: differential privacy parameter.

  Returns:
    An tff.federated_computation function to perform central tree aggregation.
  """

  if epsilon == 0:
    stddev = 0.
  else:
    stddev = _find_noise_multiplier(
        epsilon, delta, steps=math.ceil(math.log(num_bins, arity)))
  central_tree_aggregation_factory = hierarchical_histogram_factory.create_central_hierarchical_histogram_factory(
      stddev, arity, max_records_per_user)

  return build_hierarchical_histogram_computation(
      HistogramClientWork(data_range, num_bins),
      central_tree_aggregation_factory)


def build_distributed_hierarchical_histogram_computation(
    data_range: Tuple[float, float],
    num_bins: int,
    client_num_lower_bound: int,
    arity: int = 2,
    max_records_per_user: int = 100,
    epsilon: float = 1,
    delta: float = 1e-5):
  """Create the tff federated computation for local hierarchical histogram aggregation.

  Args:
    data_range: a list of two floats specifying the start and the end of the
      domain.
    num_bins: The integer number of bins to compute.
    client_num_lower_bound: The expected lower bound for the number of clients.
    arity: the branching factor of the tree, default 2.
    max_records_per_user: the maximum number of records each user is allowed to
      contribute.
    epsilon: differential privacy parameter.
    delta: differential privacy parameter.

  Returns:
    An tff.federated_computation function to perform distributed tree
    aggregation.
  """
  if epsilon == 0:
    stddev = 0.
  else:
    stddev = _find_noise_multiplier(
        epsilon, delta, steps=math.ceil(math.log(num_bins, arity)))
    stddev = stddev / math.sqrt(client_num_lower_bound)
  distributed_tree_aggregation_factory = hierarchical_histogram_factory.create_distributed_hierarchical_histogram_factory(
      stddev, arity, max_records_per_user)

  return build_hierarchical_histogram_computation(
      HistogramClientWork(data_range, num_bins),
      distributed_tree_aggregation_factory)
