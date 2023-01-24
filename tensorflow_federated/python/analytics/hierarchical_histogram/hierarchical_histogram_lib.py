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

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.hierarchical_histogram import clipping_factory
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_factory
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import iterative_process


@attr.s(eq=False, frozen=True)
class ServerOutput:
  """The container of results.

  Attributes:
    aggregated_hierarchical_histogram: A `tf.RaggedTensor` of the aggregated
      hierarchical histogram this round.
    round_timestamp: An int64 scalar timestamp of the beginning of the round.
      The value is expressed as seconds since the Unix epoch (1970-01-01
      00:00:00 UTC).
  """

  aggregated_hierarchical_histogram = attr.ib()
  round_timestamp = attr.ib()


def _discretized_histogram_counts(
    client_data: tf.data.Dataset,
    lower_bound: float,
    upper_bound: float,
    num_bins: int,
) -> tf.Tensor:
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
    raise ValueError(
        f'upper_bound: {upper_bound} is smaller than '
        f'lower_bound: {lower_bound}.'
    )

  if num_bins <= 0:
    raise ValueError(f'num_bins: {num_bins} smaller or equal to zero.')

  data_type = client_data.element_spec.dtype

  if data_type != tf.float32:
    raise ValueError(
        f'`client_data` contains {data_type} values.`tf.float32` is expected.'
    )

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
      raise ValueError(f'Expected shape ({num_bins}, ). Get {histogram.shape}.')

    if record < lower_bound or record >= upper_bound:
      return histogram
    else:
      bin_index = tf.cast(
          tf.math.floor((record - lower_bound) / precision), tf.int32
      )
    return tf.tensor_scatter_nd_add(
        tensor=histogram, indices=[[bin_index]], updates=[1]
    )

  histogram = client_data.reduce(
      tf.zeros([num_bins], dtype=tf.int32), insert_record
  )

  return histogram


def build_hierarchical_histogram_computation(
    lower_bound: float,
    upper_bound: float,
    num_bins: int,
    arity: int = 2,
    clip_mechanism: str = 'sub-sampling',
    max_records_per_user: int = 10,
    dp_mechanism: str = 'no-noise',
    noise_multiplier: float = 0.0,
    expected_clients_per_round: int = 10,
    bits: int = 22,
    enable_secure_sum: bool = True,
):
  """Creates the TFF computation for hierarchical histogram aggregation.

  Args:
    lower_bound: A `float` specifying the lower bound of the data range.
    upper_bound: A `float` specifying the upper bound of the data range.
    num_bins: The integer number of bins to compute.
    arity: The branching factor of the tree. Defaults to 2.
    clip_mechanism: A `str` representing the clipping mechanism. Currently
      supported mechanisms are - 'sub-sampling': (Default) Uniformly sample up
      to `max_records_per_user` records without replacement from the client
      dataset. - 'distinct': Uniquify client dataset and uniformly sample up to
      `max_records_per_user` records without replacement from it.
    max_records_per_user: An `int` representing the maximum of records each user
      can include in their local histogram. Defaults to 10.
    dp_mechanism: A `str` representing the differentially private mechanism to
      use. Currently supported mechanisms are - 'no-noise': (Default) Tree
      aggregation mechanism without noise. - 'central-gaussian': Tree
      aggregation with central Gaussian mechanism. -
      'distributed-discrete-gaussian': Tree aggregation mechanism with the
      distributed discrete Gaussian mechanism in "The Distributed Discrete
      Gaussian Mechanism for Federated Learning with Secure Aggregation. Peter
      Kairouz, Ziyu Liu, Thomas Steinke".
     noise_multiplier: A `float` specifying the noise multiplier (central noise
       stddev / L2 clip norm) for model updates. Defaults to 0.0.
    expected_clients_per_round: An `int` specifying the lower bound on the
      expected number of clients. Only needed when `dp_mechanism` is
      'distributed-discrete-gaussian'. Defaults to 10.
    bits: A positive integer specifying the communication bit-width B (where
      2**B will be the field size for SecAgg operations). Only needed when
      `dp_mechanism` is 'distributed-discrete-gaussian'. Please read the below
      precautions carefully and set `bits` accordingly. Otherwise, unexpected
      overflow or accuracy degradation might happen. (1) Should be in the
      inclusive range [1, 22] to avoid overflow inside secure aggregation; (2)
      Should be at least as large as `log2(4 * sqrt(expected_clients_per_round)*
      noise_multiplier * l2_norm_bound + expected_clients_per_round *
      max_records_per_user) + 1` to avoid accuracy degradation caused by
      frequent modular clipping; (3) If the number of clients exceed
      `expected_clients_per_round`, overflow might happen.
    enable_secure_sum: Whether to aggregate client's update by secure sum or
      not. Defaults to `True`. When `dp_mechanism` is set to
      `'distributed-discrete-gaussian'`, `enable_secure_sum` must be `True`.

  Returns:
    A federated computation that performs hierarchical histogram aggregation.
  """
  _check_greater_than_equal(
      upper_bound, lower_bound, 'upper_bound', 'lower_bound'
  )
  _check_positive(num_bins, 'num_bins')
  _check_greater_than_equal_thres(arity, 2, 'arity')
  _check_membership(
      clip_mechanism, clipping_factory.CLIP_MECHANISMS, 'clip_mechanism'
  )
  _check_greater_than_equal_thres(
      max_records_per_user, 1, 'max_records_per_user'
  )
  _check_membership(
      dp_mechanism, hierarchical_histogram_factory.DP_MECHANISMS, 'dp_mechanism'
  )
  _check_greater_than_equal_thres(noise_multiplier, 0.0, noise_multiplier)
  _check_positive(expected_clients_per_round, 'expected_clients_per_round')
  _check_in_range(bits, 'bits', 1, 22)
  _check_greater_than_equal_thres(
      bits, math.log2(expected_clients_per_round), 'bits'
  )
  if (
      not enable_secure_sum
      and dp_mechanism
      in hierarchical_histogram_factory.DISTRIBUTED_DP_MECHANISMS
  ):
    raise ValueError(
        'When dp_mechanism is '
        f'{hierarchical_histogram_factory.DISTRIBUTED_DP_MECHANISMS}, '
        'enable_secure_sum must be set to True to preserve distributed DP.'
    )

  @tensorflow_computation.tf_computation(
      computation_types.SequenceType(tf.float32)
  )
  def client_work(client_data):
    return _discretized_histogram_counts(
        client_data, lower_bound, upper_bound, num_bins
    )

  agg_factory = hierarchical_histogram_factory.create_hierarchical_histogram_aggregation_factory(
      num_bins=num_bins,
      arity=arity,
      clip_mechanism=clip_mechanism,
      max_records_per_user=max_records_per_user,
      dp_mechanism=dp_mechanism,
      noise_multiplier=noise_multiplier,
      expected_clients_per_round=expected_clients_per_round,
      bits=bits,
      enable_secure_sum=enable_secure_sum,
  )

  process = agg_factory.create(client_work.type_signature.result)

  @federated_computation.federated_computation(
      computation_types.at_clients(client_work.type_signature.parameter)
  )
  def hierarchical_histogram_computation(federated_client_data):
    round_timestamp = intrinsics.federated_eval(
        tensorflow_computation.tf_computation(
            lambda: tf.cast(tf.timestamp(), tf.int64)
        ),
        placements.SERVER,
    )
    client_histogram = intrinsics.federated_map(
        client_work, federated_client_data
    )

    server_output = intrinsics.federated_zip(
        ServerOutput(
            process.next(process.initialize(), client_histogram).result,
            round_timestamp,
        )
    )
    return server_output

  return hierarchical_histogram_computation


def build_hierarchical_histogram_process(
    lower_bound: float,
    upper_bound: float,
    num_bins: int,
    arity: int = 2,
    clip_mechanism: str = 'sub-sampling',
    max_records_per_user: int = 10,
    dp_mechanism: str = 'no-noise',
    noise_multiplier: float = 0.0,
    expected_clients_per_round: int = 10,
    bits: int = 22,
    enable_secure_sum: bool = True,
) -> iterative_process.IterativeProcess:
  """Creates an IterativeProcess for hierarchical histogram aggregation.

  This function wraps the `tff.computation` created by the
  `build_hierarchical_histogram_computation` in an `IterativeProcess` that
  is compatible with `tff.backends.mapreduce.MapReduceForm`.

  Args:
    lower_bound: A `float` specifying the lower bound of the data range.
    upper_bound: A `float` specifying the upper bound of the data range.
    num_bins: The integer number of bins to compute.
    arity: The branching factor of the tree. Defaults to 2.
    clip_mechanism: A `str` representing the clipping mechanism. Currently
      supported mechanisms are - 'sub-sampling': (Default) Uniformly sample up
      to `max_records_per_user` records without replacement from the client
      dataset. - 'distinct': Uniquify client dataset and uniformly sample up to
      `max_records_per_user` records without replacement from it.
    max_records_per_user: An `int` representing the maximum of records each user
      can include in their local histogram. Defaults to 10.
    dp_mechanism: A `str` representing the differentially private mechanism to
      use. Currently supported mechanisms are - 'no-noise': (Default) Tree
      aggregation mechanism without noise. - 'central-gaussian': Tree
      aggregation with central Gaussian mechanism. -
      'distributed-discrete-gaussian': Tree aggregation mechanism with the
      distributed discrete Gaussian mechanism in "The Distributed Discrete
      Gaussian Mechanism for Federated Learning with Secure Aggregation. Peter
      Kairouz, Ziyu Liu, Thomas Steinke".
     noise_multiplier: A `float` specifying the noise multiplier (central noise
       stddev / L2 clip norm) for model updates. Defaults to 0.0.
    expected_clients_per_round: An `int` specifying the lower bound on the
      expected number of clients. Only needed when `dp_mechanism` is
      'distributed-discrete-gaussian'. Defaults to 10.
    bits: A positive integer specifying the communication bit-width B (where
      2**B will be the field size for SecAgg operations). Only needed when
      `dp_mechanism` is 'distributed-discrete-gaussian'. Please read the below
      precautions carefully and set `bits` accordingly. Otherwise, unexpected
      overflow or accuracy degradation might happen. (1) Should be in the
      inclusive range [1, 22] to avoid overflow inside secure aggregation; (2)
      Should be at least as large as `log2(4 * sqrt(expected_clients_per_round)*
      noise_multiplier * l2_norm_bound + expected_clients_per_round *
      max_records_per_user) + 1` to avoid accuracy degradation caused by
      frequent modular clipping; (3) If the number of clients exceed
      `expected_clients_per_round`, overflow might happen.
    enable_secure_sum: Whether to aggregate client's update by secure sum or
      not. Defaults to `True`. When `dp_mechanism` is set to
      `'distributed-discrete-gaussian'`, `enable_secure_sum` must be `True`.

  Returns:
    A federated computation that performs hierarchical histogram aggregation.
  """
  _check_greater_than_equal(
      upper_bound, lower_bound, 'upper_bound', 'lower_bound'
  )
  _check_positive(num_bins, 'num_bins')
  _check_greater_than_equal_thres(arity, 2, 'arity')
  _check_membership(
      clip_mechanism, clipping_factory.CLIP_MECHANISMS, 'clip_mechanism'
  )
  _check_greater_than_equal_thres(
      max_records_per_user, 1, 'max_records_per_user'
  )
  _check_membership(
      dp_mechanism, hierarchical_histogram_factory.DP_MECHANISMS, 'dp_mechanism'
  )
  _check_greater_than_equal_thres(noise_multiplier, 0.0, noise_multiplier)
  _check_positive(expected_clients_per_round, 'expected_clients_per_round')
  _check_in_range(bits, 'bits', 1, 22)
  _check_greater_than_equal_thres(
      bits, math.log2(expected_clients_per_round), 'bits'
  )
  if (
      not enable_secure_sum
      and dp_mechanism
      in hierarchical_histogram_factory.DISTRIBUTED_DP_MECHANISMS
  ):
    raise ValueError(
        'When dp_mechanism is '
        f'{hierarchical_histogram_factory.DISTRIBUTED_DP_MECHANISMS}, '
        'enable_secure_sum must be set to True to preserve distributed DP.'
    )
  one_round_computation = build_hierarchical_histogram_computation(
      lower_bound=lower_bound,
      upper_bound=upper_bound,
      num_bins=num_bins,
      arity=arity,
      clip_mechanism=clip_mechanism,
      max_records_per_user=max_records_per_user,
      dp_mechanism=dp_mechanism,
      noise_multiplier=noise_multiplier,
      expected_clients_per_round=expected_clients_per_round,
      bits=bits,
      enable_secure_sum=enable_secure_sum,
  )

  parameter_type_signature = one_round_computation.type_signature.parameter
  result_type_signature = one_round_computation.type_signature.result

  @tensorflow_computation.tf_computation
  def initialize():
    value_type, _ = result_type_signature.member
    # Creates a `tf.RaggedTensor` that has the same `type_signature` as the
    # result returned by `one_round_computation`. This is to make sure the
    # generated IterativeProcess is compatible with
    # `tff.backends.mapreduce.MapReduceForm`.
    flat_values_shape = value_type[0].shape
    flat_values_dtype = value_type[0].dtype
    nested_row_splits = np.zeros(shape=value_type[1][0].shape)
    # To generated a valid `tf.RaggedTensor`, the first element in
    # `nested_row_splits` must be 0, and the last element in `nested_row_splits`
    # must be the length of `flat_values`.
    nested_row_splits[-1] = flat_values_shape[0]
    initial_hierarchical_histogram = tf.RaggedTensor.from_nested_row_splits(
        flat_values=tf.zeros(shape=flat_values_shape, dtype=flat_values_dtype),
        nested_row_splits=[nested_row_splits],
    )
    initial_timestamp = tf.constant(0, dtype=tf.int64)
    return ServerOutput(initial_hierarchical_histogram, initial_timestamp)

  @federated_computation.federated_computation
  def init_fn():
    return intrinsics.federated_eval(initialize, placements.SERVER)

  @federated_computation.federated_computation(
      init_fn.type_signature.result, parameter_type_signature
  )
  def next_fn(_, client_data):
    return one_round_computation(client_data), intrinsics.federated_value(
        (), placements.SERVER
    )

  return iterative_process.IterativeProcess(init_fn, next_fn)


def _check_greater_than_equal(lvalue, rvalue, llabel, rlabel):
  if lvalue < rvalue:
    raise ValueError(
        f'`{llabel}` should be no smaller than '
        f'`{rlabel}`. Found {lvalue} and '
        f'{rvalue}.'
    )


def _check_greater_than_equal_thres(value, threshold, label):
  if value < threshold:
    raise ValueError(f'`{label}` must be at least {threshold}. Found {value}.')


def _check_positive(value, label):
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {value}.')


def _check_non_negative(value, label):
  if value < 0:
    raise ValueError(f'{label} must be non-negative. Found {value}.')


def _check_membership(value, valid_set, label):
  if value not in valid_set:
    raise ValueError(f'`{label}` must be one of {valid_set}. Found {value}.')


def _check_in_range(value, label, left, right):
  """Checks that a scalar value is in specified range."""
  if not value >= left or not value <= right:
    raise ValueError(
        f'{label} should be within [{left}, {right}]. Found {value}.'
    )
