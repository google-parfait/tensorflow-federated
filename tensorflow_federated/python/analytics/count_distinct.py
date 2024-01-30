# Copyright 2023, The TensorFlow Federated Authors.
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
"""Implementation of federated HyperLogLog for counting distinct items.

See https://en.wikipedia.org/wiki/HyperLogLog for additional details on this
algorithm.
"""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements

# See https://en.wikipedia.org/wiki/HyperLogLog for usage of these constants.
# Setting HLL_SKETCH_SIZE = 64 is not currently supported because it is not
# clear how to hash to a uint64 value in tensorflow.  By default, we can hash
# to int64 values which only gives numbers in the range [0, 2^63-1].
# Moreover, HLL_SKETCH_SIZE must be a power of 2, hence the choice of 32.
# We can expect to count up to 2^32 records using this sketch size, which is
# more than enough for most applications.
HLL_SKETCH_SIZE = 32  # Hash function will return int mod 2^{HLL_SKETCH_SIZE}
HLL_BIT_INDEX_HEAD = 5  # log2(HLL_SKETCH_SIZE)
HLL_BIT_INDEX_TAIL = HLL_SKETCH_SIZE - HLL_BIT_INDEX_HEAD
HLL_ALPHA = 0.697


@tf.function
def _log2(u: tf.Tensor) -> tf.Tensor:
  """Compute integer log base 2."""
  # Implemented in terms of bit-wise operations instead of
  # tf.numpy.experimental.log so that it works with large integers up to 2^63.
  ans = tf.constant(0, dtype=tf.int64)
  u = tf.cast(u, dtype=tf.int64)
  while u > 0:
    ans += 1
    u = u // 2
  return ans - 1


def build_client_hyperloglog_computation() -> computation_base.Computation:
  """Builds a `tff.Computation` for computing client hyperloglog sketches.

  Specifically, the returned computation consumes a dataset of integer hashes
  and returns the HyperLogLog sketch of size HLL_SKETCH_SIZE.

  Returns:
    A `tff.Computation` for computing client hyperloglog sketches.
  """

  @tensorflow_computation.tf_computation(
      computation_types.SequenceType(np.int64)
  )
  @tf.function
  def _client_hyperloglog(client_data: tf.data.Dataset) -> tf.Tensor:
    """Computes a HyperLogLog sketch on a client dataset.

    Args:
      client_data: a dataset of integer hashes.

    Returns:
      The HyperLogLog sketch of size HLL_SKETCH_SIZE.
    """
    initial_state = tf.zeros(HLL_SKETCH_SIZE, dtype=tf.int64)

    def reduce_func(state, hash_value):
      j = hash_value // 2**HLL_BIT_INDEX_TAIL
      w = tf.bitwise.bitwise_and(hash_value, 2**HLL_BIT_INDEX_TAIL - 1)
      rho = HLL_BIT_INDEX_TAIL - _log2(w)
      sketch = tf.one_hot(tf.cast(j, tf.int32), HLL_SKETCH_SIZE, dtype=tf.int64)
      return tf.maximum(sketch * rho, state)

    return client_data.reduce(initial_state, reduce_func)

  return _client_hyperloglog


def build_federated_secure_max_computation() -> computation_base.Computation:
  """Builds a `tff.Computation` for computing max in a secure fashion.

    Specifically, the returned computation consumes sketches at @CLIENTS and
    returns the element-wise max of the inpt sketches @SERVER.

    Note: this returned computation assumes the values to be maxed are
    in the range [0, HLL_BIT_INDEX_TAIL+1].  This function works by onehot
    encoding the values which allows us to sum the values securely and then
    infer the max based on the non-zero entries of the sum.  This approach is
    feasible because the inputs are small non-negative integers.
    Generalizations of this function would require communication proportional to
    (upper_bound - lower_bound).

  Returns:
    A `tff.Computation` for computing max of client vectors.
  """

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          computation_types.TensorType(np.int64, shape=[HLL_SKETCH_SIZE]),
          placements.CLIENTS,
      )
  )
  def federated_secure_max(sketch):
    """Computes the max of client sketches in a secure federated manner.

    Note: this function assumes the values to be maxed are in the range
    [0, HLL_BIT_INDEX_TAIL+1].  This function works by onehot encoding the
    values which allows us to sum the values securely and then infer the max
    based on the non-zero entries of the sum.  This approach is feasible because
    the inputs are small non-negative integers.  Generalizations of this
    function would require communication proportional to
    (upper_bound - lower_bound).

    Args:
      sketch: sketches at clients

    Returns:
      the element-wise max of the input sketches at the server.
    """

    @tensorflow_computation.tf_computation
    @tf.function
    def _onehot_sketch(sketch: tf.Tensor) -> tf.Tensor:
      return tf.cast(
          tf.one_hot(
              tf.cast(sketch, tf.int32), HLL_BIT_INDEX_TAIL + 1, dtype=tf.int64
          ),
          tf.int32,
      )

    @tensorflow_computation.tf_computation
    @tf.function
    def maxes_from_onehots(x):
      mult = tf.reshape(
          tf.range(HLL_BIT_INDEX_TAIL + 1, dtype=tf.int64),
          (-1, HLL_BIT_INDEX_TAIL + 1),
      )
      return tf.reduce_max(tf.cast(x > 0, tf.int64) * mult, axis=1)

    onehot_sketches = intrinsics.federated_map(_onehot_sketch, sketch)
    server_sketch = intrinsics.federated_secure_sum(onehot_sketches, 1)
    maxes = intrinsics.federated_map(maxes_from_onehots, server_sketch)
    return maxes

  return federated_secure_max


def create_federated_hyperloglog_computation(
    *, use_secagg: bool = False
) -> computation_base.Computation:
  """Creates a `tff.Computation` to estimate the number of distinct strings.

  The returned computation consumes data @CLIENTS and produces an estimate of
  the number of unique words across all clients @SERVER.

  Args:
    use_secagg: Flag to specify if secure aggregation is necessary when
      computing the hyperloglog sketch.

  Returns:
    A `tff.Computation` for running the HyperLogLog algorithm.
  """

  @tensorflow_computation.tf_computation
  @tf.function
  def hash_client_data(client_data: tf.data.Dataset) -> tf.data.Dataset:
    # Ideally we'd like to use to_hash_bucket_strong here for cryptographically
    # secure hashing.  Unfortunately, it doesn't work in the contet of a
    # tf.function because the seed needs to be a list of python ints which are
    # baked into the graph rather than a tensor, however we want to be able to
    # parameterize the seed by an external input so that it can change, which
    # requires using tensors.
    return client_data.map(
        lambda item: tf.strings.to_hash_bucket_fast(item, 2**HLL_SKETCH_SIZE)
    )

  @tensorflow_computation.tf_computation
  @tf.function
  def estimate_count_from_sketch(sketch: tf.Tensor) -> tf.int64:
    """Estimate the number of unique items from a HyperLogLog sketch.

    Args:
      sketch: The HyperLogLog sketch.

    Returns:
      The estimated count for the number of distinct items.
    """
    z = 1 / tf.math.reduce_sum(0.5 ** tf.cast(sketch, tf.float64))
    return tf.cast(HLL_ALPHA * HLL_SKETCH_SIZE**2 * z, tf.int64)

  client_hyperloglog = build_client_hyperloglog_computation()
  federated_secure_max = build_federated_secure_max_computation()

  @federated_computation.federated_computation(
      computation_types.FederatedType(
          computation_types.SequenceType(np.str_), placements.CLIENTS
      )
  )
  def federated_hyperloglog(client_data):
    client_hash = intrinsics.federated_map(hash_client_data, client_data)
    sketches = intrinsics.federated_map(client_hyperloglog, client_hash)
    if use_secagg:
      server_sketch = federated_secure_max(sketches)
    else:
      server_sketch = intrinsics.federated_max(sketches)

    return intrinsics.federated_map(estimate_count_from_sketch, server_sketch)

  return federated_hyperloglog
