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

import tensorflow as tf

from tensorflow_federated.python.aggregators import primitives
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types

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


def _log2(u: tf.Tensor) -> tf.Tensor:
  """Integer log_2."""
  return tf.cast(tf.experimental.numpy.log2(tf.cast(u, tf.float64)), tf.int64)


@tensorflow_computation.tf_computation(
    computation_types.SequenceType(tf.string)
)
@tf.function
def _hash_client_data(client_data: tf.data.Dataset) -> tf.data.Dataset:
  # Ideally we'd like to use to_hash_bucket_strong here for cryptographically
  # secure hashing.  Unfortunately, it doesn't work in because the seed needs
  # to be a list of python ints which are baked into the graph rather than a
  # tensor, however we want to be able to parameterize the seed by an external
  # input so that it can change, which requires using tensors.
  return client_data.map(
      lambda item: tf.strings.to_hash_bucket_fast(item, 2**HLL_SKETCH_SIZE)
  )


@tensorflow_computation.tf_computation(computation_types.SequenceType(tf.int64))
@tf.function
def client_hyperloglog(client_data: tf.data.Dataset) -> tf.Tensor:
  """Computes a (one-hot encoding of a) HyperLogLog sketch on a client dataset.

  Args:
    client_data: a dataset of integer hashes.

  Returns:
    A one hot encoding of the HyperLogLog sketch of size 64.
  """

  initial_state = tf.zeros(HLL_SKETCH_SIZE, dtype=tf.int64)

  def reduce_func(state, hash_value):
    j = tf.bitwise.right_shift(hash_value, HLL_BIT_INDEX_TAIL)
    w = tf.bitwise.bitwise_and(hash_value, 2**HLL_BIT_INDEX_TAIL - 1)
    rho = HLL_BIT_INDEX_TAIL - _log2(w)
    sketch = tf.one_hot(j, HLL_SKETCH_SIZE, dtype=tf.int64) * rho
    return tf.maximum(sketch, state)

  return client_data.reduce(initial_state, reduce_func)


@federated_computation.federated_computation(
    computation_types.at_clients(
        computation_types.TensorType(tf.int64, shape=[HLL_SKETCH_SIZE])
    )
)
def federated_secure_max(sketch):
  """Computes the max of client sketches in a secure federated manner.

  Note: this function assumes the values to be maxed are in the range
  [0, HLL_BIT_INDEX_TAIL+1].  This function works by onehot encoding the values
  which allows us to sum the values securely and then infer the max based on the
  non-zero entries of the sum.  This approach is feasible because the inputs are
  small non-negative integers.  Generalizations of this function would require
  communication proportional to (upper_bound - lower_bound).

  Args:
    sketch: sketches at clients

  Returns:
    the element-wise max of the input sketches at the server.
  """

  @tensorflow_computation.tf_computation
  @tf.function
  def _onehot_sketch(sketch: tf.Tensor) -> tf.Tensor:
    return tf.one_hot(sketch, HLL_BIT_INDEX_TAIL + 1, dtype=tf.int32)

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


@tensorflow_computation.tf_computation(
    computation_types.TensorType(tf.int64, shape=[HLL_SKETCH_SIZE])
)
@tf.function
def _estimate_count_from_sketch(sketch: tf.Tensor) -> tf.int64:
  """Estimate the number of unique items from a HyperLogLog sketch.

  Args:
    sketch: The HyperLogLog sketch.

  Returns:
    The estimated count for the number of distinct items.
  """
  z = 1 / tf.math.reduce_sum(0.5 ** tf.cast(sketch, tf.float64))
  return tf.cast(HLL_ALPHA * HLL_SKETCH_SIZE**2 * z, tf.int64)


def create_federated_hyperloglog_computation(
    *, secagg: bool = False
) -> computation_base.Computation:
  """Creates a federated_computation to estimate the number of distinct strings.

  Args:
    secagg: Flag to specify if secure aggregation is necessary when computing
      the hyperloglog sketch.

  Returns:
    A tff.federated_computation for running the HyperLogLog algorithm.
  """

  @federated_computation.federated_computation(
      computation_types.at_clients(computation_types.SequenceType(tf.string))
  )
  def federated_hyperloglog(client_data):
    client_hash = intrinsics.federated_map(_hash_client_data, client_data)
    sketches = intrinsics.federated_map(client_hyperloglog, client_hash)
    if secagg:
      server_sketch = federated_secure_max(sketches)
    else:
      server_sketch = primitives.federated_max(sketches)

    return intrinsics.federated_map(_estimate_count_from_sketch, server_sketch)

  return federated_hyperloglog
