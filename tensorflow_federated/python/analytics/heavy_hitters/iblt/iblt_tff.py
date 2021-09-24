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
"""Heavy hitters discovery using IBLT."""

from typing import Callable, Optional, Tuple

import attr
import tensorflow as tf

from tensorflow_federated.python.analytics import data_processing
from tensorflow_federated.python.analytics import histogram_processing
from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_lib

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


@attr.s(eq=False, frozen=True)
class ServerOutput():
  """The container of results.

  Attributes:
    clients: An int32 scalar number of clients that added their data to the
      state.
    heavy_hitters: A 1-d tensor of decoded heavy hitters.
    heavy_hitters_counts: A 1-d tensor of the counts for decoded heavy hitters.
    num_not_decoded: An int32 scalar number of strings that are not decoded.
  """
  clients = attr.ib()
  heavy_hitters = attr.ib()
  heavy_hitters_counts = attr.ib()
  num_not_decoded = attr.ib()


def build_iblt_computation(
    *,
    capacity: int = 1000,
    max_string_length: int = 10,
    repetitions: int = 3,
    seed: int = 0,
    dtype: tf.dtypes.DType = tf.int64,
    max_heavy_hitters: Optional[int] = None,
    max_words_per_user: Optional[int] = None,
    k_anonymity: int = 1,
    secure_sum_bitwidth: Optional[int] = None,
    batch_size: int = 1,
    multi_contribution: bool = True,
    decode_iblt_fn: Optional[Callable[..., Tuple[tf.Tensor, tf.Tensor,
                                                 tf.Tensor]]] = None,
) -> computation_base.Computation:
  """Builds the `tff.Computation` for heavy-hitters discovery with IBLT.

  Args:
    capacity: The capacity of the IBLT sketch. Defaults to `1000`.
    max_string_length: The maximum length of a string in the IBLT. Defaults to
      `10`. Must be positive.
    repetitions: The number of repetitions in IBLT data structure (must be >=
      3). Defaults to `3`. Must be at least `3`.
    seed: An integer seed for hash functions. Defaults to `0`.
    dtype: A tensorflow data type which determines the type of the IBLT values.
      Must be `tf.int32` or `tf.int64`. Defaults to `tf.int64`.
    max_heavy_hitters: The maximum number of items to return. If the decoded
      results have more than this number of items, will order decreasingly by
      the estimated counts and return the top max_heavy_hitters items. Default
      max_heavy_hitters == `None`, which means to return all the heavy hitters
      in the result.
    max_words_per_user: The maximum number of words each client is allowed to
      contribute. If not `None`, must be a positive integer. Defaults to `None`,
      which means all the clients contribute all their words.
    k_anonymity: Only return words that appear in at least k clients. Must be a
      positive integer. Defaults to `1`.
    secure_sum_bitwidth: The bitwidth used for secure sum. The default value is
      `None`, which disables secure sum. If not `None`, must be in the range
      `[1,62]`. See `tff.federated_secure_sum_bitwidth`.
    batch_size: The number of elements in each batch of the dataset.  Defaults
      to `1`, means the input dataset is processed by
      `tf.data.Dataset.batch(1)`.  Must be a positive.
    multi_contribution: Whether each client is allowed to contribute multiple
      counts or only a count of one for each unique word. Defaults to `True`.
    decode_iblt_fn: A function to decode key-value pairs from an IBLT sketch.
      Defaults to `None`, in this case `decode_iblt_fn` will be set to
      `iblt.decode_iblt_tf`.

  Returns:
    A `tff.Computation` that performs federated heavy hitter discovery.

  Raises:
    ValueError: if parameters don't meet expectations.
  """
  if max_string_length < 1:
    raise ValueError('max_string_length should be at least 1, got '
                     f'{max_string_length}')
  if repetitions < 3:
    raise ValueError(f'repetitions should be at least 3, got {repetitions}')
  if dtype not in [tf.int32, tf.int64]:
    raise ValueError(
        f'`dtype` must be one of tf.int32 or tf.int64, got {dtype}')
  if max_heavy_hitters is not None and max_heavy_hitters < 1:
    raise ValueError(
        'max_heavy_hitters should be at least 1 when it is not None, '
        f'got {max_heavy_hitters}')
  if max_words_per_user is not None and max_words_per_user < 1:
    raise ValueError(
        'max_words_per_user should be at least 1 when it is not None, '
        f'got {max_words_per_user}')
  if k_anonymity < 1:
    raise ValueError(f'k_anonymity must be at least 1, got {k_anonymity}')
  if secure_sum_bitwidth is not None:
    if not isinstance(secure_sum_bitwidth, int) or (secure_sum_bitwidth <= 0 or
                                                    secure_sum_bitwidth > 62):
      raise ValueError(
          'If set, secure_sum_bitwidth requires an integer in the range [1,62],'
          f' got {secure_sum_bitwidth}')
  if batch_size < 1:
    raise ValueError(f'batch_size must be at least 1, got {batch_size}')

  if decode_iblt_fn is None:
    decode_iblt_fn = iblt_lib.decode_iblt_tf

  dataset_type = computation_types.SequenceType(
      computation_types.TensorType(shape=[None], dtype=tf.string))

  @computations.tf_computation(dataset_type)
  @tf.function
  def compute_sketch(dataset):
    """The TF computation to compute the frequency sketches."""
    encoder = iblt_lib.IbltEncoder(
        capacity=capacity,
        string_max_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        dtype=dtype)
    if max_words_per_user is not None:
      if multi_contribution:
        k_words = data_processing.get_capped_elements(
            dataset,
            max_words_per_user,
            batch_size=batch_size,
            max_string_length=max_string_length)
      else:
        # `tff.analytics.data_processing.get_top_elements` returns the top
        # `max_words_per_user` words in client's local histogram. Each element
        # appears at most once in the list.
        k_words = data_processing.get_top_elements(
            dataset, max_words_per_user, max_string_length=max_string_length)
    else:
      if multi_contribution:
        k_words = data_processing.get_all_elements(
            dataset, max_string_length=max_string_length)
      else:
        k_words = data_processing.get_unique_elements(
            dataset, max_string_length=max_string_length)
    return encoder.compute_iblt(k_words)

  @computations.tf_computation(dataset_type)
  def compute_unique_sketch(dataset):
    """The TF computation to compute the unique frequency sketches."""
    encoder = iblt_lib.IbltEncoder(
        capacity=capacity,
        string_max_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        dtype=dtype)
    k_words = data_processing.get_unique_elements(
        dataset, max_string_length=max_string_length)
    return encoder.compute_iblt(k_words)

  def get_heavy_hitters_mask(heavy_hitters, unique_heavy_hitters):
    """A boolean mask for items in both heavy_hitters and unique_heavy_hitters."""

    def item_in_unique_heavy_hitters(element):
      mask = tf.equal(unique_heavy_hitters, element)
      return tf.reduce_any(mask)

    heavy_hitters_mask = tf.map_fn(
        fn=item_in_unique_heavy_hitters,
        elems=heavy_hitters,
        fn_output_signature=tf.bool)
    return heavy_hitters_mask

  num_chunks = chunkers.UTF8Chunker(
      max_string_length,
      max_chunk_value=iblt_lib.DEFAULT_FIELD_SIZE).get_num_chunks()
  num_chunks_for_hash_check = 1
  num_chunks_for_value = 1
  sketch_shape = (repetitions, None,
                  num_chunks + num_chunks_for_hash_check + num_chunks_for_value)

  @computations.tf_computation(
      computation_types.TensorType(dtype=dtype, shape=sketch_shape),
      computation_types.TensorType(dtype=dtype, shape=sketch_shape))
  @tf.function
  def decode_heavy_hitters(sketch, unique_sketch):
    """The TF computation to decode the heavy hitters."""
    iblt_decoded = decode_iblt_fn(
        iblt=sketch,
        capacity=capacity,
        string_max_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        dtype=dtype)
    unique_decoded = decode_iblt_fn(
        iblt=unique_sketch,
        capacity=capacity,
        string_max_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        dtype=dtype)
    heavy_hitters, heavy_hitters_counts, num_not_decoded = iblt_decoded
    unique_heavy_hitters, unique_heavy_hitters_counts, _ = unique_decoded
    if k_anonymity > 1:
      unique_heavy_hitters, unique_heavy_hitters_counts = histogram_processing.threshold_histogram(
          unique_heavy_hitters, unique_heavy_hitters_counts, k_anonymity)
      heavy_hitters_mask = get_heavy_hitters_mask(heavy_hitters,
                                                  unique_heavy_hitters)
      heavy_hitters = tf.boolean_mask(heavy_hitters, heavy_hitters_mask)
      heavy_hitters_counts = tf.boolean_mask(heavy_hitters_counts,
                                             heavy_hitters_mask)
    if max_heavy_hitters is not None and tf.shape(
        heavy_hitters)[0] > max_heavy_hitters:
      _, top_indices = tf.math.top_k(heavy_hitters_counts, max_heavy_hitters)
      heavy_hitters = tf.gather(heavy_hitters, top_indices)
      heavy_hitters_counts = tf.gather(heavy_hitters_counts, top_indices)
    return heavy_hitters, heavy_hitters_counts, num_not_decoded

  def secure_sum(x):
    return intrinsics.federated_secure_sum_bitwidth(x, secure_sum_bitwidth)

  @computations.federated_computation(
      computation_types.at_clients(dataset_type))
  def one_round_computation(examples):
    """The TFF computation to compute the aggregated IBLT sketch."""
    if secure_sum_bitwidth is not None:
      sum_fn = secure_sum
    else:
      sum_fn = intrinsics.federated_sum
    clients = sum_fn(intrinsics.federated_value(1, placements.CLIENTS))
    sketch = sum_fn(intrinsics.federated_map(compute_sketch, examples))
    unique_sketch = sum_fn(
        intrinsics.federated_map(compute_unique_sketch, examples))
    heavy_hitters, heavy_hitters_counts, num_not_decoded = intrinsics.federated_map(
        decode_heavy_hitters, (sketch, unique_sketch))
    server_output = intrinsics.federated_zip(
        ServerOutput(
            clients=clients,
            heavy_hitters=heavy_hitters,
            heavy_hitters_counts=heavy_hitters_counts,
            num_not_decoded=num_not_decoded))
    return server_output

  return one_round_computation
