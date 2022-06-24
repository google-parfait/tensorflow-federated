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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics import data_processing
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_lib
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_tensor
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


@attr.s(eq=False, frozen=True)
class ServerOutput():
  """The container of results.

  Attributes:
    clients: An int32 scalar number of clients that added their data to the
      state.
    heavy_hitters: A 1-d tensor of decoded heavy hitters.
    heavy_hitters_unique_counts: A 1-d tensor of the number of unique users
      contributed to the decoded heavy hitters.
    heavy_hitters_counts: A 1-d tensor of the counts for decoded heavy hitters.
    num_not_decoded: An int32 scalar number of strings that are not decoded.
    round_timestamp: An int64 scalar of the timestamp of the beginning of the
      round. The value is in seconds since the epoch, in UTC.
  """
  clients = attr.ib()
  heavy_hitters = attr.ib()
  heavy_hitters_unique_counts = attr.ib()
  heavy_hitters_counts = attr.ib()
  num_not_decoded = attr.ib()
  round_timestamp = attr.ib()


def build_iblt_computation(
    *,
    capacity: int = 1000,
    max_string_length: int = 10,
    repetitions: int = 3,
    seed: int = 0,
    max_heavy_hitters: Optional[int] = None,
    max_words_per_user: Optional[int] = None,
    k_anonymity: int = 1,
    secure_sum_bitwidth: Optional[int] = None,
    batch_size: int = 1,
    multi_contribution: bool = True,
    string_postprocessor: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    decode_iblt_fn: Optional[Callable[..., Tuple[tf.Tensor, tf.Tensor,
                                                 tf.Tensor, tf.Tensor]]] = None,
) -> computation_base.Computation:
  """Builds the `tff.Computation` for heavy-hitters discovery with IBLT.

  Args:
    capacity: The capacity of the IBLT sketch. Defaults to `1000`.
    max_string_length: The maximum length of a string in the IBLT. Defaults to
      `10`. Must be positive.
    repetitions: The number of repetitions in IBLT data structure (must be >=
      3). Defaults to `3`. Must be at least `3`.
    seed: An integer seed for hash functions. Defaults to `0`.
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
    secure_sum_bitwidth: The bitwidth used for federated secure sum. The default
      value is `None`, which disables secure sum. If not `None`, must be in the
      range `[1,62]`. Note that when this parameter is not `None`, the IBLT
      sketches are summed via `federated_secure_modular_sum` with modulus equal
      to IBLT's default field size, and other values (client count, string count
      tensor) are aggregated via `federated_secure_sum` with
      `max_input=2**secure_sum_bitwidth - 1`.
    batch_size: The number of elements in each batch of the dataset.  Defaults
      to `1`, means the input dataset is processed by
      `tf.data.Dataset.batch(1)`.  Must be a positive.
    multi_contribution: Whether each client is allowed to contribute multiple
      counts or only a count of one for each unique word. Defaults to `True`.
    string_postprocessor: A callable function that is run after strings are
      decoded from the IBLT in order to postprocess them. It should accept a
      single string tensor and output a single string tensor of the same shape.
      If `None`, no postprocessing is done.
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
    decode_iblt_fn = iblt_tensor.decode_iblt_tensor_tf

  dataset_type = computation_types.SequenceType(
      computation_types.TensorType(shape=[None], dtype=tf.string))

  @tensorflow_computation.tf_computation(dataset_type)
  @tf.function
  def compute_sketch(dataset):
    """The TF computation to compute the frequency sketches."""
    encoder = iblt_tensor.IbltTensorEncoder(
        capacity=capacity,
        string_max_length=max_string_length,
        repetitions=repetitions,
        value_shape=(1,),
        seed=seed)
    if max_words_per_user is not None:
      if multi_contribution:
        k_words, counts = data_processing.get_capped_elements_with_counts(
            dataset,
            max_words_per_user,
            batch_size=batch_size,
            max_string_length=max_string_length)
      else:
        # `tff.analytics.data_processing.get_top_elements` returns the top
        # `max_words_per_user` words in client's local histogram. Each element
        # appears at most once in the list.
        k_words, counts = data_processing.get_top_elements_with_counts(
            dataset, max_words_per_user, max_string_length=max_string_length)
        counts = tf.ones_like(counts)
    else:
      k_words, counts = data_processing.get_unique_elements_with_counts(
          dataset, max_string_length=max_string_length)
      if not multi_contribution:
        counts = tf.ones_like(counts)
    counts = tf.reshape(counts, shape=[-1, 1])
    return encoder.compute_iblt(k_words, counts)

  @tensorflow_computation.tf_computation(compute_sketch.type_signature.result)
  @tf.function
  def decode_heavy_hitters(sketch, count_tensor):
    """The TF computation to decode the heavy hitters."""
    iblt_decoded = decode_iblt_fn(
        iblt=sketch,
        iblt_values=count_tensor,
        value_shape=(1,),
        capacity=capacity,
        string_max_length=max_string_length,
        repetitions=repetitions,
        seed=seed)

    (heavy_hitters, heavy_hitters_unique_counts, heavy_hitters_counts,
     num_not_decoded) = iblt_decoded

    heavy_hitters_counts = tf.squeeze(heavy_hitters_counts, axis=1)

    if k_anonymity > 1:
      heavy_hitters_mask = tf.math.greater_equal(heavy_hitters_unique_counts,
                                                 k_anonymity)
      heavy_hitters = tf.boolean_mask(heavy_hitters, heavy_hitters_mask)
      heavy_hitters_unique_counts = tf.boolean_mask(heavy_hitters_unique_counts,
                                                    heavy_hitters_mask)
      heavy_hitters_counts = tf.boolean_mask(heavy_hitters_counts,
                                             heavy_hitters_mask)

    if max_heavy_hitters is not None and tf.shape(
        heavy_hitters)[0] > max_heavy_hitters:
      _, top_indices = tf.math.top_k(heavy_hitters_counts, max_heavy_hitters)
      heavy_hitters = tf.gather(heavy_hitters, top_indices)
      heavy_hitters_unique_counts = tf.gather(heavy_hitters_unique_counts,
                                              top_indices)
      heavy_hitters_counts = tf.gather(heavy_hitters_counts, top_indices)

    if string_postprocessor is not None:
      heavy_hitters = string_postprocessor(heavy_hitters)

    return (heavy_hitters, heavy_hitters_unique_counts, heavy_hitters_counts,
            num_not_decoded)

  def secure_sum(x):
    return intrinsics.federated_secure_sum(
        x, max_input=2**secure_sum_bitwidth - 1)

  def secure_modular_sum(x):
    return intrinsics.federated_secure_modular_sum(
        x, modulus=np.int64(iblt_lib.DEFAULT_FIELD_SIZE))

  @federated_computation.federated_computation(
      computation_types.at_clients(dataset_type))
  def one_round_computation(examples):
    """The TFF computation to compute the aggregated IBLT sketch."""
    if secure_sum_bitwidth is not None:
      # Use federated secure modular sum for IBLT sketches, because IBLT
      # sketches are decoded by taking modulo over the field size.
      sketch_sum_fn = secure_modular_sum
      count_sum_fn = secure_sum
    else:
      sketch_sum_fn = intrinsics.federated_sum
      count_sum_fn = intrinsics.federated_sum
    round_timestamp = intrinsics.federated_eval(
        tensorflow_computation.tf_computation(
            lambda: tf.cast(tf.timestamp(), tf.int64)), placements.SERVER)
    clients = count_sum_fn(intrinsics.federated_value(1, placements.CLIENTS))
    sketch, count_tensor = intrinsics.federated_map(compute_sketch, examples)
    sketch = sketch_sum_fn(sketch)
    count_tensor = count_sum_fn(count_tensor)

    (heavy_hitters, heavy_hitters_unique_counts, heavy_hitters_counts,
     num_not_decoded) = intrinsics.federated_map(decode_heavy_hitters,
                                                 (sketch, count_tensor))
    server_output = intrinsics.federated_zip(
        ServerOutput(
            clients=clients,
            heavy_hitters=heavy_hitters,
            heavy_hitters_unique_counts=heavy_hitters_unique_counts,
            heavy_hitters_counts=heavy_hitters_counts,
            num_not_decoded=num_not_decoded,
            round_timestamp=round_timestamp))
    return server_output

  return one_round_computation
