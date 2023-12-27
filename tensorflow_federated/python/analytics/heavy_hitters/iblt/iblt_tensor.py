# Copyright 2022, The TensorFlow Federated Authors.
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
"""IBLTTensor encoder and decoder.

The IBLT Tensor implemented is an extension of IBLT implemented in iblt_lib.py.
The major difference is we have the ability to insert a tensor value vt along
with a count value v for each key. To insert a tuple (k,v,vt), we compute hashes
h_i(k) and q_i(k), and update these counters, for i=1,2,3:
count[i,h_i(k)] += v key[i,h_i(k)] += vk check[i,h_i(k)] += vq_i(k)
value[i, h_i(k)] += vt

Using this rule, IBLT Tensor is a "somewhat"-linear data structure. That is, we
support adding 2 IBLTs together but not subtracting. So, if a key has value
(v1, vt1) in IBLT1 and value (v2, vt2) in IBLT2, it has value (v1+v2, vt1+vt2)
in IBLT1+IBLT2 but IBLT1-IBLT2 is not a valid IBLT. We also enforce that every
tuple inserted into IBLT Tensor (k, v, vt), v is positive (strictly greater
than 0).

To make effective use of fixed precision arithmetic our TensorFlow
implementation makes use of arithmetic in a finite prime field of size
field_size, i.e., does arithmetic modulo field_size for counters key, count and
check. For the counter, value, we use the default overflow logic. Only
dtypes that satisfy the constraint (a + b) - b = a when (a + b) overflows are
guaranteed to produce valid results.

The value tensor, vt, can be arbitrary rank `(x, y, ..., z)`. It can also be
empty.

Decoding this data structure returns a list of strings, a list of counts, a list
of `(x, y, ..., z)` shaped tensors, and a scalar denoting the number of
non-decoded strings. If the value tensor is empty, value_shape = (), the output
value tensor is also empty (returned as a tf.constant([])).
"""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_lib

# Convenience Aliases
_CharacterEncoding = chunkers.CharacterEncoding


class IbltTensorDecoder(iblt_lib.IbltDecoder):
  """Decodes the strings and counts stored in an IBLT data structure."""

  def __init__(
      self,
      iblt: tf.Tensor,
      iblt_values: tf.Tensor,
      value_shape: Sequence[int],
      *args,
      **kwargs,
  ):
    """Initializes the IBLT Tensor Decoder.

    Args:
      iblt: Tensor representing the IBLT computed by the IbltEncoder
      iblt_values: Tensor representing the IBLT values computed by the
        IbltEncoder.
      value_shape: Shape of the values tensor.
      *args: See IbltDecoder.
      **kwargs: See IbltDecoder.
    """
    super().__init__(iblt=iblt, *args, **kwargs)
    self.iblt_values = iblt_values
    self.value_shape = value_shape
    self.num_values = np.prod(self.value_shape)

  @tf.function
  def _check_if_queue_is_empty(
      self,
      iblt: tf.Tensor,
      iblt_values: tf.Tensor,
      out_strings: tf.TensorArray,
      out_counts: tf.TensorArray,
      out_tensor_values: tf.TensorArray,
  ) -> bool:
    """Checks if `self.q` is empty."""
    del iblt, iblt_values, out_strings, out_counts, out_tensor_values
    return self.q.size() > 0

  @tf.function
  def _peel_element_from_iblt(
      self,
      iblt: tf.Tensor,
      iblt_values: tf.Tensor,
      out_strings: tf.TensorArray,
      out_counts: tf.TensorArray,
      out_tensor_values: tf.TensorArray,
  ) -> tuple[
      tf.Tensor, tf.Tensor, tf.TensorArray, tf.TensorArray, tf.TensorArray
  ]:
    """Peels an element from IBLT and adds new peelable elements to queue."""
    repetition, index = self.q.dequeue()
    iblt, hash_indices, data_string, count = self._decode_and_remove(
        iblt, repetition, index
    )
    tensor_value = self._decode_value(iblt_values, repetition, index)
    iblt_values = self._remove_value(iblt_values, hash_indices, tensor_value)
    if tf.strings.length(data_string) > 0:
      index = out_counts.size()
      out_counts = out_counts.write(index, count)
      out_strings = out_strings.write(index, data_string)
      out_tensor_values = out_tensor_values.write(index, tensor_value)
      for r in tf.range(self.repetitions, dtype=self._dtype):
        if self._is_peelable(iblt, r, hash_indices[r]):
          self.q.enqueue((r, hash_indices[r]))
    return iblt, iblt_values, out_strings, out_counts, out_tensor_values

  def _decode_value(
      self, iblt_values: tf.Tensor, repetition: int, index: int
  ) -> tf.Tensor:
    """Returns tensor value at `repetition` and `index`."""
    return tf.reshape(iblt_values[repetition][index], shape=self.value_shape)

  def _remove_value(
      self,
      iblt_values: tf.Tensor,
      hash_indices: tf.Tensor,
      tensor_value: tf.Tensor,
  ):
    """Removes `tensor_value` from `iblt_values`."""
    indices, values = [], []

    for repetition in range(self.repetitions):
      index = hash_indices[repetition]

      indices.append(tf.stack([repetition, index], axis=0))
      values.append(iblt_values[repetition][index] - tensor_value)
    iblt_values = tf.tensor_scatter_nd_update(iblt_values, indices, values)
    return iblt_values

  @tf.function
  def get_freq_estimates_tf(
      self,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decodes key-value pairs from an IBLT.

    Returns:
      `(out_strings, out_counts, out_tensor_values, num_not_decoded)` where
      `out_strings` is a `tf.Tensor` containing all the decoded strings,
      `out_counts` is a `tf.Tensor` containing the counts of each string,
      `out_tensor_values` is a `tf.Tensor` (of shape `values_shape`) containing
      a `tf.Tensor` of values for each string, and `num_not_decoded` is a
      `tf.Tensor` with the number of items not decoded in the IBLT.

      If `self.value_shape is` `()`, then out_tensor_values is also empty
      (returned as a `tf.constant([])`).
    """

    if not self.value_shape:
      # If the value shapes are empty, just do regular IBLT decoding and return
      # an empty tensor for `out_tensor_values`.
      strings, counts, num_not_decoded = super().get_freq_estimates_tf()
      return (
          strings,
          counts,
          tf.constant([], dtype=self._dtype),
          num_not_decoded,
      )

    iblt = tf.math.floormod(
        tf.cast(self.iblt, dtype=self._dtype),
        tf.constant(self.field_size, dtype=self._dtype),
    )

    iblt_values = self.iblt_values

    # Initialize queue with all locations that can be decoded:
    for repetition in tf.range(self.repetitions, dtype=self._dtype):
      for index in tf.range(self.table_size, dtype=self._dtype):
        if self._is_peelable(iblt, repetition, index):
          self.q.enqueue((repetition, index))

    out_strings = tf.TensorArray(
        tf.string, size=0, dynamic_size=True, clear_after_read=False
    )
    out_counts = tf.TensorArray(
        self._dtype, size=0, dynamic_size=True, clear_after_read=False
    )
    out_tensor_values = tf.TensorArray(
        iblt_values.dtype, size=0, dynamic_size=True, clear_after_read=False
    )

    # While queue is non-empty, pop and subtract from IBLT, add new peelable
    # locations to queue.
    iblt, _, out_strings, out_counts, out_tensor_values = tf.while_loop(
        self._check_if_queue_is_empty,
        self._peel_element_from_iblt,
        loop_vars=(
            iblt,
            iblt_values,
            out_strings,
            out_counts,
            out_tensor_values,
        ),
        parallel_iterations=1,
    )

    # Count of entries that could not be decoded:
    num_not_decoded = tf.reduce_sum(iblt[:, :, self.count]) / self.repetitions
    num_not_decoded = tf.cast(num_not_decoded, dtype=self._dtype)

    return (
        out_strings.stack(),
        out_counts.stack(),
        out_tensor_values.stack(),
        num_not_decoded,
    )

  def get_freq_estimates(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
      self,
  ) -> tuple[dict[Optional[str], int], dict[Optional[str], np.ndarray]]:
    """Decodes key-value pairs from an IBLT.

    Note that this method only works for UTF-8 strings, and when running TF in
    Eager mode.

    Returns:
      A dictionary containing a decoded key with its frequency and
      tensor_values.
    """
    if not tf.compat.v1.executing_eagerly():
      raise NotImplementedError("This method only works with Eager execution.")

    out_strings, out_counts, out_tensor_values, num_not_decoded = (
        self.get_freq_estimates_tf()
    )

    output_strings = [
        string.decode("utf-8", "ignore")
        for string in out_strings.numpy().tolist()
    ]
    output_counts = out_counts.numpy().tolist()
    output_tensor_values = out_tensor_values.numpy().tolist()

    string_counts = dict(zip(output_strings, output_counts))
    string_tensor_values = dict(zip(output_strings, output_tensor_values))

    num_not_decoded = num_not_decoded.numpy()
    if num_not_decoded:
      string_counts[None] = num_not_decoded
    return string_counts, string_tensor_values


class IbltTensorEncoder(iblt_lib.IbltEncoder):
  """Encodes the strings into an IBLT data structure."""

  def __init__(self, value_shape: Sequence[int], *args, **kwargs):
    """Initializes internal IBLT parameters.

    Args:
      value_shape: Shape of the values.
      *args: See IbltEncoder.
      **kwargs: See IbltEncoder.
    """
    super().__init__(*args, **kwargs)
    self.value_shape = value_shape or ()
    self.num_values = int(np.prod(self.value_shape))
    self.iblt_values_shape = (
        self.repetitions,
        self.table_size,
    ) + self.value_shape

    # TODO: b/199440652 - remove when compute_values is implemented with scatter
    if len(self.value_shape) > 1:
      # In `compute_values` we need to tile the flattened value tensors based on
      # the number of repetitions. The tiling is _always_ [1, reps, 1 ... 1]
      # with (rank of value_shape) + 1 elements.
      # We prepare this shape in advance to make it clear it is static.
      self._tile_shape = [1] + [1 for _ in self.value_shape]
      self._tile_shape[1] = self.repetitions
    else:
      self._tile_shape = [1, self.repetitions]

  def _compute_values(
      self,
      sparse_indices: tf.Tensor,
      input_values: tf.Tensor,
      input_length: int,
  ) -> tf.SparseTensor:
    """Returns SparseTensor with tensor value for each (string, repetition)."""

    # TODO: b/199440652 - replace the sparse tensor construction with
    # `scatter_nd` when it is available.

    indices = tf.reshape(sparse_indices, [-1, 3])
    repeated_indices = tf.repeat(indices, self.num_values, axis=0)

    tensor_indices = tf.tile(
        tf.range(self.num_values, dtype=tf.int64),
        [input_length * self.repetitions],
    )
    tensor_indices_reshaped = tf.reshape(tensor_indices, shape=[-1, 1])
    fused_indices = tf.concat(
        [repeated_indices, tensor_indices_reshaped], axis=-1
    )

    repeated_values = tf.tile(input_values, self._tile_shape)
    flattened_values = tf.reshape(repeated_values, [-1])

    value = tf.SparseTensor(
        indices=fused_indices,
        values=flattened_values,
        dense_shape=(input_length,)
        + (self.repetitions, self.table_size, self.num_values),
    )

    return value

  @tf.function
  def compute_iblt(
      self, input_strings: tf.Tensor, input_values: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Returns Tensor containing the values of the IBLT data structure.

    Args:
      input_strings: A 1D tensor of strings.
      input_values: A tensor of shape `(num_input_strings, value_shape)`
        containing values for each string.

    Returns:
      A tuple of tensors where the first one is of shape
      `[repetitions, table_size, num_chunks+2]` whose value at index `(r, h, c)`
      corresponds to chunk `c` of the keys if `c < num_chunks`, to the counts if
      `c == num_chunks`, and to the checks if `c == num_chunks + 1`. The second
      one is of shape `[repetitions, table_size, product(value_shape)]` and
      contains the tensor values at each key.
    """
    tf.debugging.assert_rank(input_strings, 1)
    tf.debugging.assert_type(input_strings, tf.string)

    tf.debugging.assert_equal(
        tf.shape(input_values),
        (tf.shape(input_strings)[0],) + self.value_shape
        if self.value_shape
        else tf.constant([], dtype=tf.int32),
    )

    chunks, trimmed_input_strings = self.compute_chunks(input_strings)
    if self.drop_strings_above_max_length:
      indices_to_keep = tf.equal(trimmed_input_strings, input_strings)
      trimmed_input_strings = trimmed_input_strings[indices_to_keep]
      chunks = chunks[indices_to_keep]
      input_values = input_values[indices_to_keep]

    hash_check = self._compute_hash_check(trimmed_input_strings)

    sparse_indices = self.hyperedge_hasher.get_hash_indices_tf(
        trimmed_input_strings
    )

    input_length = tf.size(trimmed_input_strings)
    counts = self._compute_counts(sparse_indices, input_length)
    checks = self._compute_checks(sparse_indices, hash_check, input_length)
    keys = self._compute_keys(sparse_indices, chunks, input_length)
    sparse_iblt = tf.sparse.add(keys, counts)
    sparse_iblt = tf.sparse.add(sparse_iblt, checks)
    iblt = tf.sparse.reduce_sum(sparse_iblt, 0)
    iblt = tf.cast(iblt, self._dtype)
    iblt = tf.math.floormod(iblt, self.field_size)
    # Force the result shape so that it can be staticly checked and analyzed.
    # Otherwise the shape is returned as `[None]`.
    iblt = tf.reshape(iblt, self.iblt_shape)

    if not self.value_shape:
      # If the value shapes are empty, we can short-circuit the iblt values
      # tensor construction and just return an empty tensor.
      iblt_values = tf.constant([], dtype=self._dtype)
    else:
      sparse_values = self._compute_values(
          sparse_indices, input_values, input_length
      )
      iblt_values = tf.sparse.reduce_sum(sparse_values, 0)
      iblt_values = tf.reshape(iblt_values, self.iblt_values_shape)
    return iblt, iblt_values


def decode_iblt_tensor_tf(
    iblt: tf.Tensor,
    iblt_values: tf.Tensor,
    capacity: int,
    string_max_bytes: int,
    value_shape: Sequence[int],
    *,
    encoding: _CharacterEncoding = _CharacterEncoding.UTF8,
    seed: int = 0,
    repetitions: int = iblt_lib.DEFAULT_REPETITIONS,
    hash_family: Optional[str] = None,
    hash_family_params: Optional[dict[str, Union[int, float]]] = None,
    field_size: int = iblt_lib.DEFAULT_FIELD_SIZE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Decode a IBLT sketch.

  This function wraps `IbltDecoder` to decode `iblt` and returns frequencies
  of decoded strings.

  Args:
    iblt: Tensor representing the IBLT computed by the IbltEncoder.
    iblt_values: Tensor representing the IBLT values computed by the
      IbltEncoder.
    capacity: Number of distinct strings that we expect to be inserted.
    string_max_bytes: Maximum length of a string in bytes that can be inserted.
    value_shape: Shape of the values tensor.
    encoding: The character encoding of the string data to decode. For
      non-character binary data or strings with unknown encoding, specify
      `CharacterEncoding.UNKNOWN`.
    seed: Integer seed for hash functions. Defaults to 0.
    repetitions: Number of repetitions in IBLT data structure (must be >= 3).
      Defaults to 3.
    hash_family: A `str` specifying the hash family to use to construct IBLT.
      Options include coupled or random, default is chosen based on capacity.
    hash_family_params: An optional `dict` of parameters that the hash family
      hasher expects. Defaults are chosen based on capacity.
    field_size: The field size for all values in IBLT. Defaults to 2**31 - 1.

  Returns:
    `(out_strings, out_counts, out_tensor_counts num_not_decoded)` where
    `out_strings` is a `tf.Tensor` containing all the decoded strings,
    `out_counts` is a `tf.Tensor` containing the counts of each string,
    `out_tensor_counts` is a `tf.Tensor` containing tensor counts for each
    decoded string and `num_not_decoded` is a `tf.Tensor` with the number of
    items not decoded in the IBLT.
  """
  iblt_decoder = IbltTensorDecoder(
      iblt=iblt,
      iblt_values=iblt_values,
      capacity=capacity,
      string_max_bytes=string_max_bytes,
      encoding=encoding,
      seed=seed,
      value_shape=value_shape,
      repetitions=repetitions,
      hash_family=hash_family,
      hash_family_params=hash_family_params,
      field_size=field_size,
  )
  return iblt_decoder.get_freq_estimates_tf()
