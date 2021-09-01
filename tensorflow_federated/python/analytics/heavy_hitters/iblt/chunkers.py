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
"""Chunkers that implement different serialization functions to use with IBLTs.

A chunker encodes and decodes the input strings into integer tensors to be used
with IBLTs.
"""

import math
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf


class UTF8Chunker:
  """Encodes and decodes strings into integer tensors using UTF-8 encoding."""

  def __init__(self,
               string_max_length: int,
               *,
               max_chunk_value: Optional[int] = None,
               dtype: tf.dtypes.DType = tf.int64):
    """Initializes the chunker.

    Args:
      string_max_length: Maximum length of the string to encode. Note that this
        is measured in bytes and some unicode characters may take more than 1
        byte. In the case that `string_max_length` does not divide
        `self._dtype_size_bytes` (calculated below), it is rounded up to the
        smallest integer that divides it.
      max_chunk_value: Maximum value in each chunk. Defaults to the maximum
        possible value in dtype.
      dtype: `tf.dtypes.DType` indicating the data type of the output. Must be
        either `tf.int32` or `tf.int64`. Defaults to `tf.int64`.

    Raises:
      ValueError: If arguments do not meet expectations.
    """
    if string_max_length < 1:
      raise ValueError('string_max_length must be at least 1.')

    if dtype not in [tf.int32, tf.int64]:
      raise ValueError('If set, dtype must be tf.int32 or tf.int64.')

    self._dtype = dtype
    self._np_dtype = dtype.as_numpy_dtype
    self._utf8_size_bits = 8
    self._max_chunk_value = max_chunk_value
    if self._max_chunk_value is not None:
      if max_chunk_value < 1:
        raise ValueError('If set, max_chunk_value must be at least 1.')
      if self._max_chunk_value > self._dtype.max:
        raise ValueError('max_chunk_value {} cannot fit in dtype {}'.format(
            max_chunk_value, dtype))
      self._dtype_size_bytes = math.floor(
          math.log2(self._max_chunk_value) / self._utf8_size_bits)
    else:
      self._dtype_size_bytes = self._dtype.size

    self._num_chunks = math.ceil(
        float(string_max_length) / self._dtype_size_bytes)
    self._max_length = self._num_chunks * self._dtype_size_bytes
    self._bit_lengths = [
        self._utf8_size_bits * i for i in range(self._dtype_size_bytes)
    ]
    # Each string is encoded into a fixed length of `self._max_length` bytes. If
    # a string takes less bytes, `\x00` are padded in the end. Replace `\x00`
    # with an empty string `''` here to removed the padded bytes in decoding.
    self._int_to_byte_map = [b''] + [
        i.tobytes() for i in np.arange(1, 256, dtype=np.uint8)
    ]

  def get_num_chunks(self) -> int:
    return self._num_chunks

  def encode_tensorflow(
      self, input_strings: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Encodes `input_strings` to tensors.

    Args:
      input_strings: A 1-D `tf.Tensor` of type `tf.string`. Denote the shape of
        `input_strings` as `(num_strings,)`.

    Returns:
      A Tuple `(encoded_strings, trimmed_input_strings)`
      - encoded_strings: A `tf.Tensor` of shape
        `(num_strings, self._num_chunks)` containing encoded `input_strings`
      - trimmed_input_strings: A `tf.Tensor` of shape `(num_strings,)`
        containing trimmed `input_strings` that the length of each string in it
        is no more than `self._max_length` bytes.
      Note that a utf-8 character might take morethan one byte, so both the
      encoded and trimmed strings could contain characters that are cut in the
      middle. The caller needs to be aware of this when decoding these strings,
      i.g. decode a byte string s by `s.decode('utf-8', 'ignore')` to avoid
      decoding errors.
    """
    string_bytes = tf.io.decode_raw(
        input_strings, out_type=tf.uint8, fixed_length=self._max_length)
    string_bytes_reshaped = tf.reshape(string_bytes,
                                       (-1, self._dtype_size_bytes))
    string_bytes_cast = tf.cast(string_bytes_reshaped, self._dtype)
    dtype_multipliers = tf.constant([[2**(bit)] for bit in self._bit_lengths],
                                    dtype=self._dtype)
    encoded_as_dtype = tf.matmul(string_bytes_cast, dtype_multipliers)
    if self._max_chunk_value:
      tf.assert_less(encoded_as_dtype,
                     tf.constant(self._max_chunk_value, dtype=self._dtype))
    encoded_strings = tf.reshape(encoded_as_dtype, (-1, self._num_chunks))

    int_to_char_map = tf.constant(self._int_to_byte_map, dtype=tf.string)
    trimmed_input_strings = tf.nn.embedding_lookup(
        int_to_char_map, tf.cast(string_bytes, dtype=tf.int32))
    trimmed_input_strings = tf.strings.reduce_join(
        trimmed_input_strings, axis=1)

    return encoded_strings, trimmed_input_strings

  @tf.function
  def decode_tensorflow(self, encoded_chunks: tf.Tensor) -> tf.Tensor:
    """Decodes `encoded_chunks` of shape `(n, self._num_chunks)` to `n` strings.

    Args:
      encoded_chunks: A `tf.Tensor` of shape `(num_strings, self._num_chunks)`
        and `self._dtype`.

    Returns:
      A `tf.Tensor` of shape `(num_strings,)` and type `tf.string`.
    """
    encoded_chunks_reshaped = tf.reshape(encoded_chunks, (-1, 1))
    encoded_chunks_tiled = tf.tile(encoded_chunks_reshaped,
                                   [1, self._dtype_size_bytes])
    bit_lengths = tf.constant(self._bit_lengths, dtype=self._dtype)
    encoded_chunks_shifted = tf.bitwise.right_shift(encoded_chunks_tiled,
                                                    bit_lengths)
    encoded_chunks_modulo = encoded_chunks_shifted % 2**(self._utf8_size_bits)
    encoded_chunks_reshaped = tf.reshape(encoded_chunks_modulo,
                                         (-1, self._max_length))
    encoded_chunks_bytes = tf.cast(encoded_chunks_reshaped, dtype=tf.int32)
    int_to_char_map = tf.constant(self._int_to_byte_map, dtype=tf.string)
    decoded_strings = tf.nn.embedding_lookup(int_to_char_map,
                                             encoded_chunks_bytes)
    decoded_strings = tf.strings.reduce_join(decoded_strings, axis=1)

    return decoded_strings

  def decode_python(self, encoded_chunks: np.ndarray) -> np.ndarray:
    """Decodes `encoded_chunks` of shape `(n, self._num_chunks)` to `n` strings.

    Args:
      encoded_chunks: A `np.ndarray` of shape `(num_strings, self._num_chunks)`
        and `self._dtype`.

    Returns:
      A `np.ndarray` of shape `(num_strings,)` and type `np.string`.
    """
    encoded_chunks_reshaped = encoded_chunks.reshape((-1, 1))
    encoded_chunks_tiles = np.tile(encoded_chunks_reshaped,
                                   [1, self._dtype_size_bytes])
    encoded_chunks_bytes_shifted = np.right_shift(encoded_chunks_tiles,
                                                  self._bit_lengths)
    encoded_chunks_bytes = encoded_chunks_bytes_shifted % 2**(
        self._utf8_size_bits)
    int_to_char_fn = lambda x: (dict(enumerate(self._int_to_byte_map)).get(x))

    # Added `otypes=(np.string_,)` as an additional arg to np.vectorize to avoid
    # numpy crashes with empty strings (not able to identify the type).
    decoded_chars = np.vectorize(
        int_to_char_fn, otypes=(np.string_,))(
            encoded_chunks_bytes)
    decoded_chars_reshaped = decoded_chars.reshape(-1, self._max_length)
    decoded_strings = np.apply_along_axis(
        lambda r: r.tobytes(), arr=decoded_chars_reshaped, axis=1)

    return decoded_strings
