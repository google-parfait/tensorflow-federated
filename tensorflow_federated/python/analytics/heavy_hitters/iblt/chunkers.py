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

import abc
import enum
import math
from typing import Optional

from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


class Chunker(abc.ABC):
  """Abstract base class for IBLT chunkers.

  Chunkers implement the methods of this base class to encode and decode strings
  for use in IBLT sketches.
  """

  @property
  @abc.abstractmethod
  def dtype(self) -> tf.dtypes.DType:
    """The dtype used for encoded chunks."""

  @abc.abstractmethod
  def get_num_chunks(self) -> int:
    """The number of chunks that will be used to encode each string."""

  @abc.abstractmethod
  def encode_tensorflow(
      self, input_strings: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor]:
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
    """

  @abc.abstractmethod
  def decode_tensorflow(self, encoded_chunks: tf.Tensor) -> tf.Tensor:
    """Decodes `encoded_chunks` to strings.

    Args:
      encoded_chunks: A `tf.Tensor` of shape `(num_strings,
        self.get_num_chunks())`

    Returns:
      A `tf.Tensor` of shape `(num_strings,)` and type `tf.string`.
    """


class CharacterEncoding(enum.Enum):
  """Supported character encodings for IBLT strings."""

  # Unknown character encoding, or binary non-character data.
  UNKNOWN = 0

  # UTF-8 encoded Unicode string.
  UTF8 = enum.auto()


_DEFAULT_DTYPE = tf.int64


def _pack_into_int(value, input_bitrange, target_bitrange):
  """Pack integers in range [0, 2**`input_bitrange`-1] into integer values.

  Args:
    value: An integer Tensor to be packed.
    input_bitrange: An integer. The number of relevant bits in `value`.
    target_bitrange: An integer. The number of bits to be used in packed
      representation.

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  # TODO: b/161433177 - Provide a general solution without memory overhead.
  # Special cases implemented without extra memory overhead.
  if input_bitrange == 6 and target_bitrange == 28:
    return _pack_into_int_6_28(value)
  if input_bitrange == 7 and target_bitrange == 28:
    return _pack_into_int_7_28(value)
  if input_bitrange == 8 and target_bitrange == 28:
    return _pack_into_int_8_28(value)
  if input_bitrange == 12 and target_bitrange == 28:
    return _pack_into_int_12_28(value)

  # General solution with possible extra memory overhead.
  logging.warning(
      'This code path can temporarily allocate extra memory. If '
      'memory footprint is a problem, consider different bitpacking'
      ' method or turning this functionality off. See b/161433177'
  )
  if input_bitrange > 1:
    value = tf.reshape(value, [-1, 1])
    value = _expand_to_binary_form(value, input_bitrange)
  return _pack_binary_form(value, target_bitrange)


def _unpack_from_int(value, original_bitrange, target_bitrange, shape):
  """Unpack integers into the range of [0, 2**`original_bitrange`-1].

  Args:
    value: An integer Tensor to be unpacked.
    original_bitrange: An integer. The number of bits used in the original
      representation.
    target_bitrange: An integer. The number of bits used in the packed
      representation.
    shape: The shape of the original input.

  Returns:
    An integer Tensor representing the unpacked `value` of the same dtype as
    `value`.
  """
  # TODO: b/161433177 - Provide a general solution without memory overhead.
  # Special cases implemented without extra memory overhead.
  if original_bitrange == 6 and target_bitrange == 28:
    return _unpack_from_int_6_28(value, shape)
  if original_bitrange == 7 and target_bitrange == 28:
    return _unpack_from_int_7_28(value, shape)
  if original_bitrange == 8 and target_bitrange == 28:
    return _unpack_from_int_8_28(value, shape)
  if original_bitrange == 12 and target_bitrange == 28:
    return _unpack_from_int_12_28(value, shape)

  # General solution with extra memory overhead.
  logging.warning(
      'This code path can temporarily allocate extra memory. If '
      'memory footprint is a problem, consider different bitpacking'
      ' method or turning this functionality off. See b/161433177'
  )
  value = _expand_to_binary_form(value, target_bitrange)
  value = tf.slice(value, [0], [tf.reduce_prod(shape) * original_bitrange])
  if original_bitrange > 1:
    return tf.reshape(_pack_binary_form(value, original_bitrange), shape)
  else:
    return tf.reshape(value, shape)


def _pack_binary_form(value, target_bits):
  # Reshape the binary input to have target_bits columns, padding with zeros if
  # necessary to fit the dimension. The bitpacked representation is computed
  # as product with vector [1, 2, 4, ..., 2**target_bits].
  packing_vector = tf.constant(
      [[2**i] for i in range(target_bits)], value.dtype
  )
  extra_zeros = tf.zeros(
      tf.math.mod(-tf.shape(value), target_bits), value.dtype
  )
  reshaped_x = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, target_bits])
  return tf.matmul(reshaped_x, packing_vector)


def _expand_to_binary_form(value, input_bits):
  # This operation is inverse of _pack_binary_form, except padded zeros are not
  # removed.
  expand_vector = tf.constant([2**i for i in range(input_bits)], value.dtype)
  bits = tf.math.mod(tf.math.floordiv(value, expand_vector), 2)
  return tf.reshape(bits, [-1])


def _pack_into_int_6_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(6, 28)`. This method relies on the fact that
  14 values in 6-bit bitrange can be packed into 3 values in 28-bitrange
  (14 = least_common_multiple(6, 28) / 6).

  It reshapes the input into matrix of 14 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**6 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 14), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 14])

  a = (
      val[:, 0]
      + val[:, 1] * 2**6
      + val[:, 2] * 2**12
      + val[:, 3] * 2**18
      + tf.math.mod(val[:, 4], 2**4) * 2**24
  )
  b = (
      tf.math.floordiv(val[:, 4], 2**4)
      + val[:, 5] * 2**2
      + val[:, 6] * 2**8
      + val[:, 7] * 2**14
      + val[:, 8] * 2**20
      + tf.math.mod(val[:, 9], 2**2) * 2**26
  )
  c = (
      tf.math.floordiv(val[:, 9], 2**2)
      + val[:, 10] * 2**4
      + val[:, 11] * 2**10
      + val[:, 12] * 2**16
      + val[:, 13] * 2**22
  )

  packed_val = tf.reshape(tf.stack([a, b, c], 1), [-1, 1])
  if extra_zeros.shape[0] in [5, 6, 7, 8, 9]:
    # We added unnecessary product of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 1, 1])
  if extra_zeros.shape[0] in [10, 11, 12, 13]:
    # We added unnecessary two products of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 2, 1])
  return packed_val


def _pack_into_int_7_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(7, 28)`. This method relies on the fact that
  4 values in 7-bit bitrange can be packed into 1 value in 28-bitrange
  (4 = least_common_multiple(7, 28) / 7).

  It reshapes the input into matrix of 4 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**7 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 4), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 4])

  packed_val = (
      val[:, 0] + val[:, 1] * 2**7 + val[:, 2] * 2**14 + val[:, 3] * 2**21
  )

  return tf.reshape(packed_val, [-1, 1])


def _pack_into_int_8_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(8, 28)`. This method relies on the fact that
  7 values in 8-bit bitrange can be packed into 2 values in 28-bitrange
  (7 = least_common_multiple(8, 28) / 8).

  It reshapes the input into matrix of 7 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**8 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 7), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 7])

  a = (
      val[:, 0]
      + val[:, 1] * 2**8
      + val[:, 2] * 2**16
      + tf.math.mod(val[:, 3], 2**4) * 2**24
  )
  b = (
      tf.math.floordiv(val[:, 3], 2**4)
      + val[:, 4] * 2**4
      + val[:, 5] * 2**12
      + val[:, 6] * 2**20
  )

  packed_val = tf.reshape(tf.stack([a, b], 1), [-1, 1])
  if extra_zeros.shape[0] in [4, 5, 6]:
    # We added unnecessary sum of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 1, 1])
  return packed_val


def _pack_into_int_12_28(value):
  """Implementation of `pack_into_int` for specific bitranges.

  This method corresponts to `(input_bitrange, target_bitrange)` form the
  `pack_into_int` method equal to `(12, 28)`. This method relies on the fact
  that 7 values in 12-bit bitrange can be packed into 3 values in 28-bitrange
  (7 = least_common_multiple(12, 28) / 12).

  It reshapes the input into matrix of 7 columns and performs operations on the
  columns of the matrix, thus vectorizing the operations and avoiding memory
  overhead of an earlier general implementation.

  Args:
    value: An integer Tensor to be packed with values in [0, 2**8 - 1].

  Returns:
    An integer Tensor representing `value` of the same dtype as `value`.
  """
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 7), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 7])

  a = val[:, 0] + val[:, 1] * 2**12 + tf.math.mod(val[:, 2], 2**4) * 2**24
  b = (
      tf.math.floordiv(val[:, 2], 2**4)
      + val[:, 3] * 2**8
      + tf.math.mod(val[:, 4], 2**8) * 2**20
  )
  c = tf.math.floordiv(val[:, 4], 2**8) + val[:, 5] * 2**4 + val[:, 6] * 2**16

  packed_val = tf.reshape(tf.stack([a, b, c], 1), [-1, 1])
  if extra_zeros.shape[0] in [3, 4]:
    # We added unnecessary sum of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 1, 1])
  if extra_zeros.shape[0] in [5, 6]:
    # We added unnecessary two sums of zeros to the representation.
    packed_val = tf.slice(packed_val, [0, 0], [packed_val.shape[0] - 2, 1])
  return packed_val


def _unpack_from_int_6_28(value, shape):
  """Inverse operation of `_pack_into_int_6_28`."""
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 3), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 3])

  a = tf.math.mod(val[:, 0], 2**6)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**6), 2**6)
  c = tf.math.mod(tf.math.floordiv(val[:, 0], 2**12), 2**6)
  d = tf.math.mod(tf.math.floordiv(val[:, 0], 2**18), 2**6)
  e = tf.math.floordiv(val[:, 0], 2**24) + tf.math.mod(val[:, 1], 2**2) * 2**4
  f = tf.math.mod(tf.math.floordiv(val[:, 1], 2**2), 2**6)
  g = tf.math.mod(tf.math.floordiv(val[:, 1], 2**8), 2**6)
  h = tf.math.mod(tf.math.floordiv(val[:, 1], 2**14), 2**6)
  i = tf.math.mod(tf.math.floordiv(val[:, 1], 2**20), 2**6)
  j = tf.math.floordiv(val[:, 1], 2**26) + tf.math.mod(val[:, 2], 2**4) * 2**2
  k = tf.math.mod(tf.math.floordiv(val[:, 2], 2**4), 2**6)
  l = tf.math.mod(tf.math.floordiv(val[:, 2], 2**10), 2**6)
  m = tf.math.mod(tf.math.floordiv(val[:, 2], 2**16), 2**6)
  n = tf.math.mod(tf.math.floordiv(val[:, 2], 2**22), 2**6)

  unpacked_val = tf.reshape(
      tf.stack([a, b, c, d, e, f, g, h, i, j, k, l, m, n], 1),
      [
          -1,
      ],
  )
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def _unpack_from_int_7_28(value, shape):
  """Inverse operation of `_pack_into_int_7_28`."""
  val = tf.reshape(value, [-1, 1])

  a = tf.math.mod(val[:, 0], 2**7)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**7), 2**7)
  c = tf.math.mod(tf.math.floordiv(val[:, 0], 2**14), 2**7)
  d = tf.math.mod(tf.math.floordiv(val[:, 0], 2**21), 2**7)

  unpacked_val = tf.reshape(
      tf.stack([a, b, c, d], 1),
      [
          -1,
      ],
  )
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def _unpack_from_int_8_28(value, shape):
  """Inverse operation of `_pack_into_int_8_28`."""
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 2), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 2])

  a = tf.math.mod(val[:, 0], 2**8)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**8), 2**8)
  c = tf.math.mod(tf.math.floordiv(val[:, 0], 2**16), 2**8)
  d = tf.math.floordiv(val[:, 0], 2**24) + tf.math.mod(val[:, 1], 2**4) * 2**4
  e = tf.math.mod(tf.math.floordiv(val[:, 1], 2**4), 2**8)
  f = tf.math.mod(tf.math.floordiv(val[:, 1], 2**12), 2**8)
  g = tf.math.floordiv(val[:, 1], 2**20)

  unpacked_val = tf.reshape(
      tf.stack([a, b, c, d, e, f, g], 1),
      [
          -1,
      ],
  )
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def _unpack_from_int_12_28(value, shape):
  """Inverse operation of `_pack_into_int_12_28`."""
  value = tf.reshape(value, [-1])
  extra_zeros = tf.zeros(tf.math.mod(-tf.shape(value), 3), value.dtype)
  val = tf.reshape(tf.concat([value, extra_zeros], 0), [-1, 3])

  a = tf.math.mod(val[:, 0], 2**12)
  b = tf.math.mod(tf.math.floordiv(val[:, 0], 2**12), 2**12)
  c = tf.math.floordiv(val[:, 0], 2**24) + tf.math.mod(val[:, 1], 2**8) * 2**4
  d = tf.math.mod(tf.math.floordiv(val[:, 1], 2**8), 2**12)
  e = tf.math.floordiv(val[:, 1], 2**20) + tf.math.mod(val[:, 2], 2**4) * 2**8
  f = tf.math.mod(tf.math.floordiv(val[:, 2], 2**4), 2**12)
  g = tf.math.mod(tf.math.floordiv(val[:, 2], 2**16), 2**12)

  unpacked_val = tf.reshape(
      tf.stack([a, b, c, d, e, f, g], 1),
      [
          -1,
      ],
  )
  unpacked_val = tf.slice(unpacked_val, [0], [tf.reduce_prod(shape)])
  return tf.reshape(unpacked_val, shape)


def create_chunker(
    *,
    string_max_bytes: int,
    encoding: CharacterEncoding = CharacterEncoding.UTF8,
    max_chunk_value: Optional[int] = None,
    dtype: tf.dtypes.DType = _DEFAULT_DTYPE,
) -> Chunker:
  """Creates a `Chunker` for the given specification.

  Note: not all parameters are supported by all Chunkers. See chunker subclass
  documentation for details.

  Args:
    string_max_bytes: Maximum length of the string to encode, in bytes.
    encoding: The character encoding of the string data to encode. For
      non-character binary data or strings with unknown encoding, specify
      `CharacterEncoding.UNKNOWN`. Defaults to `CharacterEncoding.UTF8`.
    max_chunk_value: Maximum value in each chunk. Defaults to the maximum
      possible value in dtype. Only used with `CharacterEncoding.UTF8`.
    dtype: `tf.dtypes.DType` indicating the data type of the output. Must be
      either `tf.int32` or `tf.int64`. Defaults to `tf.int64`.

  Returns:
    A `Chunker` instance.
  """
  if encoding == CharacterEncoding.UTF8:
    return UTF8Chunker(
        string_max_bytes=string_max_bytes,
        max_chunk_value=max_chunk_value,
        dtype=dtype,
    )
  if encoding == CharacterEncoding.UNKNOWN:
    return BinaryChunker(
        string_max_bytes=string_max_bytes,
        max_chunk_value=max_chunk_value,
        dtype=dtype,
    )

  raise ValueError(f'Unsupported encoding: {encoding}')


# The DType used by `BinaryChunker` internally for string lengths.
_BINARY_CHUNKER_STRING_LEN_DTYPE = tf.int32


class BinaryChunker(Chunker):
  """Encodes and decodes binary data strings into integer chunks.

  This chunker makes no assumptions about the type of data or encoding stored
  within the input strings. Thus it is suitable as a generic binary encoder
  for non-character data, or string data with unknown encoding.

  This chunker encodes data into chunks of the specified `tf.dtypes.DType`, with
  encoded values in the range `[0, max_chunk_value]`.

  The chunker encoding uses a prefix header containing a single integer denoting
  the size in bytes of the encoded string. This is necessary to remove the
  zero padding used to ensure all encoded tensors have the same size.
  """

  def __init__(
      self,
      *,
      string_max_bytes: int,
      max_chunk_value: Optional[int] = None,
      dtype: tf.dtypes.DType = _DEFAULT_DTYPE,
  ):
    """Initializes the chunker.

    Args:
      string_max_bytes: Maximum length of the binary string to encode, in bytes.
      max_chunk_value: Maximum encoded value each chunk can hold. Encoded chunk
        values will be in the range `[0, max_chunk_value]`. Defaults to the
        maximum possible value in dtype.
      dtype: `tf.dtypes.DType` indicating the data type of the output. Must be
        either `tf.int32` or `tf.int64`. Defaults to `tf.int64`.

    Raises:
      ValueError: If arguments do not meet expectations.
    """
    py_typecheck.check_type(string_max_bytes, int, label='string_max_bytes')
    py_typecheck.check_type(dtype, tf.dtypes.DType, label='dtype')
    if max_chunk_value is None:
      max_chunk_value = dtype.max
    py_typecheck.check_type(max_chunk_value, int, label='max_chunk_value')

    if string_max_bytes <= 0 or string_max_bytes > max_chunk_value:
      raise ValueError(
          f'string_max_bytes must be between [1, {max_chunk_value=}]. '
          f'Found: {string_max_bytes}'
      )

    if dtype not in (tf.int32, tf.int64):
      raise ValueError(
          f'`dtype` must be either `tf.int32` or `tf.int64`.Found: {dtype}'
      )

    if max_chunk_value < tf.uint8.max or max_chunk_value > dtype.max:
      raise ValueError(
          f'`max_chunk_value must be between [{tf.uint8.max}, {dtype.max=}]. '
          f'Found: {max_chunk_value}'
      )

    self._string_max_bytes = string_max_bytes
    self._dtype = dtype
    self._max_chunk_value = max_chunk_value

    self._chunk_value_bitrange = math.floor(math.log2(max_chunk_value))
    self._num_data_chunks = math.ceil(
        self._string_max_bytes * 8 / self._chunk_value_bitrange
    )
    # The header always fits into a single chunk because
    # `string_max_bytes <= max_chunk_value`
    self._num_header_chunks = 1

    logging.debug(
        'BinaryChunker: %s %s %s %s %s %s',
        f'{self._string_max_bytes=}',
        f'{self._dtype=}',
        f'{self._max_chunk_value=}',
        f'{self._chunk_value_bitrange=}',
        f'{self._num_data_chunks=}',
        f'{self._num_header_chunks=}',
    )

  @property
  def dtype(self) -> tf.dtypes.DType:
    return self._dtype

  def get_num_chunks(self) -> int:
    return self._num_data_chunks + self._num_header_chunks

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
  def encode_tensorflow(
      self, input_strings: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor]:
    """Encodes `input_strings` to tensors.

    Args:
      input_strings: A 1-D `tf.Tensor` of type `tf.string`. Denote the shape of
        `input_strings` as `(num_strings,)`.

    Returns:
      A Tuple `(encoded_strings, trimmed_input_strings)`
      - encoded_strings: A `tf.Tensor` of shape
        `(num_strings, self.get_num_chunks())` containing encoded
        `input_strings`
      - trimmed_input_strings: A `tf.Tensor` of shape `(num_strings,)`
        containing trimmed `input_strings` that the length of each string in it
        is no more than `self._max_length` bytes.
    """
    # `input_strings` is a 1-D tensor of binary strings. ex:
    # [
    #      b'abc',
    #   b'123456',
    # ]

    # `trimmed` contains the input strings after they've been trimmed to
    # `string_max_bytes`. ex:
    # [
    #     b'abc',
    #   b'12345',
    # ]
    trimmed = tf.strings.substr(
        input_strings, pos=0, len=self._string_max_bytes, unit='BYTE'
    )

    # `trimmed_lengths_in_bytes` is a 1-D vector of the byte size of each
    # trimmed string. ex:
    # [
    #   3,
    #   5,
    # ]
    trimmed_lengths_in_bytes = tf.strings.length(trimmed, unit='BYTE')

    # `split_bytes` is a 2-D array containing the byte-size integer
    # representation of trimmed input strings, padded with zeros to
    # `string_max_bytes`. ex:
    # [
    #   [97, 98, 99,  0,  0],
    #   [49, 50, 51, 52, 53],
    # ]
    split_bytes = tf.io.decode_raw(
        trimmed, out_type=tf.uint8, fixed_length=self._string_max_bytes
    )

    def pack_string_bytes_to_int(string_bytes: tf.Tensor) -> tf.Tensor:
      return _pack_into_int(
          string_bytes,
          input_bitrange=8,
          target_bitrange=self._chunk_value_bitrange,
      )

    # `data_chunks` is a 2-D array containing the chunked binary encoding
    # of each trimmed string, with binary data packed into each integer chunk
    # up to `self._chunk_value_bitrange` bits. ex:
    # [
    #   [25185,   198,     0],
    #   [12849, 26726,   212],
    # ]
    data_chunks = tf.squeeze(
        tf.map_fn(
            pack_string_bytes_to_int, elems=tf.cast(split_bytes, self._dtype)
        ),
        axis=-1,
    )

    # `header` is a 2-D tensor containing the trimmed length of each string in
    # the encoding dtype. ex:
    # [
    #   [3],
    #   [5],
    # ]
    header = tf.expand_dims(
        tf.cast(trimmed_lengths_in_bytes, self._dtype), axis=1
    )

    # `encoding` is a 2-D array containing the header byte size for each string
    # followed by its byte encoding. ex:
    # [
    #   [3, 25185,   198,   0],
    #   [5, 12849, 26726, 212],
    # ]
    encoding = tf.concat((header, tf.cast(data_chunks, self._dtype)), axis=1)

    tf.debugging.assert_type(encoding, self._dtype)
    tf.debugging.assert_shapes([
        # $NUM_STRINGS is a placeholder for the assertion, to verify that
        # both tensors have the same size for in the corresponding dimension.
        (input_strings, ('$NUM_STRINGS',)),
        (encoding, ('$NUM_STRINGS', self.get_num_chunks())),
    ])
    if self._max_chunk_value < self._dtype.max:
      tf.debugging.assert_less_equal(
          encoding,
          tf.cast(self._max_chunk_value, self._dtype),
          message='Encoded chunk values must be less than `max_chunk_value.',
      )
    if self._dtype.min < 0:
      tf.debugging.assert_non_negative(
          encoding, message='Encoded chunk values must be non-negative'
      )

    return (encoding, trimmed)

  @tf.function
  def decode_tensorflow(self, encoded_chunks: tf.Tensor) -> tf.Tensor:
    """Decodes `encoded_chunks` to strings.

    Args:
      encoded_chunks: A `tf.Tensor` of shape `(num_strings,
        self.get_num_chunks())`

    Returns:
      A `tf.Tensor` of shape `(num_strings,)` and type `tf.string`.
    """
    # `encoded_chunks` is either
    #
    # 1) A 2-D array containing the header byte size for each string followed by
    #    its byte encoding. ex:
    #    [
    #      [3, 25185,   198,   0],
    #      [5, 12849, 26726, 212],
    #    ]
    # 2) A 1-D array representing a single encoded string, with the same
    #    semantics as described above. ex:
    #    [3, 25185,   198,   0]
    try:
      encoded_chunks = tf.convert_to_tensor(encoded_chunks, dtype=self._dtype)
    except Exception as e:
      raise ValueError(
          '`encoded_chunks` must be a {self._dtype} tensor. '
          f'Found: {encoded_chunks}'
      ) from e
    if (
        len(encoded_chunks.shape) == 1
        and encoded_chunks.shape[0] == self.get_num_chunks()
    ):
      # Invoked with a single encoded string (case #2 above);
      # expand dims to a 2-D
      encoded_chunks = tf.expand_dims(encoded_chunks, axis=0)
    tf.debugging.assert_shapes(
        [(encoded_chunks, (None, self.get_num_chunks()))],
        message=(
            '`encoded_chunks` must be a 2-D vector '
            f'with second dimension size `{self.get_num_chunks()}`.'
        ),
    )

    # `header` is a 2-D array contains the header bytes split off from the
    # encoded string data. ex:
    # [
    #   [3],
    #   [5],
    # ]
    header, encoded_data = tf.split(
        encoded_chunks,
        axis=1,
        num_or_size_splits=[self._num_header_chunks, self._num_data_chunks],
    )

    def unpack_string_bytes_from_int(string_chunks: tf.Tensor) -> tf.Tensor:
      return _unpack_from_int(
          string_chunks,
          original_bitrange=8,
          target_bitrange=self._chunk_value_bitrange,
          shape=[self._string_max_bytes],
      )

    # `decoded_bytes` is a 2-D array containing the byte-size integer
    # representation of each string padded with zeros. ex:
    # [
    #   [97, 98, 99,  0,  0],
    #   [49, 50, 51, 52, 53],
    # ]
    decoded_bytes = tf.map_fn(
        unpack_string_bytes_from_int,
        elems=tf.expand_dims(encoded_data, axis=-1),
    )

    # `byte_strings` contains the encoded binary data as indiviual bytes
    # of tf.string binary, padded with 0 bytes to `string_max_bytes`. ex:
    # [
    #   [b'a', b'b', b'c', b'\x00', b'\x00', b'\x00', b'\x00', b'\x00']
    #   [b'1', b'2', b'3', b'4',    b'5',    b'\x00', b'\x00', b'\x00'],
    # ]
    int_to_byte_map = tf.constant([bytes([i]) for i in range(256)], tf.string)
    byte_strings = tf.nn.embedding_lookup(int_to_byte_map, decoded_bytes)

    # `string_lengths` is a 1-D vector of the actual data byte size of each
    # encoded string. ex:
    # [
    #   3,
    #   5
    # ]
    string_lengths = tf.squeeze(
        tf.cast(header, _BINARY_CHUNKER_STRING_LEN_DTYPE)
    )

    # `decoded` is the final decoded string data, in a 1-D `tf.string` tensor
    # with all padding removed. ex:
    # [
    #   b'abc'
    #   b'1234'
    # ]
    padded_strings = tf.strings.reduce_join(byte_strings, axis=1)
    decoded = tf.strings.substr(
        padded_strings,
        pos=tf.zeros_like(string_lengths),
        len=string_lengths,
        unit='BYTE',
    )

    tf.debugging.assert_type(decoded, tf.string)
    tf.debugging.assert_shapes(
        # $NUM_STRINGS is a placeholder for the assertion, to verify that
        # both tensors have the same size for in the corresponding dimension.
        [(encoded_chunks, ('$NUM_STRINGS', self.get_num_chunks()))],
        [(decoded, '$NUM_STRINGS')],
    )

    return decoded


class UTF8Chunker(Chunker):
  """Encodes and decodes strings into integer tensors using UTF-8 encoding."""

  def __init__(
      self,
      string_max_bytes: int,
      *,
      max_chunk_value: Optional[int] = None,
      dtype: tf.dtypes.DType = _DEFAULT_DTYPE,
  ):
    """Initializes the chunker.

    Args:
      string_max_bytes: Maximum length of the string to encode. Note that this
        is measured in bytes and some unicode characters may take more than 1
        byte. In the case that `string_max_bytes` does not divide
        `self._dtype_size_bytes` (calculated below), it is rounded up to the
        smallest integer that divides it.
      max_chunk_value: Maximum value in each chunk. Defaults to the maximum
        possible value in dtype.
      dtype: `tf.dtypes.DType` indicating the data type of the output. Must be
        either `tf.int32` or `tf.int64`. Defaults to `tf.int64`.

    Raises:
      ValueError: If arguments do not meet expectations.
    """
    if string_max_bytes < 1:
      raise ValueError('string_max_bytes must be at least 1.')

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
        raise ValueError(
            'max_chunk_value {} cannot fit in dtype {}'.format(
                max_chunk_value, dtype
            )
        )
      self._dtype_size_bytes = math.floor(
          math.log2(self._max_chunk_value) / self._utf8_size_bits
      )
    else:
      self._dtype_size_bytes = self._dtype.size

    self._num_chunks = math.ceil(
        float(string_max_bytes) / self._dtype_size_bytes
    )
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

  @property
  def dtype(self) -> tf.dtypes.DType:
    return self._dtype

  def get_num_chunks(self) -> int:
    return self._num_chunks

  def encode_tensorflow(
      self, input_strings: tf.Tensor
  ) -> tuple[tf.Tensor, tf.Tensor]:
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

      Note that a utf-8 character might take more than one byte, so both the
      encoded and trimmed strings could contain characters that are cut in the
      middle. The caller needs to be aware of this when decoding these strings,
      i.g. decode a byte string `s` by `s.decode('utf-8', 'ignore')` to avoid
      decoding errors.
    """
    string_bytes = tf.io.decode_raw(
        input_strings, out_type=tf.uint8, fixed_length=self._max_length
    )
    string_bytes_reshaped = tf.reshape(
        string_bytes, (-1, self._dtype_size_bytes)
    )
    string_bytes_cast = tf.cast(string_bytes_reshaped, self._dtype)
    dtype_multipliers = tf.constant(
        [[2 ** (bit)] for bit in self._bit_lengths], dtype=self._dtype
    )
    encoded_as_dtype = tf.matmul(string_bytes_cast, dtype_multipliers)
    if self._max_chunk_value:
      tf.assert_less(
          encoded_as_dtype,
          tf.constant(self._max_chunk_value, dtype=self._dtype),
      )
    encoded_strings = tf.reshape(encoded_as_dtype, (-1, self._num_chunks))

    int_to_char_map = tf.constant(self._int_to_byte_map, dtype=tf.string)
    trimmed_input_strings = tf.nn.embedding_lookup(
        int_to_char_map, tf.cast(string_bytes, dtype=tf.int32)
    )
    trimmed_input_strings = tf.strings.reduce_join(
        trimmed_input_strings, axis=1
    )

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
    encoded_chunks_tiled = tf.tile(
        encoded_chunks_reshaped, [1, self._dtype_size_bytes]
    )
    bit_lengths = tf.constant(self._bit_lengths, dtype=self._dtype)
    encoded_chunks_shifted = tf.bitwise.right_shift(
        encoded_chunks_tiled, bit_lengths
    )
    encoded_chunks_modulo = encoded_chunks_shifted % 2 ** (self._utf8_size_bits)
    encoded_chunks_reshaped = tf.reshape(
        encoded_chunks_modulo, (-1, self._max_length)
    )
    encoded_chunks_bytes = tf.cast(encoded_chunks_reshaped, dtype=tf.int32)
    int_to_char_map = tf.constant(self._int_to_byte_map, dtype=tf.string)
    decoded_strings = tf.nn.embedding_lookup(
        int_to_char_map, encoded_chunks_bytes
    )
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
    encoded_chunks_tiles = np.tile(
        encoded_chunks_reshaped, [1, self._dtype_size_bytes]
    )
    encoded_chunks_bytes_shifted = np.right_shift(
        encoded_chunks_tiles, self._bit_lengths
    )
    encoded_chunks_bytes = encoded_chunks_bytes_shifted % 2 ** (
        self._utf8_size_bits
    )
    int_to_char_fn = lambda x: (dict(enumerate(self._int_to_byte_map)).get(x))

    # Added `otypes=(np.bytes_,)` as an additional arg to np.vectorize to avoid
    # numpy crashes with empty strings (not able to identify the type).
    decoded_chars = np.vectorize(int_to_char_fn, otypes=(np.bytes_,))(
        encoded_chunks_bytes
    )
    decoded_chars_reshaped = decoded_chars.reshape(-1, self._max_length)
    decoded_strings = np.apply_along_axis(
        lambda r: r.tobytes(), arr=decoded_chars_reshaped, axis=1
    )

    return decoded_strings
