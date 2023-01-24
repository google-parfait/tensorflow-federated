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
"""Hashers that implement different hash function families to use with IBLTs."""

import farmhash
import numpy as np
import tensorflow as tf

_SEPARATOR = ';'


@tf.function
def _to_sparse_indices_format(hash_indices: tf.Tensor) -> tf.Tensor:
  """Formats `hash_indices` into the format expected by the IBLT encoder.

  Args:
    hash_indices: A `tf.Tensor` of shape `(repetitions, input_length)`
      containing the hash_indices.

  Returns:
    A `tf.Tensor` of shape `(input_length, repetitions, 3)` containing value `i`
    at index `(i, r, 0)`, value `r` at index `(i, r, 1)` and the `i-th`
    hash-index at repetition `r` at index `(i, r, 2)`.
  """
  repetitions = tf.shape(hash_indices)[0]
  input_length = tf.shape(hash_indices)[1]

  hash_indices = tf.reshape(
      tf.transpose(tf.stack(hash_indices)), shape=[input_length, repetitions, 1]
  )
  hash_indices = tf.cast(hash_indices, dtype=tf.int64)
  row_indices = tf.reshape(tf.range(input_length), shape=[input_length, 1, 1])
  row_indices = tf.tile(row_indices, [1, repetitions, 1])
  row_indices = tf.cast(row_indices, dtype=tf.int64)

  column_indices = tf.reshape(tf.range(repetitions), shape=[1, repetitions, 1])
  column_indices = tf.tile(column_indices, [input_length, 1, 1])
  column_indices = tf.cast(column_indices, dtype=tf.int64)

  sparse_indices = tf.concat(
      [row_indices, column_indices, hash_indices], axis=2
  )
  return sparse_indices


class RandomHyperEdgeHasher:
  """Hashes a string to a list of independently sampled indices.

  For a string, generates a set of indices such that each index is independently
  sampled uniformly at random.
  """

  def __init__(self, seed: int, table_size: int, repetitions: int):
    """Initializes `RandomHyperEdgeHasher`.

    Args:
      seed: An integer seed for hash functions.
      table_size: The hash table size of the IBLT. Must be a positive integer.
      repetitions: The number of repetitions in IBLT data structure. Must be an
        integer at least 3.

    Raises:
      ValueError: If arguments do not meet expectations.
    """
    if table_size < 1:
      raise ValueError('table_size must be at least 1.')

    if repetitions < 3:
      raise ValueError('repetitions must be at least 3.')

    self._seed = seed
    self._salt = [str(seed + i) + _SEPARATOR for i in range(repetitions)]
    self._table_size = table_size
    self._repetitions = repetitions

  def get_hash_indices(self, data_strings: list[str]) -> list[list[int]]:
    """Computes the indices at which `data_strings` in IBLT.

    Args:
      data_strings: A list of strings to be hashed.

    Returns:
      hash_indices: A vector of `repetitions` hash values of `data_string`,
      in {0,...,`table_size`-1}.
    """
    all_hash_indices = []
    for data_string in data_strings:
      hash_indices = []
      for i in range(self._repetitions):
        hash_indices.append(
            farmhash.fingerprint64(str(self._salt[i]) + data_string)
            % self._table_size
        )
      all_hash_indices.append(hash_indices)

    return all_hash_indices

  def get_hash_indices_tf(self, input_strings: tf.Tensor) -> tf.Tensor:
    """Returns a `tf.Tensor` containing the indices of `input_string` in IBLT.

    Args:
      input_strings: A 1-d `tf.Tensor` of strings.

    Returns:
      A `tf.Tensor` of shape `(input_length, repetitions, 3)` containing value
      `i` at index `(i, r, 0)`, value `r` at index `(i, r, 1)` and the
      hash-index of the `i-th` input string in repetition `r` at index
      `(i, r, 2)`.
    """
    hash_indices = []
    salted_input = []
    for i in range(self._repetitions):
      salted_input.append(tf.strings.join([self._salt[i], input_strings]))
      hash_indices.append(
          tf.strings.to_hash_bucket_fast(
              salted_input[i], num_buckets=self._table_size
          )
      )

    sparse_indices = _to_sparse_indices_format(hash_indices)
    return sparse_indices


class CoupledHyperEdgeHasher:
  """Hashes a string to an hyper-edge with coupled indices.

  For a string, generates a set of indices such that the indices are close to
  each as described in https://arxiv.org/pdf/2001.10500.pdf.
  """

  def __init__(
      self, seed: int, table_size: int, repetitions: int, rescale_factor: float
  ):
    """Initialize CoupledHyperEdgeHasher.

    Args:
      seed: An integer seed for hash functions.
      table_size: The hash table size of the IBLT. Must be a positive integer.
      repetitions: The number of repetitions in IBLT data structure. Must be at
        least 3.
      rescale_factor: A float to rescale `table_size` to `table_size /
        rescale_factor + 1`. This number is denoted as `z` in
         https://arxiv.org/pdf/2001.10500.pdf. Must be non-negative and no
           greater than `table_size - 1`.

    Raises:
      ValueError: If arguments do not meet expectations.
    """

    if table_size < 1:
      raise ValueError('table_size must be at least 1.')

    if repetitions < 3:
      raise ValueError('repetitions must be at least 3.')

    if rescale_factor <= 0 or rescale_factor > table_size - 1:
      raise ValueError(
          'rescale_factor must be positive and no greater than'
          f' table_size - 1. Found table_size = {table_size} and'
          f' rescale_factor = {rescale_factor}'
      )

    self._seed = seed
    self._salt = [str(seed + i) + _SEPARATOR for i in range(repetitions)]
    self._table_size = table_size
    self._repetitions = repetitions
    self._rescale_factor = rescale_factor
    self._rescaled_table_size = table_size / (self._rescale_factor + 1.0)

  def _get_hash_indices_single(self, data_string: str) -> list[int]:
    """Computes the indices of `data_string` in IBLT."""
    position = self._hash_to_float(
        data_string, (0.5, self._rescale_factor + 0.5)
    )
    hash_indices = []
    for i in range(self._repetitions):
      salted_string = str(self._salt[i]) + data_string
      offset = self._hash_to_float(salted_string, (-0.5, 0.5))
      hash_indices.append(
          int(np.floor((position + offset) * self._rescaled_table_size))
      )
    return hash_indices

  def get_hash_indices(self, data_strings: list[str]) -> list[list[int]]:
    """Computes the indices at which the given strings in IBLT.

    Args:
      data_strings: A list of strings to be hashed.

    Returns:
      hash_indices: vector of `repetitions` hash values of `data_string`,
      in {0,...,`table_size`-1}.
    """
    all_hash_indices = []
    for data_string in data_strings:
      all_hash_indices.append(self._get_hash_indices_single(data_string))
    return all_hash_indices

  def _hash_to_float(
      self,
      input_string: str,
      hash_range: tuple[float, float],
      precision: int = tf.int32.max,
  ) -> float:
    """Hashes a string and returns a `float`.

    `hash_range` is evenly divided into a number of buckets. The hashed value of
    `input_string` is mapped to one of the bucket.

    TODO(b/158684105): Update this function to directly map the hash to the
    index rather than converting hash -> float -> index.

    Args:
      input_string: An input string.
      hash_range: A tuple representing the range for the hashed value.
      precision: The number of buckets in `hash_range`. Must be a positive
        integer.

    Returns:
      A float value being the lower bound of the bucket that the hashed string
      value falls into.
    """
    (low, high) = hash_range
    hashed_value = farmhash.fingerprint64(input_string)
    hashed_value = hashed_value % precision
    hashed_value = ((float(hashed_value) / precision) * (high - low)) + low
    return hashed_value

  def _hash_to_float_tf(
      self,
      input_strings: tf.Tensor,
      hash_range: tuple[float, float],
      precision: int = tf.int32.max,
  ) -> tf.Tensor:
    """Hashes a `tf.strings` 1-d tensor and returns a `tf.float32` 1-d tensor.

    `hash_range` is evenly divided into a number of buckets. The hashed value of
    `input_strings` is mapped to one of the bucket.

    Args:
      input_strings: A 1-d `tf.Tensor` of the input strings.
      hash_range: A tuple representing the range for the hashed value.
      precision: The number of buckets in `hash_range`. Must be a positive
        integer.

    Returns:
      A 1-d `tf.float` tensor of the lower bound of each bucket that each hashed
      string value falls into.
    """
    (low, high) = hash_range
    hashed_value = tf.strings.to_hash_bucket_fast(
        input_strings, num_buckets=precision
    )
    hashed_value = hashed_value % precision
    hashed_value = (
        (tf.cast(hashed_value, tf.float32) / precision) * (high - low)
    ) + low
    return hashed_value

  def get_hash_indices_tf(self, data_strings):
    """Returns Tensor containing hash-position of `(input string, repetition)`.

    Args:
      data_strings: A `tf.Tensor` of strings.

    Returns:
      A `tf.Tensor` of shape `(input_length, repetitions, 3)` containing value
      `i` at index `(i, r, 0)`, value `r` at index `(i, r, 1)` and the
      hash-index of the `i-th` input string in repetition `r` at index
      `(i, r, 2)`.
    """
    positions = self._hash_to_float_tf(
        data_strings, (0.5, self._rescale_factor + 0.5)
    )
    hash_indices = []
    for i in range(self._repetitions):
      salted_inputs = tf.strings.join([self._salt[i], data_strings])
      offset = self._hash_to_float_tf(salted_inputs, (-0.5, 0.5))
      hash_indices.append(
          tf.floor((positions + offset) * self._rescaled_table_size)
      )

    sparse_indices = _to_sparse_indices_format(hash_indices)
    return sparse_indices
