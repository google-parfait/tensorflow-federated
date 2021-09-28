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
"""IBLT encoder and decoder for SecAgg-based heavy hitters.

The Invertible Bloom Lookup Tables (IBLT) is a sketch for representing
dictionaries of key-value pairs, where keys and values are integers.
Other key data types can be supported by encoding keys as integers (discussed
later). The IBLT differs from ordinary dictionaries in that it is a linear data
structure, that is, we can add and subtract IBLTs. If a key has value v1 in
IBLT1 and value v2 in IBLT2, it has value v1+v2 in IBLT1+IBLT2 and value v1-v2
in IBLT1-IBLT2. If a key is not present in an IBLT it implicitly has value 0.

We describe the IBLT variant implemented below. To insert a pair (k,v),
we compute hashes h_i(k) and q_i(k), and update these counters, for i=1,2,3:
count[i,h_i(k)] += v key[i,h_i(k)] += vk check[i,h_i(k)] += vq_i(k)

An IBLT has a fixed capacity for n key-value pairs. If the capacity is exceeded,
then it is usually not possible to recover all key-value pairs, but otherwise
all key-value pairs can be decoded with high probability. There is no efficient
lookup operation, but decoding all keys and values can be done in expected time
O(n). The decoding procedure works by identifying pairs (i,j) such that only one
key k in the key set has h_i(k)=j. This is possible because from (v, v * k) we
can compute v and k, and check that v * q_i(k) matches check[i,h(k)]. If there
is any colliding key the check will fail with high probability. Once a pair has
been decoded we can subtract its contribution to the IBLT and recurse on the
remainder.

Theoretical property: If the range of h_i is more than about 0.41 n, decoding
n keys will succeed with constant probability.
(The error bound holds with high probability if we increase the number of hash
functions, "repetitions", to 4 or more, at the cost of higher space usage.)

Reference: Invertible Bloom Lookup Tables, M.T. Goodrich, M. Mitzenmacher:
https://arxiv.org/abs/1101.2245

To make effective use of fixed precision arithmetic our TensorFlow
implementation makes use of arithmetic in a finite prime field of size
field_size, i.e., does arithmetic modulo field_size.

The IbltEncoder class below uses tf.strings.to_hash_bucket_fast() for hashing
whereas the IbltDecoder class uses farmhash.fingerprint64(). These two functions
are consistent since tf.strings.to_hash_bucket_fast() is implemented using
farmhash.fingerprint64(). Note that the TensorFlow documentation of the
tf.strings.to_hash_bucket_fast() states that the values of this hash function
are deterministic on the content of the string and will never change.
CAUTION: If this is no longer the case and the implementation of
tf.strings.to_hash_bucket_fast() changes, then IbltDecoder.get_freq_estimates()
would fail.
"""

from typing import Dict, Optional, Union, Tuple

import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers
from tensorflow_federated.python.analytics.heavy_hitters.iblt import hyperedge_hashers

DEFAULT_FIELD_SIZE = 2**31 - 1
DEFAULT_REPETITIONS = 3
# Theoretical IBLT space bounds, Table 1 in https://arxiv.org/pdf/1101.2245.pdf
_REPETITIONS_TO_SPACE_OVERHEAD = {3: 1.222, 4: 1.295, 5: 1.425, 6: 1.570}
_HASH_FAMILY_RANDOM = "random"
_HASH_FAMILY_COUPLED = "coupled"


def _internal_parameters(
    capacity: int,
    repetitions: int,
    hash_family: Optional[str] = None,
    hash_family_params: Optional[Dict[str, Union[int, float]]] = None,
):
  """Computes internal IBLT parameters based on constructor parameters.

  Shared between `IbltDecoder` and `IbltEncoder`.

  Args:
    capacity: Number of distinct strings that we expect to be inserted
    repetitions: Number of repetitions in IBLT data structure (must be >= 3).
    hash_family: A string specifying the hash family to use to construct IBLT.
      Options include coupled or random, default is chosen based on capacity.
    hash_family_params: A `dict` of parameters that the hash family hasher
      expects. Defaults are chosen based on capacity.

  Returns:
    table_size: the hash table size of the IBLT
    hash_family: string indicating which hash family to use.
    hash_family_params: dict containing the hash family parameters to use.
  """
  assert repetitions >= 3
  if repetitions in _REPETITIONS_TO_SPACE_OVERHEAD:
    minimum_space = _REPETITIONS_TO_SPACE_OVERHEAD[repetitions] * capacity
  else:
    minimum_space = capacity * repetitions / 4  # Rough upper bound
  # Pick slightly larger table size for robustness:
  table_size = int(1.1 * minimum_space / repetitions + 10)

  # go/iblt-coupled-hash-analysis for analysis of coupled hash family.
  suggested_hash_family = _HASH_FAMILY_COUPLED if capacity >= 100000 else _HASH_FAMILY_RANDOM
  if hash_family is None:
    hash_family = suggested_hash_family
  suggested_hash_family_params = {}
  if hash_family == _HASH_FAMILY_COUPLED:
    suggested_hash_family_params["rescale_factor"] = 0.25 * capacity**0.5

  if hash_family_params is None:
    hash_family_params = suggested_hash_family_params

  return table_size, hash_family, hash_family_params


def _gcd_tf(a, b, dtype=tf.int64):
  """Calculates the greatest common denominator of 2 numbers.

  Assumes that a and b are tf.Tensor of shape () and performs the extended
  euclidean algorithm to find the gcd and the coefficients of Bézout's
  identity (https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity)

  Args:
    a: A scalar `tf.Tensor`.
    b: A scaler `tf.Tensor`.
    dtype: Data type to perform operations in. `a` and `b` are casted to this
      dtype.

  Returns:
    A tuple of `tf.Tensor`s `(g, x, y)` such that `a*x + b*y = g = gcd(a, b)`.
  """
  a = tf.cast(a, dtype=dtype)
  b = tf.cast(b, dtype=dtype)
  x0, x1, y0, y1 = (tf.constant(0, dtype=dtype), tf.constant(1, dtype=dtype),
                    tf.constant(1, dtype=dtype), tf.constant(0, dtype=dtype))

  def cond(a, b, x0, x1, y0, y1):
    del b, x0, x1, y0, y1
    return tf.math.not_equal(a, tf.constant(0, dtype=dtype))

  def body(a, b, x0, x1, y0, y1):
    (q, a), b = (tf.cast(b / a, dtype=dtype), b % a), a
    y0, y1 = y1, y0 - q * y1
    x0, x1 = x1, x0 - q * x1
    return a, b, x0, x1, y0, y1

  a, b, x0, x1, y0, y1 = tf.while_loop(
      cond, body, loop_vars=(a, b, x0, x1, y0, y1))
  return b, x0, y0


def _inverse_mod(x, p, dtype=tf.int64):
  """Calculates the multiplicative inverse of `x modulo p`.

  Requires that `x` and `p` are coprime. If not, then raises Exception.

  Args:
    x: A scalar `tf.Tensor`.
    p: A scalar `tf.Tensor`.
    dtype: Data type to perform operations in. `a` and `b` are casted to this
      dtype.

  Returns:
    A `tf.Tensor` `y` such that `x * y modulo p = 1`.

  Raises:
    tf.errors.InvalidArgumentError: if `x` and `p` are not coprime.
  """
  p = tf.cast(p, dtype=dtype)
  gcd, x0, _ = _gcd_tf(x, p, dtype=dtype)
  tf.debugging.assert_equal(gcd, tf.constant(1, dtype=dtype), "gcd(x, p) != 1")
  return x0 % p


def _get_hash_check_salt(seed: int) -> str:
  return "hash_check_" + str(seed)


def _compute_hash_check(input_strings: tf.Tensor, field_size: int, seed: int,
                        dtype: tf.dtypes.DType) -> tf.Tensor:
  """Returns the hash_check for input_strings modulo field_size."""
  hash_check_salt = _get_hash_check_salt(seed)
  salted_input = tf.strings.join([hash_check_salt, input_strings])
  hash_check = tf.strings.to_hash_bucket_fast(
      salted_input, num_buckets=field_size)
  hash_check = tf.reshape(hash_check, shape=[tf.size(hash_check), 1])
  hash_check = tf.cast(hash_check, dtype=dtype)
  return hash_check


class IbltDecoder:
  """Decode the strings and counts stored in an IBLT data structure."""

  def __init__(
      self,
      iblt: tf.Tensor,
      capacity: int,
      string_max_length: int,
      *,
      seed: int = 0,
      repetitions: int = DEFAULT_REPETITIONS,
      hash_family: Optional[str] = None,
      hash_family_params: Optional[Dict[str, Union[int, float]]] = None,
      dtype=tf.int64,
      field_size: int = DEFAULT_FIELD_SIZE,
  ):
    """Initializes the IBLT Decoder.

    The IBLT is a tensor of shape [repetitions, table_size, num_chunks + 2].
    Its value at index `(r, h, c)` corresponds to:

      - sum of chunk `c` of keys hashing to `h` in repetition `r` if
        `c < num_chunks`
      - sum of counts of keys hashing to `h` in repetition `r` if
        `c = num_chunks`
      - sum of checks of keys hashing to `h` in repetition `r` if
        `c = num_chunks + 1`.

    Since decoding is a destructive procedure, the __init__ function starts by
    making a copy of the iblt.

    Args:
      iblt: Tensor representing the IBLT computed by the IbltEncoder.
      capacity: Number of distinct strings that we expect to be inserted.
      string_max_length: Maximum length of a string that can be inserted.
      seed: Integer seed for hash functions. Defaults to 0.
      repetitions: Number of repetitions in IBLT data structure (must be >= 3).
        Defaults to 3.
      hash_family: A `str` specifying the hash family to use to construct IBLT.
        Options include coupled or random, default is chosen based on capacity.
      hash_family_params: An optional `dict` of parameters that the hash family
        hasher expects. Defaults are chosen based on capacity.
      dtype: a tensorflow data type which determines the type of the IBLT values
      field_size: The field size for all values in IBLT. Defaults to 2**31 - 1.
    """
    self._dtype = dtype
    self.iblt = iblt
    self.table_size, self.hash_family, self.hash_family_params = _internal_parameters(
        capacity, repetitions, hash_family, hash_family_params)
    self.field_size = field_size
    self.chunker = chunkers.UTF8Chunker(
        string_max_length, max_chunk_value=self.field_size, dtype=self._dtype)
    self.num_chunks = self.chunker.get_num_chunks()
    self.count = self.num_chunks
    self.check = self.num_chunks + 1
    self.repetitions = repetitions
    self.seed = seed
    self.iblt_shape = (self.repetitions, self.table_size, self.num_chunks + 2)
    self.q = tf.queue.RandomShuffleQueue(
        capacity=self.table_size * self.repetitions,
        min_after_dequeue=0,
        dtypes=(self._dtype, self._dtype))
    if self.hash_family == _HASH_FAMILY_RANDOM:
      self.hyperedge_hasher = hyperedge_hashers.RandomHyperEdgeHasher(
          seed, self.table_size, repetitions, **self.hash_family_params)
    elif self.hash_family == _HASH_FAMILY_COUPLED:
      self.hyperedge_hasher = hyperedge_hashers.CoupledHyperEdgeHasher(
          seed, self.table_size, repetitions, **self.hash_family_params)
    else:
      raise NotImplementedError(
          f"Hash family {hash_family} not supported in IBLTs.")

  def decode_string_from_chunks(self, chunks):
    """Compute string from sequence of ints each encoding 'chunk_length' bytes.

    Inverse of `IBLTEncoder.compute_iblt`.

    Args:
      chunks: A `tf.Tensor` of `num_chunks` integers.

    Returns:
      A `tf.Tensor` with the UTF-8 string encoded in the chunks.
    """
    return self.chunker.decode_tensorflow(chunks)[0]

  def get_hash_check(self, input_strings):
    """Returns a `tf.Tensor` containing hash_checks.

    Args:
      input_strings: A `tf.Tensor` of strings.

    Returns:
      A tensor of shape `(input_length, repetitions)` containing hash_check[i]
      at index (i, r).
    """
    if input_strings.dtype != tf.string:
      raise TypeError(
          "hash checks can only be computed on string tensors, got: "
          f"{input_strings.dtype}")
    return _compute_hash_check(
        input_strings, self.field_size, seed=self.seed, dtype=self._dtype)

  def is_peelable(self, iblt, repetition, index):
    """Test if can recover string and count from location (repetition, index).

    Args:
      iblt: The IBLT data structure.
      repetition: Repetition number ("hash table number").
      index: Position in table.

    Returns:
      `True` if we can recover string and count from location `(repetition,
      index)`, `False` otherwise.
    """
    return tf.strings.length(self.decode(iblt, repetition, index)[0]) > 0

  def decode(self, iblt, repetition, index):
    """Try to recover string and count from IBLT location (repetition, index).

    Args:
      iblt: the IBLT data structure
      repetition: repetition number ("hash table number")
      index: position in table

    Returns:
      (data_string, count, chunk_encoding) where data_string is the decoded
      string, count is its corresponding count and chunk_encoding is the chunks
      that represent the encoding of the data_string. If no string is decoded,
      data_string is set to '' and the rest is set to -1.
    """
    empty_return = (tf.constant(""), tf.constant(0, dtype=self._dtype),
                    tf.zeros((self.num_chunks,), dtype=self._dtype))
    if tf.math.not_equal(iblt[repetition][index][self.count],
                         tf.constant(0, dtype=self._dtype)):
      inverse_count = _inverse_mod(
          iblt[repetition][index][self.count],
          self.field_size,
          dtype=self._dtype)
      chunks = (iblt[repetition][index][0:self.num_chunks] *
                inverse_count) % self.field_size
      data_string = self.decode_string_from_chunks(chunks)
      hash_check = self.get_hash_check(data_string)
      if tf.math.equal(
          iblt[repetition][index][self.check],
          iblt[repetition][index][self.count] * hash_check % self.field_size):
        return (data_string, iblt[repetition][index][self.count], chunks)
      else:
        return empty_return
    else:
      return empty_return

  def remove_element(self, iblt, data_string, hash_indices, chunks, count):
    """Remove the key `data_string` and its `count` from the IBLT.

    Args:
      iblt: the IBLT data structure
      data_string: string to be removed from the IBLT
      hash_indices: must equal get_hash_indices(data_string), passed to avoid
        recomputation.
      chunks: must satisfy data_string = decode_string_from_chunks(chunks),
        passed to avoid recomputation.
      count: count of `data_string` in the IBLT.

    Returns:
      The IBLT data structure with the (data_string, count) removed at
        hash_indices
    """
    hash_check = self.get_hash_check(data_string)
    indices, values = [], []
    for repetition in range(self.repetitions):
      index = hash_indices[repetition]
      repetition = tf.constant(repetition, dtype=self._dtype)
      indices.append(tf.stack([repetition, index, self.count], axis=0))
      values.append(
          (iblt[repetition][index][self.count] - count) % self.field_size)
      indices.append(tf.stack([repetition, index, self.check], axis=0))
      values.append((iblt[repetition][index][self.check] -
                     (count * hash_check)) % self.field_size)
      for chunk_id in range(self.num_chunks):
        indices.append(tf.stack([repetition, index, chunk_id], axis=0))
        values.append((iblt[repetition][index][chunk_id] -
                       (count * chunks[chunk_id])) % self.field_size)
    indices = tf.stack(indices, axis=0)
    values = tf.stack([tf.squeeze(value) for value in values], axis=0)
    iblt = tf.tensor_scatter_nd_update(iblt, indices, values)
    return iblt

  def get_hash_indices(self, data_string):
    data_strings = tf.expand_dims(data_string, 0)
    hash_indices = self.hyperedge_hasher.get_hash_indices_tf(data_strings)[0, :,
                                                                           2]
    hash_indices = tf.cast(hash_indices, dtype=self._dtype)
    return hash_indices

  def decode_and_remove(self, iblt, repetition, index):
    data_string, count, chunks = self.decode(iblt, repetition, index)
    hash_indices = self.get_hash_indices(data_string)
    if tf.strings.length(data_string) > 0:
      iblt = self.remove_element(iblt, data_string, hash_indices, chunks, count)
    return iblt, hash_indices, data_string, count

  @tf.function
  def get_freq_estimates_tf(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decode key-value pairs from an IBLT.

    Returns:
      (out_strings, out_counts, num_not_decoded) where out_strings is tf.Tensor
      containing all the decoded strings, out_counts is a tf.Tensor containing
      the counts of each string and num_not_decoded is tf.Tensor with the number
      of items not decoded in the IBLT.
    """
    iblt = tf.math.floormod(
        tf.cast(self.iblt, dtype=self._dtype),
        tf.constant(self.field_size, dtype=self._dtype))

    # Initialize queue with all locations that can be decoded:
    for repetition in tf.range(self.repetitions, dtype=self._dtype):
      for index in tf.range(self.table_size, dtype=self._dtype):
        if self.is_peelable(iblt, repetition, index):
          self.q.enqueue((repetition, index))

    out_strings = tf.TensorArray(
        tf.string, size=0, dynamic_size=True, clear_after_read=False)
    out_counts = tf.TensorArray(
        self._dtype, size=0, dynamic_size=True, clear_after_read=False)

    # While queue is non-empty, pop and subtract from IBLT, add new peelable
    # locations to queue.
    def cond(iblt, out_strings, out_counts):
      del iblt, out_strings, out_counts
      return self.q.size() > 0

    def body(iblt, out_strings, out_counts):
      repetition, index = self.q.dequeue()
      iblt, hash_indices, data_string, count = self.decode_and_remove(
          iblt, repetition, index)
      if tf.strings.length(data_string) > 0:
        index = out_counts.size()
        out_counts = out_counts.write(index, count)
        out_strings = out_strings.write(index, data_string)
        for r in tf.range(self.repetitions, dtype=self._dtype):
          if self.is_peelable(iblt, r, hash_indices[r]):
            self.q.enqueue((r, hash_indices[r]))
      return iblt, out_strings, out_counts

    iblt, out_strings, out_counts = tf.while_loop(
        cond,
        body,
        loop_vars=(iblt, out_strings, out_counts),
        parallel_iterations=1)

    # Count of entries that could not be decoded:
    num_not_decoded = tf.reduce_sum(iblt[:, :, self.count]) / self.repetitions
    num_not_decoded = tf.cast(num_not_decoded, dtype=self._dtype)

    return out_strings.stack(), out_counts.stack(), num_not_decoded

  def get_freq_estimates(self):
    """Decode key-value pairs from an IBLT.

    Note that this method only works when running TF in Eager mode.

    Returns:
      A dictionary containing a decoded key with its frequency.
    """
    if not tf.compat.v1.executing_eagerly():
      raise NotImplementedError("This method only works with Eager execution.")

    out_strings, out_counts, num_not_decoded = self.get_freq_estimates_tf()
    counter = dict(
        zip(
            [
                # Set 'ignore' in `.decode()` to ignore decoding error because
                # the strings are trimmed when they are encoded, and the
                # trimming might cut in the middle of a multi-byte utf-8
                # character.
                string.decode("utf-8", "ignore")
                for string in out_strings.numpy().tolist()
            ],
            out_counts.numpy().tolist()))
    num_not_decoded = num_not_decoded.numpy()
    if num_not_decoded:
      counter[None] = num_not_decoded
    return counter


class IbltEncoder:
  """Encodes the strings into an IBLT data structure.

  The IBLT is a numpy array of shape [repetitions, table_size, num_chunks+2].
  Its value at index (r, h, c) corresponds to:
    sum of chunk c of keys hashing to h in repetition r if c < num_chunks,
    sum of counts of keys hashing to h in repetition r if c = num_chunks,
    sum of checks of keys hashing to h in repetition r if c = num_chunks + 1.
  """

  def __init__(self,
               capacity,
               string_max_length,
               *,
               drop_strings_above_max_length=False,
               seed=0,
               repetitions=DEFAULT_REPETITIONS,
               hash_family=None,
               hash_family_params=None,
               dtype=tf.int64,
               field_size=DEFAULT_FIELD_SIZE):
    """Initializes internal IBLT parameters.

    Args:
      capacity: Number of distinct strings that we expect to be inserted.
      string_max_length: Maximum length of a string that can be inserted.
      drop_strings_above_max_length: If True, strings above string_max_length
        will be dropped when constructing the IBLT. Defaults to False.
      seed: Integer seed for hash functions. Defaults to 0.
      repetitions: Number of repetitions in IBLT data structure (must be >= 3).
        Defaults to 3.
      hash_family: String specifying the hash family to use to construct IBLT.
        (options include coupled or random, default is chosen based on capacity)
      hash_family_params: A dict of parameters that the hash family hasher
        expects. (defaults are chosen based on capacity.)
      dtype: A tensorflow data type which determines the type of the IBLT
        values.
      field_size: The field size for all values in IBLT. Defaults to 2**31 - 1.
    """
    self.string_max_length = string_max_length
    self.table_size, hash_family, hash_family_params = _internal_parameters(
        capacity, repetitions, hash_family, hash_family_params)
    self.repetitions = repetitions
    self.seed = seed
    self.field_size = field_size
    self.drop_strings_above_max_length = drop_strings_above_max_length
    self.dtype = dtype
    self.internal_dtype = tf.int64
    self.chunker = chunkers.UTF8Chunker(
        string_max_length,
        max_chunk_value=self.field_size,
        dtype=self.internal_dtype)
    self.num_chunks = self.chunker.get_num_chunks()
    self.iblt_shape = (self.repetitions, self.table_size, self.num_chunks + 2)
    if hash_family == _HASH_FAMILY_RANDOM:
      self.hyperedge_hasher = hyperedge_hashers.RandomHyperEdgeHasher(
          seed, self.table_size, repetitions, **hash_family_params)
    elif hash_family == _HASH_FAMILY_COUPLED:
      self.hyperedge_hasher = hyperedge_hashers.CoupledHyperEdgeHasher(
          seed, self.table_size, repetitions, **hash_family_params)
    else:
      raise NotImplementedError(
          "Hash family {} not supported in IBLTs.".format(hash_family))

  def compute_hash_check(self, input_strings):
    """Returns Tensor containing hash_check for each (input string, repetition).

    Args:
      input_strings: A tensor of strings.

    Returns:
      A tensor of shape (input_length, repetitions) containing hash_check[i]
      at index (i, r).
    """
    hash_check = _compute_hash_check(
        input_strings,
        self.field_size,
        seed=self.seed,
        dtype=self.internal_dtype)
    hash_check = tf.tile(hash_check, [1, self.repetitions])
    return hash_check

  def compute_chunks(self, input_strings):
    """Returns Tensor containing integer chunks for input strings.

    Args:
      input_strings: A tensor of strings.

    Returns:
      A 2D tensor with rows consisting of integer chunks corresponding to the
      string indexed by the row and a trimmed input_strings that can fit in the
      IBLT.
    """
    return self.chunker.encode_tensorflow(input_strings)

  def compute_counts(self, sparse_indices, input_length, input_counts=None):
    """Returns SparseTensor with value 1 for each (input string, repetition).

    Args:
      sparse_indices: A tensor of shape (input_length, repetitions, 3).
      input_length: An integer.
      input_counts: A 1D tensor of self.dtype representing the count of each
        string.

    Returns:
      A SparseTensor of dense_shape
      [input_length, repetitions, table_size, num_chunks+2]
      containing a count of 1 for each index of the form
      (i, r, h, num_chunks) where 0 <= i < input_length, 0 <= r < repetitions,
      and h is the hash-position of the ith input string in repetition r.
    """
    counts_chunk_indices = tf.fill([input_length, self.repetitions, 1],
                                   self.num_chunks)
    counts_chunk_indices = tf.cast(
        counts_chunk_indices, dtype=self.internal_dtype)
    counts_sparse_indices = tf.concat([sparse_indices, counts_chunk_indices],
                                      axis=2)
    counts_sparse_indices = tf.reshape(counts_sparse_indices, shape=[-1, 4])
    if input_counts is not None:
      counts_values = tf.repeat(input_counts, [self.repetitions])
    else:
      counts_values = tf.fill([tf.shape(counts_sparse_indices)[0]], 1)
      counts_values = tf.cast(counts_values, dtype=self.internal_dtype)
    counts = tf.SparseTensor(
        indices=counts_sparse_indices,
        values=counts_values,
        dense_shape=(input_length,) + self.iblt_shape)
    return counts

  def compute_checks(self,
                     sparse_indices,
                     hash_check,
                     input_length,
                     input_counts=None):
    """Returns SparseTensor with hash_check for each (input string, repetition).

    Args:
      sparse_indices: A tensor of shape (input_length, repetitions, 3).
      hash_check: A tensor of shape (input_length, repetitions).
      input_length: An integer.
      input_counts: A 1D tensor of self.dtype representing the count of each
        string.

    Returns:
      A SparseTensor of dense_shape
      [input_length, repetitions, table_size, num_chunks+2]
      containing hash_check[i, r] for each index of the form
      (i, r, h, num_chunks+1) where 0 <= i < input_length, 0 <= r < repetitions,
      and h is the hash-position of the ith input string in repetition r.
    """
    if input_counts is not None:
      hash_check = hash_check * input_counts

    checks_chunk_indices = tf.fill([input_length, self.repetitions, 1],
                                   self.num_chunks + 1)
    checks_chunk_indices = tf.cast(
        checks_chunk_indices, dtype=self.internal_dtype)
    checks_sparse_indices = tf.concat([sparse_indices, checks_chunk_indices],
                                      axis=2)
    checks_sparse_indices = tf.reshape(checks_sparse_indices, shape=[-1, 4])
    checks_values = tf.reshape(hash_check, shape=[-1])
    checks = tf.SparseTensor(
        indices=checks_sparse_indices,
        values=checks_values,
        dense_shape=(input_length,) + self.iblt_shape)
    return checks

  def compute_keys(self,
                   sparse_indices,
                   chunks,
                   input_length,
                   input_counts=None):
    """Returns SparseTensor with key for each (input string, repetition, chunk).

    Args:
      sparse_indices: A tensor of shape (input_length, repetitions, 3).
      chunks: A tensor of shape (input_length, num_chunks).
      input_length: An integer.
      input_counts: A 1D tensor of self.dtype representing the count of each
        string.

    Returns:
      A SparseTensor of dense_shape
      [input_length, repetitions, table_size, num_chunks+2]
      containing chunk[i, c] for each index of the form
      (i, r, h, c) where 0 <= i < input_length, 0 <= r < repetitions,
      0 <= c < num_chunks, and h is the hash-position of the ith input string
      in repetition r.
    """
    if input_counts is not None:
      chunks = chunks * input_counts

    keys_chunk_indices = tf.range(self.num_chunks)
    keys_chunk_indices = tf.cast(keys_chunk_indices, dtype=self.internal_dtype)
    keys_chunk_indices = tf.expand_dims(keys_chunk_indices, 0)
    keys_chunk_indices = tf.expand_dims(keys_chunk_indices, 0)
    keys_chunk_indices = tf.expand_dims(keys_chunk_indices, -1)
    keys_chunk_indices = tf.tile(keys_chunk_indices,
                                 [input_length, self.repetitions, 1, 1])
    keys_sparse_indices = tf.expand_dims(sparse_indices, -2)
    keys_sparse_indices = tf.tile(keys_sparse_indices,
                                  [1, 1, self.num_chunks, 1])
    keys_sparse_indices = tf.concat([keys_sparse_indices, keys_chunk_indices],
                                    axis=-1)
    keys_sparse_indices = tf.reshape(keys_sparse_indices, shape=[-1, 4])
    keys_values = tf.expand_dims(chunks, 1)
    keys_values = tf.tile(keys_values, [1, self.repetitions, 1])
    keys_values = tf.reshape(keys_values, shape=[-1])
    keys = tf.SparseTensor(
        indices=keys_sparse_indices,
        values=keys_values,
        dense_shape=(input_length,) + self.iblt_shape)
    return keys

  @tf.function
  def compute_iblt(self, input_strings, input_counts=None):
    """Returns Tensor containing the values of the IBLT data structure.

    Args:
      input_strings: A 1D tensor of strings.
      input_counts: A 1D tensor of self.dtype representing the count of each
        string.

    Returns:
      A tensor of shape [repetitions, table_size, num_chunks+2] whose value at
      index (r, h, c) corresponds to chunk c of the keys if c < num_chunks, to
      the counts if c = num_chunks, and to the checks if c = num_chunks + 1.
    """
    tf.debugging.assert_rank(input_strings, 1)
    tf.debugging.assert_type(input_strings, tf.string)

    if input_counts is not None:
      tf.debugging.assert_rank(input_counts, 1)
      tf.debugging.assert_equal(tf.shape(input_strings), tf.shape(input_counts))
      tf.debugging.assert_type(input_counts, self.dtype)
      input_counts = tf.expand_dims(input_counts, 1)

    chunks, trimmed_input_strings = self.compute_chunks(input_strings)
    if self.drop_strings_above_max_length:
      indices_to_keep = tf.equal(trimmed_input_strings, input_strings)
      trimmed_input_strings = trimmed_input_strings[indices_to_keep]
      chunks = chunks[indices_to_keep]
      if input_counts:
        input_counts = input_counts[indices_to_keep]

    hash_check = self.compute_hash_check(trimmed_input_strings)

    sparse_indices = self.hyperedge_hasher.get_hash_indices_tf(
        trimmed_input_strings)

    input_length = tf.size(trimmed_input_strings)
    counts = self.compute_counts(sparse_indices, input_length, input_counts)
    checks = self.compute_checks(sparse_indices, hash_check, input_length,
                                 input_counts)
    keys = self.compute_keys(sparse_indices, chunks, input_length, input_counts)
    sparse_iblt = tf.sparse.add(keys, counts)
    sparse_iblt = tf.sparse.add(sparse_iblt, checks)
    iblt = tf.sparse.reduce_sum(sparse_iblt, 0)
    iblt = tf.cast(iblt, self.dtype)
    iblt = tf.math.floormod(iblt, self.field_size)
    # Force the result shape so that it can be staticly checked and analyzed.
    # Otherwise the shape is returned as `[None]`.
    iblt = tf.reshape(iblt, self.iblt_shape)
    return iblt


def decode_iblt_tf(
    iblt: tf.Tensor,
    capacity: int,
    string_max_length: int,
    *,
    seed: int = 0,
    repetitions: int = DEFAULT_REPETITIONS,
    hash_family: Optional[str] = None,
    hash_family_params: Optional[Dict[str, Union[int, float]]] = None,
    dtype=tf.int64,
    field_size: int = DEFAULT_FIELD_SIZE,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Decode a IBLT sketch.

  This function wraps `IbltDecoder` to decode `iblt` and returns frequencies
  of decoded strings.

  Args:
    iblt: Tensor representing the IBLT computed by the IbltEncoder.
    capacity: Number of distinct strings that we expect to be inserted.
    string_max_length: Maximum length of a string that can be inserted.
    seed: Integer seed for hash functions. Defaults to 0.
    repetitions: Number of repetitions in IBLT data structure (must be >= 3).
      Defaults to 3.
    hash_family: A `str` specifying the hash family to use to construct IBLT.
      Options include coupled or random, default is chosen based on capacity.
    hash_family_params: An optional `dict` of parameters that the hash family
      hasher expects. Defaults are chosen based on capacity.
    dtype: a tensorflow data type which determines the type of the IBLT values
    field_size: The field size for all values in IBLT. Defaults to 2**31 - 1.

  Returns:
    (out_strings, out_counts, num_not_decoded) where out_strings is tf.Tensor
    containing all the decoded strings, out_counts is a tf.Tensor containing
    the counts of each string and num_not_decoded is tf.Tensor with the number
    of items not decoded in the IBLT.
  """
  iblt_decoder = IbltDecoder(
      iblt=iblt,
      capacity=capacity,
      string_max_length=string_max_length,
      seed=seed,
      repetitions=repetitions,
      hash_family=hash_family,
      hash_family_params=hash_family_params,
      dtype=dtype,
      field_size=field_size,
  )
  return iblt_decoder.get_freq_estimates_tf()
