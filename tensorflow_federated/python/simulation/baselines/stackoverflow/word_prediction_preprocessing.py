# Copyright 2019, Google LLC.
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
"""Preprocessing library for Stack Overflow next-word prediction tasks."""

import collections
from typing import Callable, List, Tuple

import attr
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations

EVAL_BATCH_SIZE = 100


@attr.s(eq=False, frozen=True)
class SpecialTokens(object):
  """Structure for Special tokens.

  Attributes:
    pad: int - Special token for padding.
    oov: list - Special tokens for out of vocabulary tokens.
    bos: int - Special token for beginning of sentence.
    eos: int - Special token for end of sentence.
  """
  pad = attr.ib()
  oov = attr.ib()
  bos = attr.ib()
  eos = attr.ib()


def split_input_target(chunk: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Generate input and target data.

  The task of language model is to predict the next word.

  Args:
    chunk: A Tensor of text data.

  Returns:
    A tuple of input and target data.
  """
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


def build_to_ids_fn(
    vocab: List[str],
    max_sequence_length: int,
    num_oov_buckets: int = 1) -> Callable[[tf.Tensor], tf.Tensor]:
  """Constructs function mapping examples to sequences of token indices."""
  special_tokens = get_special_tokens(len(vocab), num_oov_buckets)
  bos = special_tokens.bos
  eos = special_tokens.eos

  table_values = tf.range(len(vocab), dtype=tf.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=num_oov_buckets)

  def to_ids(example):

    sentence = tf.reshape(example['tokens'], shape=[1])
    words = tf.strings.split(sentence, sep=' ').values
    truncated_words = words[:max_sequence_length]
    tokens = table.lookup(truncated_words) + 1
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_sequence_length),
        lambda: tf.concat([tokens, [eos]], 0), lambda: tokens)

    return tf.concat([[bos], tokens], 0)

  return to_ids


def batch_and_split(dataset: tf.data.Dataset, sequence_length: int,
                    batch_size: int) -> tf.data.Dataset:
  return dataset.padded_batch(
      batch_size, padded_shapes=[sequence_length + 1]).map(
          split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_special_tokens(vocab_size: int,
                       num_oov_buckets: int = 1) -> SpecialTokens:
  """Gets tokens dataset preprocessing code will add to Stackoverflow."""
  return SpecialTokens(
      pad=0,
      oov=[vocab_size + 1 + n for n in range(num_oov_buckets)],
      bos=vocab_size + num_oov_buckets + 1,
      eos=vocab_size + num_oov_buckets + 2)


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    vocab: List[str],
    sequence_length: int,
    num_oov_buckets: int = 1,
    max_elements: int = -1,
    shuffle_buffer_size: int = 1000,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE
) -> computation_base.Computation:
  """Creates a preprocessing functions for Stack Overflow next-word-prediction.

  This function returns a `tff.Computation` which takes a dataset and returns a
  dataset, suitable for mapping over a set of unprocessed client datasets.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    vocab: Vocabulary which defines the embedding.
    sequence_length: Integer determining the length of preprocessed batches.
      Sequences of word tokens will be padded up to this length with a pad token
      and sequences longer than this will be truncated to this length.
    num_oov_buckets: The number of out of vocabulary buckets. Tokens that are
      not present in the `vocab` are hashed into one of these buckets.
    max_elements: Integer controlling the maximum number of elements
      to take per client. If -1, keeps all elements for each client. This is
      applied before repeating the client dataset, and is intended to contend
      with the small set of clients with tens of thousands of examples. Note
      that only the first `max_elements` will be selected.
    shuffle_buffer_size: An integer representing the shuffle buffer size on
      clients. If set to a number <= 1, no shuffling occurs. If
      `max_elements_per_client` is positive and less than `shuffle_buffer_size`,
      it will be set to `max_elements_per_client`.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A `tff.Computation` taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.

  Raises:
    ValueError: If `num_epochs` is a non-positive integer, if `batch_size` is
      not a positive integer, if `sequence_length` is not a positive integer, if
      `vocab` is empty, if `num_oov_buckets` is not a positive integer, if
      `max_elements` is not a positive integer or -1, if `sequence_length`
      is not a positive integer.
  """
  if num_epochs <= 0:
    raise ValueError('num_epochs must be a positive integer. ')
  if batch_size <= 0:
    raise ValueError('batch_size must be a positive integer.')
  if not vocab:
    raise ValueError('vocab must be non-empty.')
  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer.')
  if max_elements == 0 or max_elements < -1:
    raise ValueError('max_elements must be a positive integer or -1.')
  if num_oov_buckets <= 0:
    raise ValueError('num_oov_buckets must be a positive integer.')

  if (max_elements > 0) and (max_elements < shuffle_buffer_size):
    shuffle_buffer_size = max_elements

  feature_dtypes = collections.OrderedDict(
      creation_date=tf.string,
      title=tf.string,
      score=tf.int64,
      tags=tf.string,
      tokens=tf.string,
      type=tf.string,
  )

  @computations.tf_computation(computation_types.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    to_ids = build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=sequence_length,
        num_oov_buckets=num_oov_buckets)
    dataset = dataset.take(max_elements)
    if shuffle_buffer_size > 0:
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(num_epochs).map(
        to_ids, num_parallel_calls=num_parallel_calls)
    return batch_and_split(dataset, sequence_length, batch_size)

  return preprocess_fn
