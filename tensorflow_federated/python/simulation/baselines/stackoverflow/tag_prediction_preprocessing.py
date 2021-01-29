# Copyright 2020, Google LLC.
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
"""Preprocessing library for Stack Overflow tag prediction tasks."""

import collections
from typing import Callable, List

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations


def build_to_ids_fn(word_vocab: List[str],
                    tag_vocab: List[str]) -> Callable[[tf.Tensor], tf.Tensor]:
  """Constructs a function mapping examples to sequences of token indices."""
  word_vocab_size = len(word_vocab)
  word_table_values = tf.range(word_vocab_size, dtype=tf.int64)
  word_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(word_vocab, word_table_values),
      num_oov_buckets=1)

  tag_vocab_size = len(tag_vocab)
  tag_table_values = tf.range(tag_vocab_size, dtype=tf.int64)
  tag_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(tag_vocab, tag_table_values),
      num_oov_buckets=1)

  def to_ids(example):
    """Converts a Stack Overflow example to a bag-of-words/tags format."""
    sentence = tf.strings.join([example['tokens'], example['title']],
                               separator=' ')
    words = tf.strings.split(sentence)
    tokens = word_table.lookup(words)
    tokens = tf.one_hot(tokens, word_vocab_size + 1)
    tokens = tf.reduce_mean(tokens, axis=0)[:word_vocab_size]

    tags = example['tags']
    tags = tf.strings.split(tags, sep='|')
    tags = tag_table.lookup(tags)
    tags = tf.one_hot(tags, tag_vocab_size + 1)
    tags = tf.reduce_sum(tags, axis=0)[:tag_vocab_size]

    return (tokens, tags)

  return to_ids


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    word_vocab: List[str],
    tag_vocab: List[str],
    max_elements: int = -1,
    shuffle_buffer_size: int = 1000,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE
) -> computation_base.Computation:
  """Creates a preprocessing function for Stack Overflow tag prediction data.

  This function creates a `tff.Computation` which takes a dataset, and returns
  a preprocessed dataset. This preprocessing takes a maximum number of elements
  in the client's dataset, shuffles, repeats some number of times, and then
  maps the elements to tuples of the form (tokens, tags), where tokens are
  bag-of-words vectors, and tags are binary vectors indicating that a given
  tag is associated with the example.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    word_vocab: A list of strings representing the in-vocabulary words.
    tag_vocab: A list of tokens representing the in-vocabulary tags.
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
      not a positive integer, if `word_vocab` or `tag_vocab` is empty, if
      `max_elements` is not a positive integer or -1.
  """
  if num_epochs <= 0:
    raise ValueError('num_epochs must be a positive integer. ')
  if batch_size <= 0:
    raise ValueError('batch_size must be a positive integer.')
  if not word_vocab:
    raise ValueError('word_vocab must be non-empty.')
  if not tag_vocab:
    raise ValueError('tag_vocab must be non-empty.')
  if max_elements == 0 or max_elements < -1:
    raise ValueError('max_elements must be a positive integer or -1.')

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
    dataset = dataset.take(max_elements)
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    to_ids = build_to_ids_fn(word_vocab, tag_vocab)
    return dataset.repeat(num_epochs).map(
        to_ids, num_parallel_calls=num_parallel_calls).batch(batch_size)

  return preprocess_fn
