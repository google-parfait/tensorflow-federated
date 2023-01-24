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

from collections.abc import Callable

import attr
import tensorflow as tf

from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines.stackoverflow import constants


@attr.s(eq=False, frozen=True)
class SpecialTokens:
  """Structure for Special tokens.

  Attributes:
    padding: int - Special token for padding.
    out_of_vocab: list - Special tokens for out of vocabulary tokens.
    beginning_of_sentence: int - Special token for beginning of sentence.
    end_of_sentence: int - Special token for end of sentence.
  """

  padding = attr.ib()
  out_of_vocab = attr.ib()
  beginning_of_sentence = attr.ib()
  end_of_sentence = attr.ib()

  def get_number_of_special_tokens(self):
    return 3 + len(self.out_of_vocab)


def split_input_target(chunk: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
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
    vocab: list[str],
    max_sequence_length: int,
    num_out_of_vocab_buckets: int = 1,
) -> Callable[[tf.Tensor], tf.Tensor]:
  """Constructs function mapping examples to sequences of token indices."""
  special_tokens = get_special_tokens(len(vocab), num_out_of_vocab_buckets)
  bos = special_tokens.beginning_of_sentence
  eos = special_tokens.end_of_sentence

  table_values = tf.range(len(vocab), dtype=tf.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=num_out_of_vocab_buckets,
  )

  def to_ids(example):
    sentence = tf.reshape(example['tokens'], shape=[1])
    words = tf.strings.split(sentence, sep=' ').values
    truncated_words = words[:max_sequence_length]
    tokens = table.lookup(truncated_words) + 1
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_sequence_length),
        lambda: tf.concat([tokens, [eos]], 0),
        lambda: tokens,
    )

    return tf.concat([[bos], tokens], 0)

  return to_ids


def batch_and_split(
    dataset: tf.data.Dataset, sequence_length: int, batch_size: int
) -> tf.data.Dataset:
  return dataset.padded_batch(
      batch_size, padded_shapes=[sequence_length + 1]
  ).map(split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_special_tokens(
    vocab_size: int, num_out_of_vocab_buckets: int = 1
) -> SpecialTokens:
  """Gets tokens dataset preprocessing code will add to Stackoverflow."""
  return SpecialTokens(
      padding=0,
      out_of_vocab=[
          vocab_size + 1 + n for n in range(num_out_of_vocab_buckets)
      ],
      beginning_of_sentence=vocab_size + num_out_of_vocab_buckets + 1,
      end_of_sentence=vocab_size + num_out_of_vocab_buckets + 2,
  )


def create_preprocess_fn(
    preprocess_spec: client_spec.ClientSpec,
    vocab: list[str],
    sequence_length: int = constants.DEFAULT_SEQUENCE_LENGTH,
    num_out_of_vocab_buckets: int = 1,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing functions for Stack Overflow next-word-prediction.

  This function returns a `tff.Computation` which takes a dataset and returns a
  dataset, suitable for mapping over a set of unprocessed client datasets. This
  dataset is formed by mapping datasets of string sequences to token sequences.

  Specifically, the dataset is first shuffled, then repeated some number of
  times, and a maximum number of elements are taken. We then map these elements
  to token sequences, and batch the token sequences.

  Args:
    preprocess_spec: A `tff.simulation.baselines.ClientSpec` containing
      information on how to preprocess clients.
    vocab: A list of string values representing the vocabulary for the
      string-to-token conversion.
    sequence_length: A positive integer determining the length of preprocessed
      token sequences. Sequences of word tokens will be padded up to this length
      with a pad token and sequences longer than this will be truncated to this
      length.
    num_out_of_vocab_buckets: The number of out of vocabulary buckets. Tokens
      that are not present in the `vocab` are hashed into one of these buckets.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A callable taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  if not vocab:
    raise ValueError('vocab must be non-empty.')
  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer.')
  if num_out_of_vocab_buckets <= 0:
    raise ValueError('num_out_of_vocab_buckets must be a positive integer.')

  shuffle_buffer_size = preprocess_spec.shuffle_buffer_size
  if shuffle_buffer_size is None:
    shuffle_buffer_size = constants.DEFAULT_SHUFFLE_BUFFER_SIZE

  def preprocess_fn(dataset):
    to_ids = build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=sequence_length,
        num_out_of_vocab_buckets=num_out_of_vocab_buckets,
    )
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    if preprocess_spec.num_epochs > 1:
      dataset = dataset.repeat(preprocess_spec.num_epochs)
    if preprocess_spec.max_elements is not None:
      dataset = dataset.take(preprocess_spec.max_elements)
    dataset = dataset.map(to_ids, num_parallel_calls=num_parallel_calls)
    return batch_and_split(dataset, sequence_length, preprocess_spec.batch_size)

  return preprocess_fn
