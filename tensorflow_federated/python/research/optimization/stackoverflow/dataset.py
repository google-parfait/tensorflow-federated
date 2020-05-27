# Copyright 2019, The TensorFlow Federated Authors.
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
"""Data loader for Stackoverflow."""
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

EVAL_BATCH_SIZE = 100


def create_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common words in Stackoverflow."""
  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
  return list(vocab_dict.keys())[:vocab_size]


def split_input_target(chunk):
  """Generate input and target data.

  The task of language model is to predict the next word.

  Args:
    chunk: A Tensor of text data.

  Returns:
    A namedtuple of input and target data.
  """
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


def build_to_ids_fn(vocab, max_seq_len):
  """Constructs function mapping examples to sequences of token indices."""
  _, _, bos, eos = get_special_tokens(len(vocab))

  table_values = np.arange(len(vocab), dtype=np.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=1)

  def to_ids(example):
    sentence = tf.reshape(example['tokens'], shape=[1])
    words = tf.strings.split(sentence, sep=' ').values
    truncated_words = words[:max_seq_len]
    tokens = table.lookup(truncated_words) + 1
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_seq_len),
        lambda: tf.concat([tokens, [eos]], 0), lambda: tokens)

    return tf.concat([[bos], tokens], 0)

  return to_ids


def batch_and_split(dataset, max_seq_len, batch_size):
  return dataset.padded_batch(
      batch_size, padded_shapes=[max_seq_len + 1]).map(
          split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_special_tokens(vocab_size):
  """Gets tokens dataset preprocessing code will add to Stackoverflow."""
  pad = 0
  oov = vocab_size + 1
  bos = vocab_size + 2
  eos = vocab_size + 3
  return pad, oov, bos, eos


def create_train_dataset_preprocess_fn(vocab: List[str],
                                       client_batch_size: int,
                                       client_epochs_per_round: int,
                                       max_seq_len: int,
                                       max_training_elements_per_user: int,
                                       max_shuffle_buffer_size=10000):
  """Creates preprocessing functions for stackoverflow data.

  This function returns a function which takes a dataset and returns a dataset,
  generally for mapping over a set of unprocessed client datasets during
  training.

  Args:
    vocab: Vocabulary which defines the embedding.
    client_batch_size: Integer representing batch size to use on the clients.
    client_epochs_per_round: Number of epochs for which to repeat train client
      dataset.
    max_seq_len: Integer determining shape of padded batches. Sequences will be
      padded up to this length, and sentences longer than `max_seq_len` will be
      truncated to this length.
    max_training_elements_per_user: Integer controlling the maximum number of
      elements to take per user. If -1, takes all elements for each user.
    max_shuffle_buffer_size: Maximum shuffle buffer size.

  Returns:
    Two functions, the first `preprocess_train` and the second
    `preprocess_val_and_test`, as described above.
  """
  if client_batch_size <= 0:
    raise ValueError('client_batch_size must be a positive integer; you have '
                     'passed {}'.format(client_batch_size))
  elif client_epochs_per_round <= 0:
    raise ValueError('client_epochs_per_round must be a positive integer; you '
                     'have passed {}'.format(client_epochs_per_round))
  elif max_seq_len <= 0:
    raise ValueError('max_seq_len must be a positive integer; you have '
                     'passed {}'.format(max_seq_len))
  elif max_training_elements_per_user < -1:
    raise ValueError(
        'max_training_elements_per_user must be an integer at '
        'least -1; you have passed {}'.format(max_training_elements_per_user))

  if (max_training_elements_per_user == -1 or
      max_training_elements_per_user > max_shuffle_buffer_size):
    shuffle_buffer_size = max_shuffle_buffer_size
  else:
    shuffle_buffer_size = max_training_elements_per_user

  # TODO(b/155408842): need further investigation on why `tff.tf_compuation`
  # decorator causes b/153363900 for `to_ids`, and large memory consumption.
  def preprocess_train(dataset):
    to_ids = build_to_ids_fn(vocab, max_seq_len)
    dataset = dataset.take(max_training_elements_per_user)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(client_epochs_per_round)
    dataset = dataset.map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return batch_and_split(dataset, max_seq_len, client_batch_size)

  return preprocess_train


def create_test_dataset_preprocess_fn(vocab: List[str], max_seq_len: int):
  """Creates preprocessing functions for stackoverflow data.

  This function returns a function which represents preprocessing logic
  for use on centralized validation and test datasets outside of TFF.

  Args:
    vocab: Vocabulary which defines the embedding.
    max_seq_len: Integer determining shape of padded batches. Sequences will be
      padded up to this length, and sentences longer than `max_seq_len` will be
      truncated to this length.

  Returns:
    `preprocess_val_and_test`, as described above.
  """
  if max_seq_len <= 0:
    raise ValueError('max_seq_len must be a positive integer; you have '
                     'passed {}'.format(max_seq_len))

  def preprocess_val_and_test(dataset):
    to_ids = build_to_ids_fn(vocab, max_seq_len)
    id_dataset = dataset.map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return batch_and_split(id_dataset, max_seq_len, EVAL_BATCH_SIZE)

  return preprocess_val_and_test


def construct_word_level_datasets(vocab_size: int,
                                  client_batch_size: int,
                                  client_epochs_per_round: int,
                                  max_seq_len: int,
                                  max_training_elements_per_user: int,
                                  num_validation_examples: int,
                                  max_shuffle_buffer_size=10000):
  """Preprocessing for Stackoverflow data.

  Notice that this preprocessing function *ignores* the heldout Stackoverflow
  dataset for consistency with the other datasets in the proposed optimization
  paper, and returns a validation/test split of the Stackoverflow "test" data,
  containing more examples from users in the Stackoverflow train dataset.

  Args:
    vocab_size: Integer representing size of the vocab to use. Vocabulary will
      then be the `vocab_size` most frequent words in the Stackoverflow dataset.
    client_batch_size: Integer representing batch size to use on the clients.
    client_epochs_per_round: Number of epochs for which to repeat train client
      dataset.
    max_seq_len: Integer determining shape of padded batches. Sequences will be
      padded up to this length, and sentences longer than `max_seq_len` will be
      truncated to this length.
    max_training_elements_per_user: Integer controlling the maximum number of
      elements to take per user. If -1, takes all elements for each user.
    num_validation_examples: Number of examples from Stackoverflow test set to
      use for validation on each round.
    max_shuffle_buffer_size: Maximum shuffle buffer size.

  Returns:
    stackoverflow_train: An instance of `tff.simulation.ClientData`
      representing Stackoverflow data for training.
    stackoverflow_validation: A split of the Stackoverflow Test data as outlined
      in `tff.simulation.datasets.stackoverflow`, containing at most
      `num_validation_examples` examples.
    stackoverflow_test: A split of the same Stackoverflow Test data containing
    the examples not used in `stackoverflow_validation`.
  """
  if num_validation_examples < 1:
    raise ValueError(
        'num_validation_examples must be an integer at '
        'least 1; you have passed {}'.format(num_validation_examples))
  elif vocab_size <= 0:
    raise ValueError('vocab_size must be a positive integer; you have '
                     'passed {}'.format(vocab_size))

  (stackoverflow_train, _,
   stackoverflow_test) = tff.simulation.datasets.stackoverflow.load_data()

  vocab = create_vocab(vocab_size)

  raw_test_dataset = stackoverflow_test.create_tf_dataset_from_all_clients()

  preprocess_train = create_train_dataset_preprocess_fn(
      vocab, client_batch_size, client_epochs_per_round, max_seq_len,
      max_training_elements_per_user, max_shuffle_buffer_size)

  preprocess_val_and_test = create_test_dataset_preprocess_fn(
      vocab, max_seq_len)

  stackoverflow_train = stackoverflow_train.preprocess(preprocess_train)
  stackoverflow_val = preprocess_val_and_test(
      raw_test_dataset.take(num_validation_examples))
  stackoverflow_test = preprocess_val_and_test(
      raw_test_dataset.skip(num_validation_examples))

  return stackoverflow_train, stackoverflow_val, stackoverflow_test


def get_centralized_train_dataset(vocab_size: int,
                                  batch_size: int,
                                  max_seq_len: int,
                                  shuffle_buffer_size: int = 10000):
  """Creates centralized approximately shuffled train dataset."""

  vocab = create_vocab(vocab_size)
  to_ids = build_to_ids_fn(vocab, max_seq_len)
  train, _, _ = tff.simulation.datasets.stackoverflow.load_data()

  train = train.create_tf_dataset_from_all_clients()
  train = train.shuffle(buffer_size=shuffle_buffer_size)
  return batch_and_split(
      train.map(to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE),
      max_seq_len, batch_size)
