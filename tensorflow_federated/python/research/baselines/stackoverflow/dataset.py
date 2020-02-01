# Lint as: python3
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

import tensorflow as tf
import tensorflow_federated as tff

EVAL_BATCH_SIZE = 100
DEFAULT_SHUFFLE_BUFFER_SIZE = 10000


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

  table_values = tf.constant(range(len(vocab)), dtype=tf.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=1)

  def to_ids(example):
    sentence = tf.reshape(example['tokens'], shape=[1])
    words = tf.strings.split(sentence, sep=' ').values
    truncated_words = words[:max_seq_len]
    tokens = table.lookup(truncated_words) + 1  # Plus one for pad token.
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_seq_len),
        lambda: tf.concat([tokens, [eos]], 0),
        lambda: tokens)
    return tf.concat([[bos], tokens], 0)

  return to_ids


def batch_and_split(dataset, max_seq_len, batch_size):
  # Shape out to (max_seq_len+1) because split shortens by one.
  return (dataset.padded_batch(batch_size, padded_shapes=[max_seq_len + 1]).map(
      split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE))


def construct_word_level_datasets(vocab_size, client_batch_size,
                                  client_epochs_per_round, max_seq_len,
                                  max_elements_per_user, centralized_train,
                                  shuffle_buffer_size, num_validation_examples,
                                  num_test_examples):
  """Preprocessing for Stackoverflow data.

  Notice that this preprocessing function *ignores* the heldout Stackoverflow
  dataset for consistency with the other datasets in the proposed optimization
  paper, and returns a validation/test split of the Stackoverflow "test" data,
  containing more examples from users in the Stackoverflow train dataset.

  Args:
    vocab_size: Integer representing size of the vocab to use. Vocabulary will
      then be the `vocab_size` most frequent words in the Stackoverflow dataset.
    client_batch_size: Integer representing batch size to use.
    client_epochs_per_round: Number of epochs for which to repeat train client
      dataset (only used if centralized_train=False).
    max_seq_len: Integer determining shape of padded batches. Sequences will be
      padded up to this length, and sentences longer than `max_seq_len` will be
      truncated to this length.
    max_elements_per_user: Integer controlling the maximum number of elements to
      take per user. If -1, takes all elements for each user.
    centralized_train: If True, returns a single `tf.data.Dataset` with shuffled
      training elements. If False, returns a `tff.simulation.ClientData` for the
      training set.
    shuffle_buffer_size: Buffer size for shuffling training dataset before
      batching. If None, does not shuffle.
    num_validation_examples: Number of examples from Stackoverflow test set to
      use for validation on each round.
    num_test_examples: Number of examples from Stackoverflow test set to use for
      testing after the final round.

  Returns:
    train: Depending on `centralized_train`, either an instance of
      `tf.data.Dataset` or a `tff.simulation.ClientData` representing
      Stackoverflow data for training.
    validation: A split of the Stackoverflow Test data as outlined
      in `tff.simulation.datasets.stackoverflow`, containing at most
      `num_validation_examples` examples.
    test: A split of the same Stackoverflow Test data containing at most
      `num_test_examples` of the Stackoverflow Test examples not used in
      `stackoverflow_validation`.
  """

  if vocab_size <= 0:
    raise ValueError('vocab_size must be a positive integer')

  if client_batch_size <= 0:
    raise ValueError('client_batch_size must be a positive integer')

  if client_epochs_per_round <= 0:
    raise ValueError('client_epochs_per_round must be a positive integer')

  if max_seq_len <= 0:
    raise ValueError('max_seq_len must be a positive integer')

  if max_elements_per_user < -1:
    raise ValueError(
        'max_elements_per_user must be an integer at least -1')

  if not shuffle_buffer_size:
    shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE
  elif shuffle_buffer_size < 1:
    raise ValueError('shuffle_buffer_size must be an integer greater than 1.')

  if num_validation_examples <= 1:
    raise ValueError('num_validation_examples must be an integer at least 1')

  if num_test_examples is not None and num_test_examples <= 1:
    raise ValueError('num_test_examples must be an integer at least 1')

  # Ignoring held-out Stackoverflow users for consistency with other datasets in
  # optimization paper.
  (train, _, raw_test) = tff.simulation.datasets.stackoverflow.load_data()

  vocab = get_vocab(vocab_size)

  to_ids = build_to_ids_fn(vocab, max_seq_len)

  if centralized_train:

    def preprocess_train(dataset):
      return (dataset.take(max_elements_per_user).map(
          to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE))

    train = (
        train.preprocess(preprocess_train).create_tf_dataset_from_all_clients()
        .shuffle(shuffle_buffer_size))
    train = batch_and_split(train, max_seq_len, client_batch_size)
  else:

    def preprocess_train(dataset):
      dataset = (
          dataset.take(max_elements_per_user).shuffle(
              shuffle_buffer_size).repeat(client_epochs_per_round).map(
                  to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE))
      return batch_and_split(dataset, max_seq_len, client_batch_size)

    train = train.preprocess(preprocess_train)

  def preprocess_test(dataset):
    return (dataset.take(max_elements_per_user).map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE))

  raw_test = (
      raw_test.preprocess(preprocess_test).create_tf_dataset_from_all_clients())

  validation = batch_and_split(
      raw_test.take(num_validation_examples), max_seq_len, EVAL_BATCH_SIZE)

  test = batch_and_split(
      raw_test.skip(num_validation_examples).take(num_test_examples),
      max_seq_len, EVAL_BATCH_SIZE)

  return train, validation, test

def get_vocab(vocab_size):
  """Gets the vocab items from the Stackoverlow dataset as a list.

  Args:
    vocab_size: Integer representing size of the vocab to use. Vocabulary will
      then be the `vocab_size` most frequent words in the Stackoverflow dataset.

  Returns:
    vocab: A list of vocab elements of size `vocab_size`.
  """
  if vocab_size <= 0:
    raise ValueError('vocab_size must be a positive integer')

  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
  vocab = list(vocab_dict.keys())[:vocab_size]

  return vocab

def get_special_tokens(vocab_size):
  """Gets the ids of the four special tokens.

  The four special tokens are:
    pad: padding token
    oov: out of vocabulary
    bos: begin of sentence
    eos: end of sentence

  Args:
    vocab_size: The vocabulary size.

  Returns:
    The four-tuple (pad, oov, bos, eos).
  """
  pad = 0
  oov = vocab_size + 1
  bos = vocab_size + 2
  eos = vocab_size + 3

  return pad, oov, bos, eos
