# Copyright 2020, The TensorFlow Federated Authors.
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

from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

TEST_BATCH_SIZE = 500


def create_token_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common words in Stackoverflow."""
  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
  return list(vocab_dict.keys())[:vocab_size]


def create_tag_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common tags in Stackoverflow."""
  tag_dict = tff.simulation.datasets.stackoverflow.load_tag_counts()
  return list(tag_dict.keys())[:vocab_size]


def build_to_ids_fn(vocab_tokens, vocab_tags):
  """Constructs function mapping examples to sequences of token indices."""
  vocab_tokens_size = len(vocab_tokens)
  table_values = tf.constant(range(vocab_tokens_size), dtype=tf.int64)
  table_tokens = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab_tokens, table_values),
      num_oov_buckets=1)

  vocab_tags_size = len(vocab_tags)
  table_values = tf.constant(range(vocab_tags_size), dtype=tf.int64)
  table_tags = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab_tags, table_values),
      num_oov_buckets=1)

  def to_ids(example):
    """Converts tf example to bag of words."""
    sentence = tf.strings.join([example['tokens'], example['title']],
                               separator=' ')
    words = tf.strings.split(sentence)
    tokens = table_tokens.lookup(words)
    tokens = tf.one_hot(tokens, vocab_tokens_size+1)
    tokens = tf.reduce_mean(tokens, axis=0)[:vocab_tokens_size]

    tags = example['tags']
    tags = tf.strings.split(tags, sep='|')
    tags = table_tags.lookup(tags)
    tags = tf.one_hot(tags, vocab_tags_size+1)
    tags = tf.reduce_sum(tags, axis=0)[:vocab_tags_size]

    return (tokens, tags)

  return to_ids


def get_stackoverflow_datasets(
    vocab_tokens_size=10000,
    vocab_tags_size=500,
    max_training_elements_per_user=500,
    client_batch_size=100,
    client_epochs_per_round=1,
    max_batches_per_user=-1,
    num_validation_examples=10000,
):
  """Preprocessing for Stackoverflow data.

  Notice that this preprocessing function *ignores* the heldout Stackoverflow
  dataset for consistency with the other datasets in the proposed optimization
  paper, and returns a validation/test split of the Stackoverflow "test" data,
  containing more examples from users in the Stackoverflow train dataset.

  Args:
    vocab_tokens_size: Integer representing size of the token vocab to use.
      Vocabulary will then be the `vocab_tokens_size` most frequent words
      in the Stackoverflow dataset.
    vocab_tags_size: Integer representing size of the tags vocab to use.
      Vocabulary will then be the `vocab_tags_size` most frequent tags
      in the Stackoverflow dataset.
    max_training_elements_per_user: Integer controlling the maximum number of
      elements to take per user. If -1, takes all elements for each user.
    client_batch_size: Integer representing the client batch size.
    client_epochs_per_round: Number of client epochs per round
    max_batches_per_user: If set to a positive integer, the maximum number of
      batches in each client's dataset.
    num_validation_examples: Number of elements to use for validation

  Returns:
    stackoverflow_train: An instance of `tff.simulation.ClientData`
      representing Stackoverflow data for training.
    stackoverflow_test: A split of the Stackoverflow data containing
      the test examples.
  """
  if vocab_tokens_size <= 0:
    raise ValueError('vocab_tokens_size must be a positive integer; you have '
                     'passed {}'.format(vocab_tokens_size))
  elif vocab_tags_size <= 0:
    raise ValueError('vocab_tags_size must be a positive integer; you have '
                     'have passed {}'.format(vocab_tokens_size))
  elif max_training_elements_per_user < -1:
    raise ValueError(
        'max_training_elements_per_user must be an integer at '
        'least -1; you have passed {}'.format(max_training_elements_per_user))
  elif client_epochs_per_round == -1 and max_batches_per_user == -1:
    raise ValueError('Argument client_epochs_per_round is set to -1. If this is'
                     ' intended, then max_batches_per_user must be set to '
                     'some positive integer.')

  # Ignoring held-out Stackoverflow users for consistency with other
  # StackOverflow experiments.
  stackoverflow_train, _, stackoverflow_test = tff.simulation.datasets.stackoverflow.load_data(
  )

  vocab_tokens = create_token_vocab(vocab_tokens_size)
  vocab_tags = create_tag_vocab(vocab_tags_size)
  to_ids = build_to_ids_fn(vocab_tokens, vocab_tags)

  def preprocess_train_dataset(dataset):
    """Preprocess StackOverflow training dataset."""
    return (dataset
            # Take up to a max number of training elements
            .take(max_training_elements_per_user)
            # Shuffle the client datasets
            .shuffle(buffer_size=max_training_elements_per_user)
            # Repeat for multiple local client epochs
            .repeat(client_epochs_per_round)
            # Map sentences to bag of words
            .map(to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Batch
            .batch(client_batch_size)
            # Take a maximum number of batches
            .take(max_batches_per_user))

  def preprocess_test_dataset(dataset):
    """Preprocess StackOverflow testing dataset."""
    return dataset.map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            TEST_BATCH_SIZE, drop_remainder=False)

  stackoverflow_train = stackoverflow_train.preprocess(preprocess_train_dataset)

  raw_test_dataset = stackoverflow_test.create_tf_dataset_from_all_clients()

  stackoverflow_validation = preprocess_test_dataset(
      raw_test_dataset.take(num_validation_examples)).cache()

  stackoverflow_test = preprocess_test_dataset(
      raw_test_dataset.skip(num_validation_examples))

  return stackoverflow_train, stackoverflow_validation, stackoverflow_test


def get_centralized_datasets(train_batch_size: int,
                             validation_batch_size: Optional[int] = 500,
                             test_batch_size: Optional[int] = 500,
                             max_train_batches: Optional[int] = None,
                             max_validation_batches: Optional[int] = None,
                             max_test_batches: Optional[int] = None,
                             vocab_tokens_size=10000,
                             vocab_tags_size=500,
                             num_validation_examples=10000,
                             shuffle_buffer_size=10000):
  """Loads centralized StackOverflow training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    validation_batch_size: The batch size for the validation dataset.
    test_batch_size: The batch size for the test dataset.
    max_train_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the training dataset.
    max_validation_batches: If set to a positive integer, this specifies the
      maximum number of batches to use from the validation dataset.
    max_test_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the test dataset.
    vocab_tokens_size: Integer representing size of the vocab to use. Vocabulary
      will be the `vocab_token_size` most frequent words.
    vocab_tags_size: Integer representing the number of tags to use. The tag
      labels will be the `vocab_tag_size` most frequent tags.
    num_validation_examples: Number of examples from Stackoverflow test set to
      use for validation on each round.
    shuffle_buffer_size: The shuffle buffer size for the training dataset. If
      set to nonpositive number, no shuffling occurs.

  Returns:
    train_dataset: A `tf.data.Dataset` instance representing the training
      dataset.
    validation_dataset: A `tf.data.Dataset` instance representing the validation
      dataset.
    test_dataset: A `tf.data.Dataset` instance representing the test dataset.
  """

  # Ignoring held-out Stackoverflow users for consistency with other datasets in
  # optimization paper.
  stackoverflow_train, _, stackoverflow_test = tff.simulation.datasets.stackoverflow.load_data(
  )

  vocab_tokens = create_token_vocab(vocab_tokens_size)
  vocab_tags = create_tag_vocab(vocab_tags_size)
  to_ids = build_to_ids_fn(vocab_tokens, vocab_tags)

  def preprocess(dataset, batch_size, buffer_size=10000, shuffle_data=True):
    if shuffle_data:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    return (dataset.map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            batch_size))

  train_dataset = preprocess(
      stackoverflow_train.create_tf_dataset_from_all_clients(),
      train_batch_size,
      shuffle_buffer_size,
      shuffle_data=True)
  validation_dataset = preprocess(
      stackoverflow_test.create_tf_dataset_from_all_clients().take(
          num_validation_examples),
      validation_batch_size,
      shuffle_data=False)
  test_dataset = preprocess(
      stackoverflow_test.create_tf_dataset_from_all_clients().skip(
          num_validation_examples),
      test_batch_size,
      shuffle_data=False)

  if max_train_batches is not None and max_train_batches > 0:
    train_dataset = train_dataset.take(max_train_batches)
  if max_validation_batches is not None and max_validation_batches > 0:
    validation_dataset = validation_dataset.take(max_validation_batches)
  if max_test_batches is not None and max_test_batches > 0:
    test_dataset = test_dataset.take(max_test_batches)

  return train_dataset, validation_dataset, test_dataset
