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
"""Libraries to prepare Shakespeare datasets for CharRNN experiments."""

import functools
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_federated as tff

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)
EVAL_BATCH_SIZE = 10


def get_special_tokens():
  """Gets tokens dataset preprocessing code will add to Shakespeare.

  Returns:
    A tuple of the four special characters, (pad, oov, bos, eos).

  """
  vocab_size = len(CHAR_VOCAB)
  pad = 0
  oov = vocab_size + 1
  bos = vocab_size + 2
  eos = vocab_size + 3
  return pad, oov, bos, eos


def _build_tokenize_fn(split_length=SEQUENCE_LENGTH + 1):
  """Create a tf.function that converts a Shakespeare example to character ids.

  The function converts each example to its corresponding character ids. It then
  pads the sequence until its length is a multiple of split_length.

  Args:
    split_length: An integer used to determine the padding length for a given
      snippet. The tf.function pads until the sequence is of length divisible by
      split_length. This function is intended to be used in combination with
      something such as batching, in order to create token sequences of length
      split_length.

  Returns:
    A `tf.function`.
  """
  _, _, bos, eos = get_special_tokens()

  ids = tf.range(len(CHAR_VOCAB), dtype=tf.int64)
  lookup_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(CHAR_VOCAB, ids), num_oov_buckets=1)

  @tf.function
  def _to_tokens_and_pad(example: tf.Tensor) -> tf.Tensor:
    """Convert a Shakespeare example to a int64 tensor of token ids, and pad."""
    chars = tf.strings.bytes_split(example['snippets'])
    tokens = lookup_table.lookup(keys=chars) + 1  # Reserve 0 for pad.
    tokens = tf.concat([[bos], tokens, [eos]], 0)
    pad_length = (-tf.shape(tokens)[0]) % split_length
    return tf.concat([tokens, tf.zeros(pad_length, dtype=tf.int64)], 0)

  return _to_tokens_and_pad


@tf.function
def _split_target(sequence_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Split a N + 1 sequence into shifted-by-1 sequences for input and output."""
  input_text = tf.map_fn(lambda x: x[:-1], sequence_batch)
  target_text = tf.map_fn(lambda x: x[1:], sequence_batch)
  return (input_text, target_text)


def convert_snippets_to_character_sequence_examples(
    dataset: tf.data.Dataset,
    batch_size: int,
    epochs: int,
    shuffle_buffer_size: int = 50,
    sequence_length: int = SEQUENCE_LENGTH,
    max_batches_per_client: int = -1) -> tf.data.Dataset:
  """Convert a dataset of string snippets to a dataset of input/output character ID sequences.

  Args:
    dataset: the `tf.data.Dataset` to apply preprocessing to.
    batch_size: the number of examples per yielded batch
    epochs: the number of times to repeat the dataset in one epoch.
    shuffle_buffer_size: Buffer size for shuffling the dataset. If nonpositive,
      no shuffling occurs.
    sequence_length: the length of each example in the batch.
    max_batches_per_client: If set to a positive integer, the maximum number of
      batches in each client's dataset.

  Returns:
    A `tf.data.Dataset` yielding `(sequence of character IDs, sequence of
    character IDs)` where each sequence has `sequence_length` values.
  """
  to_tokens = _build_tokenize_fn(split_length=sequence_length + 1)
  dataset = dataset.repeat(epochs)
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  return (
      # Convert snippets to int64 tokens and pad.
      dataset.map(to_tokens, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Separate into individual tokens
      .unbatch()
      # Join into sequences of the desired length. The previous call of
      # map(to_ids,...) ensures that the collection of tokens has length
      # divisible by sequence_length + 1, so no batch dropping is expected.
      .batch(sequence_length + 1, drop_remainder=True)
      # Batch sequences together for mini-batching purposes.
      .batch(batch_size)
      # Convert batches into training examples.
      .map(_split_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Take a maximum number of batches
      .take(max_batches_per_client))


def construct_character_level_datasets(client_batch_size: int,
                                       client_epochs_per_round: int,
                                       sequence_length: int = SEQUENCE_LENGTH,
                                       max_batches_per_client: int = -1,
                                       shuffle_buffer_size: int = 50):
  """Loads and preprocesses a federated Shakespeare training dataset."""

  if client_epochs_per_round == -1 and max_batches_per_client == -1:
    raise ValueError('Argument client_epochs_per_round is set to -1. If this is'
                     ' intended, then max_batches_per_client must be set to '
                     'some positive integer.')

  train_client_data, _ = (tff.simulation.datasets.shakespeare.load_data())

  preprocessed_train_client_data = train_client_data.preprocess(
      functools.partial(
          convert_snippets_to_character_sequence_examples,
          batch_size=client_batch_size,
          epochs=client_epochs_per_round,
          shuffle_buffer_size=shuffle_buffer_size,
          sequence_length=sequence_length,
          max_batches_per_client=max_batches_per_client))

  return preprocessed_train_client_data


def get_centralized_datasets(train_batch_size: int,
                             test_batch_size: Optional[int] = 100,
                             max_train_batches: Optional[int] = None,
                             max_test_batches: Optional[int] = None,
                             sequence_length: Optional[int] = SEQUENCE_LENGTH,
                             shuffle_buffer_size: Optional[int] = 10000):
  """Loads and preprocesses centralized Shakespeare datasets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    max_train_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the training dataset.
    max_test_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the test dataset.
    sequence_length: The number of characters in each example.
    shuffle_buffer_size: The shuffle buffer size for the training dataset. If
      set to nonpositive number, no shuffling occurs.

  Returns:
    train_dataset: A `tf.data.Dataset` instance representing the training
      dataset.
    test_dataset: A `tf.data.Dataset` instance representing the test dataset.
  """
  train_client_data, test_client_data = (
      tff.simulation.datasets.shakespeare.load_data())

  train_dataset = convert_snippets_to_character_sequence_examples(
      train_client_data.create_tf_dataset_from_all_clients(),
      batch_size=train_batch_size,
      epochs=1,
      shuffle_buffer_size=shuffle_buffer_size,
      sequence_length=sequence_length)

  test_dataset = convert_snippets_to_character_sequence_examples(
      test_client_data.create_tf_dataset_from_all_clients(),
      batch_size=test_batch_size,
      epochs=1,
      shuffle_buffer_size=0,
      sequence_length=sequence_length)

  if max_train_batches is not None and max_train_batches > 0:
    train_dataset = train_dataset.take(max_train_batches)
  if max_test_batches is not None and max_test_batches > 0:
    test_dataset = test_dataset.take(max_test_batches)

  return train_dataset, test_dataset
