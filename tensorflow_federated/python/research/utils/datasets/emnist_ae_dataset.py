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
"""Library for loading and preprocessing EMNIST autoencoder data."""

from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

EMNIST_TRAIN_DIGITS_ONLY_SIZE = 341873
EMNIST_TRAIN_FULL_SIZE = 671585
TEST_BATCH_SIZE = 500
MAX_CLIENT_DATASET_SIZE = 418


def reshape_emnist_element(element):
  x = 1 - tf.reshape(element['pixels'], (-1, 28 * 28))
  return (x, x)


def get_emnist_datasets(client_batch_size,
                        client_epochs_per_round,
                        max_batches_per_client=-1,
                        only_digits=False):
  """Loads and preprocesses EMNIST training and testing sets.

  Args:
    client_batch_size: Integer representing the batch size on the clients.
    client_epochs_per_round: Integer representing the number of epochs for which
      each client should perform training. This is done by repeating the
      dataset. If set to -1, the dataset is repeated indefinitely. In this case,
      the `max_batches_per_client` argument should be set to some positive
      integer, to ensure finite training time.
    max_batches_per_client: The maximum number of batches (of size
      `client_batch_size`) in the client dataset. This is enforced by using
      `tf.data.Dataset.take`. If set to -1 (the default value), then no maximum
      number of batches is enforced.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.

  Returns:
    emnist_train: An instance of a `tff.simulation.ClientData` representing the
      training data.
    emnist_test: An instance of a `tf.data.Dataset` representing the testing
      data.
  """

  if client_epochs_per_round == -1 and max_batches_per_client == -1:
    raise ValueError('Argument client_epochs_per_round is set to -1. If this is'
                     ' intended, then max_batches_per_client must be set to '
                     'some positive integer.')

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits)

  def preprocess_train_dataset(dataset):
    """Preprocess EMNIST training dataset."""
    return (dataset
            # Shuffle according to the largest client dataset
            .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)
            # Repeat to do multiple local epochs
            .repeat(client_epochs_per_round)
            # Batch to a fixed client batch size
            .batch(client_batch_size, drop_remainder=False)
            # Take a maximum number of batches
            .take(max_batches_per_client)
            # Preprocessing step
            .map(
                reshape_emnist_element,
                num_parallel_calls=tf.data.experimental.AUTOTUNE))

  def preprocess_test_dataset(dataset):
    """Preprocess EMNIST testing dataset."""
    return (dataset.batch(TEST_BATCH_SIZE, drop_remainder=False).map(
        reshape_emnist_element,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache())

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients()).cache()
  return emnist_train, emnist_test


def get_centralized_datasets(train_batch_size: int,
                             test_batch_size: Optional[int] = 500,
                             max_train_batches: Optional[int] = None,
                             max_test_batches: Optional[int] = None,
                             only_digits: Optional[bool] = False,
                             shuffle_train: Optional[bool] = True):
  """Loads and preprocesses centralized EMNIST autoencoder datasets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    max_train_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the training dataset.
    max_test_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the test dataset.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.
    shuffle_train: A boolean indicating whether to shuffle the centralized train
      dataset.

  Returns:
    train_dataset: A `tf.data.Dataset` instance representing the training
      dataset.
    test_dataset: A `tf.data.Dataset` instance representing the test dataset.
  """
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits)

  def preprocess(dataset, batch_size, buffer_size=10000, shuffle_data=True):
    if shuffle_data:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    return (dataset.batch(batch_size).map(
        reshape_emnist_element,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache())

  train_dataset = preprocess(
      emnist_train.create_tf_dataset_from_all_clients(),
      train_batch_size,
      shuffle_data=shuffle_train)
  test_dataset = preprocess(
      emnist_test.create_tf_dataset_from_all_clients(),
      test_batch_size,
      shuffle_data=False)

  if max_train_batches is not None and max_train_batches > 0:
    train_dataset = train_dataset.take(max_train_batches)
  if max_test_batches is not None and max_test_batches > 0:
    test_dataset = test_dataset.take(max_test_batches)

  return train_dataset, test_dataset
