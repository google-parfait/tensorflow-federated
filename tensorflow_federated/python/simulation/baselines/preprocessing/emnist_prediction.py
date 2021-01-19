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
"""Preprocessing library for EMNIST prediction tasks."""

import collections
from typing import Tuple

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.simulation.datasets import emnist

MAX_CLIENT_DATASET_SIZE = 418


def _reshape_for_digit_recognition(element):
  return tf.expand_dims(element['pixels'], axis=-1), element['label']


def _reshape_for_autoencoder(element):
  x = 1 - tf.reshape(element['pixels'], (-1, 28 * 28))
  return (x, x)


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    shuffle_buffer_size: int = MAX_CLIENT_DATASET_SIZE,
    emnist_task: str = 'digit_recognition',
    num_parallel_calls: tf.Tensor = tf.data.experimental.AUTOTUNE
) -> computation_base.Computation:
  """Creates a preprocessing function for EMNIST client datasets.

  The preprocessing shuffles, repeats, batches, and then reshapes, using
  the `shuffle`, `repeat`, `batch`, and `map` attributes of a
  `tf.data.Dataset`, in that order.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    shuffle_buffer_size: An integer representing the shuffle buffer size on
      clients. If set to a number <= 1, no shuffling occurs.
    emnist_task: A string indicating the EMNIST task being performed. Must be
      one of 'digit_recognition' or 'autoencoder'. If the former, then elements
      are mapped to tuples of the form (pixels, label), if the latter then
      elements are mapped to tuples of the form (pixels, pixels).
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A `tff.Computation` performing the preprocessing discussed above.
  """
  if num_epochs < 1:
    raise ValueError('num_epochs must be a positive integer.')

  if emnist_task == 'digit_recognition':
    mapping_fn = _reshape_for_digit_recognition
  elif emnist_task == 'autoencoder':
    mapping_fn = _reshape_for_autoencoder
  else:
    raise ValueError('emnist_task must be one of "digit_recognition" or '
                     '"autoencoder".')

  feature_dtypes = collections.OrderedDict(
      pixels=computation_types.TensorType(tf.float32, shape=(28, 28)),
      label=computation_types.TensorType(tf.int32))

  @computations.tf_computation(computation_types.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.repeat(num_epochs).batch(
        batch_size, drop_remainder=False).map(
            mapping_fn, num_parallel_calls=num_parallel_calls)

  return preprocess_fn


def get_federated_datasets(
    train_client_batch_size: int = 20,
    test_client_batch_size: int = 100,
    train_client_epochs_per_round: int = 1,
    test_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: int = MAX_CLIENT_DATASET_SIZE,
    test_shuffle_buffer_size: int = 1,
    only_digits: bool = False,
    emnist_task: str = 'digit_recognition'
) -> Tuple[client_data.ClientData, client_data.ClientData]:
  """Loads and preprocesses federated EMNIST training and testing sets.

  Args:
    train_client_batch_size: The batch size for all train clients.
    test_client_batch_size: The batch size for all test clients.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    test_client_epochs_per_round: The number of epochs each test client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each test client's dataset. If set
      to some integer less than or equal to 1, no shuffling occurs.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.
    emnist_task: A string indicating the EMNIST task being performed. Must be
      one of 'digit_recognition' or 'autoencoder'. If the former, then elements
      are mapped to tuples of the form (pixels, label), if the latter then
      elements are mapped to tuples of the form (pixels, pixels).

  Returns:
    A tuple (emnist_train, emnist_test) of `ClientData` instances
      representing the federated training and test datasets.
  """
  if train_client_epochs_per_round < 1:
    raise ValueError(
        'train_client_epochs_per_round must be a positive integer.')
  if test_client_epochs_per_round < 0:
    raise ValueError('test_client_epochs_per_round must be a positive integer.')

  emnist_train, emnist_test = emnist.load_data(only_digits=only_digits)

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      emnist_task=emnist_task)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=test_client_epochs_per_round,
      batch_size=test_client_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      emnist_task=emnist_task)

  emnist_train = emnist_train.preprocess(train_preprocess_fn)
  emnist_test = emnist_test.preprocess(test_preprocess_fn)
  return emnist_train, emnist_test


def get_centralized_datasets(
    train_batch_size: int = 20,
    test_batch_size: int = 500,
    train_shuffle_buffer_size: int = 10000,
    test_shuffle_buffer_size: int = 1,
    only_digits: bool = False,
    emnist_task: str = 'digit_recognition'
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Loads and preprocesses centralized EMNIST training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.
    emnist_task: A string indicating the EMNIST task being performed. Must be
      one of 'digit_recognition' or 'autoencoder'. If the former, then elements
      are mapped to tuples of the form (pixels, label), if the latter then
      elements are mapped to tuples of the form (pixels, pixels).

  Returns:
    A tuple (train_dataset, test_dataset) of `tf.data.Dataset` instances
    representing the centralized training and test datasets.
  """
  emnist_train, emnist_test = emnist.load_data(only_digits=only_digits)

  emnist_train = emnist_train.create_tf_dataset_from_all_clients()
  emnist_test = emnist_test.create_tf_dataset_from_all_clients()

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=train_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      emnist_task=emnist_task)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=test_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      emnist_task=emnist_task)

  emnist_train = train_preprocess_fn(emnist_train)
  emnist_test = test_preprocess_fn(emnist_test)

  return emnist_train, emnist_test
