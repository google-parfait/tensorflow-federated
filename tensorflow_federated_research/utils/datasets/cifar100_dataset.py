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
"""Library for loading and preprocessing CIFAR-100 training and testing data."""

import collections
import functools
from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_federated as tff

CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
NUM_EXAMPLES_PER_CLIENT = 500
TEST_BATCH_SIZE = 100


def preprocess_cifar_example(example, crop_shape, distort=False):
  """Preprocesses a CIFAR-100 example by cropping, flipping, and normalizing."""
  image = tf.cast(example['image'], tf.float32)
  if distort:
    image = tf.image.random_crop(image, size=crop_shape)
    image = tf.image.random_flip_left_right(image)
  else:
    image = tf.image.resize_with_crop_or_pad(
        image, target_height=crop_shape[1], target_width=crop_shape[2])
  image = tf.image.per_image_standardization(image)
  return (image, example['label'])


def get_federated_cifar100(client_epochs_per_round,
                           train_batch_size,
                           crop_shape=CIFAR_SHAPE,
                           max_batches_per_client=-1,
                           serializable=False):
  """Loads and preprocesses federated CIFAR100 training and testing sets.

  Args:
    client_epochs_per_round: An integer specifying the number of local epochs
      performed per training round by each client. Used to replicate each client
      dataset an appropriate number of times.
    train_batch_size: The batch size for the training dataset.
    crop_shape: An iterable of integers specifying the desired crop
      shape for pre-processing. Must be convertable to a tuple of integers
      (CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS) which cannot have elements that
      exceed (32, 32, 3), element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.
    max_batches_per_client: If set to a positive integer, the maximum number of
      batches in each client's dataset.
    serializable: Boolean indicating whether the returned datasets are intended
      to be serialized and shipped across RPC channels. If `True`, stateful
      transformations will be disallowed.

  Returns:
    A tuple of `tff.simulation.ClientData` and `tf.data.Datset` objects.
  """
  if not isinstance(crop_shape, collections.Iterable):
    raise TypeError(
        'Argument crop_shape must be an iterable.')
  crop_shape = tuple(crop_shape)
  if len(crop_shape) != 3:
    raise ValueError('The crop_shape must have length 3, corresponding to a '
                     'tensor of shape [height, width, channels].')
  if not isinstance(serializable, bool):
    raise TypeError(
        'serializable must be a Boolean; you passed {} of type {}.'.format(
            serializable, type(serializable)))
  if client_epochs_per_round == -1 and max_batches_per_client == -1:
    raise ValueError('Argument client_epochs_per_round is set to -1. If this is'
                     ' intended, then max_batches_per_client must be set to '
                     'some positive integer.')

  cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
  train_crop_shape = (train_batch_size,) + crop_shape
  test_crop_shape = (TEST_BATCH_SIZE,) + crop_shape
  train_image_map = functools.partial(
      preprocess_cifar_example,
      crop_shape=train_crop_shape,
      distort=not serializable)
  test_image_map = functools.partial(
      preprocess_cifar_example, crop_shape=test_crop_shape, distort=False)

  def preprocess_train_dataset(dataset):
    """Preprocess CIFAR100 training dataset."""
    if not serializable:
      dataset = dataset.shuffle(buffer_size=NUM_EXAMPLES_PER_CLIENT)
    return dataset.repeat(client_epochs_per_round).batch(
        train_batch_size,
        drop_remainder=True).map(train_image_map).take(max_batches_per_client)

  def preprocess_test_dataset(dataset):
    """Preprocess CIFAR100 testing dataset."""
    return dataset.batch(
        TEST_BATCH_SIZE, drop_remainder=False).map(test_image_map)

  cifar_train = cifar_train.preprocess(preprocess_train_dataset)
  cifar_test = preprocess_test_dataset(
      cifar_test.create_tf_dataset_from_all_clients()).cache()
  return cifar_train, cifar_test


def get_centralized_datasets(train_batch_size: int,
                             test_batch_size: Optional[int] = 100,
                             max_train_batches: Optional[int] = None,
                             max_test_batches: Optional[int] = None,
                             crop_shape: Optional[Tuple[int, int,
                                                        int]] = CIFAR_SHAPE):
  """Loads and preprocesses centralized CIFAR100 training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    max_train_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the training dataset.
    max_test_batches: If set to a positive integer, this specifies the maximum
      number of batches to use from the test dataset.
    crop_shape: An iterable of integers specifying the desired crop
      shape for pre-processing. Must be convertable to a tuple of integers
      (CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS) which cannot have elements that
      exceed (32, 32, 3), element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.

  Returns:
    train_dataset: A `tf.data.Dataset` instance representing the training
      dataset.
    test_dataset: A `tf.data.Dataset` instance representing the test dataset.
  """
  try:
    crop_shape = tuple(crop_shape)
  except:
    raise ValueError(
        'Argument crop_shape must be able to coerced into a length 3 tuple.')
  if len(crop_shape) != 3:
    raise ValueError('The crop_shape must have length 3, corresponding to a '
                     'tensor of shape [height, width, channels].')
  cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()
  train_crop_shape = (train_batch_size,) + crop_shape
  test_crop_shape = (test_batch_size,) + crop_shape
  train_image_map = functools.partial(
      preprocess_cifar_example, crop_shape=train_crop_shape, distort=True)
  test_image_map = functools.partial(
      preprocess_cifar_example, crop_shape=test_crop_shape, distort=False)

  train_dataset = cifar_train.create_tf_dataset_from_all_clients().shuffle(
      buffer_size=10000).batch(
          train_batch_size, drop_remainder=True).map(
              train_image_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_dataset = cifar_test.create_tf_dataset_from_all_clients().batch(
      test_batch_size, drop_remainder=False).map(test_image_map).cache()

  if max_train_batches is not None and max_train_batches > 0:
    train_dataset = train_dataset.take(max_train_batches)
  if max_test_batches is not None and max_test_batches > 0:
    test_dataset = test_dataset.take(max_test_batches)

  return train_dataset, test_dataset
