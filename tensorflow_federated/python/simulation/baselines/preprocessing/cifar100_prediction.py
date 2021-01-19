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
"""Preprocessing library for CIFAR-100 classification tasks."""

import collections
from typing import Callable, Sequence, Tuple, Union

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.simulation.datasets import cifar100

CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
NUM_EXAMPLES_PER_CLIENT = 500


def build_image_map(
    crop_shape: Union[tf.Tensor, Sequence[int]],
    distort: bool = False
) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
  """Builds a function that crops and normalizes CIFAR-100 elements.

  The image is first converted to a `tf.float32`, then cropped (according to
  the `distort` argument). Finally, its values are normalized via
  `tf.image.per_image_standardization`.

  Args:
    crop_shape: A tuple (crop_height, crop_width, channels)
      specifying the desired crop shape for pre-processing batches. This cannot
      exceed (32, 32, 3) element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.
    distort: A boolean indicating whether to distort the image via random crops
      and flips. If set to False, the image is resized to the `crop_shape` via
      `tf.image.resize_with_crop_or_pad`.

  Returns:
    A callable accepting a tensor and performing the crops and normalization
    discussed above.
  """

  if distort:

    def crop_fn(image):
      image = tf.image.random_crop(image, size=crop_shape)
      image = tf.image.random_flip_left_right(image)
      return image

  else:

    def crop_fn(image):
      return tf.image.resize_with_crop_or_pad(
          image, target_height=crop_shape[0], target_width=crop_shape[1])

  def image_map(example):
    image = tf.cast(example['image'], tf.float32)
    image = crop_fn(image)
    image = tf.image.per_image_standardization(image)
    return (image, example['label'])

  return image_map


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    shuffle_buffer_size: int = NUM_EXAMPLES_PER_CLIENT,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE,
    distort_image=False,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE
) -> computation_base.Computation:
  """Creates a preprocessing function for CIFAR-100 client datasets.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    shuffle_buffer_size: An integer representing the shuffle buffer size on
      clients. If set to a number <= 1, no shuffling occurs.
    crop_shape: A tuple (crop_height, crop_width, num_channels) specifying the
      desired crop shape for pre-processing. This tuple cannot have elements
      exceeding (32, 32, 3), element-wise. The element in the last index should
      be set to 3 to maintain the RGB image structure of the elements.
    distort_image: A boolean indicating whether to perform preprocessing that
      includes image distortion, including random crops and flips.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A `tff.Computation` performing the preprocessing described above.

  Raises:
    TypeError: If `crop_shape` is not an iterable.
    ValueError: If `num_epochs` is a non-positive integer, if `crop_shape` is
      iterable but not length 3.
  """
  if num_epochs < 1:
    raise ValueError('num_epochs must be a positive integer.')
  if not isinstance(crop_shape, collections.abc.Iterable):
    raise TypeError('Argument crop_shape must be an iterable.')
  crop_shape = tuple(crop_shape)
  if len(crop_shape) != 3:
    raise ValueError('The crop_shape must have length 3, corresponding to a '
                     'tensor of shape [height, width, channels].')

  feature_dtypes = collections.OrderedDict(
      coarse_label=computation_types.TensorType(tf.int64),
      image=computation_types.TensorType(tf.uint8, shape=(32, 32, 3)),
      label=computation_types.TensorType(tf.int64))

  image_map_fn = build_image_map(crop_shape, distort_image)

  @computations.tf_computation(computation_types.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    return (
        dataset.repeat(num_epochs)
        # We map before batching to ensure that the cropping occurs
        # at an image level (eg. we do not perform the same crop on
        # every image within a batch)
        .map(image_map_fn,
             num_parallel_calls=num_parallel_calls).batch(batch_size))

  return preprocess_fn


def get_federated_datasets(
    train_client_batch_size: int = 20,
    test_client_batch_size: int = 100,
    train_client_epochs_per_round: int = 1,
    test_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: int = NUM_EXAMPLES_PER_CLIENT,
    test_shuffle_buffer_size: int = 1,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE,
    serializable: bool = False
) -> Tuple[client_data.ClientData, client_data.ClientData]:
  """Loads and preprocesses federated CIFAR100 training and testing sets.

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
    crop_shape: An iterable of integers specifying the desired crop
      shape for pre-processing. Must be convertable to a tuple of integers
      (CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS) which cannot have elements that
      exceed (32, 32, 3), element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.
    serializable: Boolean indicating whether the returned datasets are intended
      to be serialized and shipped across RPC channels. If `True`, stateful
      transformations will be disallowed.

  Returns:
    A tuple (cifar_train, cifar_test) of `tff.simulation.ClientData` instances
      representing the federated training and test datasets.

  Raises:
    TypeError: If `serializable` is not a boolean.
  """
  if not isinstance(serializable, bool):
    raise TypeError(
        'serializable must be a Boolean; you passed {} of type {}.'.format(
            serializable, type(serializable)))
  cifar_train, cifar_test = cifar100.load_data()

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=not serializable)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=test_client_epochs_per_round,
      batch_size=test_client_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=False)

  cifar_train = cifar_train.preprocess(train_preprocess_fn)
  cifar_test = cifar_test.preprocess(test_preprocess_fn)
  return cifar_train, cifar_test


def get_centralized_datasets(
    train_batch_size: int = 20,
    test_batch_size: int = 100,
    train_shuffle_buffer_size: int = 10000,
    test_shuffle_buffer_size: int = 1,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Loads and preprocesses centralized CIFAR100 training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    crop_shape: An iterable of integers specifying the desired crop
      shape for pre-processing. Must be convertable to a tuple of integers
      (CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS) which cannot have elements that
      exceed (32, 32, 3), element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.

  Returns:
    A tuple (cifar_train, cifar_test) of `tf.data.Dataset` instances
    representing the centralized training and test datasets.
  """
  cifar_train, cifar_test = cifar100.load_data()
  cifar_train = cifar_train.create_tf_dataset_from_all_clients()
  cifar_test = cifar_test.create_tf_dataset_from_all_clients()

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=train_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=True)
  cifar_train = train_preprocess_fn(cifar_train)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=test_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=False)
  cifar_test = test_preprocess_fn(cifar_test)

  return cifar_train, cifar_test
