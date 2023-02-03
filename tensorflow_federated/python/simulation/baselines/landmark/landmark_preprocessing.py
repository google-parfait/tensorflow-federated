# Copyright 2022, The TensorFlow Federated Authors.
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
"""Preprocessing library for LANDMARK baseline tasks."""

import collections
from collections.abc import Callable
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.simulation.baselines import client_spec


# The largest client of LANDMARK has 3500 examples.
_MAX_CLIENT_DATASET_SIZE = 3500
# We use 128 as the image size (not 224 as in https://arxiv.org/abs/2003.08082)
# for faster experiments.
IMAGE_SIZE = 128


def _map_fn(
    element: collections.OrderedDict[str, tf.Tensor], is_training: bool
) -> tuple[tf.Tensor, tf.Tensor]:
  """Preprocesses an image for training/eval using Keras data augmentation.

  The original GLD images have various image size. We map a single image at a
  time instead of a batch to the target image size. For training images, we
  randomly crop the image to the target size, randomly flipped the image
  horizontally, and rescale the input values to the range of [-1.0, 1.0]. For
  testing images, we crop the center portion of the image at the target size,
  and rescale the input values to the range of [-1.0, 1.0].

  Args:
    element: An OrderedDict with the keys of `image/decoded` and `class`
      representing an image.
    is_training: Boolean. If true it would preprocess an image for train,
      otherwise it would preprocess it for evaluation.

  Returns:
    A tuple or two tensors. The first tensor represents the processed image.
    The second tensor inherits the `class` of the input element.
  """
  preprocessing_image_for_train = tf.keras.Sequential([
      tf.keras.layers.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.Rescaling(
          scale=2.0 / 255, offset=-1.0
      ),  # rescale the values to the range of [-1.0, 1.0]
  ])
  preprocessing_image_for_eval = tf.keras.Sequential([
      tf.keras.layers.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
      tf.keras.layers.Rescaling(
          scale=2.0 / 255, offset=-1.0
      ),  # rescale the values to the range of [-1.0, 1.0]
  ])

  if is_training:
    image = preprocessing_image_for_train(element['image/decoded'])
  else:
    image = preprocessing_image_for_eval(element['image/decoded'])

  label = element['class']
  return image, label


def create_preprocess_fn(
    preprocess_spec: client_spec.ClientSpec,
    is_training: bool,
    num_parallel_calls: tf.Tensor = tf.data.experimental.AUTOTUNE,
    debug_seed: Optional[int] = None,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing function for LANDMARK client datasets.

  The preprocessing function crops & reshapes, shuffles, repeats, and batches
  images, using the `map`, `shuffle`, `repeat`, `take`, and `batch` of a
  `tf.data.Dataset`, in that order.

  Args:
    preprocess_spec: A `tff.simulation.baselines.ClientSpec` containing
      information on how to preprocess clients.
    is_training: Boolean. If `True`, we randomly crop the image to the target
      size, randomly flipped the image horizontally, and rescale the input
      values to the range of [-1.0, 1.0]. Otherwise, we crop the center portion
      of the image at the target size, and rescale the input values to the range
      of [-1.0, 1.0].
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.
    debug_seed: An optional integer seed for deterministic shuffling and
      mapping. Intended for unittesting.

  Returns:
    A callable function taking `tf.data.Dataset` as an input, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """

  shuffle_buffer_size = preprocess_spec.shuffle_buffer_size
  if shuffle_buffer_size is None:
    shuffle_buffer_size = _MAX_CLIENT_DATASET_SIZE

  num_epochs = preprocess_spec.num_epochs
  batch_size = preprocess_spec.batch_size
  max_elements = preprocess_spec.max_elements

  def preprocess_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Create a callable function to preprocess `tf.data.Dataset`.

    Args:
      dataset: `tf.data.Dataset` to be preprocessed.

    Returns:
      `tf.data.Dataset` preprocessed according to the input arguments.
    """
    if is_training:
      data_map_fn = lambda element: _map_fn(element, is_training=True)
    else:
      data_map_fn = lambda element: _map_fn(element, is_training=False)

    dataset = dataset.map(data_map_fn, num_parallel_calls=num_parallel_calls)
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size, seed=debug_seed)
    if num_epochs > 1:
      dataset = dataset.repeat(num_epochs)
    if max_elements is not None:
      dataset = dataset.take(max_elements)
    return dataset.batch(batch_size)

  return preprocess_fn
