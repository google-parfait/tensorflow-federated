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
"""Preprocessing library for EMNIST baseline tasks."""

from typing import Callable

import tensorflow as tf

from tensorflow_federated.python.simulation.baselines import client_spec

MAX_CLIENT_DATASET_SIZE = 418


def _reshape_for_character_recognition(element):
  return tf.expand_dims(element['pixels'], axis=-1), element['label']


def _reshape_for_autoencoder(element):
  x = 1 - tf.reshape(element['pixels'], (-1, 28 * 28))
  return (x, x)


def create_preprocess_fn(
    preprocess_spec: client_spec.ClientSpec,
    emnist_task: str = 'character_recognition',
    num_parallel_calls: tf.Tensor = tf.data.experimental.AUTOTUNE
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing function for EMNIST client datasets.

  The preprocessing shuffles, repeats, batches, and then reshapes, using
  the `shuffle`, `repeat`, `take`, `batch`, and `map` attributes of a
  `tf.data.Dataset`, in that order.

  Args:
    preprocess_spec: A `tff.simulation.baselines.ClientSpec` containing
      information on how to preprocess clients.
    emnist_task: A string indicating the EMNIST task being performed. Must be
      one of 'character_recognition' or 'autoencoder'. If the former, then
      elements are mapped to tuples of the form (pixels, label), if the latter
      then elements are mapped to tuples of the form (pixels, pixels).
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A callable taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  shuffle_buffer_size = preprocess_spec.shuffle_buffer_size
  if shuffle_buffer_size is None:
    shuffle_buffer_size = MAX_CLIENT_DATASET_SIZE

  if emnist_task == 'character_recognition':
    mapping_fn = _reshape_for_character_recognition
  elif emnist_task == 'autoencoder':
    mapping_fn = _reshape_for_autoencoder
  else:
    raise ValueError('emnist_task must be one of "character_recognition" or '
                     '"autoencoder".')

  def preprocess_fn(dataset):
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    if preprocess_spec.num_epochs > 1:
      dataset = dataset.repeat(preprocess_spec.num_epochs)
    if preprocess_spec.max_elements is not None:
      dataset = dataset.take(preprocess_spec.max_elements)
    dataset = dataset.batch(preprocess_spec.batch_size, drop_remainder=False)
    return dataset.map(mapping_fn, num_parallel_calls=num_parallel_calls)

  return preprocess_fn
