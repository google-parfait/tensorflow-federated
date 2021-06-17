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
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.types import computation_types

MAX_CLIENT_DATASET_SIZE = 418


def _reshape_for_digit_recognition(element):
  return tf.expand_dims(element['pixels'], axis=-1), element['label']


def _reshape_for_autoencoder(element):
  x = 1 - tf.reshape(element['pixels'], (-1, 28 * 28))
  return (x, x)


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    max_elements: Optional[int] = None,
    shuffle_buffer_size: Optional[int] = None,
    emnist_task: str = 'digit_recognition',
    num_parallel_calls: tf.Tensor = tf.data.experimental.AUTOTUNE
) -> computation_base.Computation:
  """Creates a preprocessing function for EMNIST client datasets.

  The preprocessing shuffles, repeats, batches, and then reshapes, using
  the `shuffle`, `take`, `repeat`, `batch`, and `map` attributes of a
  `tf.data.Dataset`, in that order.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    max_elements: An optional integer governing the maximum number of examples
      used by each client. Must be `None` or a positive integer. If set to
      `None`, all examples are used, otherwise a maximum of `max_elements` are
      used for each client dataset.
    shuffle_buffer_size: An optional integer representing the shuffle buffer
      size on clients. Must be `None` or a positive integer. If set to `1`, no
      shuffling occurs. If set to `None`, this will be set to `418`, the largest
      number of elements in any EMNIST client dataset.
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
  if batch_size < 1:
    raise ValueError('batch_size must be a positive integer.')
  if max_elements is not None and max_elements <= 0:
    raise ValueError('max_elements must be `None` or a positive integer.')
  if shuffle_buffer_size is not None and shuffle_buffer_size <= 0:
    raise ValueError(
        'shuffle_buffer_size must be `None` or a positive integer.')

  if max_elements is None:
    max_elements = -1
  if shuffle_buffer_size is None:
    shuffle_buffer_size = MAX_CLIENT_DATASET_SIZE

  if emnist_task == 'digit_recognition':
    mapping_fn = _reshape_for_digit_recognition
  elif emnist_task == 'autoencoder':
    mapping_fn = _reshape_for_autoencoder
  else:
    raise ValueError('emnist_task must be one of "digit_recognition" or '
                     '"autoencoder".')

  # Features are intentionally sorted lexicographically by key for consistency
  # across datasets.
  feature_dtypes = collections.OrderedDict(
      label=computation_types.TensorType(tf.int32),
      pixels=computation_types.TensorType(tf.float32, shape=(28, 28)))

  @computations.tf_computation(computation_types.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.take(max_elements)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.map(mapping_fn, num_parallel_calls=num_parallel_calls)

  return preprocess_fn
