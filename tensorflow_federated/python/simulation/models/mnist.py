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
"""An example of an MNIST model function for use with TensorFlow Federated."""

import collections
import random

import tensorflow as tf

from tensorflow_federated.python.learning.metrics import counters


def create_simple_keras_model(learning_rate=0.1):
  """Returns an instance of `tf.Keras.Model` with just one dense layer.

  Args:
    learning_rate: The learning rate to use with the SGD optimizer.

  Returns:
    An instance of `tf.Keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(784,)),
      tf.keras.layers.Dense(10, tf.nn.softmax, kernel_initializer='zeros'),
  ])

  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate),
      metrics=[
          tf.keras.metrics.SparseCategoricalAccuracy(),
          counters.NumExamplesCounter(),
      ],
  )
  return model


def keras_dataset_from_emnist(dataset):
  """Converts `dataset` for use with the output of `create_simple_keras_model`.

  Args:
    dataset: An instance of `tf.data.Dataset` to read from.

  Returns:
    An instance of `tf.data.Dataset` after conversion.
  """

  def map_fn(example):
    return collections.OrderedDict([
        ('x', tf.reshape(example['pixels'], [-1])),
        ('y', example['label']),
    ])

  return dataset.map(map_fn)


# TODO(b/235837441): Move this functionality to a more general location.
class _DeterministicInitializer:
  """Wrapper to produce different deterministic initialization values."""

  def __init__(
      self,
      initializer_type: type[tf.keras.initializers.Initializer],
      base_seed: int,
  ):
    self._initializer_type = initializer_type
    if base_seed is None:
      base_seed = random.randint(1, 1e9)
    self._base_seed = base_seed

  def __call__(self):
    self._base_seed += 1
    return self._initializer_type(seed=self._base_seed)


def create_keras_model(compile_model=False):
  """Returns an instance of `tf.keras.Model` for use with the MNIST example.

  This code is based on the following target, which unfortunately cannot be
  imported as it is a Python binary, not a library:

  https://github.com/tensorflow/models/blob/master/official/r1/mnist/mnist.py

  Args:
    compile_model: If True, compile the model with a basic optimizer and loss.

  Returns:
    A `tf.keras.Model`.
  """
  # TODO(b/120157713): Find a way to import this code.
  data_format = 'channels_last'
  input_shape = [28, 28, 1]
  initializer = _DeterministicInitializer(
      tf.keras.initializers.RandomNormal, base_seed=0
  )
  max_pool = tf.keras.layers.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format
  )
  model = tf.keras.Sequential([
      tf.keras.layers.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
      tf.keras.layers.Conv2D(
          32,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer=initializer(),
      ),
      max_pool,
      tf.keras.layers.Conv2D(
          64,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer=initializer(),
      ),
      max_pool,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          1024, activation=tf.nn.relu, kernel_initializer=initializer()
      ),
      tf.keras.layers.Dropout(0.4, seed=1),
      tf.keras.layers.Dense(10, kernel_initializer=initializer()),
  ])
  if compile_model:
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    )
  return model
