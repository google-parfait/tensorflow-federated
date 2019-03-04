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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

# TODO(b/123578208): Remove deep keras imports after updating TF version.
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff


def create_simple_keras_model(learning_rate=0.1):
  """Returns an instance of `tf.Keras.Model` with just one dense layer.

  Args:
    learning_rate: The learning rate to use with the SGD optimizer.

  Returns:
    An instance of `tf.Keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(784,)),
      tf.keras.layers.Dense(10, tf.nn.softmax, kernel_initializer='zeros')
  ])

  def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))

  model.compile(
      loss=loss_fn,
      optimizer=gradient_descent.SGD(learning_rate),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model


def keras_dataset_from_emnist(dataset):
  """Converts `dataset` for use with the output of `create_simple_keras_model`.

  Args:
    dataset: An instance of `tf.data.Dataset` to read from.

  Returns:
    An instance of `tf.data.Dataset` after conversion.
  """

  def map_fn(example):
    return collections.OrderedDict([('x', tf.reshape(example['pixels'], [-1])),
                                    ('y', example['label'])])

  return dataset.map(map_fn)


def create_keras_model(compile_model=False):
  """Returns an instance of `tf.keras.Model` for use with the MNIST example.

  This code is based on the following target, which unfortunately cannot be
  imported as it is a Python binary, not a library:

  https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py

  Args:
    compile_model: If True, compile the model with a basic optimizer and loss.

  Returns:
    A `tf.keras.Model`.
  """
  # TODO(b/120157713): Find a way to import this code.
  data_format = 'channels_last'
  input_shape = [28, 28, 1]
  l = tf.keras.layers
  initializer = tf.keras.initializers.RandomNormal(seed=0)
  max_pool = l.MaxPooling2D((2, 2), (2, 2),
                            padding='same',
                            data_format=data_format)
  model = tf.keras.Sequential([
      l.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
      l.Conv2D(
          32,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer=initializer), max_pool,
      l.Conv2D(
          64,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer=initializer), max_pool,
      l.Flatten(),
      l.Dense(1024, activation=tf.nn.relu, kernel_initializer=initializer),
      l.Dropout(0.4, seed=1),
      l.Dense(10, kernel_initializer=initializer)
  ])
  if compile_model:
    # TODO(b/124534248): Currently we need the extra reduce_mean to ensure
    # the loss is a scalar.
    def loss_fn(y_true, y_pred):
      return tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              y_true, y_pred, from_logits=True))

    model.compile(
        loss=loss_fn, optimizer=gradient_descent.SGD(learning_rate=0.1))
  return model


Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name


def create_random_batch():
  """Returns an instance of `Batch` populated with random tensors."""
  return Batch(
      x=tf.random.uniform(tf.TensorShape([1, 784]), dtype=tf.float32),
      y=tf.constant(1, dtype=tf.int64, shape=[1, 1]))


def model_fn():
  """Constructs the MNIST model wrapped for use with TensorFlow Federated.

  The model constructed by this function can be passed as an argument to
  `tff.learning.build_federated_averaging_process` to create a federated
  training process.

  Returns:
    An instance of `tff.learning.Model` that represents a trainable model.
  """
  keras_model = create_keras_model(compile_model=True)
  dummy_batch = create_random_batch()
  return tff.learning.from_compiled_keras_model(keras_model, dummy_batch)
