# Lint as: python3
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
"""Build a model for EMNIST classification."""

import functools

import tensorflow as tf


def create_conv_dropout_model(only_digits=True):
  """Recommended model to use for EMNIST experiments.

  When `only_digits=True`, the summary of returned model is
  ```
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 28, 28, 1)         0
  _________________________________________________________________
  conv2d (Conv2D)              (None, 26, 26, 32)        320
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
  _________________________________________________________________
  dropout (Dropout)            (None, 12, 12, 64)        0
  _________________________________________________________________
  flatten (Flatten)            (None, 9216)              0
  _________________________________________________________________
  dense (Dense)                (None, 128)               1179776
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 128)               0
  _________________________________________________________________
  dense_1 (Dense)              (None, 10)                1290
  =================================================================
  Total params: 1,199,882
  Trainable params: 1,199,882
  Non-trainable params: 0
  ```
  For `only_digits=False`, the last dense layer is slightly larger.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model


def create_original_fedavg_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  The number of parameters when `only_digits=True` is (1,663,370), which matches
  what is reported in the paper.

  When `only_digits=True`, the summary of returned model is
  ```
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 28, 28, 1)         0
  _________________________________________________________________
  conv2d (Conv2D)              (None, 28, 28, 32)        832
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
  _________________________________________________________________
  max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
  _________________________________________________________________
  flatten (Flatten)            (None, 3136)              0
  _________________________________________________________________
  dense (Dense)                (None, 512)               1606144
  _________________________________________________________________
  dense_1 (Dense)              (None, 10)                5130
  =================================================================
  Total params: 1,663,370
  Trainable params: 1,663,370
  Non-trainable params: 0
  ```
  For `only_digits=False`, the last dense layer is slightly larger.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model


def create_two_hidden_layer_model(only_digits=True, hidden_units=200):
  """Create a two hidden-layer fully connected neural network.

  When `only_digits=True`, the summary of returned model is

  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 784)               0
  _________________________________________________________________
  dense (Dense)                (None, 200)               157000
  _________________________________________________________________
  dense_1 (Dense)              (None, 200)               40200
  _________________________________________________________________
  dense_2 (Dense)              (None, 10)                2010
  =================================================================
  Total params: 199,210
  Trainable params: 199,210
  Non-trainable params: 0

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    hidden_units: An integer specifying the number of units in the hidden layer.
      We default to 200 units, which matches that in
      https://arxiv.org/abs/1602.05629.

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu),
      tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model
