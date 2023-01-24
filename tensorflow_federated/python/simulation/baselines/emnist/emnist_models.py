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
"""Library for building EMNIST classification and autoencoder models."""

import functools
import random
from typing import Optional

import tensorflow as tf


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


def create_conv_dropout_model(
    only_digits: bool = True, debug_seed: Optional[int] = None
) -> tf.keras.Model:
  """Create a convolutional network with dropout.

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
    only_digits: If `True`, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If `False`, uses 62 outputs for the larger
      dataset.
    debug_seed: An optional integer seed for deterministic weights
      initialization. This is intened for unittesting.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  glorot_uniform = _DeterministicInitializer(
      tf.keras.initializers.GlorotUniform, base_seed=debug_seed
  )
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=glorot_uniform(),
      ),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=glorot_uniform(),
      ),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=glorot_uniform(),
      ),
  ])

  return model


def create_original_fedavg_cnn_model(
    only_digits: bool = True, debug_seed: Optional[int] = None
) -> tf.keras.Model:
  """Create a convolutional network without dropout.

  This recreates the CNN model used in the original FedAvg paper,
  https://arxiv.org/abs/1602.05629. The number of parameters when
  `only_digits=True` is (1,663,370), which matches what is reported in the
  paper. When `only_digits=True`, the summary of returned model is
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
    only_digits: If `True`, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If `False`, uses 62 outputs for the larger
      dataset.
    debug_seed: An optional integer seed for deterministic weights
      initialization. This is intended for unittesting.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format,
  )

  glorot_uniform = _DeterministicInitializer(
      tf.keras.initializers.GlorotUniform, base_seed=debug_seed
  )

  def conv2d(**kwargs):
    return tf.keras.layers.Conv2D(
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu,
        kernel_initializer=glorot_uniform(),
        **kwargs,
    )

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          512, activation=tf.nn.relu, kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=glorot_uniform(),
      ),
  ])
  return model


def create_two_hidden_layer_model(
    only_digits: bool = True,
    hidden_units: int = 200,
    debug_seed: Optional[int] = None,
) -> tf.keras.Model:
  """Create a two hidden-layer fully connected neural network.

  When `only_digits=True`, the summary of returned model summary is
  ```
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
  ```

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If `True`, uses a final layer with
      10 outputs, for use with the digit-only EMNIST dataset. If `False`, uses
      62 outputs for the larger dataset.
    hidden_units: An integer specifying the number of units in the hidden layer.
      By default, this is set to `200`, which matches the original FedAvg paper,
      https://arxiv.org/abs/1602.05629.
    debug_seed: An optional integer seed for deterministic weights
      initialization. This is intended for unittesting.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  if hidden_units < 1:
    raise ValueError('hidden_units must be a positive integer.')

  glorot_uniform = _DeterministicInitializer(
      tf.keras.initializers.GlorotUniform, base_seed=debug_seed
  )

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      tf.keras.layers.Dense(
          hidden_units,
          activation=tf.nn.relu,
          kernel_initializer=glorot_uniform(),
      ),
      tf.keras.layers.Dense(
          hidden_units,
          activation=tf.nn.relu,
          kernel_initializer=glorot_uniform(),
      ),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=glorot_uniform(),
      ),
  ])
  return model


def create_autoencoder_model(
    debug_seed: Optional[int] = None,
) -> tf.keras.Model:
  """Create a bottleneck autoencoder model for use with EMNIST.

  The model is based of the MNIST autoencoder from:
  Reducing the Dimensionality of Data with Neural Networks
    G. E. Hinton and R. R. Salakhutdinov, science 313(5786), 504-507.

  The model has the following layer structure:
  ```
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  dense (Dense)                (None, 1000)              785000
  _________________________________________________________________
  dense_1 (Dense)              (None, 500)               500500
  _________________________________________________________________
  dense_2 (Dense)              (None, 250)               125250
  _________________________________________________________________
  dense_3 (Dense)              (None, 30)                7530
  _________________________________________________________________
  dense_4 (Dense)              (None, 250)               7750
  _________________________________________________________________
  dense_5 (Dense)              (None, 500)               125500
  _________________________________________________________________
  dense_6 (Dense)              (None, 1000)              501000
  _________________________________________________________________
  dense_7 (Dense)              (None, 784)               784784
  =================================================================
  Total params: 2,837,314
  Trainable params: 2,837,314
  Non-trainable params: 0
  ```
  AutoEncoder model

  Args:
    debug_seed: An optional integer seed for deterministic weights
      initialization. This is intended for unittesting.

  Returns:
    An uncompiled `tf.keras.Model`.
  """

  glorot_uniform = _DeterministicInitializer(
      tf.keras.initializers.GlorotUniform, base_seed=debug_seed
  )

  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          1000,
          activation='sigmoid',
          input_shape=(784,),
          kernel_initializer=glorot_uniform(),
      ),
      tf.keras.layers.Dense(
          500, activation='sigmoid', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          250, activation='sigmoid', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          30, activation='linear', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          250, activation='sigmoid', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          500, activation='sigmoid', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          1000, activation='sigmoid', kernel_initializer=glorot_uniform()
      ),
      tf.keras.layers.Dense(
          784, activation='sigmoid', kernel_initializer=glorot_uniform()
      ),
  ])
  return model
