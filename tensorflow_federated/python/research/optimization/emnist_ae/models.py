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

import tensorflow as tf


def create_autoencoder_model():
  """Recommended model to use for EMNIST AutoEncoder experiments.

  The model from Hinton and Salakhutdinov 2010 looks like
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

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(1000, activation='sigmoid', input_shape=(784,)),
      tf.keras.layers.Dense(500, activation='sigmoid'),
      tf.keras.layers.Dense(250, activation='sigmoid'),
      tf.keras.layers.Dense(30, activation='linear'),
      tf.keras.layers.Dense(250, activation='sigmoid'),
      tf.keras.layers.Dense(500, activation='sigmoid'),
      tf.keras.layers.Dense(1000, activation='sigmoid'),
      tf.keras.layers.Dense(784, activation='sigmoid'),
  ])

  return model
