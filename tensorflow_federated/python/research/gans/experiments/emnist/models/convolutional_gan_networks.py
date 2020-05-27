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
"""Networks for GAN for basic image datasets, e.g. MNIST/EMNIST."""

import tensorflow as tf

layers = tf.keras.layers


# These two networks are borrowed from a TF-GAN MNIST tutorial:
# https://github.com/tensorflow/gan/blob/master/tensorflow_gan/examples/mnist/networks.py
def get_gan_discriminator_model(weight_decay=2.5e-5):
  """Discriminator (as Keras model) to use in an MNIST/EMNIST GAN."""
  input_shape = (28, 28, 1)

  model = tf.keras.Sequential()

  model.add(
      layers.Conv2D(
          filters=64,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          input_shape=input_shape))
  model.add(layers.LeakyReLU(alpha=0.01))

  model.add(
      layers.Conv2D(
          filters=128,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding='same',
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
  model.add(layers.LeakyReLU(alpha=0.01))

  model.add(layers.Flatten())

  model.add(
      layers.Dense(
          1024,
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
  _batch_norm_with_reshape_to_4d(model, n_channels=1024)
  model.add(layers.LeakyReLU(alpha=0.01))

  model.add(
      layers.Dense(
          1,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))

  return model


def get_gan_generator_model(latent_dim=64, weight_decay=2.5e-5):
  """Generator (as Keras model) to use in an MNIST/EMNIST GAN."""
  input_shape = (latent_dim,)

  model = tf.keras.Sequential()

  model.add(
      layers.Dense(
          1024,
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay),
          input_shape=input_shape))
  _batch_norm_with_reshape_to_4d(model, n_channels=1024)
  model.add(layers.ReLU())

  model.add(
      layers.Dense(
          7 * 7 * 256,
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
  _batch_norm_with_reshape_to_4d(model, n_channels=7 * 7 * 256)
  model.add(layers.ReLU())

  model.add(layers.Reshape((7, 7, 256)))

  model.add(
      layers.Conv2DTranspose(
          filters=64,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding='same',
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
  # Don't need to use _batch_norm_with_reshape_to_4d to get fused batch norm
  # usage b/c 'x' is already 4D (incl. the batch dim) here.
  model.add(layers.BatchNormalization(momentum=0.999, scale=False))
  model.add(layers.ReLU())

  model.add(
      layers.Conv2DTranspose(
          filters=32,
          kernel_size=(4, 4),
          strides=(2, 2),
          padding='same',
          use_bias=False,
          kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * weight_decay)))
  # Don't need to use _batch_norm_with_reshape_to_4d to get fused batch norm
  # usage b/c 'x' is already 4D (incl. the batch dim) here.
  model.add(layers.BatchNormalization(momentum=0.999, scale=False))
  model.add(layers.ReLU())

  model.add(
      layers.Conv2D(
          filters=1, kernel_size=(4, 4), padding='same', activation='tanh'))

  return model


def _batch_norm_with_reshape_to_4d(model, n_channels):
  # We temporarily reshape to 4D (incl. the batch dim) to induce the Keras
  # BatchNormalization layer to use the fused batch norm implementation under
  # the hood (nn.fused_batch_norm) instead of the general batch norm
  # implementation (nn.batch_normalization). (This keeps things matching the
  # TF-GAN behavior as closely as possible.)
  model.add(layers.Reshape((1, 1, n_channels)))
  model.add(layers.BatchNormalization(momentum=0.999, scale=False))
  model.add(layers.Reshape((n_channels,)))
