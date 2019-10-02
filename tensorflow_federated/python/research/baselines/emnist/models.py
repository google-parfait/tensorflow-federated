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
  input_shape = [28, 28, 1]

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28 * 28,), target_shape=input_shape),
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape,
          data_format=data_format),
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
  input_shape = [28, 28, 1]

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
      tf.keras.layers.Reshape(input_shape=(28 * 28,), target_shape=input_shape),
      conv2d(filters=32, input_shape=input_shape),
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

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    hidden_units: An integer specifying the number of units in the hidden layer.

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          hidden_units, activation=tf.nn.relu, input_shape=(28 * 28,)),
      tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model


# Defining global constants for ResNet model
L2_WEIGHT_DECAY = 2e-4


def _residual_block(input_tensor, kernel_size, filters, base_name):
  """A block of two conv layers with an identity residual connection.

  Args:
    input_tensor: The input tensor for the residual block.
    kernel_size: An integer specifying the kernel size of the convolutional
      layers in the residual blocks.
    filters: A list of two integers specifying the filters of the conv layers in
      the residual blocks. The first integer specifies the number of filters on
      the first conv layer within each residual block, the second applies to the
      remaining conv layers within each block.
    base_name: A string used to generate layer names.

  Returns:
    The output tensor of the residual block evaluated at the input tensor.
  """
  filters1, filters2 = filters

  x = tf.keras.layers.Conv2D(
      filters1,
      kernel_size,
      padding='same',
      use_bias=False,
      name='{}_conv_1'.format(base_name))(
          input_tensor)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      name='{}_conv_2'.format(base_name))(
          x)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def _conv_residual_block(input_tensor,
                         kernel_size,
                         filters,
                         base_name,
                         strides=(2, 2)):
  """A block of two conv layers with a convolutional residual connection.

  Args:
    input_tensor: The input tensor for the residual block.
    kernel_size: An integer specifying the kernel size of the convolutional
      layers in the residual blocks.
    filters: A list of two integers specifying the filters of the conv layers in
      the residual blocks. The first integer specifies the number of filters on
      the first conv layer within each residual block, the second applies to the
      remaining conv layers within each block.
    base_name: A string used to generate layer names.
    strides: A tuple of integers specifying the strides lengths in the first
      conv layer in the block.

  Returns:
    The output tensor of the residual block evaluated at the input tensor.
  """
  filters1, filters2 = filters

  x = tf.keras.layers.Conv2D(
      filters1,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      name='{}_conv_1'.format(base_name))(
          input_tensor)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(
      filters2,
      kernel_size,
      padding='same',
      use_bias=False,
      name='{}_conv_2'.format(base_name))(
          x)

  shortcut = tf.keras.layers.Conv2D(
      filters2, (1, 1),
      strides=strides,
      use_bias=False,
      name='{}_conv_shortcut'.format(base_name))(
          input_tensor)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def _resnet_block(input_tensor,
                  size,
                  kernel_size,
                  filters,
                  stage,
                  conv_strides=(2, 2)):
  """A block which applies multiple residual blocks to a given input.

   The resnet block applies a single conv residual block followed by multiple
   identity residual blocks to a given input.

  Args:
    input_tensor: The input tensor for the resnet block.
    size: An integer specifying the number of residual blocks. A conv residual
      block is applied once, followed by (size - 1) identity residual blocks.
    kernel_size: An integer specifying the kernel size of the convolutional
      layers in the residual blocks.
    filters: A list of two integers specifying the filters of the conv layers in
      the residual blocks. The first integer specifies the number of filters on
      the first conv layer within each residual block, the second applies to the
      remaining conv layers within each block.
    stage: An integer representing the the position of the resnet block within
      the resnet. Used for generating layer names.
    conv_strides: A tuple of integers specifying the strides in the first conv
      layer within each conv residual block.

  Returns:
    The output tensor of the resnet block evaluated at the input tensor.
  """

  x = _conv_residual_block(
      input_tensor,
      kernel_size,
      filters,
      base_name='res_{}_block_0'.format(stage),
      strides=conv_strides)
  for i in range(size - 1):
    x = _residual_block(
        x,
        kernel_size,
        filters,
        base_name='res_{}_block_{}'.format(stage, i + 1))
  return x


def create_resnet(num_blocks=5, only_digits=True):
  """Instantiates a ResNet model for EMNIST classification.

  Instantiates the ResNet architecture from https://arxiv.org/abs/1512.03385.
  The ResNet contains 3 stages of ResNet blocks with each block containing one
  conv residual block followed by (num_blocks - 1) idenity residual blocks. Each
  residual block has 2 convolutional layers. With the input convolutional
  layer and the final dense layer, this brings the total number of trainable
  layers in the network to (6*num_blocks + 2). This number is often used to
  identify the ResNet, so for example ResNet56 has num_blocks = 9.

  Args:
    num_blocks: An integer representing the number of residual blocks within
      each ResNet block.
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.

  Returns:
    A `tf.keras.Model`.
  """

  num_classes = 10 if only_digits else 62

  target_shape = (28, 28, 1)
  img_input = tf.keras.layers.Input(shape=(28 * 28,))

  x = img_input

  x = tf.keras.layers.Reshape(
      target_shape=target_shape, input_shape=(28 * 28,))(
          x)
  x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='initial_pad')(x)
  x = tf.keras.layers.Conv2D(
      16, (3, 3),
      strides=(1, 1),
      padding='valid',
      use_bias=False,
      name='initial_conv')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = _resnet_block(
      x,
      size=num_blocks,
      kernel_size=3,
      filters=[16, 16],
      stage=2,
      conv_strides=(1, 1))

  x = _resnet_block(
      x,
      size=num_blocks,
      kernel_size=3,
      filters=[32, 32],
      stage=3,
      conv_strides=(2, 2))

  x = _resnet_block(
      x,
      size=num_blocks,
      kernel_size=3,
      filters=[64, 64],
      stage=4,
      conv_strides=(2, 2))

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      num_classes,
      activation=tf.nn.softmax,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      name='fully_connected')(
          x)

  inputs = img_input
  model = tf.keras.models.Model(
      inputs, x, name='resnet{}'.format(6 * num_blocks + 2))

  return model
