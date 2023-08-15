# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet v2 model for Keras using Batch or Group Normalization.

Related papers/blogs:
- http://arxiv.org/pdf/1603.05027v2.pdf
"""

from collections.abc import Iterable
import enum
from typing import Optional

import tensorflow as tf


BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 1e-4


def _check_iterable_with_positive_ints(structure):
  if not isinstance(structure, Iterable):
    return False
  return all(isinstance(a, int) and a > 0 for a in structure)


class ResidualBlock(enum.Enum):
  BASIC = 'basic'
  BOTTLENECK = 'bottleneck'


class NormLayer(enum.Enum):
  GROUP_NORM = 'group_norm'
  BATCH_NORM = 'batch_norm'


def _norm_relu(input_tensor, norm):
  """Applies normalization and ReLU activation to an input tensor.

  Args:
    input_tensor: The `tf.Tensor` to apply the block to.
    norm: A `NormLayer` specifying the type of normalization layer used.

  Returns:
    A `tf.Tensor`.
  """
  channel_axis = -1  # last non-batch dimension in keras GroupNormalization

  if norm is NormLayer.GROUP_NORM:
    x = tf.keras.layers.GroupNormalization(axis=channel_axis)(input_tensor)
  elif norm is NormLayer.BATCH_NORM:
    x = tf.keras.layers.BatchNormalization(
        axis=channel_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON
    )(input_tensor)
  else:
    raise ValueError('The norm argument must be of type `NormLayer`.')
  return tf.keras.layers.Activation('relu')(x)


def _conv_norm_relu(input_tensor, filters, kernel_size, norm, strides=(1, 1)):
  """Applies convolution, normalization, and ReLU activation to an input tensor.

  These functions are applied to `input_tensor` in that exact order.

  Args:
    input_tensor: The `tf.Tensor` to apply the block to.
    filters: An integer specifying the number of filters in the convolutional
      layer.
    kernel_size: A tuple of specifying the kernel height and width
      (respectively) in the convolutional layer.
    norm: A `NormLayer` specifying the type of normalization layer used.
    strides:  A tuple of two integers specifying the stride height and width
      (respectively) in the convolutional layers.

  Returns:
    A `tf.Tensor`.
  """
  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
  )(input_tensor)
  return _norm_relu(x, norm=norm)


def _norm_relu_conv(input_tensor, filters, kernel_size, norm, strides=(1, 1)):
  """Applies normalization, ReLU activation, and convolution to an input tensor.

  These functions are applied to `input_tensor` in that exact order.

  Args:
    input_tensor: The `tf.Tensor` to apply the block to.
    filters: An integer specifying the number of filters in the convolutional
      layer.
    kernel_size: A tuple of specifying the kernel height and width
      (respectively) in the convolutional layer.
    norm: A `NormLayer` specifying the type of normalization layer used.
    strides:  A tuple of two integers specifying the stride height and width
      (respectively) in the convolutional layers.

  Returns:
    A `tf.Tensor`.
  """
  x = _norm_relu(input_tensor, norm=norm)
  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
  )(x)
  return x


def _shortcut(input_tensor, residual, norm):
  """Computes the output of a shortcut block between an input and residual.

  More specifically, this block takes `input` and adds it to `residual`. If
  `input` is not the same shape as `residual`, then we first apply an
  appropriately-sized convolutional layer to alter its shape to that of
  `residual` and normalize via `norm` before adding it to `residual`.

  Args:
    input_tensor: The `tf.Tensor` to apply the block to.
    residual: A `tf.Tensor` added to `input_tensor` after it has been passed
      through a convolution and normalization.
    norm: A `NormLayer` specifying the type of normalization layer used.

  Returns:
    A `tf.Tensor`.
  """
  input_shape = tf.keras.backend.int_shape(input_tensor)
  residual_shape = tf.keras.backend.int_shape(residual)

  row_axis = 1
  col_axis = 2
  channel_axis = -1  # last non-batch dimension in keras GroupNormalization

  stride_width = int(round(input_shape[row_axis] / residual_shape[row_axis]))
  stride_height = int(round(input_shape[col_axis] / residual_shape[col_axis]))
  equal_channels = input_shape[channel_axis] == residual_shape[channel_axis]

  shortcut = input_tensor
  # Use a 1-by-1 kernel if the strides are greater than 1, or there the input
  # and residual tensors have different numbers of channels.
  if stride_width > 1 or stride_height > 1 or not equal_channels:
    shortcut = tf.keras.layers.Conv2D(
        filters=residual_shape[channel_axis],
        kernel_size=(1, 1),
        strides=(stride_width, stride_height),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
    )(shortcut)

    if norm is NormLayer.GROUP_NORM:
      shortcut = tf.keras.layers.GroupNormalization(axis=channel_axis)(shortcut)
    elif norm is NormLayer.BATCH_NORM:
      shortcut = tf.keras.layers.BatchNormalization(
          axis=channel_axis,
          momentum=BATCH_NORM_DECAY,
          epsilon=BATCH_NORM_EPSILON,
      )(shortcut)
    else:
      raise ValueError('The norm argument must be of type `NormLayer`.')

  return tf.keras.layers.add([shortcut, residual])


def _basic_block(
    input_tensor, filters, norm, strides=(1, 1), normalize_first=True
):
  """Computes the forward pass of an input tensor through a basic block.

  Specifically, the basic block consists of two convolutional layers with
  `filters` filters, with additional layers for ReLU activation and
  normalization layers. All kernels are of shape (3, 3). Finally, the output
  is passed through a shortcut block.

  Args:
    input_tensor: The `tf.Tensor` to apply the block to.
    filters: An integer specifying the number of filters in the convolutional
      layers.
    norm: A `NormLayer` specifying the type of normalization layer used.
    strides:  A tuple of two integers specifying the stride height and width
      (respectively) in the convolutional layers.
    normalize_first: If set to `True`, normalization is performed before the
      first convolution. If `False`, no normalization is performed before the
      first convolutional layer.

  Returns:
    A `tf.Tensor`.
  """
  if normalize_first:
    x = _norm_relu_conv(
        input_tensor,
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        norm=norm,
    )
  else:
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
    )(input_tensor)

  x = _norm_relu_conv(
      x, filters=filters, kernel_size=(3, 3), strides=strides, norm=norm
  )
  return _shortcut(input_tensor, x, norm=norm)


def _bottleneck_block(
    input_tensor, filters, norm, strides=(1, 1), normalize_first=True
):
  """Applies a bottleneck convolutional block to a given input tensor.

  Specifically, this applies a sequence of 3 normalization, ReLU, and
  convolutional layers to the input tensor, followed by a shortcut block. The
  convolutions use filters of shape (1, 1), (3, 3), and (1, 1), respectively.

  Args:
    input_tensor: The `tf.Tensor` to apply the block to.
    filters: An integer specifying the number of filters in the first two
      convolutional layers. The third uses `4*filters`.
    norm: A `NormLayer` specifying the type of normalization layer used.
    strides:  A tuple of two integers specifying the stride height and width
      (respectively) in the convolutional layers.
    normalize_first: If set to `True`, normalization is performed before the
      first convolution. If `False`, no normalization is performed before the
      first convolutional layer.

  Returns:
    A `tf.Tensor`.
  """
  if normalize_first:
    x = _norm_relu_conv(
        input_tensor,
        filters=filters,
        kernel_size=(1, 1),
        strides=strides,
        norm=norm,
    )
  else:
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
    )(input_tensor)

  x = _norm_relu_conv(
      x, filters=filters, kernel_size=(3, 3), strides=strides, norm=norm
  )

  x = _norm_relu_conv(
      x, filters=filters * 4, kernel_size=(1, 1), strides=strides, norm=norm
  )
  return _shortcut(input_tensor, x, norm=norm)


def _residual_block(
    input_tensor,
    block_function,
    filters,
    num_blocks,
    norm,
    strides=(1, 1),
    is_first_layer=False,
):
  """Builds a residual block with repeating bottleneck or basic blocks."""
  x = input_tensor
  for i in range(num_blocks):
    if is_first_layer and i == 0:
      normalize_first = False
    else:
      normalize_first = True

    x = block_function(
        input_tensor=x,
        filters=filters,
        strides=strides,
        normalize_first=normalize_first,
        norm=norm,
    )
  return x


def create_resnet(
    input_shape: tuple[int, int, int],
    num_classes: int = 10,
    residual_block: ResidualBlock = ResidualBlock.BOTTLENECK,
    repetitions: Optional[list[int]] = None,
    initial_filters: int = 64,
    initial_strides: tuple[int, int] = (2, 2),
    initial_kernel_size: tuple[int, int] = (7, 7),
    initial_max_pooling: bool = True,
    norm_layer: NormLayer = NormLayer.GROUP_NORM,
) -> tf.keras.Model:
  """Creates a ResNet v2 model with batch or group normalization.

  Instantiates the architecture from http://arxiv.org/pdf/1603.05027v2.pdf.
  The ResNet contains stages of residual blocks, each with sequences of
  convolutional, ReLU, and normalization layers. The order depends on the
  choice of `block`, and the type of normalization is governed by `norm`.

  Args:
    input_shape: A length 3 tuple of positive integeres dictating the number of
      rows, columns, and channels of an input. Restricted to the `channels_last`
      format.
    num_classes: A positive integer describing the number of output classes.
    residual_block: A `ResidualBlock` describing what type of residual block is
      used throughout the ResNet.
    repetitions: An optional list of integers describing the number of blocks
      within each stage. If None, defaults to the resnet50 repetitions of [3, 4,
      6, 3].
    initial_filters: An integer specifying the number of filters in the initial
      convolutional layer.
    initial_strides: A tuple of two integers specifying the stride height and
      width (respectively) in the initial convolutional layer.
    initial_kernel_size: A tuple of two integers specifying the kernel height
      and width (respectively) in the initial convolutional layer.
    initial_max_pooling: Whether to use max pooling after the initial
      convolutional layer.
    norm_layer: A `NormLayer` describing which normalization layer is used in
      the resulting model.

  Returns:
    An uncompiled `tf.keras.Model`.

  Raises:
    ValueError: If `input_shape` is not a length three iterable with positive
      integer values, if image data format is not `channels_last` (the format is
      `channels_first`), if `num_classes` is not a positive integer, if
      `residual_block` is not of type `ResidualBlock`, if `repetitions` is not
      `None` and is not an iterable with positive integer elements, if
      `initial_filters` is not positive, if `initial_strides` and
      `initial_kernel_size` are not length 2 iterables with positive integer
      elements, if `norm_layer` is not of type `NormLayer`.
  """

  if (
      not _check_iterable_with_positive_ints(input_shape)
      or len(input_shape) != 3
  ):
    raise ValueError(
        'input_shape must be an iterable of length 3 containing '
        'only positive integers.'
    )

  # TODO: b/265363369 - Support `channels_first` image format once the
  # GroupNormalizaiton index issue is fixed.
  if tf.keras.backend.image_data_format() != 'channels_last':
    raise ValueError(
        'Image data needs to be represented in the `channels_last` format with'
        ' a three-dimensional array where the last channel represents the color'
        ' channel. '
    )

  if num_classes < 1:
    raise ValueError('num_classes must be a positive integer.')

  if residual_block is ResidualBlock.BASIC:
    block_fn = _basic_block
  elif residual_block is ResidualBlock.BOTTLENECK:
    block_fn = _bottleneck_block
  else:
    raise ValueError('residual_block must be of type `ResidualBlock`.')

  if not repetitions:
    repetitions = [3, 4, 6, 3]
  elif not _check_iterable_with_positive_ints(repetitions):
    raise ValueError(
        'repetitions must be None or an iterable containing positive integers'
    )

  if initial_filters < 1:
    raise ValueError('initial_filters must be a positive integer.')

  if (
      not _check_iterable_with_positive_ints(initial_strides)
      or len(initial_strides) != 2
  ):
    raise ValueError(
        'initial_strides must be an iterable of length 2 '
        'containing only positive integers.'
    )

  if (
      not _check_iterable_with_positive_ints(initial_kernel_size)
      or len(initial_kernel_size) != 2
  ):
    raise ValueError(
        'initial_kernel_size must be an iterable of length 2 '
        'containing only positive integers.'
    )

  if not isinstance(norm_layer, NormLayer):
    raise ValueError('norm_layer must be of type `NormLayer`.')

  img_input = tf.keras.layers.Input(shape=input_shape)
  x = _conv_norm_relu(
      img_input,
      filters=initial_filters,
      kernel_size=initial_kernel_size,
      strides=initial_strides,
      norm=norm_layer,
  )

  if initial_max_pooling:
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), strides=initial_strides, padding='same'
    )(x)

  filters = initial_filters

  for i, r in enumerate(repetitions):
    x = _residual_block(
        x,
        block_fn,
        filters=filters,
        num_blocks=r,
        is_first_layer=(i == 0),
        norm=norm_layer,
    )
    filters *= 2

  # Final activation in the residual blocks
  x = _norm_relu(x, norm=norm_layer)

  # Classification block
  x = tf.keras.layers.GlobalAveragePooling2D()(x)

  x = tf.keras.layers.Dense(
      num_classes,
      activation='softmax',
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
      bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
  )(x)

  model = tf.keras.models.Model(img_input, x)
  return model


def create_resnet18(
    input_shape: tuple[int, int, int],
    num_classes: int,
    norm_layer: NormLayer = NormLayer.GROUP_NORM,
) -> tf.keras.Model:
  """Creates a ResNet-18 with basic residual blocks.

  Args:
    input_shape: A length 3 tuple of positive integeres dictating the number of
      rows, columns, and channels of an input. Can be in channel-first or
      channel-last format.
    num_classes: A positive integer describing the number of output classes.
    norm_layer: A `NormLayer` describing which normalization layer is used in
      the resulting model.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  return create_resnet(
      input_shape,
      num_classes,
      residual_block=ResidualBlock.BASIC,
      repetitions=[2, 2, 2, 2],
      norm_layer=norm_layer,
  )


def create_resnet34(
    input_shape: tuple[int, int, int],
    num_classes: int,
    norm_layer: NormLayer = NormLayer.GROUP_NORM,
) -> tf.keras.Model:
  """Creates a ResNet-34 with basic residual blocks.

  Args:
    input_shape: A length 3 tuple of positive integeres dictating the number of
      rows, columns, and channels of an input. Can be in channel-first or
      channel-last format.
    num_classes: A positive integer describing the number of output classes.
    norm_layer: A `NormLayer` describing which normalization layer is used in
      the resulting model.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  return create_resnet(
      input_shape,
      num_classes,
      residual_block=ResidualBlock.BASIC,
      repetitions=[3, 4, 6, 3],
      norm_layer=norm_layer,
  )


def create_resnet50(
    input_shape: tuple[int, int, int],
    num_classes: int,
    norm_layer: NormLayer = NormLayer.GROUP_NORM,
) -> tf.keras.Model:
  """Creates a ResNet-50 model with bottleneck residual blocks.

  Args:
    input_shape: A length 3 tuple of positive integeres dictating the number of
      rows, columns, and channels of an input. Can be in channel-first or
      channel-last format.
    num_classes: A positive integer describing the number of output classes.
    norm_layer: A `NormLayer` describing which normalization layer is used in
      the resulting model.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  return create_resnet(
      input_shape,
      num_classes,
      residual_block=ResidualBlock.BOTTLENECK,
      repetitions=[3, 4, 6, 3],
      norm_layer=norm_layer,
  )


def create_resnet101(
    input_shape: tuple[int, int, int],
    num_classes: int,
    norm_layer: NormLayer = NormLayer.GROUP_NORM,
) -> tf.keras.Model:
  """Creates a ResNet-101 model with bottleneck residual blocks.

  Args:
    input_shape: A length 3 tuple of positive integeres dictating the number of
      rows, columns, and channels of an input. Can be in channel-first or
      channel-last format.
    num_classes: A positive integer describing the number of output classes.
    norm_layer: A `NormLayer` describing which normalization layer is used in
      the resulting model.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  return create_resnet(
      input_shape,
      num_classes,
      residual_block=ResidualBlock.BOTTLENECK,
      repetitions=[3, 4, 23, 3],
      norm_layer=norm_layer,
  )


def create_resnet152(
    input_shape: tuple[int, int, int],
    num_classes: int,
    norm_layer: NormLayer = NormLayer.GROUP_NORM,
) -> tf.keras.Model:
  """Creates a ResNet-152 model with bottleneck residual blocks.

  Args:
    input_shape: A length 3 tuple of positive integeres dictating the number of
      rows, columns, and channels of an input. Can be in channel-first or
      channel-last format.
    num_classes: A positive integer describing the number of output classes.
    norm_layer: A `NormLayer` describing which normalization layer is used in
      the resulting model.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  return create_resnet(
      input_shape,
      num_classes,
      residual_block=ResidualBlock.BOTTLENECK,
      repetitions=[3, 8, 36, 3],
      norm_layer=norm_layer,
  )
