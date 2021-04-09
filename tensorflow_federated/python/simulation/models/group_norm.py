# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# Orginal implementation from keras_contrib/layer/normalization
# =============================================================================
"""Lightweight implementation of GroupNorm."""

import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
  """Group normalization layer.

    Source: 'Group Normalization' (Yuxin Wu & Kaiming He, 2018)
    https://arxiv.org/abs/1803.08494

    Group Normalization divides the channels into groups and computes within
    each group the mean and variance for normalization. Empirically, its
    accuracy is more stable than batch norm in a wide range of small batch
    sizes, if learning rate is adjusted linearly with batch sizes.

    In this lightweight implementation, we normalize each group by subtracting
    the mean of the group, and dividing by the variance. We do not provide
    functionality for additional scaling and offsets, as in more complex
    GroupNorm and BatchNorm implementations.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization.
  """

  def __init__(self,
               groups: int = 2,
               axis: int = -1,
               epsilon: float = 1e-3,
               **kwargs):
    """Constructs a Group Normalization layer.

    Args:
      groups: An integer specifying the number of groups used by Group
        Normalization. Must be in the range `[1, N]` where `N` is the input
        dimension of the axis to be normalized. The input dimension must be
        divisible by the number of groups.
      axis: An integer indicating which axis of the input should be normalized.
      epsilon: A float added to the variance estimate of each group to avoid
        numerical instability for values close to zero.
      **kwargs: Additional arguments for constructing the layer. For details on
        supported arguments, see `tf.keras.layers.Layer`.
    """

    super().__init__(**kwargs)
    self.supports_masking = True
    self.groups = groups
    self.axis = axis
    self.epsilon = epsilon
    self._check_axis()

  def build(self, input_shape):
    """Used after layer initialization to create a `tf.keras.layers.InputSpec`.

    This method also runs compatibility checks to ensure that the given
    `input_shape` is compatible with the GroupNorm layer.

    Args:
      input_shape: A tuple representing the expected input shape of any tensor
        passed through the layer.
    """
    self._check_if_input_shape_is_none(input_shape)
    self._check_size_of_dimensions(input_shape)
    self._create_input_spec(input_shape)
    self.built = True
    super().build(input_shape)

  def call(self, inputs):
    """Computes the output of the layer on a given tensor."""
    input_shape = tf.keras.backend.int_shape(inputs)
    tensor_input_shape = tf.shape(inputs)
    reshaped_inputs, _ = self._reshape_into_groups(inputs, input_shape,
                                                   tensor_input_shape)
    normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

    is_instance_norm = (input_shape[self.axis] // self.groups) == 1
    if not is_instance_norm:
      outputs = tf.reshape(normalized_inputs, tensor_input_shape)
    else:
      outputs = normalized_inputs

    return outputs

  def get_config(self):
    """Returns a dictionary representing the configuration of the layer."""
    config = {
        'groups': self.groups,
        'axis': self.axis,
        'epsilon': self.epsilon,
    }
    base_config = super().get_config()
    return {**base_config, **config}

  def compute_output_shape(self, input_shape):
    return input_shape

  def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
    """Reshapes an input tensor into separate groups."""
    group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
    is_instance_norm = (input_shape[self.axis] // self.groups) == 1
    if not is_instance_norm:
      group_shape[self.axis] = input_shape[self.axis] // self.groups
      group_shape.insert(self.axis, self.groups)
      group_shape = tf.stack(group_shape)
      inputs = tf.reshape(inputs, group_shape)
    return inputs, group_shape

  def _apply_normalization(self, reshaped_inputs, input_shape):
    """Normalizes a reshaped tensor across the `axis` attribute of the layer.

    In this lightweight implementation, we normalize by subtracting the mean,
    and dividing by the variance. We do not provide functionality for additional
    scaling and offsets, as in more complex GroupNorm and BatchNorm
    implementations.

    Args:
      reshaped_inputs: A `tf.Tensor` representing the input to the layer after
        it has been reshaped into groups.
      input_shape: A tuple representing the original input shape of the
        `tf.Tensor` passed to the layer.

    Returns:
      A `tf.Tensor` formed by normalizing `reshaped_inputs` according to the
      mean and variance of each group.
    """
    group_shape = tf.keras.backend.int_shape(reshaped_inputs)
    group_reduction_axes = list(range(1, len(group_shape)))
    is_instance_norm = (input_shape[self.axis] // self.groups) == 1
    if not is_instance_norm:
      axis = -2 if self.axis == -1 else self.axis - 1
    else:
      axis = -1 if self.axis == -1 else self.axis - 1
    group_reduction_axes.pop(axis)

    mean, variance = tf.nn.moments(
        reshaped_inputs, group_reduction_axes, keepdims=True)

    normalized_inputs = tf.nn.batch_normalization(
        reshaped_inputs,
        mean=mean,
        variance=variance,
        scale=None,
        offset=None,
        variance_epsilon=self.epsilon,
    )
    return normalized_inputs

  def _check_if_input_shape_is_none(self, input_shape):
    dim = input_shape[self.axis]
    if dim is None:
      raise ValueError('Axis {} of input tensor must have a defined dimension, '
                       'but the layer received an input with shape {}.'.format(
                           self.axis, input_shape))

  def _check_size_of_dimensions(self, input_shape):
    """Ensures that `input_shape` is compatible with the number of groups."""
    dim = input_shape[self.axis]
    if dim < self.groups:
      raise ValueError('Number of groups {} cannot be more than the number of '
                       'channels {}.'.format(self.groups, dim))

    if dim % self.groups != 0:
      raise ValueError('The number of channels {} must be a multiple of the '
                       'number of groups {}.'.format(dim, self.groups))

  def _check_axis(self):
    if self.axis == 0:
      raise ValueError(
          'You are trying to normalize your batch axis, axis 0, which is '
          'incompatible with GroupNorm. Consider using '
          '`tf.keras.layers.BatchNormalization` instead.')

  def _create_input_spec(self, input_shape):
    """Creates a `tf.keras.layers.InputSpec` for the GroupNorm layer."""
    dim = input_shape[self.axis]
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=len(input_shape), axes={self.axis: dim})
