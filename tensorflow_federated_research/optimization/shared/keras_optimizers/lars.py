# Copyright 2020 The TensorFlow Federated Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keras implementation of the Layer-wise Adaptive Rate Scaling (LARS) optimizer.

Original paper: [Large batch training of convolutional networks]
(https://arxiv.org/pdf/1708.03888.pdf).

Code adapted from Algorithm 1 in [Large Batch Optimization for Deep Learning:
Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962):
https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/optimizers/lamb.py
"""

import re
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf


FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


class LARS(tf.keras.optimizers.Optimizer):
  """Optimizer that implements Layer-wise Adaptive Rate Scaling (LARS)."""

  def __init__(self,
               learning_rate: Union[FloatTensorLike] = 0.001,
               momentum: FloatTensorLike = 0.9,
               weight_decay_rate: FloatTensorLike = 0.0,
               epsilon: FloatTensorLike = 0.001,
               exclude_from_weight_decay: Optional[List[str]] = None,
               exclude_from_layer_adaptation: Optional[List[str]] = None,
               name: str = 'LARS',
               **kwargs):
    """Construct a new LARS optimizer.

    Args:
        learning_rate: A `Tensor` or floating point value representing the
          learning rate of the optimizer.
        momentum: A `float` value or a constant `float` tensor representing the
          momentum parameter.
        weight_decay_rate: A `float` value representing the weight decay rate.
        epsilon: A `float` value used for numerical stability.
        exclude_from_weight_decay: A list of regex patterns of
          variables excluded from weight decay. Variables whose name contain
          a substring matching the pattern will be excluded.
        exclude_from_layer_adaptation: A list of regex patterns of
          variables excluded from layer adaptation. Variables whose name
          contain a substring matching the pattern will be excluded. If not
          provided, this will default to the same value as
          `exclude_from_weight_decay`.
        name: Optional name for the operations created when applying
          gradients. Defaults to 'LARS'.
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
          `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue`
          is clip gradients by value, `decay` is included for backward
          compatibility to allow time inverse decay of learning rate. The kwarg
          `lr` is included for backward compatibility, it is recommended to use
          `learning_rate` instead.
    """
    super().__init__(name, **kwargs)

    # We handle L2 regularization/weight decay generically, via a
    # 'weight_decay_rate' hyperparameter (distinct from the default
    # Keras learning rate decay optionally supplied via the 'decay' kwarg).
    self._set_hyper('weight_decay_rate', weight_decay_rate)
    self._set_hyper('epsilon', epsilon)
    self._set_hyper('decay', self._initial_decay)   # Keras default, not used
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('momentum', momentum)
    self.exclude_from_weight_decay = exclude_from_weight_decay
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay

  def _create_slots(self, var_list):
    # Create slots for the first moment.
    for var in var_list:
      self.add_slot(var, 'm')

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_dtype = var.dtype.base_dtype
    lr = self._get_hyper('learning_rate', var_dtype)
    momentum = self._get_hyper('momentum', var_dtype)
    weight_decay_rate = self._get_hyper('weight_decay_rate', var_dtype)
    epsilon = self._get_hyper('epsilon', var_dtype)
    var_name = self._get_variable_name(var.name)

    # m_t = beta * m_{t-1} + (1 - beta) * (g_t + lambda * x_t)
    m = self.get_slot(var, 'm')
    grad_with_decay = grad
    if self._do_use_weight_decay(var_name):
      grad_with_decay += weight_decay_rate * var
    scaled_grad_with_decay = grad_with_decay * (1 - momentum)
    m_t = m.assign(
        m * momentum + scaled_grad_with_decay, use_locking=self._use_locking)

    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      w_norm = tf.norm(var, ord=2)
      m_norm = tf.norm(m_t, ord=2)
      ratio = tf.where(
          tf.greater(w_norm, 0),
          tf.where(tf.greater(m_norm, 0), (w_norm / (m_norm + epsilon)), 1.0),
          1.0,
      )

    var_update = var - ratio * lr * m_t
    return var.assign(var_update, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_dtype = var.dtype.base_dtype
    lr = self._get_hyper('learning_rate', var_dtype)
    momentum = self._get_hyper('momentum', var_dtype)
    weight_decay_rate = self._get_hyper('weight_decay_rate', var_dtype)
    epsilon = self._get_hyper('epsilon', var_dtype)
    var_name = self._get_variable_name(var.name)

    # m_t = beta * m_{t-1} + (1 - beta) * (g_t + lambda * x_t)
    m = self.get_slot(var, 'm')
    grad_with_decay = grad
    if self._do_use_weight_decay(var_name):
      grad_with_decay += weight_decay_rate * var
    scaled_grad_with_decay = grad_with_decay * (1 - momentum)
    m_t = m.assign(m * momentum, use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, scaled_grad_with_decay)

    ratio = 1.0
    if self._do_layer_adaptation(var_name):
      w_norm = tf.norm(var, ord=2)
      m_norm = tf.norm(m_t, ord=2)
      ratio = tf.where(
          tf.greater(w_norm, 0),
          tf.where(tf.greater(m_norm, 0), (w_norm / (m_norm + epsilon)), 1.0),
          1.0,
      )

    var_update = var.assign_sub(
        ratio * lr * m_t, use_locking=self._use_locking)
    return tf.group(*[var_update, m_t])

  def get_config(self):
    config = super().get_config()
    config.update({
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'weight_decay_rate':
            self._serialize_hyperparameter('weight_decay_rate'),
        'decay':
            self._serialize_hyperparameter('decay'),
        'momentum':
            self._serialize_hyperparameter('momentum'),
        'epsilon':
            self._serialize_hyperparameter('epsilon'),
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match('^(.*):\\d+$', param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
