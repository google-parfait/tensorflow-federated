# Lint as: python3
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
# ==============================================================================
"""Layer-wise Adaptive Rate Scaling optimizer for federated optimization."""

import tensorflow as tf


class FLARSOptimizer(tf.keras.optimizers.Optimizer):
  """Layer-wise Adaptive Rate Scaling for federated optimization.

  FLARS optimizer is an extension of LARS optimizer from
  https://arxiv.org/abs/1708.03888. The main distinction is that FLARS normalize
  the gradient with  E_i|| g_i||_2 instead of
  ||E_i[g_i]||_2. Note, FLARS scaling is currently only enabled for
  dense tensors.
  """

  def __init__(self,
               learning_rate,
               momentum=0.9,
               eeta=0.01,
               epsilon=0.0,
               nesterov=False,
               max_ratio=1.0,
               name="FLARSOptimizer",
               **kwargs):
    """Construct a new FLARS Optimizer.

    Args:
      learning_rate: A `Tensor` or floating point value. The base learning rate.
      momentum: A floating point value. Momentum hyperparameter.
      eeta: Default set to 0.01.
      epsilon: Optional epsilon parameter to be set in models that have very
        small gradients. Default set to 0.0.
      nesterov: When set to `True`, nesterov momentum will be enabled.
      max_ratio: Upper bound for the entire ratio being applied to the base
        learning rate.
      name: Optional name prefix for variables and ops created by
        `FLARSOptimizer`.
      **kwargs: Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate.

    Raises:
      ValueError: If a hyperparameter is set to a non-sensical value.
    """
    super(FLARSOptimizer, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("decay", self._initial_decay)

    self._momentum = False
    if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self._max_ratio = max_ratio
    self._eeta = eeta
    self._epsilon = epsilon
    self._nesterov = nesterov
    self._grads_norm = None

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def compute_lr(self, base_lr, grad, var, grad_norm):
    w_norm = tf.norm(var)
    trust_ratio = tf.where(
        tf.math.greater(w_norm, 0),
        tf.where(
            tf.math.greater(grad_norm, 0),
            (self._eeta * w_norm / (grad_norm + self._epsilon)), 1.0), 1.0)

    scaled_lr = base_lr * tf.minimum(trust_ratio, self._max_ratio)
    return scaled_lr

  def update_grads_norm(self, vars_list, grads_norm):
    self._grads_norm = dict([
        (var.name, grad_norm) for var, grad_norm in zip(vars_list, grads_norm)
    ])

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    base_lr = self._get_hyper("learning_rate", var_dtype)
    assert (
        self._grads_norm), "Call update_grads_norm() before apply_gradients()!"
    grads_norm = self._grads_norm.get(var.name, 1.)
    scaled_lr = self.compute_lr(base_lr, grad, var, grads_norm)

    # Use resource_apply_keras_momentum in the future.
    if self._momentum:
      accum = self.get_slot(var, "momentum")
      momentum = self._get_hyper("momentum", var_dtype)
      with tf.control_dependencies([
          accum.assign(accum * momentum - grad * scaled_lr),
      ]):
        update_ops = tf.cond(
            tf.constant(self._nesterov, dtype=tf.bool),
            lambda: var.assign_add(accum * momentum - grad * scaled_lr),
            lambda: var.assign_add(accum))
    else:
      update_ops = var.assign_sub(grad * scaled_lr)

    return update_ops

  def get_config(self):
    config = super(FLARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self._nesterov,
        "eeta": self._eeta,
        "max_ratio": self._max_ratio,
        "epsilon": self._epsilon
    })
    return config
