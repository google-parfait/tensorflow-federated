# Copyright 2019 The TensorFlow Federated Authors. All Rights Reserved.
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
"""Keras implementation of the Yogi adaptive optimizer.

Implementation of Yogi is based on additive updates, as opposed to
multiplicative updates (as in Adam). The updates are governed by:
m_t+1 = beta1*m_t + (1-beta1)*g_t
v_t+1 = v_t + sign(g_t-v_t)(g_t^2)
Experiments show better performance across NLP and Vision tasks.

Original paper:
https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf
"""

import tensorflow as tf


def _solve(a, b, c):
  """Return solution of a quadratic minimization.

  The optimization equation is:
       f(a, b, c) = argmin_w{1/2 * a * w^2 + b * w + c * |w|}
  we get optimal solution w*:
       w* = -(b - sign(b)*c)/a if |b| > c else w* = 0
  REQUIRES: Dimensionality of a and b must be same
  Args:
    a: A Tensor
    b: A Tensor
    c: A Tensor with one element.

  Returns:
    A Tensor w, which is solution for the equation
  """
  w = (c * tf.sign(b) - b) / a
  w = tf.cast(tf.abs(b) > c, dtype=b.dtype) * w
  return w


class Yogi(tf.keras.optimizers.Optimizer):
  """Optimizer that implements the Yogi algorithm in Keras.

  See Algorithm 2 of
  https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization.pdf.
  """

  def __init__(self,
               learning_rate=0.01,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-3,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               initial_accumulator_value=1.0,
               activation='sign',
               name='Yogi',
               **kwargs):
    """Construct a new Yogi optimizer.

    Args:
      learning_rate: A Tensor or a floating point value. The learning rate.
      beta1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the 2nd moment estimates.
      epsilon: A constant trading off adaptivity and noise.
      l1_regularization_strength: A float value, must be greater than or equal
        to zero.
      l2_regularization_strength: A float value, must be greater than or equal
        to zero.
      initial_accumulator_value: The starting value for accumulators. Only
        positive values are allowed.
      activation: Use hard sign or soft tanh to determin sign.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Yogi".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super().__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta1)
    self._set_hyper('beta_2', beta2)
    self._set_hyper('epsilon', epsilon)
    self._set_hyper('l1_regularization_strength', l1_regularization_strength)
    self._set_hyper('l2_regularization_strength', l2_regularization_strength)

    self._beta1 = beta1
    self._activation = activation
    self._initial_accumulator_value = initial_accumulator_value
    self._l1_regularization_strength = l1_regularization_strength
    self._l2_regularization_strength = l2_regularization_strength

  def _create_slots(self, var_list):
    """See `tf.train.Optimizer._create_slots()`."""
    # Create slots for the first and second moments, and maximum second moments.
    for var in var_list:
      init = tf.constant_initializer(self._initial_accumulator_value)
      self.add_slot(var, 'v', init)
      if self._beta1 > 0.0:
        self.add_slot(var, 'm')

  def _resource_apply_dense(self, grad, var):
    """See `tf.train.Optimizer._apply_dense()`."""
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    beta1_t = self._get_hyper('beta_1', var_dtype)
    beta2_t = self._get_hyper('beta_2', var_dtype)
    epsilon_t = self._get_hyper('epsilon', var_dtype)
    l1_t = self._get_hyper('l1_regularization_strength', var_dtype)
    l2_t = self._get_hyper('l2_regularization_strength', var_dtype)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta1_power = tf.pow(beta1_t, local_step)
    beta2_power = tf.pow(beta2_t, local_step)

    lr = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))

    update_vs = []
    if self._beta1 == 0.0:
      # v_t = v + sign(g_t^2-v)(g_t^2)
      v = self.get_slot(var, 'v')
      grad2 = grad * grad
      if self._activation == 'sign':
        sign = tf.sign(grad2 - v)
      elif self._activation == 'tanh':
        sign = tf.tanh(10 * (grad2 - v))
      else:
        raise NotImplementedError('Activation function can be sign or tanh')
      v_t = v.assign_add(
          (1 - beta2_t) * sign * grad2, use_locking=self._use_locking)
      v_sqrt = tf.sqrt(v_t)

      # Yogi effective LR
      per_coord_lr = lr / (v_sqrt + epsilon_t)

      # Variable update
      # Step 1: Gradient descent
      new_var = var - per_coord_lr * grad
      # Step 2: Prox operator
      if self._l1_regularization_strength > 0:
        new_var = _solve(1 + l2_t * per_coord_lr, -new_var, l1_t * per_coord_lr)
      elif self._l2_regularization_strength > 0:
        new_var = new_var / (1 + l2_t * per_coord_lr)
      # Step 3: Update
      var_update = var.assign(new_var, use_locking=self._use_locking)

      update_vs.append(var_update)
      update_vs.append(v_t)

    else:
      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, 'm')
      m_t = m.assign(
          m * beta1_t + grad * (1 - beta1_t), use_locking=self._use_locking)

      # v_t = v + sign(g_t^2-v)(g_t^2)
      v = self.get_slot(var, 'v')
      grad2 = grad * grad
      if self._activation == 'sign':
        sign = tf.sign(grad2 - v)
      elif self._activation == 'tanh':
        sign = tf.tanh(10 * (grad2 - v))
      else:
        raise NotImplementedError('Activation function can be sign or tanh')
      v_t = v.assign_add(
          (1 - beta2_t) * sign * grad2, use_locking=self._use_locking)
      v_sqrt = tf.sqrt(v_t)

      # Yogi effective LR
      per_coord_lr = lr / (v_sqrt + epsilon_t)

      # Variable update
      # Step 1: Gradient descent
      new_var = var - per_coord_lr * m_t
      # Step 2: Prox operator
      if self._l1_regularization_strength > 0:
        new_var = _solve(1 + l2_t * per_coord_lr, -new_var, l1_t * per_coord_lr)
      elif self._l2_regularization_strength > 0:
        new_var = new_var / (1 + l2_t * per_coord_lr)
      # Step 3: Update
      var_update = var.assign(new_var, use_locking=self._use_locking)
      update_vs.append(var_update)
      update_vs.append(m_t)
      update_vs.append(v_t)

    # Create an op that groups all the above operations
    return tf.group(*update_vs)

  def _resource_apply_sparse(self, grad, var, indices):
    """Applies sparse gradients to a variable.

    Args:
      grad: A tensor for the `values` of `tf.IndexedSlices`.
      var: A `tf.Variable` object.
      indices: A tensor for the `indices` of `tf.IndexedSlices`.

    Returns:
      An op which updates `var` with `grad` and `indices`.
    """

    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    beta1_t = self._get_hyper('beta_1', var_dtype)
    beta2_t = self._get_hyper('beta_2', var_dtype)
    epsilon_t = self._get_hyper('epsilon', var_dtype)
    l1_t = self._get_hyper('l1_regularization_strength', var_dtype)
    l2_t = self._get_hyper('l2_regularization_strength', var_dtype)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta1_power = tf.pow(beta1_t, local_step)
    beta2_power = tf.pow(beta2_t, local_step)

    lr = (lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))

    update_vs = []
    if self._beta1 == 0.0:
      # v_t = v + sign(g_t^2-v)(g_t^2)
      v = self.get_slot(var, 'v')
      grad2 = grad * grad
      v_slice = tf.gather(v, indices)
      if self._activation == 'sign':
        sign = tf.sign(grad2 - v_slice)
      elif self._activation == 'tanh':
        sign = tf.tanh(10 * (grad2 - v_slice))
      else:
        raise NotImplementedError('Activation function can be sign or tanh')
      v_scaled_g_values = v_slice + (1 - beta2_t) * sign * grad2
      v_t = self._resource_scatter_update(v, indices, v_scaled_g_values)
      v_sqrt = tf.sqrt(v_scaled_g_values)

      # Yogi effective LR
      per_coord_lr = lr / (v_sqrt + epsilon_t)

      # Variable update
      # Step 1: Gradient descent
      var_slice = tf.gather(var, indices)
      new_var = var_slice - per_coord_lr * grad
      # Step 2: Prox operator
      if self._l1_regularization_strength > 0:
        new_var = _solve(1 + l2_t * per_coord_lr, -new_var, l1_t * per_coord_lr)
      elif self._l2_regularization_strength > 0:
        new_var = new_var / (1 + l2_t * per_coord_lr)
      # Step 3: Update
      var_update = self._resource_scatter_update(var, indices, new_var)
      update_vs.append(var_update)
      update_vs.append(v_t)

    else:
      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, 'm')
      m_scaled_g_values = grad * (1 - beta1_t)
      m_t = m.assign(m * beta1_t, use_locking=self._use_locking)
      with tf.control_dependencies([m_t]):
        m_slice = tf.gather(m, indices) + m_scaled_g_values
        m_t = self._resource_scatter_update(m, indices, m_slice)

      # v_t = v + sign(g_t^2-v)(g_t^2)
      v = self.get_slot(var, 'v')
      grad2 = grad * grad
      v_slice = tf.gather(v, indices)
      if self._activation == 'sign':
        sign = tf.sign(grad2 - tf.gather(v, indices))
      elif self._activation == 'tanh':
        sign = tf.tanh(10 * (grad2 - tf.gather(v, indices)))
      else:
        raise NotImplementedError('Activation function can be sign or tanh')
      v_scaled_g_values = v_slice + (1 - beta2_t) * sign * grad2
      v_t = self._resource_scatter_update(v, indices, v_scaled_g_values)
      v_sqrt = tf.sqrt(v_scaled_g_values)

      # Yogi effective LR
      per_coord_lr = lr / (v_sqrt + epsilon_t)

      # Variable update
      # Step 1: Gradient descent
      var_slice = tf.gather(var, indices)
      new_var = var_slice - per_coord_lr * m_slice
      # Step 2: Prox operator
      if self._l1_regularization_strength > 0:
        new_var = _solve(1 + l2_t * per_coord_lr, -new_var, l1_t * per_coord_lr)
      elif self._l2_regularization_strength > 0:
        new_var = new_var / (1 + l2_t * per_coord_lr)
      # Step 3: Update
      var_update = self._resource_scatter_update(var, indices, new_var)
      update_vs.append(var_update)
      update_vs.append(m_t)
      update_vs.append(v_t)

    # Create an op that groups all the above operations
    return tf.group(*update_vs)

  def get_config(self):
    config = super().get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta1': self._serialize_hyperparameter('beta_1'),
        'beta2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self._serialize_hyperparameter('epsilon'),
        'l1_t': self._serialize_hyperparameter('l1_regularization_strength'),
        'l2_t': self._serialize_hyperparameter('l2_regularization_strength'),
        'activation': self._activation,
        'initial_accumulator_value': self._initial_accumulator_value,
    })
    return config
