# Copyright 2019 The TensorFlow Federated Authors. All Rights Reserved.
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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared.keras_optimizers import yogi


def yogi_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.01,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-3,
                      l1reg=0.0,
                      l2reg=0.0):
  """Performs Yogi parameter update using numpy.

  Args:
    param: An numpy ndarray of the current parameter.
    g_t: An numpy ndarray of the current gradients.
    t: An numpy ndarray of the current time step.
    m: An numpy ndarray of the 1st moment estimates.
    v: An numpy ndarray of the 2nd moment estimates.
    alpha: A float value of the learning rate.
    beta1: A float value of the exponential decay rate for the 1st moment
      estimates.
    beta2: A float value of the exponential decay rate for the 2nd moment
      estimates.
    epsilon: A float of a small constant for numerical stability.
    l1reg: A float value of L1 regularization
    l2reg: A float value of L2 regularization

  Returns:
    A tuple of numpy ndarrays (param_t, m_t, v_t) representing the
    updated parameters for `param`, `m`, and `v` respectively.
  """
  beta1 = np.array(beta1, dtype=param.dtype)
  beta2 = np.array(beta2, dtype=param.dtype)

  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  g2_t = g_t * g_t
  v_t = v - (1 - beta2) * np.sign(v - g2_t) * g2_t

  per_coord_lr = alpha_t / (np.sqrt(v_t) + epsilon)
  param_t = param - per_coord_lr * m_t

  if l1reg > 0:
    param_t = (param_t - l1reg * per_coord_lr * np.sign(param_t)) / (
        1 + l2reg * per_coord_lr)
    print(param_t.dtype)
    param_t[np.abs(param_t) < l1reg * per_coord_lr] = 0.0
  elif l2reg > 0:
    param_t = param_t / (1 + l2reg * per_coord_lr)
  return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
  local_step = tf.cast(opt.iterations + 1, dtype)
  beta_1_t = tf.cast(opt._get_hyper('beta_1'), dtype)
  beta_1_power = tf.math.pow(beta_1_t, local_step)
  beta_2_t = tf.cast(opt._get_hyper('beta_2'), dtype)
  beta_2_power = tf.math.pow(beta_2_t, local_step)
  return (beta_1_power, beta_2_power)


dtypes_to_test = [('float32', tf.dtypes.float32),
                  ('float64', tf.dtypes.float64), ('half', tf.dtypes.half)]


class YogiOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def do_test_sparse(self, dtype, beta1=0.0, l1reg=0.0, l2reg=0.0):
    if tf.test.is_gpu_available() and dtype is tf.dtypes.half:
      return

    # Initialize variables for numpy implementation.
    m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0_np_indices = np.array([0, 1], dtype=np.int32)
    grads0 = tf.IndexedSlices(
        tf.constant(grads0_np), tf.constant(grads0_np_indices),
        tf.constant([2]))
    grads1_np_indices = np.array([0, 1], dtype=np.int32)
    grads1 = tf.IndexedSlices(
        tf.constant(grads1_np), tf.constant(grads1_np_indices),
        tf.constant([2]))
    opt = yogi.Yogi(
        beta1=beta1,
        l1_regularization_strength=l1reg,
        l2_regularization_strength=l2reg)
    if not tf.executing_eagerly():
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(tf.compat.v1.global_variables_initializer())

    # Fetch params to validate initial values.
    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run 3 steps of Yogi.
    for t in range(1, 4):
      beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
      self.assertAllCloseAccordingToType(beta1**t, self.evaluate(beta1_power))
      self.assertAllCloseAccordingToType(0.999**t, self.evaluate(beta2_power))
      if not tf.executing_eagerly():
        self.evaluate(update)
      else:
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m0, v0 = yogi_update_numpy(
          var0_np, grads0_np, t, m0, v0, beta1=beta1, l1reg=l1reg, l2reg=l2reg)
      var1_np, m1, v1 = yogi_update_numpy(
          var1_np, grads1_np, t, m1, v1, beta1=beta1, l1reg=l1reg, l2reg=l2reg)

      # Validate updated params.
      self.assertAllCloseAccordingToType(
          var0_np,
          self.evaluate(var0),
          msg='Updated params 0 do not match in NP and TF')
      self.assertAllCloseAccordingToType(
          var1_np,
          self.evaluate(var1),
          msg='Updated params 1 do not match in NP and TF')

  @parameterized.named_parameters(dtypes_to_test)
  def test_sparse(self, dtype):
    self.do_test_sparse(dtype)

  @parameterized.named_parameters(dtypes_to_test)
  def test_sparse_regularization(self, dtype):
    self.do_test_sparse(dtype, l1reg=0.1, l2reg=0.2)

  @parameterized.named_parameters(dtypes_to_test)
  def test_sparse_momentum(self, dtype):
    self.do_test_sparse(dtype, beta1=0.9)

  @parameterized.named_parameters(dtypes_to_test)
  def test_sparse_momentum_regularization(self, dtype):
    self.do_test_sparse(dtype, beta1=0.9, l1reg=0.1, l2reg=0.2)

  @parameterized.named_parameters(dtypes_to_test)
  def test_sparse_repeated_indices(self, dtype):
    if tf.test.is_gpu_available() and dtype is tf.dtypes.half:
      return

    repeated_index_update_var = tf.Variable([[1.0], [2.0]], dtype=dtype)
    aggregated_update_var = tf.Variable([[1.0], [2.0]], dtype=dtype)
    grad_repeated_index = tf.IndexedSlices(
        tf.constant([0.1, 0.1], shape=[2, 1], dtype=dtype), tf.constant([1, 1]),
        tf.constant([2, 1]))
    grad_aggregated = tf.IndexedSlices(
        tf.constant([0.2], shape=[1, 1], dtype=dtype), tf.constant([1]),
        tf.constant([2, 1]))
    opt1 = yogi.Yogi()
    opt2 = yogi.Yogi()

    if not tf.executing_eagerly():
      repeated_update = opt1.apply_gradients([(grad_repeated_index,
                                               repeated_index_update_var)])
      aggregated_update = opt2.apply_gradients([(grad_aggregated,
                                                 aggregated_update_var)])
      self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose(
        self.evaluate(aggregated_update_var),
        self.evaluate(repeated_index_update_var))

    for _ in range(3):
      if not tf.executing_eagerly():
        self.evaluate(repeated_update)
        self.evaluate(aggregated_update)
      else:
        opt1.apply_gradients([(grad_repeated_index, repeated_index_update_var)])
        opt2.apply_gradients([(grad_aggregated, aggregated_update_var)])

      self.assertAllClose(
          self.evaluate(aggregated_update_var),
          self.evaluate(repeated_index_update_var))

  def do_test_basic(self, dtype, beta1=0.0, l1reg=0.0, l2reg=0.0):
    if tf.test.is_gpu_available() and dtype is tf.dtypes.half:
      return

    # Initialize variables for numpy implementation.
    m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)

    opt = yogi.Yogi(
        beta1=beta1,
        l1_regularization_strength=l1reg,
        l2_regularization_strength=l2reg)

    if not tf.executing_eagerly():
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(tf.compat.v1.global_variables_initializer())

    # Fetch params to validate initial values.
    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run 3 steps of Yogi.
    for t in range(1, 4):
      beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
      self.assertAllCloseAccordingToType(beta1**t, self.evaluate(beta1_power))
      self.assertAllCloseAccordingToType(0.999**t, self.evaluate(beta2_power))

      if not tf.executing_eagerly():
        self.evaluate(update)
      else:
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m0, v0 = yogi_update_numpy(
          var0_np, grads0_np, t, m0, v0, beta1=beta1, l1reg=l1reg, l2reg=l2reg)
      var1_np, m1, v1 = yogi_update_numpy(
          var1_np, grads1_np, t, m1, v1, beta1=beta1, l1reg=l1reg, l2reg=l2reg)

      # Validate updated params.
      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @parameterized.named_parameters(dtypes_to_test)
  def test_basic(self, dtype):
    self.do_test_basic(dtype)

  @parameterized.named_parameters(dtypes_to_test)
  def test_basic_regularization(self, dtype):
    self.do_test_basic(dtype, l1reg=0.1, l2reg=0.2)

  @parameterized.named_parameters(dtypes_to_test)
  def test_basic_momentum(self, dtype):
    self.do_test_basic(dtype, beta1=0.9)

  @parameterized.named_parameters(dtypes_to_test)
  def test_basic_momentum_regularization(self, dtype):
    self.do_test_basic(dtype, beta1=0.9, l1reg=0.1, l2reg=0.2)

  @parameterized.named_parameters(dtypes_to_test)
  def test_tensor_learning_rate(self, dtype):
    if tf.test.is_gpu_available() and dtype is tf.dtypes.half:
      return

    # Initialize variables for numpy implementation.
    m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)
    opt = yogi.Yogi(tf.constant(0.01))

    if not tf.executing_eagerly():
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(tf.compat.v1.global_variables_initializer())

    # Fetch params to validate initial values.
    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run 3 steps of Yogi.
    for t in range(1, 4):
      beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
      self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
      self.assertAllCloseAccordingToType(0.999**t, self.evaluate(beta2_power))

      if not tf.executing_eagerly():
        self.evaluate(update)
      else:
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m0, v0 = yogi_update_numpy(var0_np, grads0_np, t, m0, v0)
      var1_np, m1, v1 = yogi_update_numpy(var1_np, grads1_np, t, m1, v1)

      # Validate updated params.
      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @parameterized.named_parameters(dtypes_to_test)
  def test_sharing(self, dtype):
    if tf.test.is_gpu_available() and dtype is tf.dtypes.half:
      return

    # Initialize variables for numpy implementation.
    m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)
    opt = yogi.Yogi()

    if not tf.executing_eagerly():
      update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(tf.compat.v1.global_variables_initializer())

    # Fetch params to validate initial values.
    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

    # Run 3 steps of intertwined Yogi1 and Yogi2.
    for t in range(1, 4):
      beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
      self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
      self.assertAllCloseAccordingToType(0.999**t, self.evaluate(beta2_power))
      if not tf.executing_eagerly():
        if t % 2 == 0:
          self.evaluate(update1)
        else:
          self.evaluate(update2)
      else:
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m0, v0 = yogi_update_numpy(var0_np, grads0_np, t, m0, v0)
      var1_np, m1, v1 = yogi_update_numpy(var1_np, grads1_np, t, m1, v1)

      # Validate updated params.
      self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
      self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def test_get_config(self):
    opt = yogi.Yogi(1e-4)
    config = opt.get_config()
    self.assertEqual(config['learning_rate'], 1e-4)


if __name__ == '__main__':
  tf.test.main()
