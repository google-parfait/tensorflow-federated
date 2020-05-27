# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0. Licensed to the Apache
# Software Foundation. You may not use this file except in compliance with the
# License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for Layer-wise Adaptive Rate Scaling optimizer."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.flars import flars_optimizer


class FLARSOptimizerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('tf.float32 m=0', tf.float32, 0), ('tf.float32 m=0.9', tf.float32, 0.9),
      ('tf.float64 m=0', tf.float64, 0), ('tf.float64 m=0.9', tf.float64, 0.9))
  def testFLARSGradientOneStep(self, dtype, momentum):
    shape = [3, 3]
    var_np = np.ones(shape)
    grad_np = np.ones(shape)
    lr_np = 0.1
    m_np = momentum
    ep_np = 1e-5
    eeta = 0.1
    vel_np = np.zeros(shape)

    var = tf.Variable(var_np, dtype=dtype, name='a')
    grad = tf.Variable(grad_np, dtype=dtype)

    opt = flars_optimizer.FLARSOptimizer(
        learning_rate=lr_np, momentum=m_np, eeta=eeta, epsilon=ep_np)

    g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
    opt.update_grads_norm([var], [g_norm])

    self.evaluate(tf.compat.v1.global_variables_initializer())

    pre_var = self.evaluate(var)

    self.assertAllClose(var_np, pre_var)

    opt.apply_gradients([(grad, var)])

    post_var = self.evaluate(var)

    w_norm = np.linalg.norm(var_np.flatten(), ord=2)
    trust_ratio = eeta * w_norm / (g_norm + ep_np)
    scaled_lr = lr_np * trust_ratio

    vel_np = m_np * vel_np - scaled_lr * grad_np
    var_np += vel_np

    self.assertAllClose(var_np, post_var)
    if m_np != 0:
      post_vel = self.evaluate(opt.get_slot(var, 'momentum'))
      self.assertAllClose(vel_np, post_vel)

  @parameterized.named_parameters(
      ('tf.float32 m=0', tf.float32, 0), ('tf.float32 m=0.9', tf.float32, 0.9),
      ('tf.float64 m=0', tf.float64, 0), ('tf.float64 m=0.9', tf.float64, 0.9))
  def testFLARSGradientMultiStep(self, dtype, momentum):
    shape = [3, 3]
    var_np = np.ones(shape)
    grad_np = np.ones(shape)
    lr_np = 0.1
    m_np = momentum
    ep_np = 1e-5
    eeta = 0.1
    vel_np = np.zeros(shape)

    var = tf.Variable(var_np, dtype=dtype, name='a')
    grad = tf.Variable(grad_np, dtype=dtype)
    opt = flars_optimizer.FLARSOptimizer(
        learning_rate=lr_np, momentum=m_np, eeta=eeta, epsilon=ep_np)

    g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
    opt.update_grads_norm([var], [g_norm])

    self.evaluate(tf.compat.v1.global_variables_initializer())

    pre_var = self.evaluate(var)
    self.assertAllClose(var_np, pre_var)

    for _ in range(10):
      opt.apply_gradients([(grad, var)])

      post_var = self.evaluate(var)

      w_norm = np.linalg.norm(var_np.flatten(), ord=2)
      trust_ratio = eeta * w_norm / (g_norm + ep_np)
      scaled_lr = lr_np * trust_ratio

      vel_np = m_np * vel_np - scaled_lr * grad_np
      var_np += vel_np

      self.assertAllClose(var_np, post_var)
      if m_np != 0:
        post_vel = self.evaluate(opt.get_slot(var, 'momentum'))
        self.assertAllClose(vel_np, post_vel)

  @parameterized.named_parameters(('tf.float32', tf.float32),
                                  ('tf.float64', tf.float64))
  def testComputeLRMaxRatio(self, dtype):
    shape = [3, 3]
    var_np = np.ones(shape)
    grad_np = np.ones(shape) * 0.0001

    var = tf.Variable(var_np, dtype=dtype, name='a')
    grad = tf.Variable(grad_np, dtype=dtype)

    base_lr = 1.0
    opt = flars_optimizer.FLARSOptimizer(base_lr)
    scaled_lr = opt.compute_lr(base_lr, grad, var, tf.norm(grad))
    self.assertAlmostEqual(base_lr, scaled_lr)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
