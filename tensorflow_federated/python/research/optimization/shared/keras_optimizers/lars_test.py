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

from absl import logging
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared.keras_optimizers import lars


def lars_update_numpy(current_x,
                      first_moment,
                      gradient,
                      learning_rate,
                      momentum=0.9,
                      weight_decay_rate=0.0,
                      epsilon=0.0):
  """Performs a LARS update using numpy.

  Args:
    current_x: A numpy array of the current parameter.
    first_moment: A numpy array representing the accumulated first moment
      estimate.
    gradient: A numpy array representing the current gradient.
    learning_rate: A float representing the learning rate.
    momentum: A nonnegative float representing the momentum parameter.
    weight_decay_rate: A nonnegative float representing the weight decay rate.
    epsilon: A nonnegative float used for numerical stability.

  Returns:
    A tuple of the updated x parameters, and the updated first moment.
  """
  current_x = np.array(current_x)
  gradient = np.array(gradient)
  first_moment = np.array(first_moment)

  updated_first_moment = momentum * first_moment + (1 - momentum) * (
      gradient + (weight_decay_rate * current_x))

  ratio = (np.linalg.norm(current_x)) / (
      np.linalg.norm(updated_first_moment) + epsilon)
  updated_x = current_x - ratio * learning_rate * updated_first_moment
  return updated_x, updated_first_moment


DTYPES_TO_TEST = [('float32', tf.float32), ('float64', tf.float64)]


class LarsTest(tf.test.TestCase, parameterized.TestCase):

  def run_lars_steps(self,
                     dtype,
                     learning_rate=0.01,
                     momentum=0.9,
                     weight_decay_rate=0.0,
                     epsilon=0.001,
                     atol=1e-3):

    # Initialize variables for numpy implementation.
    x0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    x1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)

    m0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
    m1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)

    g0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    g1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    x0_tf = tf.Variable(x0_np)
    x1_tf = tf.Variable(x1_np)

    g0_tf = tf.constant(g0_np)
    g1_tf = tf.constant(g1_np)

    opt = lars.LARS(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay_rate=weight_decay_rate,
        epsilon=epsilon)

    # Fetch params to validate initial values.
    self.assertAllClose(x0_np, self.evaluate(x0_tf))
    self.assertAllClose(x1_np, self.evaluate(x1_tf))
    logging.info('x0_tf: %s', x0_tf)
    logging.info('x1_tf: %s', x1_tf)
    logging.info('x0_np: %s', x0_np)
    logging.info('x1_np: %s', x1_np)

    # Run 3 steps of LARS.
    for _ in range(3):
      opt.apply_gradients(zip([g0_tf, g1_tf], [x0_tf, x1_tf]))
      logging.info('Optimizer config: %s', opt.variables())
      logging.info('x0_tf: %s', x0_tf)
      logging.info('x1_tf: %s', x1_tf)

      x0_np, m0_np = lars_update_numpy(
          current_x=x0_np,
          first_moment=m0_np,
          gradient=g0_np,
          learning_rate=learning_rate,
          momentum=momentum,
          weight_decay_rate=weight_decay_rate,
          epsilon=epsilon)

      x1_np, m1_np = lars_update_numpy(
          current_x=x1_np,
          first_moment=m1_np,
          gradient=g1_np,
          learning_rate=learning_rate,
          momentum=momentum,
          weight_decay_rate=weight_decay_rate,
          epsilon=epsilon)

      logging.info('x0_np: %s', x0_np)
      logging.info('x1_np: %s', x1_np)

      # Validate updated params.
      self.assertAllCloseAccordingToType(
          x0_np,
          self.evaluate(x0_tf),
          msg='Updated params 0 do not match in NP and TF',
          atol=atol)
      self.assertAllCloseAccordingToType(
          x1_np,
          self.evaluate(x1_tf),
          msg='Updated params 1 do not match in NP and TF',
          atol=atol)

  @parameterized.named_parameters(DTYPES_TO_TEST)
  def test_lars_on_tf_and_np(self, dtype):
    self.run_lars_steps(dtype)

  @parameterized.named_parameters([('lr1', 0.002), ('lr2', 1.0), ('lr3', 0.05)])
  def test_lars_on_different_lrs(self, learning_rate):
    self.run_lars_steps(dtype=tf.float32, learning_rate=learning_rate)

  @parameterized.named_parameters([('m1', 0.1), ('m2', 0.5), ('m3', 0.0),
                                   ('m4', 1.0)])
  def test_lars_on_different_momentums(self, momentum):
    self.run_lars_steps(dtype=tf.float32, momentum=momentum, epsilon=0.001)

  @parameterized.named_parameters([('w1', 0.0), ('w2', 0.5), ('w3', 1.0)])
  def test_lars_on_different_weight_decay_rates(self, weight_decay_rate):
    self.run_lars_steps(dtype=tf.float32, weight_decay_rate=weight_decay_rate)

  @parameterized.named_parameters([('e1', 0.0), ('e2', 0.01), ('e3', 1.0)])
  def test_lars_on_different_epsilon(self, epsilon):
    self.run_lars_steps(dtype=tf.float32, epsilon=epsilon)


if __name__ == '__main__':
  tf.test.main()
