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
"""Tests for shampoo."""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared.keras_optimizers import shampoo


class ShampooTest(tf.test.TestCase):
  """A test that demonstrates the use of matrix preconditioner."""

  def testShampooWithMatrixShapedTensors(self):
    # Parameter matrix of size [4,2] would result in L_{t}, and R_{t} of
    # sizes [4, 4] and [2, 2]
    size = [4, 2]
    init_var_np = np.zeros(size)
    # Initialize gradient as random tensor.
    grad_np = np.random.rand(size[0], size[1])

    var = tf.Variable(init_var_np, dtype=tf.float32)
    grad = tf.constant(grad_np, dtype=tf.float32)

    epsilon = 1e-3
    momentum = 0.9
    opt = shampoo.Shampoo(
        learning_rate=1.0,
        momentum=momentum,
        epsilon=epsilon,
        start_preconditioning_steps=0)

    init_val = self.evaluate(var)
    self.assertAllCloseAccordingToType(init_var_np, init_val)

    def np_power(mat_g, alpha, matrix_epsilon=1e-6):
      """Computes mat_g^alpha for a square symmetric matrix mat_g."""
      mat_for_svd = mat_g + np.eye(mat_g.shape[0]) * matrix_epsilon
      mat_u, diag_d, mat_v = np.linalg.svd(mat_for_svd, full_matrices=True)
      diag_d = np.power(np.maximum(diag_d, matrix_epsilon), alpha)
      return np.dot(mat_u, np.dot(np.diag(diag_d), mat_v))

    def norm(val):
      return np.sqrt(np.sum(np.square(val)))

    opt.apply_gradients(zip([grad], [var]))
    mat_g1 = np.dot(grad_np, grad_np.transpose())
    expected_mat_g1 = self.evaluate(opt.get_slot(var, 'mat_statistics_0'))
    self.assertAllCloseAccordingToType(mat_g1, expected_mat_g1, atol=1e-1)

    mat_g2 = np.dot(grad_np.transpose(), grad_np)
    expected_mat_g2 = self.evaluate(opt.get_slot(var, 'mat_statistics_1'))
    self.assertAllCloseAccordingToType(mat_g2, expected_mat_g2, atol=1e-1)

    mat_left = np_power(mat_g1, -0.25)
    expected_mat_left = self.evaluate(opt.get_slot(var, 'mat_preconditioner_0'))
    self.assertAllCloseAccordingToType(mat_left, expected_mat_left, atol=1e-1)

    mat_right = np_power(mat_g2, -0.25)
    expected_mat_right = self.evaluate(
        opt.get_slot(var, 'mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(mat_right, expected_mat_right, atol=1e-1)

    var_step_1_val = self.evaluate(var)

    # New update has the scale of the second diagonal adagrad update.
    adagrad_update = grad_np / (np.sqrt(np.square(grad_np)) + epsilon) \
        * (1.0 - momentum)
    preconditioned_grad_update = np.dot(np.dot(mat_left, grad_np), mat_right)

    # With normalization by diagonal enabled.
    var_step_1_np = init_var_np - preconditioned_grad_update * norm(
        adagrad_update) / norm(preconditioned_grad_update)
    # TODO(b/167281303) Investigate why the test fails when the precision is
    # less than 1e-1.
    self.assertAllCloseAccordingToType(var_step_1_np, var_step_1_val, atol=1e-1)

    # Gradients are summed over time.
    opt.apply_gradients(zip([grad], [var]))
    mat_g1 += np.dot(grad_np, grad_np.transpose())
    mat_left = np_power(mat_g1, -0.25)
    expected_mat_left = self.evaluate(opt.get_slot(var, 'mat_preconditioner_0'))
    self.assertAllCloseAccordingToType(mat_left, expected_mat_left, atol=1e-1)

    mat_g2 += np.dot(grad_np.transpose(), grad_np)
    mat_right = np_power(mat_g2, -0.25)
    expected_mat_right = self.evaluate(
        opt.get_slot(var, 'mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(mat_right, expected_mat_right, atol=1e-1)

  def testShampooWithMatrixShapedTensorsRightOnlyPreconditioner(self):
    # Parameter matrix of size [4,2] would result in L_{t}, and R_{t} of
    # sizes [4, 4] and [2, 2]. Since max_any_dim is set to 3, it would skip
    # L_{t} and only use R_{t}. The exponent in the inverse used to compute
    # the preconditioner becomes -1/2.
    size = [4, 2]
    init_var_np = np.zeros(size)
    # Initialize gradient as random tensor.
    grad_np = np.random.rand(size[0], size[1])

    var = tf.Variable(init_var_np, dtype=tf.float32)
    grad = tf.constant(grad_np, dtype=tf.float32)

    epsilon = 1e-3
    momentum = 0.9
    opt = shampoo.Shampoo(
        learning_rate=1.0,
        momentum=momentum,
        epsilon=epsilon,
        fallback_to_diagonal_dim=3,
        start_preconditioning_steps=0,
    )

    init_val = self.evaluate(var)
    self.assertAllCloseAccordingToType(init_var_np, init_val)

    def np_power(mat_g, alpha, matrix_epsilon=1e-6):
      """Computes mat_g^alpha for a square symmetric matrix mat_g."""
      mat_for_svd = mat_g + np.eye(mat_g.shape[0]) * matrix_epsilon
      mat_u, diag_d, mat_v = np.linalg.svd(mat_for_svd, full_matrices=True)
      diag_d = np.power(np.maximum(diag_d, matrix_epsilon), alpha)
      return np.dot(mat_u, np.dot(np.diag(diag_d), mat_v))

    def norm(val):
      return np.sqrt(np.sum(np.square(val)))

    # Run a single step of gradient update.
    opt.apply_gradients(zip([grad], [var]))

    mat_g2 = np.dot(grad_np.transpose(), grad_np)
    expected_mat_g2 = self.evaluate(opt.get_slot(var, 'mat_statistics_1'))
    self.assertAllCloseAccordingToType(mat_g2, expected_mat_g2, atol=1e-1)

    mat_right = np_power(mat_g2, -0.5)
    expected_mat_right = self.evaluate(
        opt.get_slot(var, 'mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(mat_right, expected_mat_right, atol=1e-1)

    var_step_1_val = self.evaluate(var)

    # New update has the scale of the second diagonal adagrad update.
    adagrad_update = grad_np / (np.sqrt(np.square(grad_np)) + epsilon) \
        * (1.0 - momentum)
    preconditioned_grad_update = np.matmul(grad_np, mat_right)

    # With normalization by diagonal enabled.
    var_step_1_np = init_var_np - preconditioned_grad_update * norm(
        adagrad_update) / norm(preconditioned_grad_update)

    self.assertAllCloseAccordingToType(var_step_1_np, var_step_1_val, atol=1e-1)

    # Gradients are summed over time.
    opt.apply_gradients(zip([grad], [var]))
    mat_g2 += np.dot(grad_np.transpose(), grad_np)
    mat_right = np_power(mat_g2, -0.5)
    expected_mat_right = self.evaluate(
        opt.get_slot(var, 'mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(mat_right, expected_mat_right, atol=1e-1)

  def testShampooWithMatrixShapedTensorsWithBlocks(self):
    # Parameter matrix of size [4,2] would result in 4 L_{t}, and R_{t} of
    # sizes [2, 2] and [2, 2].
    size = [4, 2]
    init_var_np = np.zeros(size)
    # Initialize gradient as random tensor.
    grad_np = np.random.rand(size[0], size[1])

    var = tf.Variable(init_var_np, dtype=tf.float32)
    grad = tf.constant(grad_np, dtype=tf.float32)

    epsilon = 1e-3
    momentum = 0.9
    opt = shampoo.Shampoo(
        learning_rate=1.0,
        momentum=momentum,
        epsilon=epsilon,
        block_partition_threshold_size=3,
        block_size=2,
        start_preconditioning_steps=0,
    )

    init_val = self.evaluate(var)
    self.assertAllCloseAccordingToType(init_var_np, init_val)

    def np_power(mat_g, alpha, matrix_epsilon=1e-6):
      """Computes mat_g^alpha for a square symmetric matrix mat_g."""
      mat_for_svd = mat_g + np.eye(mat_g.shape[0]) * matrix_epsilon
      mat_u, diag_d, mat_v = np.linalg.svd(mat_for_svd, full_matrices=True)
      diag_d = np.power(np.maximum(diag_d, matrix_epsilon), alpha)
      return np.dot(mat_u, np.dot(np.diag(diag_d), mat_v))

    def norm(val):
      return np.sqrt(np.sum(np.square(val)))

    # Run a single step of gradient update.
    opt.apply_gradients(zip([grad], [var]))

    block_0_grad_np = grad_np[:2, :2]
    block_1_grad_np = grad_np[2:4, :2]

    block_0_mat_g1 = np.dot(block_0_grad_np, block_0_grad_np.transpose())
    expected_block_0_mat_g1 = self.evaluate(
        opt.get_slot(var, '0_mat_statistics_0'))

    self.assertAllCloseAccordingToType(
        block_0_mat_g1, expected_block_0_mat_g1, atol=1e-1)

    block_0_mat_g2 = np.dot(block_0_grad_np.transpose(), block_0_grad_np)
    expected_block_0_mat_g2 = self.evaluate(
        opt.get_slot(var, '0_mat_statistics_1'))
    self.assertAllCloseAccordingToType(
        block_0_mat_g2, expected_block_0_mat_g2, atol=1e-1)

    block_1_mat_g1 = np.dot(block_1_grad_np, block_1_grad_np.transpose())
    expected_block_1_mat_g1 = self.evaluate(
        opt.get_slot(var, '1_mat_statistics_0'))
    self.assertAllCloseAccordingToType(
        block_1_mat_g1, expected_block_1_mat_g1, atol=1e-1)

    block_1_mat_g2 = np.dot(block_1_grad_np.transpose(), block_1_grad_np)
    expected_block_1_mat_g2 = self.evaluate(
        opt.get_slot(var, '1_mat_statistics_1'))
    self.assertAllCloseAccordingToType(
        block_1_mat_g2, expected_block_1_mat_g2, atol=1e-1)

    block_0_mat_left = np_power(block_0_mat_g1, -0.25)
    expected_block_0_mat_left = self.evaluate(
        opt.get_slot(var, '0_mat_preconditioner_0'))
    self.assertAllCloseAccordingToType(
        block_0_mat_left, expected_block_0_mat_left, atol=1e-1)

    block_0_mat_right = np_power(block_0_mat_g2, -0.25)
    expected_block_0_mat_right = self.evaluate(
        opt.get_slot(var, '0_mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(
        block_0_mat_right, expected_block_0_mat_right, atol=1e-1)

    block_1_mat_left = np_power(block_1_mat_g1, -0.25)
    expected_block_1_mat_left = self.evaluate(
        opt.get_slot(var, '1_mat_preconditioner_0'))
    self.assertAllCloseAccordingToType(
        block_1_mat_left, expected_block_1_mat_left, atol=1e-1)

    block_1_mat_right = np_power(block_1_mat_g2, -0.25)
    expected_block_1_mat_right = self.evaluate(
        opt.get_slot(var, '1_mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(
        block_1_mat_right, expected_block_1_mat_right, atol=1e-1)

    var_step_1_val = self.evaluate(var)

    # New update has the scale of the second diagonal adagrad update.
    adagrad_update = grad_np / (np.sqrt(np.square(grad_np)) + epsilon) \
        * (1.0 - momentum)

    block_0_update = np.dot(
        np.dot(block_0_mat_left, block_0_grad_np), block_0_mat_right)
    block_1_update = np.dot(
        np.dot(block_1_mat_left, block_1_grad_np), block_1_mat_right)
    preconditioned_grad_update = np.concatenate(
        (block_0_update, block_1_update), axis=0)
    # With normalization by diagonal enabled.
    var_step_1_np = init_var_np - preconditioned_grad_update * norm(
        adagrad_update) / norm(preconditioned_grad_update)
    self.assertAllCloseAccordingToType(var_step_1_np, var_step_1_val, atol=1e-1)

    # Gradients are summed over time.
    opt.apply_gradients(zip([grad], [var]))
    block_0_mat_g1 += np.dot(block_0_grad_np, block_0_grad_np.transpose())
    block_0_mat_left = np_power(block_0_mat_g1, -0.25)
    expected_block_0_mat_left = self.evaluate(
        opt.get_slot(var, '0_mat_preconditioner_0'))
    self.assertAllCloseAccordingToType(
        block_0_mat_left, expected_block_0_mat_left, atol=1e-1)

    block_0_mat_g2 += np.dot(block_0_grad_np.transpose(), block_0_grad_np)
    block_0_mat_right = np_power(block_0_mat_g2, -0.25)
    expected_block_0_mat_right = self.evaluate(
        opt.get_slot(var, '0_mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(
        block_0_mat_right, expected_block_0_mat_right, atol=1e-1)

    block_1_mat_g1 += np.dot(block_1_grad_np, block_1_grad_np.transpose())
    block_1_mat_left = np_power(block_1_mat_g1, -0.25)
    expected_block_1_mat_left = self.evaluate(
        opt.get_slot(var, '1_mat_preconditioner_0'))
    self.assertAllCloseAccordingToType(
        block_1_mat_left, expected_block_1_mat_left, atol=1e-1)

    block_1_mat_g2 += np.dot(block_1_grad_np.transpose(), block_1_grad_np)
    block_1_mat_right = np_power(block_1_mat_g2, -0.25)
    expected_block_1_mat_right = self.evaluate(
        opt.get_slot(var, '1_mat_preconditioner_1'))
    self.assertAllCloseAccordingToType(
        block_1_mat_right, expected_block_1_mat_right, atol=1e-1)


class PartitionConfigTest(tf.test.TestCase):
  """Partition config tests."""

  def testPartitionConfig(self):
    with self.assertRaises(ValueError):
      shampoo.PartitionConfig(-1, 2)

    with self.assertRaises(ValueError):
      shampoo.PartitionConfig(2, -1)

    with self.assertRaises(ValueError):
      shampoo.PartitionConfig(2, 3)


class TensorPartitionerTest(tf.test.TestCase):
  """Tensor partitioner tests."""

  def testTensorPartitioner(self):
    initial_value = np.ones((255, 255))
    w1 = tf.Variable(initial_value, dtype=tf.float32)
    partition_info = shampoo.PartitionConfig(200, 128)
    grad = tf.constant(initial_value)
    metadata = shampoo.partition_metadata(w1, partition_info)
    partitioned_grad = shampoo.partition_tensor(w1, partition_info)
    reformed_grad = shampoo.reform_tensor(partitioned_grad,
                                          metadata.num_splits_per_dim)
    self.assertAllCloseAccordingToType(reformed_grad, grad)

  def testPartitionMetadata(self):
    initial_value = np.ones((255, 255))
    w1 = tf.Variable(initial_value, dtype=tf.float32)
    partition_info = shampoo.PartitionConfig(200, 128)
    metadata = shampoo.partition_metadata(w1, partition_info)
    self.assertAllEqual(metadata.split_sizes_per_dim, [[128, 127], [128, 127]])
    self.assertAllEqual(metadata.num_splits_per_dim, [2, 2])

  def testPartitionTensor(self):
    initial_value = np.ones((255, 255))
    w1 = tf.Variable(initial_value, dtype=tf.float32)
    partition_info = shampoo.PartitionConfig(200, 128)
    partitioned_grad = shampoo.partition_tensor(w1, partition_info)
    partitioned_shape = [grad.get_shape() for grad in partitioned_grad]
    self.assertEqual(partitioned_shape,
                     [[128, 128], [127, 128], [128, 127], [127, 127]])


if __name__ == '__main__':
  tf.test.main()
