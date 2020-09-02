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
"""Keras implementation of the Shampoo optimizer.

Original paper: [Second Order Optimization Made Practical]
(https://arxiv.org/pdf/2002.09018.pdf).

Code adapted from Algorithm 1 in [Second Order Optimization Made Practical]:
https://github.com/tensorflow/lingvo/blob/master/lingvo/core/distributed_shampoo.py
The new features in federated shampoo optimizer:
  * Computing preconditioner on the server.
  * Selecting to use gradient_norm_adjuster, 'adagrad' or None.
"""

import functools
from typing import Sequence, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

TfValue = Union[tf.Variable, tf.Tensor]


class PartitionConfig:
  """Config for tensor partitioning."""

  def __init__(self, max_dim_size: int, partition_size: int):
    """Initialize the `PartitionConfig`.

    Args:
      max_dim_size: Partitions dimensions with size greater than this value.
      partition_size: Size of each partition
    """
    if partition_size < 1 or partition_size > max_dim_size:
      raise ValueError('Parition size must be no less than 1 and no greater'
                       'than max_dim_size.')

    self.max_dim_size = max_dim_size
    self.partition_size = partition_size


class PartitionMetadata:
  """Metadata for partitioning."""

  def __init__(self, split_sizes_per_dim: Sequence[Sequence[int]],
               num_splits_per_dim: Sequence[int]):
    """Initialize the `PartitionMetadata`.

    Args:
      split_sizes_per_dim: Split sizes per dimemsion.
      num_splits_per_dim: Number of splits per dimension (inferred from
        split_sizes_per_dim).
    """
    self.split_sizes_per_dim = split_sizes_per_dim
    self.num_splits_per_dim = num_splits_per_dim


def partition_metadata(
    tensor: TfValue,
    partition_info: PartitionConfig,
) -> PartitionMetadata:
  """Returns metadata required for partitioning and reforming tensors.

  Args:
    tensor: Tensor to partition.
    partition_info: Partitioning info.

  Returns:
    split_sizes_per_dim and num_splits_per_dim.
  """
  shape = tensor.get_shape()
  # Split if dim is greater than max_dim.
  axis_to_shard = [s > partition_info.max_dim_size for s in shape]
  split_sizes_per_dim = []
  # Compute the number of splits, and the sizes of splits for each dimension.
  for sharded, dim in zip(axis_to_shard, shape):
    dim = int(dim)
    if sharded:
      num_shards = dim // partition_info.partition_size
      last_shard_size = dim % partition_info.partition_size
      split_sizes = [partition_info.partition_size] * num_shards
      if last_shard_size > 0:
        split_sizes.append(last_shard_size)
      split_sizes_per_dim.append(split_sizes)
    else:
      split_sizes_per_dim.append([dim])
  num_splits_per_dim = [len(v) for v in split_sizes_per_dim]
  return PartitionMetadata(split_sizes_per_dim, num_splits_per_dim)


def partition_tensor(tensor: TfValue,
                     partition_info: PartitionConfig) -> List[TfValue]:
  """Returns partitioned tensors."""
  metadata = (partition_metadata(tensor, partition_info))
  # Split from last to first axis.
  partitioned_tensors = [tensor]
  rank = len(metadata.num_splits_per_dim)
  for raxis, (num_splits, sizes) in enumerate(
      zip(
          reversed(metadata.num_splits_per_dim),
          reversed(metadata.split_sizes_per_dim))):
    if num_splits > 1:
      tmp_partitioned_tensors = []
      for item in partitioned_tensors:
        tmp_partitioned_tensors += tf.split(item, sizes, axis=rank - raxis - 1)
      partitioned_tensors = tmp_partitioned_tensors
  return partitioned_tensors


def reform_tensor(partitioned_tensors: Sequence[TfValue],
                  num_splits_per_dim: Sequence[int]) -> TfValue:
  """Returns a tensor concatenated from the given partitions."""
  # Concatenates tensors across all dimension. Assumes the `partitions` tensor
  # was created by partition_tensor.
  for axis, num_splits in enumerate(num_splits_per_dim):
    if num_splits > 1:
      tmp_partitioned_tensors = []
      num_concat = len(partitioned_tensors) // num_splits
      for i in range(num_concat):
        tensors_to_concat = (
            partitioned_tensors[i * num_splits:(i + 1) * num_splits])
        tmp_partitioned_tensors.append(tf.concat(tensors_to_concat, axis=axis))
      partitioned_tensors = tmp_partitioned_tensors
  return partitioned_tensors[0]


class Shampoo(tf.keras.optimizers.Optimizer):
  """Approximates full-matrix AdaGrad per layer.

  Approximates full-matrix AdaGrad with kronecker-products of two statistics
  matrices based on only the first-order gradients of the layer.

  "Second-order optimization made practical.", 2019
  Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer.
  """

  def __init__(self,
               learning_rate: int,
               momentum: float = 0.9,
               gradient_norm_adjuster: Optional[str] = 'adagrad',
               initial_accumulator_value: float = 0.0,
               start_preconditioning_steps: int = 10,
               statistics_computation_frequency: int = 1,
               epsilon: float = 1e-3,
               matrix_epsilon: float = 1e-6,
               second_moment_averaging: float = 1.0,
               fallback_to_diagonal_dim: int = 4096,
               max_any_dim: int = 6656,
               block_size: int = 4096,
               block_partition_threshold_size: int = 1000000,
               exponent_multiplier: float = 1.0,
               name: str = 'Shampoo',
               **kwargs):
    """Construct a Shampoo optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value. Momentum is not applied to
        sparse updates. Similar to beta_1 in Adam.
      gradient_norm_adjuster: A string value which indicates the optimizer used
        to adjust the norm of the gradient. Supported ['adagrad', None]
      initial_accumulator_value: A floating point value.
      start_preconditioning_steps: A int32 value which indicates when to start
        preconditioning.
      statistics_computation_frequency: A int32 step value which indicates how
        often to compute statistics for preconditioning.
      epsilon: An epsilon value for diagnoal second-order moment.
      matrix_epsilon: An epsilon regularizer to make the matrices positive
        definite.
      second_moment_averaging: 1.0 means sum of gradients squares, while less
        than 1.0 switches to RMSProp style exponential moving averages of the
        second moments.
      fallback_to_diagonal_dim: Fallback to diagonal version of AFMA if the any
        of the dimension is larger than fallback_to_diagonal_dim.
      max_any_dim: If maximum value for any dimension is greater than this value
        we skip preconditioning and fall back to the diagonal.
      block_size: Dimension of the partitioned tensors.
      block_partition_threshold_size: Partitions diemnsions beyond this size.
      exponent_multiplier: A multiplier 'e` for the exponent for the inverse
        calculation. e * -1/(2*rank). Only applies when calculating inverses
        through svd.
      name: Optional name prefix for the operations created when applying
        gradients.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. The kwarg `lr` is included
        for backward compatibility, it is recommended to use `learning_rate`
        instead.
    """
    super().__init__(name, **kwargs)
    self._set_hyper('learning_rate', learning_rate)
    self._set_hyper('epsilon', epsilon)
    self._set_hyper('first_moment_averaging', momentum)
    self._set_hyper('start_preconditioning_steps', start_preconditioning_steps)
    self._set_hyper('matrix_epsilon', matrix_epsilon)
    self._set_hyper('second_moment_averaging', second_moment_averaging)

    # Computes statistics every K steps.
    self._statistics_computation_frequency = statistics_computation_frequency
    self._max_any_dim = max_any_dim
    self._block_size = block_size
    self._fallback_to_diagonal_dim = fallback_to_diagonal_dim
    self._second_moment_averaging = second_moment_averaging
    self._first_moment_averaging = momentum
    self._exponent_multiplier = exponent_multiplier
    self._initial_accumulator_value = initial_accumulator_value
    self._gradient_norm_adjuster = gradient_norm_adjuster

    # All vars that are preconditioned.
    self._all_vars_for_preconditioning = []
    self._partition_info = PartitionConfig(block_partition_threshold_size,
                                           block_size)
    self._partitioner_metadata = {}

  def _fallback_to_diagonal_for_shape(self, shape):
    """Returns whether we should fallback to the diagonal update given shape."""
    if len(shape) <= 1:
      return True
    if any(d > self._max_any_dim for d in shape):
      return True
    if all(d == 1 for d in shape):
      return True
    return False

  def _preconditioner_available_for_dims(
      self, shape: Sequence[tf.TensorShape]) -> List[bool]:
    """Returns indicator vector if preconditioner exists for each axis."""
    # If any of the dims < fallback_to_diagonal_dim and not 1, we run a
    # a preconditioner for that particular dimension.
    return [d <= self._fallback_to_diagonal_dim and d != 1 for d in shape]

  def _preconditioner_indices(self,
                              shape: Sequence[tf.TensorShape]) -> List[int]:
    """Returns indices of the available preconditioner."""
    preconditioners_available = self._preconditioner_available_for_dims(shape)
    indices = np.cumsum(preconditioners_available) - 1
    size = len(indices)
    return [
        indices[index] if avail else size
        for index, avail in enumerate(preconditioners_available)
    ]

  def _make_named_slot(self, var: TfValue, val: TfValue, slot_name: str):
    _ = self.add_slot(var, slot_name, initializer=val)

  def make_named_zeros_slot(self, var: TfValue, slot_name: str):
    self.add_slot(var, slot_name, initializer='zeros')

  def _generalized_inverse_pth_root(
      self,
      input_t: TfValue,
      exponent: float,
      epsilon: float = 1e-12) -> Tuple[float, float]:
    """Compute inverse of a square matrix."""
    input_t_f64 = tf.cast(input_t, tf.float64)
    epsilon_f64 = tf.cast(epsilon, tf.float64)
    s, u, v = tf.linalg.svd(
        input_t_f64 +
        tf.eye(tf.shape(input_t_f64)[0], dtype=tf.float64) * epsilon_f64,
        full_matrices=True)
    inv_s = tf.reshape(
        tf.pow(tf.maximum(s, epsilon_f64), tf.cast(exponent, tf.float64)),
        [1, -1])
    val = tf.matmul(u * inv_s, v, adjoint_b=True)
    return tf.cast(val, tf.float32), tf.reduce_max(tf.abs(u - v))

  def _inverse_pth_root(self,
                        input_t: TfValue,
                        exponent_t: float,
                        epsilon: float = 1e-12) -> Tuple[float, float]:
    # Apply exponent multiplier.
    exponent_t = exponent_t * self._exponent_multiplier
    output, diff = self._generalized_inverse_pth_root(input_t, exponent_t,
                                                      epsilon)
    return output, diff

  def _create_slots_for_preconditioning(self, v: TfValue):
    if self._first_moment_averaging > 0.0:
      self.make_named_zeros_slot(v, 'precond_grad_momentum')
    partitioned_v = partition_tensor(v, self._partition_info)
    num_partitions = len(partitioned_v)
    for pt_idx, pt_v in enumerate(partitioned_v):
      pt_v_shape = pt_v.get_shape()
      preconditioner_exists_for_dim = (
          self._preconditioner_available_for_dims(pt_v_shape))
      for i, d in enumerate(pt_v_shape):
        if preconditioner_exists_for_dim[i]:
          mat_stat_init = tf.zeros([d, d], dtype=pt_v.dtype)
          self._make_named_slot(
              v, mat_stat_init,
              self._statistics_key_for_partition_and_dim(
                  i, pt_idx, num_partitions))
          self._make_named_slot(
              v, mat_stat_init,
              self._preconditioner_key_for_partition_and_dim(
                  i, pt_idx, num_partitions))

  def _create_slots(self, var_list: Sequence[TfValue]):
    for v in var_list:
      self._make_named_slot(v,
                            tf.ones_like(v) * self._initial_accumulator_value,
                            'accumulator')
      if self._first_moment_averaging > 0.0:
        self.make_named_zeros_slot(v, 'momentum')
      shape = np.array(v.get_shape())
      self._partitioner_metadata[v.ref()] = partition_metadata(
          v, self._partition_info)
      if not self._fallback_to_diagonal_for_shape(shape):
        self._all_vars_for_preconditioning.append(v)
        self._create_slots_for_preconditioning(v)

  def _statistics_key_for_partition_and_dim(self, dim_index: int,
                                            partition_index: int,
                                            num_partitions: int) -> str:
    if num_partitions == 1:
      return 'mat_statistics_' + str(dim_index)
    else:
      return str(partition_index) + '_mat_statistics_' + str(dim_index)

  def _preconditioner_key_for_partition_and_dim(self, dim_index: int,
                                                partition_index: int,
                                                num_partitions: int) -> str:
    if num_partitions == 1:
      return 'mat_preconditioner_' + str(dim_index)
    else:
      return str(partition_index) + '_mat_preconditioner_' + str(dim_index)

  def _key_for_var(self, var: TfValue, dim_index: int, partition_index: int):
    return 'P_' + str(partition_index) + '_D_' + str(dim_index) + '_' + var.name

  def _updated_statistics(self, var: TfValue,
                          partitioned_grads: Sequence[TfValue]):
    """Returns updated Shampoo statistics L_t, R_t, etc.

    Args:
      var: tf.Variable associated with the gradient.
      partitioned_grads: Partitioned gradient tensor.

    Returns:
      A list of updated statistics matrices.
    """
    var_dtype = var.dtype.base_dtype
    second_moment_averaging = self._get_hyper('second_moment_averaging',
                                              var_dtype)
    precond_statistics_update = []
    num_partitions = len(partitioned_grads)
    mat_stats = []
    mat_grads = []
    mat_dims = []
    for pt_idx, pt_grad in enumerate(partitioned_grads):
      pt_shape = pt_grad.get_shape()
      preconditioner_exists_for_dim = (
          self._preconditioner_available_for_dims(pt_shape))
      rank = len(pt_shape)
      # Calculates the preconditioner statistics for each tensor.
      for i in range(rank):
        if preconditioner_exists_for_dim[i]:
          mat_stats.append(
              self.get_slot(
                  var,
                  self._statistics_key_for_partition_and_dim(
                      i, pt_idx, num_partitions)))
          mat_grads.append(pt_grad)
          mat_dims.append(i)

    # axes is the list of indices to reduce - everything but
    # the current i.
    def _update_statistics(dim, stat_var: TfValue, grad: TfValue) -> TfValue:
      """Update preconditioner statistics."""
      var_rank = len(grad.get_shape())
      axes = list(range(dim)) + list(range(dim + 1, var_rank))
      new_stat = tf.tensordot(grad, grad, axes=(axes, axes))
      if self._second_moment_averaging == 1.0:
        updated_stat = stat_var.assign_add(
            new_stat, use_locking=self._use_locking)
      else:
        updated_stat = stat_var.assign_add(
            (second_moment_averaging - 1.0) * stat_var +
            (1.0 - second_moment_averaging) * new_stat,
            use_locking=self._use_locking)
      return updated_stat

    local_steps = tf.cast(self.iterations, tf.int32)
    if self._statistics_computation_frequency <= 1:
      for mat_stat, mat_grad, dim in zip(mat_stats, mat_grads, mat_dims):
        precond_statistics_update.append(
            _update_statistics(dim, mat_stat, mat_grad))
    else:

      # NOTE: We rewrite tf.cond() as a while loop to avoid certain overheads
      # in XLA from buffer allocation.
      def _loop_body(mat_stats, mat_grads, mat_dims, unused_perform_step):
        precond_statistics_update_ops = []
        for mat_stat, mat_grad, dim in zip(mat_stats, mat_grads, mat_dims):
          precond_statistics_update_ops.append(
              _update_statistics(dim, mat_stat, mat_grad))
        with tf.control_dependencies(precond_statistics_update_ops):
          return tf.constant(False)

      loop_body_fn = functools.partial(_loop_body, mat_stats, mat_grads,
                                       mat_dims)
      run_statistics_computation = tf.equal(
          tf.math.floormod(
              local_steps,
              tf.cast(self._statistics_computation_frequency, tf.int32)), 0)
      tf.while_loop(lambda perform_step: perform_step, loop_body_fn,
                    [run_statistics_computation])
    return precond_statistics_update

  def _compute_preconditioned_raw_grad(
      self, var: TfValue, partitioned_grads: Sequence[TfValue]) -> TfValue:
    """Returns preconditioned gradient.

    Args:
      var: tf.Variable associated with the gradient.
      partitioned_grads: Partitioned gradient tensor.

    Returns:
      A preconditioned gradient tensor.
    """
    var_dtype = var.dtype.base_dtype
    matrix_epsilon = self._get_hyper('matrix_epsilon', var_dtype)
    partitioned_preconditioned_grads = []
    num_partitions = len(partitioned_grads)
    for pt_idx, pt_grad in enumerate(partitioned_grads):
      pt_shape = pt_grad.get_shape()
      rank = len(pt_shape)
      preconditioner_exists_for_dim = (
          self._preconditioner_available_for_dims(pt_shape))
      preconditioner_indices = self._preconditioner_indices(pt_shape)
      mat_preconditioner_list = []
      for i in range(rank):
        if preconditioner_exists_for_dim[i]:
          mat_preconditioner = self.get_slot(
              var,
              self._preconditioner_key_for_partition_and_dim(
                  i, pt_idx, num_partitions))
          mat_stat = self.get_slot(
              var,
              self._statistics_key_for_partition_and_dim(
                  i, pt_idx, num_partitions))
          mat_stat_inverse, _ = self._inverse_pth_root(
              mat_stat, -1.0 / (2.0 * sum(preconditioner_exists_for_dim)),
              matrix_epsilon)
          mat_preconditioner_list.append(
              mat_preconditioner.assign(
                  mat_stat_inverse, use_locking=self._use_locking))

      precond_grad = pt_grad
      if rank == 2 and all(preconditioner_exists_for_dim):
        # Fast path for speedup.
        precond_grad = tf.matmul(
            tf.matmul(mat_preconditioner_list[0], precond_grad),
            mat_preconditioner_list[1])
      else:
        for i in range(rank):
          if preconditioner_exists_for_dim[i]:
            precond_grad = tf.tensordot(
                precond_grad,
                mat_preconditioner_list[preconditioner_indices[i]],
                axes=([0], [0]))
          else:
            # if preconditioner is not available we transpose it to
            # permute the axis for the next preconditioner.
            precond_grad = tf.transpose(
                precond_grad, perm=list(range(1, rank)) + [0])

      partitioned_preconditioned_grads.append(precond_grad)
    return reform_tensor(
        partitioned_preconditioned_grads,
        self._partitioner_metadata[var.ref()].num_splits_per_dim)

  def _preconditioned_update(self, var: TfValue,
                             partitioned_grads: Sequence[TfValue],
                             diagonal_grad_update: TfValue) -> TfValue:
    """Computes the matrix preconditioned update.

    Args:
      var: Variable for which we are computing the preconditioned gradient.
      partitioned_grads: Partitioned gradients.
      diagonal_grad_update: Update as given by diagonal adagrad.

    Returns:
      scaled preconditioned gradient.
    """

    var_dtype = var.dtype.base_dtype
    first_moment_averaging = self._get_hyper('first_moment_averaging',
                                             var_dtype)
    local_steps = tf.cast(self.iterations, var_dtype)
    first_moment_averaging_t = tf.pow(first_moment_averaging, local_steps + 1)
    precond_grad = self._compute_preconditioned_raw_grad(var, partitioned_grads)
    if self._first_moment_averaging > 0.0:
      gbar = self.get_slot(var, 'precond_grad_momentum')
      matrix_preconditioned_grad = gbar.assign(
          gbar * first_moment_averaging_t + precond_grad *
          (1.0 - first_moment_averaging_t),
          use_locking=self._use_locking)
    else:
      matrix_preconditioned_grad = precond_grad

    # We use the direction from Shampoo while using the step size scale from
    # adaptive optimizers.
    precond_l2_norm = tf.norm(matrix_preconditioned_grad, ord=2)
    diagonal_l2_norm = tf.norm(diagonal_grad_update, ord=2)
    multiplier = tf.where(
        tf.greater(precond_l2_norm, 0.0),
        tf.maximum(diagonal_l2_norm, 1e-30) /
        (tf.maximum(precond_l2_norm, 1e-30)), 1.0)
    return matrix_preconditioned_grad * multiplier

  def _apply_dense(self, grad: TfValue, var: TfValue):
    """Shampoo Computes preconditioner periodically and updates model."""
    partitioned_grads = partition_tensor(grad, self._partition_info)
    shape = var.get_shape()
    fallback_to_diagonal = self._fallback_to_diagonal_for_shape(shape)

    precond_statistics_update = []
    if not fallback_to_diagonal:
      precond_statistics_update = self._updated_statistics(
          var, partitioned_grads)

    var_dtype = var.dtype.base_dtype
    lr_t = self._get_hyper('learning_rate', var_dtype)
    beta1_t = self._get_hyper('first_moment_averaging', var_dtype)
    epsilon_t = self._get_hyper('epsilon', var_dtype)
    start_preconditioning_steps = self._get_hyper('start_preconditioning_steps',
                                                  var_dtype)
    local_step = tf.cast(self.iterations + 1, var_dtype)

    lr = lr_t
    if self._gradient_norm_adjuster is None:
      per_coord_lr = 1.0
    elif self._gradient_norm_adjuster == 'adagrad':
      v = self.get_slot(var, 'accumulator')
      v_t = v.assign_add(grad * grad)
      v_sqrt = tf.sqrt(v_t)
      per_coord_lr = 1.0 / (v_sqrt + epsilon_t)
    else:
      raise NotImplementedError('Gradient norm adjuster %s is not supported!' %
                                self._gradient_norm_adjuster)

    update_vs = []
    if self._first_moment_averaging > 0.0:
      # m_t = beta1 * m + (1 - beta1) * g_t
      scaled_g = (1.0 - beta1_t) * (grad * per_coord_lr)
      m = self.get_slot(var, 'momentum')
      m_t = m.assign(m * beta1_t + scaled_g, use_locking=self._use_locking)
      gbar_updated = m_t
    else:
      gbar_updated = per_coord_lr * grad
      m_t = tf.no_op()

    if not fallback_to_diagonal:
      # Update the preconditioner statistics followed by computing the
      # preconditioned gradient.
      with tf.control_dependencies(precond_statistics_update):
        s = tf.cast(
            tf.greater_equal(
                tf.cast(local_step, tf.int32),
                tf.cast(start_preconditioning_steps, tf.int32)), tf.float32)
        preconditioned_grad = self._preconditioned_update(
            var, partitioned_grads, gbar_updated)
        # slowly adapt from diagonal to preconditioned gradient.
        w = tf.minimum(1.0, tf.maximum(
            (local_step - start_preconditioning_steps) \
            / start_preconditioning_steps, 0.0))
        warmup_update = s * lr * (
            w * preconditioned_grad + (1.0 - w) * gbar_updated)
        fallback_update = (1 - s) * (lr * gbar_updated)
        var_update = var.assign_sub(
            warmup_update + fallback_update, use_locking=self._use_locking)
    else:
      var_update = var.assign_sub(
          lr * gbar_updated, use_locking=self._use_locking)

    update_vs.append(var_update)
    update_vs.append(m_t)
    update_vs.append(v_t)

    # Create an op that groups all the above operations
    return tf.group(*update_vs)

  def _resource_apply_dense(self, grad: TfValue, var: TfValue):
    return self._apply_dense(grad, var)

  # Sparse gradients are not handled currently and is part of future work.
  def _resource_apply_sparse(self, grad_values, var, grad_indices):
    raise NotImplementedError

  def _apply_sparse(self, grad, var):
    raise NotImplementedError

  def get_config(self):
    config = super().get_config()
    config.update({
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'first_moment_averaging':
            self._serialize_hyperparameter('first_moment_averaging'),
        'second_moment_averaging':
            self._serialize_hyperparameter('second_moment_averaging'),
        'start_preconditioning_steps':
            self._serialize_hyperparameter('start_preconditioning_steps'),
        'matrix_epsilon':
            self._serialize_hyperparameter('max_epsilon'),
        'max_any_dim':
            self._max_any_dim,
        'block_size':
            self._block_size,
        'exponent_multiplier':
            self._exponent_multiplier,
        'initial_accumulator_value':
            self._initial_accumulator_value,
        'fallback_to_diagonal_dim':
            self._fallback_to_diagonal_dim,
        'statistics_computation_frequency':
            self._statistics_computation_frequency,
        'gradient_norm_adjuster':
            self._gradient_norm_adjuster,
    })
    return config
