# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for testing executors."""

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import set_default_executor
from tensorflow_federated.python.core.utils import tf_computation_utils

_mnist_model_type = computation_types.NamedTupleType([
    ('weights', computation_types.TensorType(tf.float32, [784, 10])),
    ('bias', computation_types.TensorType(tf.float32, [10]))
])

_mnist_batch_type = computation_types.NamedTupleType([
    ('x', computation_types.TensorType(tf.float32, [None, 784])),
    ('y', computation_types.TensorType(tf.int32, [None]))
])

_mnist_sample_batch = collections.OrderedDict([('x',
                                                np.ones([1, 784],
                                                        dtype=np.float32)),
                                               ('y', np.ones([1],
                                                             dtype=np.int32))])

_mnist_initial_model = collections.OrderedDict([
    ('weights', np.zeros([784, 10], dtype=np.float32)),
    ('bias', np.zeros([10], dtype=np.float32))
])


@computations.tf_computation(_mnist_model_type, _mnist_batch_type)
def _mnist_batch_loss(model, batch):
  predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
  return -tf.reduce_mean(
      tf.reduce_sum(
          tf.one_hot(batch.y, 10) * tf.log(predicted_y), reduction_indices=[1]))


@computations.tf_computation(_mnist_model_type, _mnist_batch_type)
def _mnist_batch_train(model, batch):
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  model_vars = tf_computation_utils.create_variables('v', _mnist_model_type)
  assign_vars_op = tf_computation_utils.assign(model_vars, model)
  with tf.control_dependencies([assign_vars_op]):
    train_op = optimizer.minimize(_mnist_batch_loss(model_vars, batch))
    with tf.control_dependencies([train_op]):
      return tf_computation_utils.identity(model_vars)


def test_mnist_training(test_obj, executor):
  """Tests `executor` against MNIST training in the context of test `test_obj`.

  Args:
    test_obj: The test instance.
    executor: The executor to be tested.
  """

  def _get_losses_before_and_after_training_single_batch(ex):
    set_default_executor.set_default_executor(ex)
    model = _mnist_initial_model
    losses = [_mnist_batch_loss(model, _mnist_sample_batch)]
    for _ in range(20):
      model = _mnist_batch_train(model, _mnist_sample_batch)
      losses.append(_mnist_batch_loss(model, _mnist_sample_batch))
    return losses

  for expected_loss, actual_loss in zip(
      _get_losses_before_and_after_training_single_batch(None),
      _get_losses_before_and_after_training_single_batch(executor)):
    test_obj.assertAlmostEqual(actual_loss, expected_loss, places=3)
