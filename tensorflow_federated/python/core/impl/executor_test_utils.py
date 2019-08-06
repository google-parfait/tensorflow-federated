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

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import executor_base
from tensorflow_federated.python.core.impl import executor_value_base
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
          tf.one_hot(batch.y, 10) * tf.math.log(predicted_y), axis=[1]))


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


class TracingExecutor(executor_base.Executor):
  """Tracing executor keeps a log of all calls for use in testing."""

  def __init__(self, target):
    """Creates a new instance of a tracing executor.

    The tracing executor keeps the trace of all calls. Entries in the trace
    consist of the method name followed by arguments and the returned result,
    with the executor values represented as integer indexes starting from 1.

    Args:
      target: An instance of `executor_base.Executor`.
    """
    py_typecheck.check_type(target, executor_base.Executor)
    self._target = target
    self._last_used_index = 0
    self._trace = []

  @property
  def trace(self):
    return self._trace

  def _get_new_value_index(self):
    val_index = self._last_used_index + 1
    self._last_used_index = val_index
    return val_index

  async def create_value(self, value, type_spec=None):
    target_val = await self._target.create_value(value, type_spec)
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    if type_spec is not None:
      self._trace.append(('create_value', value, type_spec, wrapped_val.index))
    else:
      self._trace.append(('create_value', value, wrapped_val.index))
    return wrapped_val

  async def create_call(self, comp, arg=None):
    if arg is not None:
      target_val = await self._target.create_call(comp.value, arg.value)
      wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                         target_val)
      self._trace.append(
          ('create_call', comp.index, arg.index, wrapped_val.index))
      return wrapped_val
    else:
      target_val = await self._target.create_call(comp.value)
      wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                         target_val)
      self._trace.append(('create_call', comp.index, wrapped_val.index))
      return wrapped_val

  async def create_tuple(self, elements):
    target_val = await self._target.create_tuple(
        anonymous_tuple.map_structure(lambda x: x.value, elements))
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_tuple',
         anonymous_tuple.map_structure(lambda x: x.index,
                                       elements), wrapped_val.index))
    return wrapped_val

  async def create_selection(self, source, index=None, name=None):
    target_val = await self._target.create_selection(
        source.value, index=index, name=name)
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_selection', source.index, index if index is not None else name,
         wrapped_val.index))
    return wrapped_val


class TracingExecutorValue(executor_value_base.ExecutorValue):
  """A value managed by `TracingExecutor`."""

  def __init__(self, owner, index, value):
    """Creates an instance of a value in the tracing executor.

    Args:
      owner: An instance of `TracingExecutor`.
      index: An integer identifying the value.
      value: An embedded value from the target executor.
    """
    py_typecheck.check_type(owner, TracingExecutor)
    py_typecheck.check_type(index, int)
    py_typecheck.check_type(value, executor_value_base.ExecutorValue)
    self._owner = owner
    self._index = index
    self._value = value

  @property
  def index(self):
    return self._index

  @property
  def value(self):
    return self._value

  @property
  def type_signature(self):
    return self._value.type_signature

  async def compute(self):
    result = await self._value.compute()
    self._owner.trace.append(('compute', self._index, result))
    return result
