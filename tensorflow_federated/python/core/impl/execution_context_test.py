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
"""Tests for execution_context.py."""

import collections

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import execution_context


def _test_ctx():
  return execution_context.ExecutionContext(eager_executor.EagerExecutor())


class ExecutionContextTest(absltest.TestCase):

  def test_simple_no_arg_tf_computation_with_int_result(self):

    @computations.tf_computation
    def comp():
      return tf.constant(10)

    with context_stack_impl.context_stack.install(_test_ctx()):
      result = comp()

    self.assertEqual(result, 10)

  def test_one_arg_tf_computation_with_int_param_and_result(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 10)

    with context_stack_impl.context_stack.install(_test_ctx()):
      result = comp(3)

    self.assertEqual(result, 13)

  def test_three_arg_tf_computation_with_int_params_and_result(self):

    @computations.tf_computation(tf.int32, tf.int32, tf.int32)
    def comp(x, y, z):
      return tf.multiply(tf.add(x, y), z)

    with context_stack_impl.context_stack.install(_test_ctx()):
      result = comp(3, 4, 5)

    self.assertEqual(result, 35)

  def test_tf_computation_with_dataset_params_and_int_result(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.reduce(np.int32(0), lambda x, y: x + y)

    with context_stack_impl.context_stack.install(_test_ctx()):
      result = comp(
          tf.data.Dataset.range(10).map(lambda x: tf.cast(x, tf.int32)))

    self.assertEqual(result, 45)

  def test_tf_computation_with_structured_result(self):

    @computations.tf_computation
    def comp():
      return collections.OrderedDict([('a', tf.constant(10)),
                                      ('b', tf.constant(20))])

    with context_stack_impl.context_stack.install(_test_ctx()):
      result = comp()

    self.assertIsInstance(result, collections.OrderedDict)
    self.assertDictEqual(result, {'a': 10, 'b': 20})


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
