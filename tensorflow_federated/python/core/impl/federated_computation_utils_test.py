# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for value_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import federated_computation_utils
from tensorflow_federated.python.core.impl import func_utils


class ComputationBuildingUtilsTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.parameters(
      (lambda f, x: f(f(x)),
       [('f', computation_types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32)],
       '(foo -> foo.f(foo.f(foo.x)))'),
      (lambda f, g, x: f(g(x)),
       [('f', computation_types.FunctionType(tf.int32, tf.int32)),
        ('g', computation_types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32)],
       '(foo -> foo.f(foo.g(foo.x)))'),
      (lambda x: (x[1], x[0]),
       (tf.int32, tf.int32),
       '(foo -> <foo[1],foo[0]>)'),
      (lambda: 'stuff', None, 'stuff'))
  # pyformat: enable
  def zero_or_one_arg_func_to_building_block(self, func, parameter_type,
                                             func_str):
    parameter_name = 'foo'
    parameter_type = computation_types.to_type(parameter_type)
    func = func_utils.wrap_as_zero_or_one_arg_callable(func, parameter_type)
    result = federated_computation_utils.zero_or_one_arg_func_to_building_block(
        func, parameter_name, parameter_type, context_stack_impl.context_stack)
    self.assertEqual(str(result), func_str)


if __name__ == '__main__':
  absltest.main()
