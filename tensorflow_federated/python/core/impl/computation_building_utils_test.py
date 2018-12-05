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

# Dependency imports
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import computation_building_utils as bu
from tensorflow_federated.python.core.impl import func_utils


class ComputationBuildingUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (lambda f, x: f(f(x)),
       [('f', types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32)],
       '(foo -> foo.f(foo.f(foo.x)))'),
      (lambda f, g, x: f(g(x)),
       [('f', types.FunctionType(tf.int32, tf.int32)),
        ('g', types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32)],
       '(foo -> foo.f(foo.g(foo.x)))'),
      (lambda x: (x[1], x[0]),
       (tf.int32, tf.int32),
       '(foo -> <foo[1],foo[0]>)'))
  def test_zero_or_one_arg_func_to_lambda(self, func, parameter_type, func_str):
    parameter_name = 'foo'
    parameter_type = types.to_type(parameter_type)
    func = func_utils.wrap_as_zero_or_one_arg_callable(func, parameter_type)
    result = bu.zero_or_one_arg_func_to_lambda(
        func, parameter_name, parameter_type)
    self.assertEqual(str(result), func_str)


if __name__ == '__main__':
  absltest.main()
