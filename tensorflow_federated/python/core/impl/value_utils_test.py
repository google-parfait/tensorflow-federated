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
from absl.testing import parameterized
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import func_utils
from tensorflow_federated.python.core.impl import value_utils
from tensorflow_federated.python.core.impl import values


class ValueUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ([tf.bool, ('a', tf.int32)], '<foo[0],a=foo.a>'),
      ([tf.bool, tf.float32, tf.int32], '<foo[0],foo[1],foo[2]>'),
      ([('a', tf.bool), ('b', tf.int32), ('c', tf.bool)],
       '<a=foo.a,b=foo.b,c=foo.c>'))
  def test_expand_tuple(self, type_spec, tuple_str):
    type_spec = types.to_type(type_spec)
    tuple_val = value_utils.expand_tuple(values.Reference('foo', type_spec))
    self.assertEqual(str(tuple_val.type_signature), str(type_spec))
    self.assertEqual(str(tuple_val), tuple_str)

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
    result = value_utils.zero_or_one_arg_func_to_lambda(
        func, parameter_name, parameter_type)
    self.assertEqual(str(result), func_str)


if __name__ == '__main__':
  unittest.main()
