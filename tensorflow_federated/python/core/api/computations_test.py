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
"""Tests for computations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import value_base


class ComputationsTest(tf.test.TestCase):

  def test_tf_comp_first_mode_of_usage_as_non_polymorphic_wrapper(self):
    # Wrapping a lambda with a parameter.
    foo = computations.tf_computation(lambda x: x > 10, tf.int32)
    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # Wrapping an existing Python function with a parameter.
    bar = computations.tf_computation(tf.add, (tf.int32, tf.int32))
    self.assertEqual(str(bar.type_signature), '(<int32,int32> -> int32)')

    # Wrapping a no-parameter lambda.
    baz = computations.tf_computation(lambda: tf.constant(10))
    self.assertEqual(str(baz.type_signature), '( -> int32)')

    # Wrapping a no-parameter Python function.
    def bak_func():
      return tf.constant(10)

    bak = computations.tf_computation(bak_func)
    self.assertEqual(str(bak.type_signature), '( -> int32)')

  def test_tf_comp_second_mode_of_usage_as_non_polymorphic_decorator(self):
    # Decorating a Python function with a parameter.
    @computations.tf_computation(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # Decorating a no-parameter Python function.
    @computations.tf_computation
    def bar():
      return tf.constant(10)

    self.assertEqual(str(bar.type_signature), '( -> int32)')

  def test_tf_comp_third_mode_of_usage_as_polymorphic_callable(self):
    # Wrapping a lambda.
    foo = computations.tf_computation(lambda x: x > 0)  # pylint: disable=unused-variable

    # Decorating a Python function.
    @computations.tf_computation
    def bar(x, y):  # pylint: disable=unused-variable
      return x > y

    # TODO(b/113112108): Include invocations of these polymorphic callables.
    # Currently polymorphic callables, even though already fully supported,
    # cannot be easily tested, since little happens under the hood until they
    # are actually invoked.

  def test_fed_comp_typical_usage_as_decorator_with_unlabeled_type(self):

    @computations.federated_computation((computation_types.FunctionType(
        tf.int32, tf.int32), tf.int32))
    def foo(f, x):
      assert isinstance(f, value_base.Value)
      assert isinstance(x, value_base.Value)
      assert str(f.type_signature) == '(int32 -> int32)'
      assert str(x.type_signature) == 'int32'
      result_value = f(f(x))
      assert isinstance(result_value, value_base.Value)
      assert str(result_value.type_signature) == 'int32'
      return result_value

    # TODO(b/113112108): Add an invocation to make the test more meaningful.

    self.assertEqual(
        str(foo.type_signature), '(<(int32 -> int32),int32> -> int32)')

  def test_fed_comp_typical_usage_as_decorator_with_labeled_type(self):

    @computations.federated_computation((('f',
                                          computation_types.FunctionType(
                                              tf.int32, tf.int32)), ('x',
                                                                     tf.int32)))
    def foo(f, x):
      return f(f(x))

    # TODO(b/113112108): Add an invocation to make the test more meaningful.

    self.assertEqual(
        str(foo.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')


if __name__ == '__main__':
  tf.test.main()
