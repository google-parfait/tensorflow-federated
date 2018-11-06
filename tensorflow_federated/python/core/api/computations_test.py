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

from tensorflow_federated.python.core.api import computations as fc


class ComputationsTest(tf.test.TestCase):

  def test_first_mode_of_usage_as_non_polymorphic_wrapper(self):
    # Wrapping a lambda with a parameter.
    foo = fc.tf_computation(lambda x: x > 10, tf.int32)
    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # Wrapping an existing Python function with a parameter.
    bar = fc.tf_computation(tf.add, (tf.int32, tf.int32))
    self.assertEqual(str(bar.type_signature), '(<int32,int32> -> int32)')

    # Wrapping a no-parameter lambda.
    baz = fc.tf_computation(lambda: tf.constant(10))
    self.assertEqual(str(baz.type_signature), '( -> int32)')

    # Wrapping a no-parameter Python function.
    def bak_func():
      return tf.constant(10)
    bak = fc.tf_computation(bak_func)
    self.assertEqual(str(bak.type_signature), '( -> int32)')

  def test_second_mode_of_usage_as_non_polymorphic_decorator(self):
    # Decorating a Python function with a parameter.
    @fc.tf_computation(tf.int32)
    def foo(x):
      return x > 10
    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # Decorating a no-parameter Python function.
    @fc.tf_computation
    def bar():
      return tf.constant(10)
    self.assertEqual(str(bar.type_signature), '( -> int32)')

  def test_third_mode_of_usage_as_polymorphic_callable(self):
    # Wrapping a lambda.
    foo = fc.tf_computation(lambda x: x > 0)  # pylint: disable=unused-variable

    # Decorating a Python function.
    @fc.tf_computation
    def bar(x, y):  # pylint: disable=unused-variable
      return x > y

    # TODO(b/113112108): Include invocations of these polymorphic callables.
    # Currently polymorphic callables, even though already fully supported,
    # cannot be easily tested, since little happens under the hood until they
    # are actually invoked.


if __name__ == '__main__':
  tf.test.main()
