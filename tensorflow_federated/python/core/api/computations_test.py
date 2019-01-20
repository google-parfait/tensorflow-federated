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

import collections

# Dependency imports

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import value_base


class FederatedComputationsTest(test_utils.TffTestCase):

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

    @computations.federated_computation((
        ('f', computation_types.FunctionType(tf.int32, tf.int32)),
        ('x', tf.int32),
    ))
    def foo(f, x):
      return f(f(x))

    # TODO(b/113112108): Add an invocation to make the test more meaningful.

    self.assertEqual(
        str(foo.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')

  def test_no_argument_fed_comp(self):

    @computations.federated_computation
    def foo():
      return 10

    self.assertEqual(str(foo.type_signature), '( -> int32)')


def tf1_and_tf2_test(test_func):
  """A decorator for testing TFF wrapping of TF.

  Args:
    test_func: A test function to be decorated. It must accept to arguments,
      self (a TestCase), and tf_computation, which is either
      computations.tf_computation or computations.tf2_computation. Optionally,
      the test_func may return something that can be compared using
      self.assertEqual.

  Returns:
    The decorated function, which executes test_func using both wrappers,
    and compares the results.
  """

  def test_tf1_and_tf2(self):
    tf2_result = test_func(self, computations.tf2_computation)
    with tf.Graph().as_default():
      tf1_result = test_func(self, computations.tf_computation)
    self.assertEqual(tf1_result, tf2_result)

  return test_tf1_and_tf2


class TensorFlowComputationsTest(test_utils.TffTestCase):

  # TODO(b/122081673): Support tf.Dataset serialization in tf2_computation.
  @test_utils.graph_mode_test
  def test_with_tf_datasets(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def foo(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    self.assertEqual(str(foo.type_signature), '(int64* -> int64)')

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.range(10)

    self.assertEqual(str(bar.type_signature), '( -> int64*)')

  @tf1_and_tf2_test
  def test_tf_comp_first_mode_of_usage_as_non_polymorphic_wrapper(
      self, tf_computation):
    # Wrapping a lambda with a parameter.
    foo = tf_computation(lambda x: x > 10, tf.int32)
    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # Wrapping an existing Python function with a parameter.
    bar = tf_computation(tf.add, (tf.int32, tf.int32))
    self.assertEqual(str(bar.type_signature), '(<int32,int32> -> int32)')

    # Wrapping a no-parameter lambda.
    baz = tf_computation(lambda: tf.constant(10))
    self.assertEqual(str(baz.type_signature), '( -> int32)')

    # Wrapping a no-parameter Python function.
    def bak_func():
      return tf.constant(10)

    bak = tf_computation(bak_func)
    self.assertEqual(str(bak.type_signature), '( -> int32)')

  @tf1_and_tf2_test
  def test_tf_comp_second_mode_of_usage_as_non_polymorphic_decorator(
      self, tf_computation):
    # Decorating a Python function with a parameter.
    @tf_computation(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(str(foo.type_signature), '(int32 -> bool)')

    # Decorating a no-parameter Python function.
    @tf_computation
    def bar():
      return tf.constant(10)

    self.assertEqual(str(bar.type_signature), '( -> int32)')

  @tf1_and_tf2_test
  def test_tf_comp_third_mode_of_usage_as_polymorphic_callable(
      self, tf_computation):
    # Wrapping a lambda.
    _ = tf_computation(lambda x: x > 0)

    # Decorating a Python function.
    @tf_computation
    def bar(x, y):  # pylint: disable=unused-variable
      return x > y

    # TODO(b/113112108): Include invocations of these polymorphic callables.
    # Currently polymorphic callables, even though already fully supported,
    # cannot be easily tested, since little happens under the hood until they
    # are actually invoked.

  @tf1_and_tf2_test
  def test_with_variable(self, tf_computation):

    v_slot = []

    @tf.contrib.eager.function(autograph=False)
    def foo(x):
      if not v_slot:
        v_slot.append(tf.Variable(0))
      v = v_slot[0]
      tf.assign(v, 1)
      return v + x

    tf_comp = tf_computation(foo, tf.int32)
    self.assertEqual(tf_comp(1), 2)

  @tf1_and_tf2_test
  def test_one_param(self, tf_computation):

    @tf.contrib.eager.function
    def foo(x):
      return x + 1

    tf_comp = tf_computation(foo, tf.int32)
    self.assertEqual(tf_comp(1), 2)

  @tf1_and_tf2_test
  def test_no_params_structured_outputs(self, tf_computation):
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.contrib.eager.function
    def foo():
      return (1, 2, {'foo': 3.0, 'bar': 5.0}, MyType(True, False))

    tf_comp = tf_computation(foo, None)
    result = tf_comp()
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 2)
    self.assertEqual(dict(anonymous_tuple.to_elements(result[2])),
                     {'foo': 3.0, 'bar': 5.0})
    self.assertEqual(tuple(list(result[3])), (True, False))
    return (tf_comp.type_signature, str(tf_comp()))

  @tf1_and_tf2_test
  def test_polymorphic(self, tf_computation):

    def foo(x, y, z=3):
      # Since we don't wrap this as a tf.function, we need to do some
      # tf.convert_to_tensor(...) in order to ensure we have TensorFlow types.
      x = tf.convert_to_tensor(x)
      y = tf.convert_to_tensor(y)
      return (x + y, tf.convert_to_tensor(z))

    tf_comp = tf_computation(foo)  # A polymorphic TFF function.

    self.assertEqual(tuple(iter(tf_comp(1, 2))), (3, 3))  # With int32
    self.assertEqual(tuple(iter(tf_comp(1.0, 2.0))), (3.0, 3.0))  # With float32
    self.assertEqual(tuple(iter(tf_comp(1, 2, z=3))), (3, 3))  # With z

  @tf1_and_tf2_test
  def test_simple_structured_input(self, tf_computation):

    def foo(t):
      return t[0] + t[1]

    tf_poly = tf_computation(foo)
    self.assertEqual(tf_poly((1, 2)), 3)

  @tf1_and_tf2_test
  def test_more_structured_input(self, tf_computation):

    @tf.contrib.eager.function(autograph=False)
    def foo(tuple1, tuple2=(1, 2)):
      return tuple1[0] + tuple1[1][0] + tuple1[1][1] + tuple2[0] + tuple2[1]

    tf_poly = tf_computation(foo)
    self.assertEqual(tf_poly((1, (2, 3))), 9)
    self.assertEqual(tf_poly((1, (2, 3)), (0, 0)), 6)

  def test_py_and_tf_args(self):

    @tf.contrib.eager.function(autograph=False)
    def foo(x, y, add=True):
      return x + y if add else x - y

    # XXX - Q - Worth discussing.
    # tf.Functions support mixing tensorflow and Python arguments, usually
    # with the semantics you would expect. Currently, TFF does not
    # support this kind of mixing, even for Polymorphic TFF functions.
    # However, you can work around this by explicitly binding any Python
    # arguments on a tf.Function:

    tf_poly_add = computations.tf_computation(lambda x, y: foo(x, y, True))
    tf_poly_sub = computations.tf_computation(lambda x, y: foo(x, y, False))
    self.assertEqual(tf_poly_add(2, 1), 3)
    self.assertEqual(tf_poly_add(2., 1.), 3.)
    self.assertEqual(tf_poly_sub(2, 1), 1)

  # TODO(b/123193055): Enable this for computations.tf_computation as well
  def test_more_structured_input_explicit_types(self):

    @tf.contrib.eager.function(autograph=False)
    def foo(tuple1, tuple2=(1, 2)):
      return tuple1[0] + tuple1[1][0] + tuple1[1][1] + tuple2[0] + tuple2[1]

    tff_type1 = [
        (tf.int32, (tf.int32, tf.int32)),
        ('tuple2', (tf.int32, tf.int32))]
    tff_type2 = [
        (tf.int32, (tf.int32, tf.int32)),
        (tf.int32, tf.int32)]

    # Both of the above are valid.
    # XXX Q - Should explicitly naming 'tuple1' also work? (It doesn't)
    for tff_type in [tff_type1, tff_type2]:
      tf_comp = computations.tf2_computation(foo, tff_type)
      self.assertEqual(tf_comp((1, (2, 3)), (0, 0)), 6)

  def test_something_that_only_works_with_tf2(self):
    # These variables will be tracked and serialized automatically.
    v1 = tf.Variable(0.0)
    v2 = tf.Variable(0.0)

    @tf.contrib.eager.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def foo(x):
      tf.assign(v1, 1.0)
      tf.assign(v2, 1.0)
      return (v1 + v2, x)

    foo_cf = foo.get_concrete_function()

    @tf.contrib.eager.function(autograph=False)
    def bar(x):
      a, b = foo_cf(x)
      return a + b

    # If we had wrapped this test in @graph_mode_test and called
    # computations.tf_computation, this example will not work.
    tf2_comp = computations.tf2_computation(bar, tf.float32)
    self.assertEqual(tf2_comp(1.0), 3.0)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
