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
"""Tests for computation_wrapper.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import computation_wrapper
from tensorflow_federated.python.core.impl import func_utils


class WrappedForTest(func_utils.ConcreteFunction):
  """A class that represents a wrapped function for testing purposes.

  Upon invocation, it returns a string of the form P : T -> R, where P is the
  parameter tuple (or None), T is its type (or None), and R is the returned
  result, all converted into strings via str().
  """

  def __init__(self, func, parameter_type):
    self._func = func
    super(WrappedForTest, self).__init__(
        types.FunctionType(parameter_type, tf.string))

  def _invoke(self, arg):
    result = self._func(arg) if self.type_signature.parameter else self._func()
    return '{} : {} -> {}'.format(
        str(arg), str(self.type_signature.parameter), str(result))


test_wrap = computation_wrapper.ComputationWrapper(WrappedForTest)


class ComputationWrapperTest(tf.test.TestCase):

  # NOTE: Many tests below silence certain linter warnings. These warnings are
  # not applicable, since it's the wrapper code, not not the dummy functions
  # that are being tested, so whether the specific function declarations used
  # here follow good practices is not really relevant. The purpose of the test
  # is to exercise various corner cases that the wrapper needs to be able to
  # correctly handle.

  def test_as_decorator_with_kwargs(self):
    with self.assertRaises(TypeError):

      @test_wrap(foo=1)
      def func():  # pylint: disable=unused-variable
        pass

  def test_as_wrapper_with_kwargs(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda: None, foo=1)  # pylint: disable=unnecessary-lambda

  def test_as_decorator_without_arguments_on_no_parameter_py_func(self):

    @test_wrap
    def my_func():
      """This is my func."""
      return 10

    self.assertEqual(my_func(), 'None : None -> 10')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_wrapper_without_arguments_on_no_parameter_lambda(self):
    self.assertEqual(test_wrap(lambda: 10)(), 'None : None -> 10')

  def test_as_decorator_with_empty_arguments_on_no_parameter_py_func(self):

    @test_wrap()
    def my_func():
      """This is my func."""
      return 10

    self.assertEqual(my_func(), 'None : None -> 10')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_decorator_with_one_argument_on_no_parameter_py_func(self):
    with self.assertRaises(TypeError):

      @test_wrap(tf.int32)
      def my_func():  # pylint: disable=unused-variable
        pass

  def test_as_wrapper_with_one_argument_on_no_parameter_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda: None, tf.int32)

  def test_as_decorator_with_one_argument_on_one_parameter_py_func(self):

    @test_wrap(tf.int32)
    def my_func(x):
      """This is my func."""
      return x + 10

    self.assertEqual(my_func(5), '5 : int32 -> 15')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_wrapper_with_one_argument_on_one_parameter_lambda(self):
    self.assertEqual(
        test_wrap(lambda x: x + 10, tf.int32)(5), '5 : int32 -> 15')

  def test_as_decorator_with_non_tuple_argument_on_two_parameter_py_func(self):
    with self.assertRaises(TypeError):

      @test_wrap(tf.int32)
      def my_func(x, y):  # pylint: disable=unused-variable,unused-argument
        pass

  def test_as_wrapper_with_non_tuple_argument_on_two_parameter_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(
          lambda x, y: None,  # pylint: disable=unused-variable,unused-argument
          tf.int32)

  def test_as_decorator_with_two_tuple_argument_on_three_param_py_func(self):
    with self.assertRaises(TypeError):

      @test_wrap((tf.int32, tf.int32))
      def my_func(x, y, z):  # pylint: disable=unused-variable,unused-argument
        pass

  def test_as_wrapper_with_two_tuple_argument_on_three_param_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda x, y, z: None, (tf.int32, tf.int32))

  def test_as_decorator_with_arg_name_mismatching_element_name_in_py_func(self):
    with self.assertRaises(TypeError):

      @test_wrap([('x', tf.int32), ('y', tf.int32)])
      def my_func(x, z):  # pylint: disable=unused-variable,unused-argument
        pass

  def test_as_wrapper_with_arg_name_mismatching_element_name_in_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda x, z: None, [('x', tf.int32), ('y', tf.int32)])

  def test_as_decorator_with_tuple_params_on_two_parameter_py_func(self):

    @test_wrap((tf.int32, tf.int32))
    def my_func(x, y):
      """This is my func."""
      return x + y

    self.assertEqual(my_func(1, 2), '<1,2> : <int32,int32> -> 3')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_wrapper_with_tuple_params_on_two_parameter_py_func(self):
    self.assertEqual(
        test_wrap(lambda x, y: x + y, (tf.int32, tf.int32))(1, 2),
        '<1,2> : <int32,int32> -> 3')

  def test_as_decorator_with_tuple_params_on_one_parameter_py_func(self):
    # Computations only have a single parameter (or none), and we allow the
    # flexibility of feeding tuple-like parameters in pieces by specifying
    # tuple elementas as multiple arguments. This is independent of how the
    # backing Python function binds to the argument on the definition side.
    # Thus, the ordinary linter check is inapplicable, as there's exists no
    # direct connection between the signature of the call and that of the
    # Python definition. The TFF type decouples one from the other.
    @test_wrap([('x', tf.int32), ('y', tf.int32)])
    def my_func(arg):
      """This is my func."""
      return arg.x + arg.y

    self.assertEqual(
        my_func(1, 2),  # pylint: disable=too-many-function-args
        '<x=1,y=2> : <x=int32,y=int32> -> 3')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_wrapper_with_tuple_params_on_one_parameter_py_func(self):
    self.assertEqual(
        test_wrap(lambda arg: arg[0] + arg[1], (tf.int32, tf.int32))(1, 2),
        '<1,2> : <int32,int32> -> 3')

  def test_as_decorator_with_named_tuple_params_on_two_param_py_func(self):

    @test_wrap([('x', tf.int32), ('y', tf.int32)])
    def my_func(x, y):
      """This is my func."""
      return x + y

    self.assertEqual(my_func(1, 2), '<x=1,y=2> : <x=int32,y=int32> -> 3')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_wrapper_with_named_tuple_params_on_two_param_py_func(self):
    wrapped = test_wrap(lambda x, y: x + y, [('x', tf.int32), ('y', tf.int32)])
    self.assertEqual(wrapped(1, 2), '<x=1,y=2> : <x=int32,y=int32> -> 3')

  def test_as_decorator_without_arguments_on_py_func_with_one_param(self):

    @test_wrap
    def my_func(x):
      """This is my func."""
      return x + 1

    self.assertEqual(my_func(10), '<10> : <int32> -> 11')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_wrapper_without_arguments_on_py_func_with_one_param(self):
    wrapped = test_wrap(lambda x: x + 1)
    self.assertEqual(wrapped(10), '<10> : <int32> -> 11')

  def test_as_decorator_without_arguments_on_py_func_with_two_params(self):

    @test_wrap
    def my_func(x, y):
      """This is my func."""
      return x + y

    self.assertEqual(my_func(10, 20), '<10,20> : <int32,int32> -> 30')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_decorator_with_empty_arguments_on_py_func_with_one_param(self):

    @test_wrap()
    def my_func(x):
      """This is my func."""
      return x + 1

    self.assertEqual(my_func(10), '<10> : <int32> -> 11')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_as_decorator_with_empty_arguments_on_py_func_with_two_params(self):

    @test_wrap()
    def my_func(x, y):
      """This is my func."""
      return x + y

    self.assertEqual(my_func(10, 20), '<10,20> : <int32,int32> -> 30')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_with_integer_args(self):
    with self.assertRaises(TypeError):
      test_wrap(10, 20)

  def test_with_varargs_no_type(self):

    @test_wrap
    def my_func(*args):
      """This is my func."""
      return sum(args)

    self.assertEqual(
        my_func(10, 20, 30), '<10,20,30> : <int32,int32,int32> -> 60')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_with_varargs_scalar_type(self):

    @test_wrap(tf.int32)
    def my_func(*args):
      """This is my func."""
      return sum(args)

    self.assertEqual(my_func(10), '10 : int32 -> 10')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_with_varargs_tuple_type(self):

    @test_wrap([tf.int32, tf.int32, tf.int32, tf.int32])
    def my_func(x, y, *args):
      """This is my func."""
      return x + y + sum(args)

    self.assertEqual(
        my_func(10, 20, 30, 40),
        '<10,20,30,40> : <int32,int32,int32,int32> -> 100')
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_with_kwargs_no_type(self):

    @test_wrap
    def my_func(**kwargs):
      """This is my func."""
      return kwargs['x'] / kwargs['y']

    self.assertIn(
        my_func(x=10, y=20), [
            '<x=10,y=20> : <x=int32,y=int32> -> 0.5',
            '<y=20,x=10> : <y=int32,x=int32> -> 0.5'
        ])
    self.assertEqual(my_func.__doc__, 'This is my func.')

  def test_with_all_kinds_or_args_no_type(self):
    # Exercising a corner case that may not follow the style guide, but is one
    # possible scenario one may throw the wrapper at.
    @test_wrap
    def my_func(  # pylint: disable=keyword-arg-before-vararg
        a,
        b,
        c=10,
        d=20,
        *args,
        **kwargs):
      """This is my func."""
      return '{},{},{},{},{},{}'.format(a, b, c, d, args, kwargs)

    self.assertEqual(my_func(1, 2), '<1,2> : <int32,int32> -> 1,2,10,20,(),{}')
    self.assertEqual(
        my_func(1, 2, 3), '<1,2,3> : <int32,int32,int32> -> 1,2,3,20,(),{}')
    self.assertEqual(
        my_func(1, 2, d=3),
        '<1,2,d=3> : <int32,int32,d=int32> -> 1,2,10,3,(),{}')
    self.assertEqual(
        my_func(1, 2, e=3),
        '<1,2,e=3> : <int32,int32,e=int32> -> 1,2,10,20,(),{\'e\': 3}')
    self.assertEqual(
        my_func(1, 2, 3, 4, 5),
        '<1,2,3,4,5> : <int32,int32,int32,int32,int32> -> 1,2,3,4,(5,),{}')
    self.assertEqual(my_func.__doc__, 'This is my func.')


if __name__ == '__main__':
  tf.test.main()
