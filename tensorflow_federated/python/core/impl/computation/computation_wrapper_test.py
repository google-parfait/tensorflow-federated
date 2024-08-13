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

import functools

from absl.testing import absltest
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


class WrappedForTest(computation_impl.ConcreteComputation):
  """A class that represents a wrapped function for testing purposes.

  Upon invocation, it returns a string of the form P : T -> R, where P is the
  parameter tuple (or None), T is its type (or None), and R is the returned
  result, all converted into strings via str().
  """

  def __init__(self, fn, parameter_type, unpack, name=None):
    del name  # Unused.
    fn_type = computation_types.FunctionType(parameter_type, np.str_)
    test_proto = pb.Computation(type=type_serialization.serialize_type(fn_type))
    super().__init__(
        computation_proto=test_proto,
        context_stack=context_stack_impl.context_stack,
        annotated_type=fn_type,
    )

    self._fn = function_utils.wrap_as_zero_or_one_arg_callable(
        fn, parameter_type, unpack
    )

  @property
  def fn(self):
    return self._fn


class ContextForTest(context_base.SyncContext):

  def invoke(self, comp, arg):
    result = comp.fn(arg) if comp.type_signature.parameter else comp.fn()
    return '{} : {} -> {}'.format(
        str(arg), str(comp.type_signature.parameter), str(result)
    )


test_wrap = computation_wrapper.ComputationWrapper(WrappedForTest)


class ComputationWrapperTest(absltest.TestCase):

  def test_as_decorator_without_arguments_on_no_parameter_py_fn(self):
    @test_wrap
    def my_fn():
      """This is my fn."""
      return 10

    self.assertEqual(my_fn(), 'None : None -> 10')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_without_arguments_on_no_parameter_lambda(self):
    self.assertEqual(test_wrap(lambda: 10)(), 'None : None -> 10')

  def test_as_wrapper_without_arguments_on_no_parameter_partial(self):
    def identity(x):
      return x

    self.assertEqual(
        test_wrap(functools.partial(identity, 10))(), 'None : None -> 10'
    )

  def test_as_decorator_with_empty_arguments_on_no_parameter_py_fn(self):
    @test_wrap()
    def my_fn():
      """This is my fn."""
      return 10

    self.assertEqual(my_fn(), 'None : None -> 10')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_decorator_with_one_argument_on_no_parameter_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap(np.int32)
      def _():
        pass

  def test_as_wrapper_with_one_argument_on_no_parameter_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda: None, np.int32)

  def test_as_decorator_with_one_argument_on_one_parameter_py_fn(self):

    @test_wrap(np.int32)
    def my_fn(x):
      """This is my fn."""
      return x + 10

    self.assertEqual(my_fn(5), '5 : int32 -> 15')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_one_argument_on_one_parameter_lambda(self):
    self.assertEqual(
        test_wrap(lambda x: x + 10, np.int32)(5), '5 : int32 -> 15'
    )

  def test_as_decorator_with_non_tuple_argument_on_two_parameter_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap(np.int32)
      def _(x, y):
        del x, y  # Unused.
        pass

  def test_as_wrapper_with_non_tuple_argument_on_two_parameter_lambda(self):
    with self.assertRaises(TypeError):

      def my_fn(x, y):
        del x, y  # Unused.
        pass

      test_wrap(my_fn, np.int32)

  def test_as_decorator_with_two_tuple_argument_on_three_param_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap((np.int32, np.int32))
      def _(x, y, z):
        del x, y, z  # Unused.
        pass

  def test_as_wrapper_with_two_tuple_argument_on_three_param_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda x, y, z: None, (np.int32, np.int32))

  def test_as_decorator_with_arg_name_mismatching_element_name_in_py_fn(self):
    with self.assertRaises(TypeError):

      @test_wrap([('x', np.int32), ('y', np.int32)])
      def _(x, z):
        del x, z  # Unused.
        pass

  def test_as_wrapper_with_arg_name_mismatching_element_name_in_lambda(self):
    with self.assertRaises(TypeError):
      test_wrap(lambda x, z: None, [('x', np.int32), ('y', np.int32)])

  def test_as_decorator_with_tuple_params_on_two_parameter_py_fn(self):

    @test_wrap((np.int32, np.int32))
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assertEqual(my_fn(1, 2), '<x=1,y=2> : <x=int32,y=int32> -> 3')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_tuple_params_on_two_parameter_py_fn(self):
    wrapped = test_wrap(lambda x, y: x + y, (np.int32, np.int32))
    self.assertEqual(wrapped(1, 2), '<x=1,y=2> : <x=int32,y=int32> -> 3')

  def test_as_decorator_with_tuple_params_on_one_parameter_py_fn(self):
    # Computations only have a single parameter (or none), and we allow the
    # flexibility of feeding tuple-like parameters in pieces by specifying
    # tuple elementas as multiple arguments. This is independent of how the
    # backing Python function binds to the argument on the definition side.
    # Thus, the ordinary linter check is inapplicable, as there's exists no
    # direct connection between the signature of the call and that of the
    # Python definition. The TFF type decouples one from the other.
    @test_wrap([('x', np.int32), ('y', np.int32)])
    def my_fn(arg):
      """This is my fn."""
      return arg.x + arg.y

    self.assertEqual(
        my_fn(1, 2),  # pylint: disable=too-many-function-args
        '<x=1,y=2> : <x=int32,y=int32> -> 3',
    )
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_tuple_params_on_one_parameter_py_fn(self):
    self.assertEqual(
        test_wrap(lambda arg: arg[0] + arg[1], (np.int32, np.int32))(1, 2),
        '<1,2> : <int32,int32> -> 3',
    )

  def test_as_decorator_with_named_tuple_params_on_two_param_py_fn(self):

    @test_wrap([('x', np.int32), ('y', np.int32)])
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assertEqual(my_fn(1, 2), '<x=1,y=2> : <x=int32,y=int32> -> 3')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_with_named_tuple_params_on_two_param_py_fn(self):
    wrapped = test_wrap(lambda x, y: x + y, [('x', np.int32), ('y', np.int32)])
    self.assertEqual(wrapped(1, 2), '<x=1,y=2> : <x=int32,y=int32> -> 3')

  def test_as_decorator_without_arguments_on_py_fn_with_one_param(self):
    @test_wrap
    def my_fn(x):
      """This is my fn."""
      return x + 1

    self.assertEqual(my_fn(10), '<10> : <int32> -> 11')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_wrapper_without_arguments_on_py_fn_with_one_param(self):
    wrapped = test_wrap(lambda x: x + 1)
    self.assertEqual(wrapped(10), '<10> : <int32> -> 11')

  def test_as_decorator_without_arguments_on_py_fn_with_two_params(self):
    @test_wrap
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assertEqual(my_fn(10, 20), '<10,20> : <int32,int32> -> 30')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_decorator_with_empty_arguments_on_py_fn_with_one_param(self):
    @test_wrap()
    def my_fn(x):
      """This is my fn."""
      return x + 1

    self.assertEqual(my_fn(10), '<10> : <int32> -> 11')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_as_decorator_with_empty_arguments_on_py_fn_with_two_params(self):
    @test_wrap()
    def my_fn(x, y):
      """This is my fn."""
      return x + y

    self.assertEqual(my_fn(10, 20), '<10,20> : <int32,int32> -> 30')
    self.assertEqual(my_fn.__doc__, 'This is my fn.')

  def test_with_integer_args(self):
    with self.assertRaises(TypeError):
      test_wrap(10, 20)

  def test_with_varargs_no_type(self):
    with self.assertRaises(TypeError):

      @test_wrap
      def _(*args):
        """This is my fn."""
        return sum(args)

  def test_with_varargs_scalar_type(self):
    with self.assertRaises(TypeError):

      @test_wrap(np.int32)
      def _(*args):
        """This is my fn."""
        return sum(args)

  def test_with_varargs_tuple_type(self):
    with self.assertRaises(TypeError):

      @test_wrap([np.int32, np.int32, np.int32, np.int32])
      def _(x, y, *args):
        """This is my fn."""
        return x + y + sum(args)

  def test_with_kwargs_no_type(self):
    with self.assertRaises(TypeError):

      @test_wrap
      def _(**kwargs):
        """This is my fn."""
        return kwargs['x'] / kwargs['y']

  def test_as_decorator_with_unbundled_arguments(self):

    @test_wrap(np.int32, np.int32)
    def foo(unused_x, unused_y):
      return 99

    self.assertEqual(
        foo(unused_y=20, unused_x=10),
        '<unused_x=10,unused_y=20> : <unused_x=int32,unused_y=int32> -> 99',
    )

  def test_as_decorator_with_named_positional_arguments(self):

    @test_wrap(np.int32, np.int32)
    def foo(unused_x, unused_y):
      return 99

    expected = (
        '<unused_x=10,unused_y=20> : <unused_x=int32,unused_y=int32> -> 99'
    )
    self.assertEqual(foo(unused_x=10, unused_y=20), expected)
    self.assertEqual(foo(10, unused_y=20), expected)
    self.assertEqual(foo(unused_y=20, unused_x=10), expected)

  def test_as_decorator_with_optional_arguments(self):
    with self.assertRaisesRegex(TypeError, 'default'):

      @test_wrap(np.int32, np.int32)
      def _(unused_x=10, unused_y=20):
        return 99

  def test_as_wrapper_with_unbundled_arguments(self):
    foo = test_wrap(lambda unused_x, unused_y: 99, np.int32, np.int32)
    self.assertEqual(
        foo(10, 20),
        '<unused_x=10,unused_y=20> : <unused_x=int32,unused_y=int32> -> 99',
    )

  def test_as_wrapper_with_one_argument_instance_method(self):
    class IntWrapper:

      def __init__(self, x):
        self._x = x

      def multiply_by(self, y):
        return self._x * y

    five = IntWrapper(5)
    wrapped = test_wrap(five.multiply_by, np.int32)
    self.assertEqual(wrapped(2), '2 : int32 -> 10')

  def test_as_wrapper_with_no_argument_instance_method(self):
    class C:

      def __init__(self, x):
        self._x = x

      def my_method(self):
        return self._x

    c = C(99)
    wrapped = test_wrap(c.my_method)
    self.assertEqual(wrapped(), 'None : None -> 99')

  def test_as_wrapper_with_class_property(self):
    class C:

      @property
      def x(self):
        return 99

    c = C()
    with self.assertRaises(TypeError):
      test_wrap(c.x)

  def test_as_wrapper_with_classmethod(self):
    class C:

      @classmethod
      def prefix(cls, msg):
        return f'{cls.__name__}_{msg}'

    wrapped = test_wrap(C.prefix)
    self.assertEqual(wrapped('foo'), '<foo> : <str> -> C_foo')


if __name__ == '__main__':
  with context_stack_impl.context_stack.install(ContextForTest()):
    absltest.main()
