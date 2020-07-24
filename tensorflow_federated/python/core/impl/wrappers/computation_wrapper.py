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
"""Utilities for constructing decorators/wrappers for functions and defuns."""

import types
from typing import Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.utils import function_utils
from tensorflow_federated.python.tensorflow_libs import function


def _wrap(fn, parameter_type, wrapper_fn):
  """Wraps a possibly-polymorphic `fn` in `wrapper_fn`.

  If `parameter_type` is `None` and `fn` takes any arguments (even with default
  values), `fn` is inferred to be polymorphic and won't be passed to
  `wrapper_fn` until invocation time (when concrete parameter types are
  available).

  `wrapper_fn` must accept three positional arguments and one defaulted argument
  `name`:

  * `target_fn`, the Python function to be wrapped.

  * `parameter_type`, the optional type of the computation's
    parameter (an instance of `computation_types.Type`).

  * `unpack`, an argument which will be passed on to
    `function_utils.wrap_as_zero_or_one_arg_callable` when wrapping `target_fn`.
    See that function for details.

  * Optional `name`, the name of the function that is being wrapped (only for
    debugging purposes).

  Args:
    fn: The function or defun to wrap as a computation.
    parameter_type: Optional type of any arguments to `fn`.
    wrapper_fn: The Python callable that performs actual wrapping. The object to
      be returned by this function should be an instance of a
      `ConcreteFunction`.

  Returns:
    Either the result of wrapping (an object that represents the computation),
    or a polymorphic callable that performs wrapping upon invocation based on
    argument types. The returned function still may accept multiple
    arguments (it has not yet had
    `function_uils.wrap_as_zero_or_one_arg_callable` applied to it).

  Raises:
    TypeError: if the arguments are of the wrong types, or the `wrapper_fn`
      constructs something that isn't a ConcreteFunction.
  """
  try:
    fn_name = fn.__name__
  except AttributeError:
    fn_name = None
  signature = function_utils.get_signature(fn)
  parameter_type = computation_types.to_type(parameter_type)
  if parameter_type is None and signature.parameters:
    # There is no TFF type specification, and the function/defun declares
    # parameters. Create a polymorphic template.
    def _wrap_polymorphic(parameter_type: computation_types.Type,
                          unpack: Optional[bool]):
      return wrapper_fn(fn, parameter_type, unpack=unpack, name=fn_name)

    polymorphic_fn = function_utils.PolymorphicFunction(_wrap_polymorphic)

    # When applying a decorator, the __doc__ attribute with the documentation
    # in triple-quotes is not automatically transferred from the function on
    # which it was applied to the wrapped object, so we must transfer it here
    # explicitly.
    polymorphic_fn.__doc__ = getattr(fn, '__doc__', None)
    return polymorphic_fn

  # Either we have a concrete parameter type, or this is no-arg function.
  concrete_fn = wrapper_fn(fn, parameter_type, unpack=None)
  py_typecheck.check_type(concrete_fn, function_utils.ConcreteFunction,
                          'value returned by the wrapper')
  if (concrete_fn.type_signature.parameter is not None and
      not concrete_fn.type_signature.parameter.is_equivalent_to(parameter_type)
     ):
    raise TypeError(
        'Expected a concrete function that takes parameter {}, got one '
        'that takes {}.'.format(
            str(parameter_type), str(concrete_fn.type_signature.parameter)))
  # When applying a decorator, the __doc__ attribute with the documentation
  # in triple-quotes is not automatically transferred from the function on
  concrete_fn.__doc__ = getattr(fn, '__doc__', None)
  return concrete_fn


class ComputationWrapper(object):
  """A class for creating wrappers that convert functions into computations.

  This class builds upon the _wrap() function defined above, adding on
  functionality shared between the `tf_computation`, `tf2_computation`, and
  `federated_computation` decorators. The shared functionality includes relating
  formal Python function parameters and call arguments to TFF types, packing and
  unpacking arguments, verifying types, and support for polymorphism.

  Here's how one can use `ComputationWrapper` to construct a decorator/wrapper
  named `xyz`:

  ```python
  def my_wrapper_fn(target_fn, parameter_type, unpack, name=None):
    ...
  xyz = computation_wrapper.ComputationWrapper(my_wrapper_fn)
  ```

  The resulting `xyz` can be used either as an `@xyz(...)` decorator or as a
  manual wrapping function: `wrapped_func = xyz(my_func, ...)`. The latter
  method may be preferable when using functions from an external module or
  for wrapping an anonymous lambda.

  The decorator can be used in two ways:
  1. Invoked with a single positional argument specifying the types of the
     function's arguments (`@xyz(some_argument_type)`).
  2. Invoked with no arguments (`@xyz` or `@xyz()`). This is used for functions
     which take no arguments, or functions which are polymorphic (used with
     multiple different argument types).

  Here's how the decorator behaves in each case:

  If the user specifies a tuple type in an unbundled form (simply by listing the
  types of its constituents as separate arguments), the tuple type is formed on
  the user's behalf for convenience.

  1. When the decorator is invoked with positional arguments:

     ```python
     @xyz(('x', tf.int32), ('y', tf.int32))
     ```

     The decorator arguments must be instances of `types.Type`, or something
     convertible to it by `types.to_type()`. The arguments are interpreted as
     the specification of the parameter of the computation being constructed by
     the decorator. Since the formal parameter to computations is always a
     single argument, multiple arguments to the decorator will be packed into a
     tuple type. This means that the following two invocations behave the same:

     ```
     @xyz(('x', tf.int32), ('y', tf.int32)) # gets packed into the below
     @xyz((('x', tf.int32), ('y', tf.int32)))
     ```

     In the above example, the computation will accept as an argument a pair
     of integers named `x` and `y`.

     The function being decorated this way must declare at least one parameter.

     a. If the Python function declares only one parameter, that parameter will
        receive all arguments packed into a single value:

        ```python
        @xyz(('x', tf.int32), ('y', tf.int32))
        def my_comp(coord):
          ... # use `coord.x` and `coord.y`
        ```

     b. If the Python function declares multiple parameters, the computation's
        parameter type must be convertible to type `tff.StructType`
        (usually a list containing types or pairs of `(str, types.Type)`.

        ```python
        # With explicitly named parameters
        @xyz(('x', tf.int32), ('y', tf.int32))
        def my_comp(x, y):
          ... # use `x` and `y`

        # Without explicitly named parameters
        @xyz(tf.int32, tf.int32)
        def my_comp(x, y):
          ... # use `x` and `y`
        ```

        The number and order of parameters in the decorator arguments and the
        Python function must match. For named elements, the names in the
        decorator and the Python function must also match.

        Python functions with multiple parameters can end the parameter list
        with `*args` or `*kwargs` to pack all remaining arguments into a single
        If the Python function declares `*args` or `*kwargs`, any remaining
        parameters will be packed into this single argument:

        ```python
        @xyz(tf.int32, tf.int32, tf.int32, tf.int32)
        def my_comp(x, y, *args):
          ... # use `x`, `y`, `args[0]`, `args[1]`
        ```

        If `*args` is the only argument to the Python function, an exception
        will be thrown, since it's ambiguous whether the function accepts a
        single `tff.StructType` argument (as in the single-argument case (a)
        above) or a list of arguments.

  2. When the decorator is specified without arguments (`@xyz` or `@xyz()`):

     a. If the Python function declares no parameters, the decorator constructs
        a no-parameter computation, as in the following example:

        ```python
        @xyz
        def my_comp():
          ...
        ```

     b. If the function does declare at least one parameter, it is treated as a
        polymorphic function that's instantiated in each concrete context in
        which it's used based on the types of its arguments. The decorator still
        handles the plumbing and parameter type inference.

        For example:

        ```python
        @xyz
        def my_comp(x, y):
          ...
        ```

        In this case, `my_comp` becomes a polymorphic callable, with the actual
        construction postponed. Suppose it's then used as follows, e.g., in an
        orchestration context:

        ```python
        my_comp(5.0, True)
        ```

        At the time of invocation, the decorator uses the information contained
        in the call arguments 5.0 and True to infer the computation's parameter
        type signature, and once the types have been determined, proceeds in
        exactly the same manner as already described in (1) above.

        It is important to note that the arguments of the invocation are not
        simply passed into the body of the Python function being decorated.
        The parameter type inference step is all that differs between the
        polymorphic case and case (1) above.

        Polymorphic functions are the only case where no constraints exist on
        the kinds of arguments that may be present: declaring default values,
        `*args` or `**kwargs`, and any combination of those are valid. The
        mapping is resolved at invocation time based on arguments of the call,
        as in the example below:

        ```python
        @xyz
        def my_comp(x, y=True, *args, **kwargs):
          ...

        my_comp(1, False, 2, 3, 'foo', name='bar')
        ```

        As with all polymorphic functions, no construction is actually performed
        until invocation, and at invocation time, the default parameter values
        are used alongside those actually used during the invocation to infer
        the computation's parameter type. The code that actually constructs the
        computation is oblivious to whether parameters of the Python function
        being decorated were driven by the default values, or by the arguments
        of the actual call.

        Note that the last argument to the function in the example above will
        be inferred as type `('name', str)`, not just `str`.

  For more examples of usage, see `computation_wrapper_test`.
  """

  def __init__(self, wrapper_fn):
    """Construct a new wrapper/decorator for the given wrapping function.

    Args:
      wrapper_fn: The Python callable that performs actual wrapping (as in the
        specification of `_wrap`).

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_callable(wrapper_fn)
    self._wrapper_fn = wrapper_fn

  def __call__(self, *args):
    """Handles the different modes of usage of the decorator/wrapper.

    This method only acts as a frontend that allows this class to be used as a
    decorator or wrapper in a variety of ways. The actual wrapping is performed
    by the private method `_wrap`.

    Args:
      *args: Positional arguments (the decorator at this point does not accept
        keyword arguments, although that might change in the future).

    Returns:
      Either a result of wrapping, or a callable that expects a function,
      method, or a defun and performs wrapping on it, depending on specific
      usage pattern.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    if not args:
      # If invoked as a decorator, and with an empty argument list as "@xyz()"
      # applied to a function definition, expect the Python function being
      # decorated to be passed in the subsequent call, and potentially create
      # a polymorphic callable. The parameter type is unspecified.
      # Deliberate wrapping with a lambda to prevent the caller from being able
      # to accidentally specify parameter type as a second argument.
      return lambda fn: _wrap(fn, None, self._wrapper_fn)
    elif (isinstance(args[0], (types.FunctionType, types.MethodType)) or
          function.is_tf_function(args[0])):
      # If the first argument on the list is a Python function, instance method,
      # or a defun, this is the one that's being wrapped. This is the case of
      # either a decorator invocation without arguments as "@xyz" applied to a
      # function definition, of an inline invocation as "... = xyz(lambda....).
      # Any of the following arguments, if present, are the arguments to the
      # wrapper that are to be interpreted as the type specification.
      if len(args) > 2:
        args = (args[0], args[1:])
      return _wrap(
          args[0],
          computation_types.to_type(args[1]) if len(args) > 1 else None,
          self._wrapper_fn)
    else:
      if len(args) > 1:
        args = (args,)
      arg_type = computation_types.to_type(args[0])
      return lambda fn: _wrap(fn, arg_type, self._wrapper_fn)
