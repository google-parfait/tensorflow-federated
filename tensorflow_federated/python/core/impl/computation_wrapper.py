# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import function_utils
from tensorflow_federated.python.core.impl import type_utils


def _wrap(fn, parameter_type, wrapper_fn):
  """Wrap a given `fn` with a given `parameter_type` using `wrapper_fn`.

  This method does not handle the multiple modes of usage as wrapper/decorator,
  as those are handled by ComputationWrapper below. It focused on the simple
  case with a function/defun (always present) and either a valid parameter type
  or an indication that there's no parameter (None).

  The only ambiguity left to resolve is whether `fn` should be immediately
  wrapped, or treated as a polymorphic callable to be wrapped upon invocation
  based on actual parameter types. The determination is based on the presence
  or absence of parameters in the declaration of `fn`. In order to be
  treated as a concrete no-argument computation, `fn` shouldn't declare any
  arguments (even with default values).

  The `wrapper_fn` must accept three arguments, and optional forth kwarg `name`:

  * `target_fn'`, the Python function that to be wrapped, accepting possibly
    *args and **kwargs.

  * Either None for a no-parameter computation, or the type of the computation's
    parameter (an instance of `computation_types.Type`) if the computation has
    one.

  * `unpack`, an argument which will be passed on to
    `function_utils.wrap_as_zero_or_one_arg_callable` when wrapping `target_fn`.
    See that function for details.

  * Optional `name`, the name of the function that is being wrapped (only for
    debugging purposes).

  Args:
    fn: The function or defun to wrap as a computation.
    parameter_type: The parameter type accepted by the computation, or None if
      there is no parameter.
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
  argspec = function_utils.get_argspec(fn)
  parameter_type = computation_types.to_type(parameter_type)
  if parameter_type is None:
    if (argspec.args or argspec.varargs or argspec.keywords):
      # There is no TFF type specification, and the function/defun declares
      # parameters. Create a polymorphic template.
      def _wrap_polymorphic(wrapper_fn, fn, parameter_type, name=fn_name):
        return wrapper_fn(fn, parameter_type, unpack=True, name=name)

      polymorphic_fn = function_utils.PolymorphicFunction(
          lambda pt: _wrap_polymorphic(wrapper_fn, fn, pt))

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
  if not type_utils.are_equivalent_types(concrete_fn.type_signature.parameter,
                                         parameter_type):
    raise TypeError(
        'Expected a concrete function that takes parameter {}, got one '
        'that takes {}.'.format(
            str(parameter_type), str(concrete_fn.type_signature.parameter)))
  # When applying a decorator, the __doc__ attribute with the documentation
  # in triple-quotes is not automatically transferred from the function on
  concrete_fn.__doc__ = getattr(fn, '__doc__', None)
  return concrete_fn


class ComputationWrapper(object):
  """A decorator/wrapper for converting functions and defuns into computations.

  This class builds upon the _wrap() function defined above, and offers several
  forms of syntactic sugar, primarily as a function decorator.

  Each decorator/wrapper, created as an instance of this class, may serve as a
  decorator/wrapper of a particular type, depending on the constructor argument.
  At this point, we plan on having two decorators/wrappers, one for TensorFlow
  code, and a separate one for orchestration logic, i.e., for anything that is
  not TensorFlow, and in future, potentially for hybrid TF/non-TF logic. We may
  or may not eventually merge these and/or define new kinds of wrappers for
  other types of computation logic. Each of these and future decorators/wrappers
  will be created with one constructor call from an appropriate API module.

  Decorators/wrappers share certain aspects of their behavior for the sake of
  offering consistent developer experience. This class does not deal with
  any particulars of how the specific decorator/wrapper converts code into a
  computation, only with the common aspects of relating formal Python function
  parameters and call arguments to TFF types, packing and unpacking arguments,
  verifying types, and support for polymorphism.

  Here's how one can construct a decorator/wrapper named `xyz`:

  ```python
  xyz = computation_wrapper.ComputationWrapper(...constructor arguments...)
  ```

  Here's how one can use it as a decorator to transform a Python function to a
  computation:

  ```python
  @xyz(...decorator arguments...)
  def my_comp(...function parameters...):
    ...
  ```
  Converting defuns works in the same way:

  ```python
  @xyz(...decorator arguments...)
  @tf.function
  def my_comp(...function parameters...):
    ...
  ```

  Alternatively, the newly defined `xyz` can also always be used inline, i.e.,
  with the Python function (or a defun) that's to be converted passed to it
  explicitly as an argument, as in the example below:

  ```python
  my_comp = xyz(fn, ...decorator arguments...)
  ```

  This mode of invocation can be used, e.g., to transform `fn` that's been
  defined in an external module, or an anonymous lambda that may be more
  convenient to define in-line in the call arguments in case where the lambda
  is short and simple (e.g., a single TF op).

  In the description that follows, the three terms "constructor arguments",
  "decorator arguments", and "function parameters" will refer to the parts of
  the code in the example shown above. Terms like "decorator" and "wrapper"
  will refer to objects constructed and returned by this function, and they
  may be used interchangeably.

  Once created, a decorator/wrapper can be used with or without arguments, and
  it can be applied to a function that does or does not accept parameters. The
  behavior of the decorator varies depending on what decorator arguments and
  parameters in the Python function are present. This is dictated by the fact
  that there is an impedance mismatch between the complex function signatures
  in Python, with multiple arguments, some positional and some keyword-based,
  default values, *args and **kwargs, etc., and the simplified type system in
  TFF in which each computation having either zero or exactly one parameter,
  as well as the fact that in Python, the syntax for specifying a function's
  parameter tuple as keyword args deviates from the syntax of specifying tuple
  or dictionary values in the general case, whereas in TFF no such discrepancy
  can exist by design. The conventions below attempt at offering some degree
  of flexibility for the Python programmer to declare and use the annotated
  functions in a manner they might find convenient, while having a consistent
  and predictable mapping between the various styles of definition and usage,
  and the type signatures of the computations in TFF.

  At the definition time, the following scenarios are supported. In case when
  the decorator is used with a defun, the term "Python function" refers to the
  abstract Python function that takes the same arguments (both ordinary Python
  functions and defuns are manipulated using helper functionality defined in
  function_utils.py).

  1. The decorator is invoked with a single positional decorator argument, as
     in the example below.

     ```python
     @xyz([('x', tf.int32), ('y', tf.int32)])
     ```

     The decorator argument must be an instance of `types.Type`, or something
     convertible to it by `types.to_type()`. The argument is interpreted as the
     specification of the (single) formal parameter of the computation being
     constructed by the decorator.

     In the above example, the computation will accept as an argument a pair
     of integers named `x` and `y`.

     The function being decorated this way must declare at least one parameter.

     a. If the Python function declares exactly one parameter, the parameter
        represents the (single, as always) parameter of the computation under
        construction in its entirety, as in the example below.

        ```python
        @xyz([('x', tf.int32), ('y', tf.int32)])
        def my_comp(coord):
          ...
        ```

        Here, in the body of `my_comp`, the parameter `coord` represents the
        entire integer pair, so its constituents would be referred to in the
        code of `my_comp` as `coord.x` and `coord.y`, respectively.

        Note that the name `coord` has no meaning outside of `my_comp` body,
        in that it does not affect the type signature of the constructed
        computation. It does, however, generally affect naming used internally,
        as we make best-effort attempt at making names that appear internally
        match those that appear in Python code for the ease of debugging. For
        example, `coord` might be used as a common naming scope for parameter
        placeholders or variables in a TensorFlow graph constructed.

     b. If the Python function declares multiple parameters, the computation's
        parameter type must be a named tuple type. The parameters of the Python
        function are interpreted as capturing elements of that parameter tuple,
        as in the example below.

        ```python
        @xyz([('x', tf.int32), ('y', tf.int32)])
        def my_comp(x, y):
          ...
        ```

        The number of parameters accepted by the Python function in this case
        must match the number of elements in the named tuple type specified in
        the decorator argument. The elements of the tuple type do not have to
        be named, and the example below is also valid.

        ```python
        @xyz([tf.int32, tf.int32])
        def my_comp(x, y):
          ...
        ```

        However, where named tuple elements are named, their names must match
        those that appear on the parameter list of the Python function, and
        they have to be listed in the same order, as we assume correspondence
        based on the position in the tuple, not based on the naming.

     c. If the Python function declares `*args` or `*kwargs`, the parameter type
        is mapped to those in the intuitive way, except if there's an ambigiuty
        in the mapping. The latter will be the case, e.g., when the parameter
        type is a typle, and the function defines *args and nothing else,
        leaving ambiguity as to whether it expects to accept the tuple in its
        entirety, or its elements. In this case, an exception is thrown.

        ```python
        @xyz([tf.int32, tf.int32, tf.int32])
        def my_comp(x, y, *args):
          ...
        ```

  2. The decorator is specified without arguments `xyz`, or invoked with an
     empty list of arguments `xyz()`. The two are treated as equivalent.

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
        handles the plumbing and parameter type inference, while delegating the
        actual computation construction to a component responsible for it.

        Thus, for example, consider the decorastor used as follows:

        ```python
        @xyz
        def my_comp(x, y):
          ...
        ```

        In this case, `my_comp` becomes a polymorphic callable, with the actual
        construction postponed. Suppose it's then used as follows, e.g., in an
        orchestration context:

          my_comp(5.0, True)

        At the time of invocation, the decorator uses the information contained
        in the call arguments 5.0 and True to infer the computation's parameter
        type signature, and once the types have been determined, proceeds in
        exactly the same manner as already described above.

        It is important to note that, in all cases, but perhaps most notably
        including this one, as it's driven by actual usage, the Python
        arguments of the invocation are not simply passed into the body of the
        Python function being decorated. The computation is being constructed
        on the fly basd on concrete usage, but the construction occurs outside
        of the context of the invocation. The parameter type inference step is
        all that relates the two.

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

  For more examples of usage, see `computation_wrapper_test`.

  If the user specifies a tuple type in an unbundled form (simply by listing the
  types of its constituents as separate arguments), the tuple type is formed on
  the user's behalf for convenience.
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
    """Handles the differents modes of usage of the decorator/wrapper.

    This method only acts as a frontend that allows this class to be used as a
    decorator or wrapper in a variety of ways. The actual wrapping is performed
    by the private method `_wrap`.

    Args:
      *args: Positional arguments (the decorator at this point does not accept
        keyword arguments, although that might change in the future).

    Returns:
      Either a result of wrapping, or a callable that expects a function or a
      defun and performs wrapping on it, depending on specific usage pattern.

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
    elif (isinstance(args[0], types.FunctionType) or
          function_utils.is_defun(args[0])):
      # If the first argument on the list is a Python function or a defun, this
      # is the one that's being wrapped. This is the case of either a decorator
      # invocation without arguments as "@xyz" applied to a function definition,
      # of an inline invocation as "... = xyz(lambda....). Any of the following
      # arguments, if present, are the arguments to the wrapper that are to be
      # interpreted as the type specification.
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
