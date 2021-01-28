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
"""Utilities for constructing decorators/wrappers for functions and tf.function."""

import collections
import functools
import inspect
import types
from typing import Optional, Tuple

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.tensorflow_libs import function


def _parameters(fn):
  return function_utils.get_signature(fn).parameters.values()


def _check_parameters(parameters):
  """Ensure only non-varargs positional-or-keyword arguments."""
  for parameter in parameters:
    if parameter.default is not inspect.Parameter.empty:
      # We don't have a way to build defaults into the function's type.
      raise TypeError(
          'TFF does not support default parameters. Found parameter '
          f'`{parameter.name}` with default value {parameter.default}')
    if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
      # We don't have a way to encode positional-only into the function's type.
      raise TypeError(
          'TFF does not support positional-only parameters. Found parameter '
          f'`{parameter.name}` which appears before a `/` entry.')
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
      # We don't have a way to encode keyword-only into the function's type.
      raise TypeError(
          'TFF does not support keyword-only arguments. Found parameter '
          f'`{parameter.name}` which appears after a `*` or `*args` entry.')
    if parameter.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
      # For concrete functions, we can't determine at tracing time which
      # arguments should be bundled into args vs. kwargs, since arguments can
      # be passed by position *or* by keyword at later call sites.
      raise TypeError('TFF does not support varargs. Found varargs parameter '
                      f'`{parameter.name}`.')
    if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
      raise AssertionError(f'Unexpected parameter kind: {parameter.kind}')


def _wrap_concrete(fn_name: Optional[str],
                   wrapper_fn,
                   parameter_type,
                   unpack=None) -> function_utils.ConcreteFunction:
  """Wraps with `wrapper_fn` given the provided `parameter_type`."""
  generator = wrapper_fn(parameter_type, fn_name)
  arg = next(generator)
  try:
    result = yield arg
  except Exception as e:  # pylint: disable=broad-except
    generator.throw(e)
  concrete_fn = generator.send(result)
  py_typecheck.check_type(concrete_fn, function_utils.ConcreteFunction,
                          'value returned by the wrapper')
  result_parameter_type = concrete_fn.type_signature.parameter
  if (result_parameter_type is not None and
      not result_parameter_type.is_equivalent_to(parameter_type)):
    raise TypeError(
        'Expected a concrete function that takes parameter {}, got one '
        'that takes {}.'.format(
            str(parameter_type), str(concrete_fn.type_signature.parameter)))
  yield concrete_fn


def _parameter_type(
    parameters, parameter_types: Tuple[computation_types.Type, ...]
) -> Optional[computation_types.Type]:
  """Bundle any user-provided parameter types into a single argument type."""
  parameter_names = [parameter.name for parameter in parameters]
  if not parameter_types and not parameters:
    return None
  if len(parameter_types) == 1:
    parameter_type = parameter_types[0]
    if parameter_type is None and not parameters:
      return None
    if len(parameters) == 1:
      return parameter_type
    # There is a single parameter type but multiple parameters.
    if not parameter_type.is_struct() or len(parameter_type) != len(parameters):
      raise TypeError(
          f'Function with {len(parameters)} parameters must have a parameter '
          f'type with the same number of parameters. Found parameter type '
          f'{parameter_type}.')
    name_list_from_types = structure.name_list(parameter_type)
    if name_list_from_types:
      if len(name_list_from_types) != len(parameter_type):
        raise TypeError(
            'Types with both named and unnamed fields cannot be unpacked into '
            f'argument lists. Found parameter type {parameter_type}.')
      if set(name_list_from_types) != set(parameter_names):
        raise TypeError(
            'Function argument names must match field names of parameter type. '
            f'Found argument names {parameter_names}, which do not match '
            f'{name_list_from_types}, the top-level fields of the parameter '
            f'type {parameter_type}.')
      # The provided parameter type has all named fields which exactly match
      # the names of the function's parameters.
      return parameter_type
    else:
      # The provided parameter type has no named fields. Apply the names from
      # the function parameters.
      parameter_types = (v for (_, v) in structure.to_elements(parameter_type))
      return computation_types.StructWithPythonType(
          list(zip(parameter_names, parameter_types)), collections.OrderedDict)
  elif len(parameters) == 1:
    # If there are multiple provided argument types but the function being
    # decorated only accepts a single argument, tuple the arguments together.
    return computation_types.to_type(parameter_types)
  if len(parameters) != len(parameter_types):
    raise TypeError(
        f'Function with {len(parameters)} parameters is '
        f'incompatible with provided argument types {parameter_types}.')
  # The function has `n` parameters and `n` parameter types.
  # Zip them up into a structure using the names from the function as keys.
  return computation_types.StructWithPythonType(
      list(zip(parameter_names, parameter_types)), collections.OrderedDict)


def is_function(maybe_fn):
  return (isinstance(maybe_fn, (types.FunctionType, types.MethodType)) or
          function.is_tf_function(maybe_fn))


class ComputationReturnedNoneError(ValueError):
  """Error for computations which return `None` or do not return."""

  def __init__(self, fn_to_wrap):
    code = fn_to_wrap.__code__
    line_number = code.co_firstlineno
    filename = code.co_filename
    message = (
        f'The function defined on line {line_number} of file {filename} '
        'returned `None` (or didn\'t explicitly `return` at all), but TFF '
        'computations must return some non-`None` value.')
    super().__init__(message)


class _TracingError(Exception):
  """Cleanup error provided to generators upon errors during tracing."""


class PythonTracingStrategy(object):
  """A wrapper strategy that directly traces the function being wrapped.

  This strategy relies on directly invoking the Python function being wrapped
  in an appropriately prepared context, feeding it synthetic arguments, and
  processing the returned result. This strategy may not be usable in cases,
  where the process of serializing a Python computation relies on an external
  serialization logic.

  The mechanics are as follows:

  * The `wrapper_fn` supplied in the constructor is treated as a generator that
    given the type of the computation parameter, understands how to prepare
    synthetic arguments for the function to be traced, process its result, and
    construct the serialized form of the computation.

  * The generator is first asked to yield synthetic arguments for the function
    being traced. These are unpacked as needed to match the Python function's
    signature, and the Python function is invoked.

  * The result of the invocation is then sent to the generator, to allow it to
    continue, and the generator yields the constructed computation.

  Here's one way one can use this helper class with a `ComputationWrapper`:

  ```python
  def my_wrapper_fn(parameter_type, name=None):
    ...generate some stand-in argument structure matching `parameter_type`...
    result_of_calling_function = yield argument_structure
    ...postprocess the result and generate a function_utils.ConcreteFunction...
    yield concrete_function

  xyz = computation_wrapper.ComputationWrapper(
    PythonTracingStrategy(my_wrapper_fn))
  ```
  """

  def __init__(self, wrapper_fn):
    """Constructs this tracing strategy given a generator-based `wrapper_fn`.

    Args:
      wrapper_fn: The Python callable that controls the tracing process.
    """
    self._wrapper_fn = wrapper_fn

  def __call__(self, fn_to_wrap, fn_name, parameter_type, unpack):
    unpack_arguments_fn = function_utils.create_argument_unpacking_fn(
        fn_to_wrap, parameter_type, unpack=unpack)
    wrapped_fn_generator = _wrap_concrete(fn_name, self._wrapper_fn,
                                          parameter_type)
    packed_args = next(wrapped_fn_generator)
    try:
      args, kwargs = unpack_arguments_fn(packed_args)
      result = fn_to_wrap(*args, **kwargs)
      if result is None:
        raise ComputationReturnedNoneError(fn_to_wrap)
    except Exception:
      # Give nested generators an opportunity to clean up, then
      # re-raise the original error without extra context.
      # We don't want to simply pass the error into the generators,
      # as that would result in the whole generator stack being added
      # to the error message.
      try:
        wrapped_fn_generator.throw(_TracingError())
      except _TracingError:
        pass
      raise
    return wrapped_fn_generator.send(result)


class ComputationWrapper(object):
  """A class for creating wrappers that convert functions into computations.

  Here's how one can use `ComputationWrapper` to construct a decorator/wrapper
  named `xyz`:

  ```python
  xyz = computation_wrapper.ComputationWrapper(...)
  ```

  The resulting `xyz` can then be used either as an `@xyz(...)` decorator or as
  a manual wrapping function: `wrapped_func = xyz(my_func, ...)`. The latter
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

  def __init__(self, strategy):
    """Construct a new wrapper/decorator for the given wrapper callable.

    Args:
      strategy: Python callable that encapsulates the mechanics of the actual
        wrapping process. It must satisfy the following requirements. First, it
        must take a tuple `(fn_to_wrap, fn_name, parameter_type, unpack)` as an
        argument, where `fn_to_wrap` is the Python function being wrapped,
        `fn_name` is a name to assign to the constructed computation (typically
        the same as the name of the Python function), `parameter_type` is an
        instance of `computation_types.Type` that represents the TFF type of the
        computation parameter, and `unpack` affects the process of mapping
        Python parameters to the TFF computation's parameter, and has the
        semantics identical to the `unpack` flag in the specification of
        `function_utils.create_argument_unpacking_fn`. Second, it must return a
        result of type `computation_impl.ComputationImpl` that represents the
        constructed computation.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_callable(strategy)
    self._strategy = strategy

  def __call__(self, *args, tff_internal_types=None):
    """Handles the different modes of usage of the decorator/wrapper.

    Args:
      *args: Positional arguments (the decorator at this point does not accept
        keyword arguments, although that might change in the future).
      tff_internal_types: TFF internal usage only. This argument should be
        considered private.

    Returns:
      Either a result of wrapping, or a callable that expects a function,
      method, or a tf.function and performs wrapping on it, depending on
      specific usage pattern.

    Raises:
      TypeError: if the arguments are of the wrong types.
      ValueError: if the function to wrap returns `None`.
    """
    if not args or not is_function(args[0]):
      # If invoked as a decorator, and with an empty argument list as "@xyz()"
      # applied to a function definition, expect the Python function being
      # decorated to be passed in the subsequent call, and potentially create
      # a polymorphic callable. The parameter type is unspecified.
      # Deliberate wrapping with a lambda to prevent the caller from being able
      # to accidentally specify parameter type as a second argument.
      # The tricky partial recursion is needed to inline the logic in the
      # "success" case below.
      if tff_internal_types is not None:
        raise TypeError(f'Expected a function to wrap, found {args}.')
      provided_types = tuple(map(computation_types.to_type, args))
      return functools.partial(self.__call__, tff_internal_types=provided_types)
    # If the first argument on the list is a Python function, instance method,
    # or a tf.function, this is the one that's being wrapped. This is the case
    # of either a decorator invocation without arguments as "@xyz" applied to
    # a function definition, of an inline invocation as
    # `... = xyz(lambda....).`
    # Any of the following arguments, if present, are the arguments to the
    # wrapper that are to be interpreted as the type specification.
    fn_to_wrap = args[0]
    if not tff_internal_types:
      tff_internal_types = tuple(map(computation_types.to_type, args[1:]))
    else:
      if len(args) > 1:
        raise TypeError(f'Expected no further arguments, found {args[1:]}.')

    parameter_types = tff_internal_types
    parameters = _parameters(fn_to_wrap)

    # NOTE: many of the properties checked here are only necessary for
    # non-polymorphic computations whose type signatures must be resolved
    # prior to use. However, we continue to enforce these requirements even
    # in the polymorphic case in order to avoid creating an inconsistency.
    _check_parameters(parameters)

    try:
      fn_name = fn_to_wrap.__name__
    except AttributeError:
      fn_name = None

    if (not parameter_types) and parameters:
      # There is no TFF type specification, and the function/tf.function
      # declares parameters. Create a polymorphic template.
      def _polymorphic_wrapper(parameter_type: computation_types.Type,
                               unpack: Optional[bool]):
        return self._strategy(
            fn_to_wrap, fn_name, parameter_type, unpack=unpack)

      wrapped_func = function_utils.PolymorphicFunction(_polymorphic_wrapper)
    else:
      # Either we have a concrete parameter type, or this is no-arg function.
      parameter_type = _parameter_type(parameters, parameter_types)
      wrapped_func = self._strategy(
          fn_to_wrap, fn_name, parameter_type, unpack=None)

    # Copy the __doc__ attribute with the documentation in triple-quotes from
    # the decorated function.
    wrapped_func.__doc__ = getattr(fn_to_wrap, '__doc__', None)

    return wrapped_func
