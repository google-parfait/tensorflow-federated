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
"""Utilities for Python functions, defuns, and other types of callables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import types

import attr
import six
from six.moves import range

from tensorflow.python.framework import function
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import context_base
from tensorflow_federated.python.core.impl import context_stack_base
from tensorflow_federated.python.core.impl import type_utils


def is_defun(fn):
  """Determines whether `fn` is one of the known types of TF defuns.

  Args:
    fn: The object to test for being a supported type of a TensorFlow defun.

  Returns:
    True iff `fn` is a supported type of a TF defun.
  """
  return isinstance(
      fn,
      (
          # TODO(b/113112885): Add support for tfe Function and
          # PolymorphicFunction,
          # currently omitted due to issues with visibility.

          # While these classes can be private to TF users, we need to peek into
          # the private interfaces of these classes in order to obtain the
          # function signatures and type information that are otherwise
          # unavailable via regular public APIs. In order to do so safelty, we
          # need to narrow the scope down to a few concrete classes, internal
          # structure we create a dependency on.
          # TODO(b/113112885): Work towards avoiding this, posisbly by
          # upstreaming some helper library or extending the public interface.
          function._DefinedFunction,  # pylint: disable=protected-access
          function._OverloadedFunction  # pylint: disable=protected-access
      )) or (
          # TODO(b/113112885): Add (cleaner) support for tf.Function and
          'def_function.Function' in py_typecheck.type_string(type(fn)))


@attr.s(slots=True, frozen=True)
class SimpleArgSpec(object):
  """A simple container class that mimics the deprecated `inspect.ArgSpec`."""
  args = attr.ib()
  varargs = attr.ib()
  keywords = attr.ib()
  defaults = attr.ib()

  def __str__(self):
    parts = []
    if self.args:
      parts.append('args={}'.format(self.args))
    if self.varargs:
      parts.append('varargs={}'.format(self.varargs))
    if self.keywords:
      parts.append('kwargs={}'.format(self.keywords))
    if self.defaults:
      parts.append('defaults={}'.format(self.defaults))
    return '({})'.format(', '.join(parts))


def get_argspec(fn):
  """Returns the `SimpleArgSpec` structure for the given function.

  Args:
    fn: The Python function or Tensorflow function to analyze.

  Returns:
    A `SimpleArgSpec`.

  Raises:
    TypeError: if the argument is not of a supported type.
  """

  def _getargspec(fn):
    """Get the argspec depending on the version of Python being used."""
    if six.PY2:
      argspec = inspect.getargspec(fn)  # pylint: disable=deprecated-method
      return SimpleArgSpec(*argspec)
    else:
      signature = inspect.signature(fn)
      args = []
      varargs = None
      keywords = None
      defaults = None
      for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
          keywords = param.name
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
          varargs = param.name
        else:
          args.append(param.name)
        if param.default != inspect.Parameter.empty:
          if defaults is None:
            defaults = [param.default]
          else:
            defaults.append(param.default)
      return SimpleArgSpec(args, varargs, keywords,
                           tuple(defaults) if defaults else None)

  if isinstance(fn, types.FunctionType):
    return _getargspec(fn)
  elif is_defun(fn):
    return _getargspec(fn.python_function)
  else:
    raise TypeError('Expected a Python function or a defun, found {}.'.format(
        py_typecheck.type_string(type(fn))))


def get_callargs_for_argspec(argspec, *args, **kwargs):
  """Similar to `inspect.getcallargs()`, but accepts SimpleArgSpec instead.

  This function allows `getcallargs()` capability to be used with defuns and
  other types of callables that aren't Python functions.

  Args:
    argspec: An instance of SimpleArgSpec to assign arguments to.
    *args: Positional arguments.
    **kwargs: Keyword-based arguments.

  Returns:
    The same type of result as what `inspect.getcallargs()` returns.

  Raises:
    TypeError: if the arguments are of the wrong types, or if the 'args' and
      'kwargs' combo is not compatible with 'argspec'.
  """
  py_typecheck.check_type(argspec, SimpleArgSpec)
  result = {}
  num_specargs = len(argspec.args) if argspec.args else 0
  num_defaults = len(argspec.defaults) if argspec.defaults else 0
  num_specargs_without_defaults = num_specargs - num_defaults
  if len(args) > num_specargs and not argspec.varargs:
    raise TypeError(
        'Too many positional arguments for the call: expected at most {}, '
        'found {}.'.format(num_specargs, len(args)))
  for idx, specarg in enumerate(argspec.args):
    if idx < len(args):
      if specarg in kwargs:
        raise TypeError('Argument {} specified twice.'.format(specarg))
      result[specarg] = args[idx]
    elif specarg in kwargs:
      result[specarg] = kwargs[specarg]
    elif idx >= num_specargs_without_defaults:
      result[specarg] = argspec.defaults[idx - num_specargs_without_defaults]
    else:
      raise TypeError(
          'Argument {} was not specified and does not have a default.'.format(
              specarg))
  unused_kwargs = {k: v for k, v in six.iteritems(kwargs) if k not in result}
  if argspec.varargs:
    result[argspec.varargs] = args[num_specargs:]
  if argspec.keywords:
    result[argspec.keywords] = unused_kwargs
  elif unused_kwargs:
    raise TypeError(
        'Unexpected keyword arguments in the call: {}'.format(unused_kwargs))
  return result


def is_argspec_compatible_with_types(argspec, *args, **kwargs):
  """Determines if functions matching 'argspec' accept given 'args'/'kwargs'.

  Args:
    argspec: An instance of `SimpleArgSpec` to verify agains the arguments.
    *args: Zero or more positional arguments, all of which must be instances of
      computation_types.Type or something convertible to it by
      computation_types.to_type().
    **kwargs: Zero or more keyword arguments, all of which must be instances of
      computation_types.Type or something convertible to it by
      computation_types.to_type().

  Returns:
    True or false, depending on the outcome of the test.

  Raises:
    TypeError: if the arguments are of the wrong computation_types.
  """
  try:
    callargs = get_callargs_for_argspec(argspec, *args, **kwargs)
    if not argspec.defaults:
      return True
  except TypeError:
    return False

  # As long as we have been able to construct 'callargs', and there are no
  # default values to verify against the given types, there is nothing more
  # to do here, otherwise we have to verify the types of defaults against
  # the types we've been given as parameters to this function.
  num_specargs_without_defaults = len(argspec.args) - len(argspec.defaults)
  for idx, default_value in enumerate(argspec.defaults):
    if default_value is not None:
      arg_name = argspec.args[num_specargs_without_defaults + idx]
      call_arg = callargs[arg_name]
      if call_arg is not default_value:
        arg_type = computation_types.to_type(call_arg)
        default_type = type_utils.infer_type(default_value)
        if not type_utils.is_assignable_from(arg_type, default_type):
          return False
  return True


def is_argument_tuple(arg):
  """Determines if 'arg' is interpretable as an argument tuple.

  Args:
    arg: A value or type to test.

  Returns:
    True iff 'arg' is either an anonymous tuple in which all unnamed elements
    precede named ones, or a named tuple typle with this property, or something
    that can be converted into the latter by computation_types.to_type().

  Raises:
    TypeError: if the argument is neither an AnonymousTuple, nor a type spec.
  """
  if isinstance(arg, anonymous_tuple.AnonymousTuple):
    elements = anonymous_tuple.to_elements(arg)
  elif isinstance(arg, value_base.Value):
    return is_argument_tuple(arg.type_signature)
  else:
    arg = computation_types.to_type(arg)
    if isinstance(arg, computation_types.NamedTupleType):
      elements = anonymous_tuple.to_elements(arg)
    else:
      return False
  max_unnamed = -1
  min_named = len(elements)
  for idx, element in enumerate(elements):
    if element[0]:
      min_named = min(min_named, idx)
    else:
      max_unnamed = idx
  return max_unnamed < min_named


def unpack_args_from_tuple(tuple_with_args):
  """Extracts argument types from a named tuple type.

  Args:
    tuple_with_args: An instance of either an AnonymousTuple or
      computation_types.NamedTupleType (or something convertible to it by
      computation_types.to_type()), on which is_argument_tuple() is True.

  Returns:
    A pair (args, kwargs) containing tuple elements from 'tuple_with_args'.

  Raises:
    TypeError: if 'tuple_with_args' is of a wrong type.
  """
  if not is_argument_tuple(tuple_with_args):
    raise TypeError('Not an argument tuple: {}.'.format(tuple_with_args))
  if isinstance(tuple_with_args, anonymous_tuple.AnonymousTuple):
    elements = anonymous_tuple.to_elements(tuple_with_args)
  elif isinstance(tuple_with_args, value_base.Value):
    elements = []
    for index, (name, _) in enumerate(
        anonymous_tuple.to_elements(tuple_with_args.type_signature)):
      if name is not None:
        elements.append((name, getattr(tuple_with_args, name)))
      else:
        elements.append((None, tuple_with_args[index]))
  else:
    tuple_with_args = computation_types.to_type(tuple_with_args)
    py_typecheck.check_type(tuple_with_args, computation_types.NamedTupleType)
    elements = anonymous_tuple.to_elements(tuple_with_args)
  args = []
  kwargs = {}
  for e in elements:
    if e[0]:
      kwargs[e[0]] = e[1]
    else:
      args.append(e[1])
  return (args, kwargs)


def pack_args_into_anonymous_tuple(args, kwargs, type_spec=None, context=None):
  """Packs positional and keyword arguments into an anonymous tuple.

  If 'type_spec' is not None, it must be a tuple type or something that's
  convertible to it by computation_types.to_type(). The assignment of arguments
  to fields of the tuple follows the same rule as during function calls. If
  'type_spec' is None, the positional arguments precede any of the keyword
  arguments, and the ordering of the keyword arguments matches the ordering in
  which they appear in kwargs. If the latter is an OrderedDict, the ordering
  will be preserved. On the other hand, if the latter is an ordinary unordered
  dict, the ordering is arbitrary.

  Args:
    args: Positional arguments.
    kwargs: Keyword arguments.
    type_spec: The optional type specification (either an instance of
      computation_types.NamedTupleType or something convertible to it), or None
      if there's no type. Used to drive the arrangements of args into fields of
      the constructed anonymous tuple, as noted in the description.
    context: The optional context (an instance of `context_base.Context`) in
      which the arguments are being packed. Required if and only if the
      `type_spec` is not `None`.

  Returns:
    An anoymous tuple containing all the arguments.

  Raises:
    TypeError: if the arguments are of the wrong computation_types.
  """
  type_spec = computation_types.to_type(type_spec)
  if not type_spec:
    return anonymous_tuple.AnonymousTuple([(None, arg) for arg in args] +
                                          list(six.iteritems(kwargs)))
  else:
    py_typecheck.check_type(type_spec, computation_types.NamedTupleType)
    py_typecheck.check_type(context, context_base.Context)
    if not is_argument_tuple(type_spec):
      raise TypeError(
          'Parameter type {} does not have a structure of an argument tuple, '
          'and cannot be populated from multiple positional and keyword '
          'arguments'.format(type_spec))
    else:
      result_elements = []
      positions_used = set()
      keywords_used = set()
      for index, (name, elem_type) in enumerate(
          anonymous_tuple.to_elements(type_spec)):
        if index < len(args):
          if name is not None and name in kwargs:
            raise TypeError('Argument {} specified twice.'.format(name))
          else:
            arg_value = args[index]
            result_elements.append((name, context.ingest(arg_value, elem_type)))
            positions_used.add(index)
        elif name is not None and name in kwargs:
          arg_value = kwargs[name]
          result_elements.append((name, context.ingest(arg_value, elem_type)))
          keywords_used.add(name)
        elif name:
          raise TypeError('Argument named {} is missing.'.format(name))
        else:
          raise TypeError('Argument at position {} is missing.'.format(index))
      positions_missing = set(range(len(args))).difference(positions_used)
      if positions_missing:
        raise TypeError(
            'Positional arguments at {} not used.'.format(positions_missing))
      keywords_missing = set(kwargs.keys()).difference(keywords_used)
      if keywords_missing:
        raise TypeError(
            'Keyword arguments at {} not used.'.format(keywords_missing))
      return anonymous_tuple.AnonymousTuple(result_elements)


def pack_args(parameter_type, args, kwargs, context):
  """Pack arguments into a single one that matches the given parameter type.

  The arguments may or may not be packed into a tuple, depending on the type of
  the parameter, and how many arguments are present.

  Args:
    parameter_type: The type of the single parameter expected by a computation,
      an instance of computation_types.Type or something convertible to it, or
      None if the computation is not expecting a parameter.
    args: Positional arguments of a call.
    kwargs: Keyword arguments of a call.
    context: The context (an instance of `context_base.Context`) in which the
      arguments are being packed.

  Returns:
    A single value object of type that matches 'parameter_type' that contains
    all the arguments, or None if the 'parameter_type' is None.

  Raises:
    TypeError: if the args/kwargs do not match the given parameter type.
  """
  py_typecheck.check_type(context, context_base.Context)
  if parameter_type is None:
    # If there's no parameter type, there should be no args of any kind.
    if args or kwargs:
      raise TypeError('Was not expecting any arguments.')
    else:
      return None
  else:
    parameter_type = computation_types.to_type(parameter_type)
    if not args and not kwargs:
      raise TypeError(
          'Declared a parameter of type {}, but got no arguments.'.format(
              parameter_type))
    else:
      single_positional_arg = (len(args) == 1) and not kwargs
      if not isinstance(parameter_type, computation_types.NamedTupleType):
        # If not a named tuple type, a single positional argument is the only
        # supported call style.
        if not single_positional_arg:
          raise TypeError(
              'Parameter type {} is compatible only with a single positional '
              'argument, but found {} positional and {} keyword args.'.format(
                  parameter_type, len(args), len(kwargs)))
        else:
          arg = args[0]
      elif single_positional_arg:
        arg = args[0]
      elif not is_argument_tuple(parameter_type):
        raise TypeError(
            'Parameter type {} does not have a structure of an argument '
            'tuple, and cannot be populated from multiple positional and '
            'keyword arguments; please construct a tuple before the '
            'call.'.format(parameter_type))
      else:
        arg = pack_args_into_anonymous_tuple(args, kwargs, parameter_type,
                                             context)
      return context.ingest(arg, parameter_type)


def infer_unpack_needed(fn, parameter_type, should_unpack=None):
  """Returns whether parameter_type must be unpacked when calling fn.

  Args:
    fn: The function to be invoked.
    parameter_type: The TFF type of the parameter bundle to be accepted by the
      returned callable, if any, or None if there's no parameter.
    should_unpack: Default or expected return value; None implies the inferred
      value should be returned. If either unpacking or packing could work, and
      should_unpack is not None, then should_unpack is returned.

  Returns:
    A `bool` indicating whether or not to unpack.
  """
  if should_unpack not in [True, False, None]:
    raise TypeError('The unpack argument has an unexpected value {!r}.'.format(
        should_unpack))
  unpack = should_unpack  # Default return value.
  argspec = get_argspec(fn)

  if parameter_type is None:
    if is_argspec_compatible_with_types(argspec):
      if should_unpack:
        raise ValueError('Requested unpacking of a no-arg function.')
      return False
    else:
      raise TypeError(
          'The argspec {} of the supplied function cannot be interpreted as a '
          'body of a no-parameter computation.'.format(argspec))

  unpack_required = not is_argspec_compatible_with_types(
      argspec, parameter_type)
  # Boolean identity comparison becaue unpack can have a non-boolean value.
  if unpack_required and should_unpack is False:  # pylint: disable=g-bool-id-comparison
    raise TypeError(
        'The supplied function "{}" with argspec {} cannot accept a value of type '
        '{} as a single argument.'.format(fn.__name__, argspec, parameter_type))
  if is_argument_tuple(parameter_type):
    arg_types, kwarg_types = unpack_args_from_tuple(parameter_type)
    unpack_possible = is_argspec_compatible_with_types(argspec, *arg_types,
                                                       **kwarg_types)
  else:
    unpack_possible = False
  # Boolean identity comparison becaue unpack can have a non-boolean value.
  if not unpack_possible and should_unpack is True:  # pylint: disable=g-bool-id-comparison
    raise TypeError(
        'The supplied function with argspec {} cannot accept a value of type '
        '{} as multiple positional and/or keyword arguments. That is, the '
        'argument cannot be unpacked, but unpacking was requested.'.format(
            argspec, parameter_type))
  if unpack_required and not unpack_possible:
    raise TypeError(
        'The supplied function "{}" with argspec {} cannot accept a value of type '
        '{} as either a single argument or multiple positional and/or keyword '
        'arguments.'.format(fn.__name__, argspec, parameter_type))
  if not unpack_required and unpack_possible and should_unpack is None:
    # The supplied function could accept a value as either a single argument,
    # or as multiple positional and/or keyword arguments, and the caller did
    # not specify any preference, leaving ambiguity in how to handle the
    # mapping. We resolve the ambiguity by defaulting to capturing the entire
    # argument, as that's the behavior suggested as expected by the users.
    unpack = False

  if unpack is None:
    # Any ambiguity at this point has been resolved, so the following
    # condition holds and need only be verified in tests.
    assert unpack_required == unpack_possible, (unpack_required,
                                                unpack_possible)
    unpack = unpack_possible

  return unpack


def wrap_as_zero_or_one_arg_callable(fn, parameter_type=None, unpack=None):
  """Wraps around `fn` so it accepts up to one positional TFF-typed argument.

  This function helps to simplify dealing with functions and defuns that might
  have diverse and complex signatures, but that represent computations and as
  such, conceptually only accept a single parameter. The returned callable has
  a single positional parameter or no parameters. If it has one parameter, the
  parameter is expected to contain all arguments required by `fn` and matching
  the supplied parameter type signature bundled together into an anonymous
  tuple, if needed. The callable unpacks that structure, and passes all of
  its elements as positional or keyword-based arguments in the call to `fn`.

  Example usage:

    @tf.function
    def my_fn(x, y, z=10, name='bar', *p, **q):
      return x + y

    type_spec = (tf.int32, tf.int32)

    wrapped_fn = wrap_as_zero_or_one_arg_callable(my_fn, type_spec)

    arg = AnonymoutTuple([('x', 10), ('y', 20)])

    ... = wrapped_fn(arg)

  Args:
    fn: The underlying backend function or defun to invoke with the unpacked
      arguments.
    parameter_type: The TFF type of the parameter bundle to be accepted by the
      returned callable, if any, or None if there's no parameter.
    unpack: Whether to break the parameter down into constituent parts and feed
      them as arguments to `fn` (True), leave the parameter as is and pass it to
      `fn` as a single unit (False), or allow it to be inferred from the
      signature of `fn` (None). In the latter case (None), if any ambiguity
      arises, an exception is thrown. If the parameter_type is None, this value
      has no effect, and is simply ignored.

  Returns:
    The zero- or one-argument callable that invokes `fn` with the unbundled
    arguments, as described above.

  Raises:
    TypeError: if arguments to this call are of the wrong types, or if the
      supplied 'parameter_type' is not compatible with `fn`.
  """
  # TODO(b/113112885): Revisit whether the 3-way 'unpack' knob is sufficient
  # for our needs, or more options are needed.
  if unpack not in [True, False, None]:
    raise TypeError(
        'The unpack argument has an unexpected value {!r}.'.format(unpack))
  argspec = get_argspec(fn)
  parameter_type = computation_types.to_type(parameter_type)
  if parameter_type is None:
    if is_argspec_compatible_with_types(argspec):
      # Deliberate wrapping to isolate the caller from `fn`, e.g., to prevent
      # the caller from mistakenly specifying args that match fn's defaults.
      return lambda: fn()  # pylint: disable=unnecessary-lambda
    else:
      raise TypeError(
          'The argspec {} of the supplied function cannot be interpreted as a '
          'body of a no-parameter computation.'.format(argspec))
  else:
    if infer_unpack_needed(fn, parameter_type, unpack):
      arg_types, kwarg_types = unpack_args_from_tuple(parameter_type)

      def _unpack_and_call(fn, arg_types, kwarg_types, arg):
        """An interceptor function that unpacks 'arg' before calling `fn`.

        The function verifies the actual parameters before it forwards the
        call as a last-minute check.

        Args:
          fn: The function or defun to invoke.
          arg_types: The list of positional argument types (guaranteed to all be
            instances of computation_types.Types).
          kwarg_types: The dictionary of keyword argument types (guaranteed to
            all be instances of computation_types.Types).
          arg: The argument to unpack.

        Returns:
          The result of invoking `fn` on the unpacked arguments.

        Raises:
          TypeError: if types don't match.
        """
        py_typecheck.check_type(
            arg, (anonymous_tuple.AnonymousTuple, value_base.Value))
        args = []
        for idx, expected_type in enumerate(arg_types):
          element_value = arg[idx]
          actual_type = type_utils.infer_type(element_value)
          if not type_utils.is_assignable_from(expected_type, actual_type):
            raise TypeError(
                'Expected element at position {} to be of type {}, found {}.'
                .format(idx, expected_type, actual_type))
          if isinstance(element_value, anonymous_tuple.AnonymousTuple):
            element_value = type_utils.convert_to_py_container(
                element_value, expected_type)
          args.append(element_value)
        kwargs = {}
        for name, expected_type in six.iteritems(kwarg_types):
          element_value = getattr(arg, name)
          actual_type = type_utils.infer_type(element_value)
          if not type_utils.is_assignable_from(expected_type, actual_type):
            raise TypeError(
                'Expected element named {} to be of type {}, found {}.'.format(
                    name, expected_type, actual_type))
          if type_utils.is_anon_tuple_with_py_container(element_value,
                                                        expected_type):
            element_value = type_utils.convert_to_py_container(
                element_value, expected_type)
          kwargs[name] = element_value
        return fn(*args, **kwargs)

      # TODO(b/132888123): Consider other options to avoid possible bugs here.
      try:
        (fn, arg_types, kwarg_types)
      except NameError:
        raise AssertionError('Args to be bound must be in scope.')
      return lambda arg: _unpack_and_call(fn, arg_types, kwarg_types, arg)
    else:
      # An interceptor function that verifies the actual parameter before it
      # forwards the call as a last-minute check.
      def _call(fn, parameter_type, arg):
        arg_type = type_utils.infer_type(arg)
        if not type_utils.is_assignable_from(parameter_type, arg_type):
          raise TypeError('Expected an argument of type {}, found {}.'.format(
              parameter_type, arg_type))
        if type_utils.is_anon_tuple_with_py_container(arg, parameter_type):
          arg = type_utils.convert_to_py_container(arg, parameter_type)
        return fn(arg)

      # TODO(b/132888123): Consider other options to avoid possible bugs here.
      try:
        (fn, parameter_type)
      except NameError:
        raise AssertionError('Args to be bound must be in scope.')
      return lambda arg: _call(fn, parameter_type, arg)


class ConcreteFunction(computation_base.Computation):
  """A base class for concretely-typed (non-polymorphic) functions."""

  def __init__(self, type_signature, context_stack):
    """Constructs this concrete function with the give type signature.

    Args:
      type_signature: An instance of computation_types.FunctionType.
      context_stack: The context stack to use.

    Raises:
      TypeError: if the arguments are of the wrong computation_types.
    """
    py_typecheck.check_type(type_signature, computation_types.FunctionType)
    py_typecheck.check_type(context_stack, context_stack_base.ContextStack)
    self._type_signature = type_signature
    self._context_stack = context_stack

  @property
  def type_signature(self):
    return self._type_signature

  def __call__(self, *args, **kwargs):
    context = self._context_stack.current
    arg = pack_args(self._type_signature.parameter, args, kwargs, context)
    return context.invoke(self, arg)


class PolymorphicFunction(object):
  """A generic polymorphic function that accepts arguments of diverse types."""

  def __init__(self, concrete_function_factory):
    """Crates a polymorphic function with a given function factory.

    Args:
      concrete_function_factory: A callable that accepts a (non-None) TFF type
        as an argument, and returns a ConcreteFunction instance that's been
        created to accept a single positional argument of this TFF type (to be
        reused for future calls with parameters of a matching type).
    """
    self._concrete_function_factory = concrete_function_factory
    self._concrete_function_cache = {}

  def __call__(self, *args, **kwargs):
    """Invokes this polymorphic function with a given set of arguments.

    Args:
      *args: Positional args.
      **kwargs: Keyword args.

    Returns:
      The result of calling a concrete function, instantiated on demand based
      on the argument types (and cached for future calls).

    Raises:
      TypeError: if the concrete functions created by the factory are of the
        wrong computation_types.
    """
    # TODO(b/113112885): We may need to normalize individuals args, such that
    # the type is more predictable and uniform (e.g., if someone supplies an
    # unordered dictionary), possibly by converting dict-like and tuple-like
    # containters into anonymous tuples.
    packed_arg = pack_args_into_anonymous_tuple(args, kwargs)
    arg_type = type_utils.infer_type(packed_arg)
    key = repr(arg_type)
    concrete_fn = self._concrete_function_cache.get(key)
    if not concrete_fn:
      concrete_fn = self._concrete_function_factory(arg_type)
      py_typecheck.check_type(concrete_fn, ConcreteFunction,
                              'concrete function')
      if concrete_fn.type_signature.parameter != arg_type:
        raise TypeError(
            'Expected a concrete function that takes parameter {}, got one '
            'that takes {}.'.format(arg_type,
                                    concrete_fn.type_signature.parameter))
      self._concrete_function_cache[key] = concrete_fn
    return concrete_fn(packed_arg)
