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
"""Implementations of the abstract interface Value in api/value_base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import zlib

# Dependency imports

from six import string_types

from tensorflow_federated.proto.v0 import computation_pb2 as pb

from tensorflow_federated.python.common_libs import py_typecheck

from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import anonymous_tuple
from tensorflow_federated.python.core.impl import func_utils
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils


class Value(value_base.Value):
  """A generic base class for values that appear in TFF computations."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, type_spec):
    """Constructs a value of the given type.

    Args:
      type_spec: An instance of types.Type, or something convertible to it via
        types.to_type().
    """
    self._type_signature = types.to_type(type_spec)

  @property
  def type_signature(self):
    return self._type_signature

  def __dir__(self):
    if not isinstance(self._type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator dir() is only suppored for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._type_signature)))
    else:
      # Not pre-creating or memoizing the list, as we do not expect this to be
      # a common enough operation to warrant doing so.
      return [e[0] for e in self._type_signature.elements if e[0]]

  def __getattr__(self, name):
    py_typecheck.check_type(name, string_types)
    try:
      return Selection(self, name=name)
    except ValueError as err:
      raise AttributeError(str(err))

  def __len__(self):
    if not isinstance(self._type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator len() is only supported for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._type_signature)))
    else:
      return len(self._type_signature.elements)

  def __getitem__(self, key):
    py_typecheck.check_type(key, int)
    try:
      return Selection(self, index=key)
    except ValueError as err:
      raise KeyError(str(err))

  def __iter__(self):
    if not isinstance(self._type_signature, types.NamedTupleType):
      raise TypeError(
          'Operator iter() is only supported for named tuples, but the object '
          'on which it has been invoked is of type {}.'.format(
              str(self._type_signature)))
    else:
      for index in range(len(self._type_signature.elements)):
        yield Selection(self, index=index)

  def __call__(self, *args, **kwargs):
    if not isinstance(self._type_signature, types.FunctionType):
      raise SyntaxError(
          'Function-like invocation is only supported for values of '
          'functional types, but the value being invoked is of type '
          '{} that does not support invocation.'.format(
              str(self._type_signature)))
    else:
      arg = func_utils.pack_args(self._type_signature.parameter, args, kwargs)
      return Call(self, arg)


class Reference(Value):
  """A reference to a name defined earlier, e.g., in a Lambda."""

  def __init__(self, name, type_spec, context=None):
    """Creates a reference to 'name' of type 'type_spec' in context 'context'.

    Args:
      name: The name of the referenced entity.
      type_spec: The type spec of the referenced entity.
      context: The optional context in which the referenced entity is defined.
        This class does not prescribe what Python type the 'context' needs to be
        and merely exposes it as a property (see below). The only requirement
        is that the context implements str() and repr().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(name, string_types)
    super(Reference, self).__init__(type_spec)
    self._name = name
    self._context = context

  @property
  def name(self):
    return self._name

  @property
  def context(self):
    return self._context

  def __repr__(self):
    return 'Reference(\'{}\', {}{})'.format(
        self._name, repr(self.type_signature),
        ', {}'.format(repr(self._context)) if self._context else '')

  def __str__(self):
    return ('{}@{}'.format(self._name, str(self._context)) if self._context
            else self._name)


class Selection(Value):
  """A selection by name or index from another tuple-typed value."""

  def __init__(self, source, name=None, index=None):
    """A selection from 'source' by a string or numeric 'name_or_index'.

    Exactly one of 'name' or 'index' must be specified (not None).

    Args:
      source: The source value to select from, either an instance of Value or
        something convertible to it by to_value().
      name: A string name of the element to be selected.
      index: A numeric index of the element to be selected.

    Raises:
      TypeError: if arguments are of the wrong types.
      ValueError: if the name is empty or index is negative, or the name/index
        is not compatible with the type signature of the source, or neither or
        both are defined (not None).
    """
    if name is None and index is None:
      raise ValueError(
          'Must define either a name or index, and neither was specified.')
    if name is not None and index is not None:
      raise ValueError(
          'Cannot simultaneously specify a name and an index, choose one.')
    if source is None:
      raise TypeError('The source of selection cannot be None.')
    self._source = to_value(source)
    source_type = self._source.type_signature
    if not isinstance(source_type, types.NamedTupleType):
      raise TypeError(
          'Expected the source of selection to be a TFF named tuple, '
          'instead found it to be of type {}.'.format(str(source_type)))
    if name is not None:
      py_typecheck.check_type(name, string_types)
      if not name:
        raise ValueError('The name of the selected element cannot be empty.')
      else:
        # Normalize, in case we are dealing with a Unicode type or some such.
        name = str(name)
        super(Selection, self).__init__(
            type_utils.get_named_tuple_element_type(source_type, name))
        self._name = name
        self._index = None
    else:
      # Index must have been specified, since name is None.
      py_typecheck.check_type(index, int)
      elements = source_type.elements
      if index >= 0 and index < len(elements):
        super(Selection, self).__init__(elements[index][1])
        self._name = None
        self._index = index
      else:
        raise ValueError(
            'The index of the selected element {} does not fit into the '
            'valid range 0..{} determined by the source type '
            'signature.'.format(index, str(len(elements) - 1)))

  @property
  def name(self):
    return self._name

  @property
  def index(self):
    return self._index

  def __repr__(self):
    if self._name is not None:
      return 'Selection({}, name={})'.format(
          repr(self._source), '\'{}\''.format(self._name))
    else:
      return 'Selection({}, index={})'.format(repr(self._source), self._index)

  def __str__(self):
    return (
        '{}.{}'.format(str(self._source), self._name) if self._name is not None
        else '{}[{}]'.format(str(self._source), self._index))


class Tuple(Value, anonymous_tuple.AnonymousTuple):
  """A tuple with one or more values as named or unnamed elements."""

  def __init__(self, elements):
    """Constructs a tuple from the given list of elements.

    Args:
      elements: The elements of the tuple, supplied as a list of (name, value)
        pairs, where 'name' can be None in case the corresponding element is
        not named and only accessible via an index (see also AnonymousTuple).

    Raises:
      TypeError: if arguments are of the wrong types.
    """
    # Not using super() here and below, as the two base classes have different
    # signatures of their constructors, and the named tuple implementation
    # of selection interfaces should override that in the generic class 'Value'
    # to favor simplified expressions where simplification is possible.
    def _map_element(e):
      if (not isinstance(e, Value) and
          isinstance(e, tuple) and
          (len(e) == 2) and
          (e[0] is None or isinstance(e[0], string_types))):
        return (e[0], to_value(e[1]))
      else:
        return (None, to_value(e))
    elements = [_map_element(e) for e in elements]
    Value.__init__(self, types.NamedTupleType([
        ((e[0], e[1].type_signature) if e[0] else e[1].type_signature)
        for e in elements]))
    anonymous_tuple.AnonymousTuple.__init__(self, elements)

  def __repr__(self):
    return 'Tuple([{}])'.format(', '.join(
        '({}, {})'.format(
            '\'{}\''.format(e[0]) if e[0] is not None else 'None', repr(e[1]))
        for e in anonymous_tuple.to_elements(self)))

  def __str__(self):
    return anonymous_tuple.AnonymousTuple.__str__(self)

  def __dir__(self):
    return anonymous_tuple.AnonymousTuple.__dir__(self)

  def __getattr__(self, name):
    return anonymous_tuple.AnonymousTuple.__getattr__(self, name)

  def __len__(self):
    return anonymous_tuple.AnonymousTuple.__len__(self)

  def __getitem__(self, index):
    return anonymous_tuple.AnonymousTuple.__getitem__(self, index)

  def __iter__(self):
    return anonymous_tuple.AnonymousTuple.__iter__(self)


class Call(Value):
  """A representation of a TFF function call."""

  def __init__(self, func, arg=None):
    """Creates a call to 'func' with argument 'arg'.

    Args:
      func: A value of a functional type that represents the function to invoke.
      arg: The optional argument, present iff 'func' expects one, of a type that
        matches the type of 'func'.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(func, value_base.Value)
    if not isinstance(func.type_signature, types.FunctionType):
      raise TypeError(
          'Expected func to be of a functional type, '
          'but found that its type is {}.'.format(str(func.type_signature)))
    if func.type_signature.parameter is not None:
      if arg is None:
        raise TypeError(
            'The invoked function expects an argument of type {}, '
            'but got None instead.'.format(str(func.type_signature.parameter)))
      arg = to_value(arg)
      if not func.type_signature.parameter.is_assignable_from(
          arg.type_signature):
        raise TypeError(
            'The parameter of the invoked function is expected to be of '
            'type {}, but the supplied argument is of an incompatible '
            'type {}.'.format(
                str(func.type_signature.parameter), str(arg.type_signature)))
    elif arg is not None:
      raise TypeError(
          'The invoked function does not expect any parameters, but got '
          'an argument of type {}.'.format(py_typecheck.type_string(type(arg))))
    super(Call, self).__init__(func.type_signature.result)
    # By now, this condition should hold, so we only double-check in debug mode.
    assert (arg is not None) == (func.type_signature.parameter is not None)
    self._function = func
    self._argument = arg

  @property
  def function(self):
    return self._function

  @property
  def argument(self):
    return self._argument

  def __repr__(self):
    return ('Call({}, {})'.format(repr(self._function), repr(self._argument))
            if self._argument is not None
            else 'Call({})'.format(repr(self._function)))

  def __str__(self):
    return ('{}({})'.format(str(self._function), str(self._argument))
            if self._argument is not None
            else '{}()'.format(str(self._function)))


class Lambda(Value):
  """A representation of a TFF lambda expression."""

  def __init__(self, parameter_name, parameter_type, result):
    """Creates a lambda expression.

    Args:
      parameter_name: The (string) name of the parameter accepted by the lambda.
        This name can be used by Reference() instances in the body of the lambda
        to refer to the parameter.
      parameter_type: The type of the parameter, an instance of types.Type or
        something convertible to it by types.to_type().
      result: The resulting value produced by the expression that forms the body
        of the lambda. Must be an instance of Value, or something convertible to
        it by to_value() below.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(parameter_name, string_types)
    if parameter_type is None:
      raise TypeError('A lambda expression must have a valid parameter type.')
    parameter_type = types.to_type(parameter_type)
    assert isinstance(parameter_type, types.Type)
    result = to_value(result)
    super(Lambda, self).__init__(
        types.FunctionType(parameter_type, result.type_signature))
    self._parameter_name = parameter_name
    self._parameter_type = parameter_type
    self._result = result

  @property
  def parameter_name(self):
    return self._parameter_name

  @property
  def parameter_type(self):
    return self._parameter_type

  @property
  def result(self):
    return self._result

  def __repr__(self):
    return ('Lambda(\'{}\', {}, {})'.format(
        self._parameter_name, repr(self._parameter_type), repr(self._result)))

  def __str__(self):
    return '({} -> {})'.format(self._parameter_name, str(self._result))


class Block(Value):
  """A representation of a block of TFF code."""

  def __init__(self, local_symbols, result):
    """Creates a block of TFF code.

    Args:
      local_symbols: The list of one or more local declarations, each of which
        is a 2-tuple (name, value), with 'name' being the string name of a
        local symbol being defined, and 'value' being the instance of Value or
        an entity convertible to Value that will be locally bound to that name.
      result: The result value, an instance of Value or something convertible
        to it by to_value() below.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    updated_locals = []
    for index, element in enumerate(local_symbols):
      if (not isinstance(element, tuple) or
          (len(element) != 2) or
          not isinstance(element[0], string_types)):
        raise TypeError(
            'Expected the locals to be a list of 2-element tuples with string '
            'name as their first element, but this is not the case for the '
            'local at position {} in the sequence: {}.'.format(
                index, str(element)))
      name = element[0]
      try:
        value = to_value(element[1])
      except TypeError as err:
        raise TypeError(
            'Expected local {} to be an instance of {} or something '
            'convertible to it, but found {}: {}.'.format(
                name,
                value_base.Value.__name__,
                py_typecheck.type_string(type(element[1])),
                str(err)))
      updated_locals.append((name, value))
    result = to_value(result)
    super(Block, self).__init__(result.type_signature)
    self._locals = updated_locals
    self._result = result

  @property
  def locals(self):
    return list(self._locals)

  @property
  def result(self):
    return self._result

  def __repr__(self):
    return ('Block([{}], {})'.format(
        ', '.join('(\'{}\', {})'.format(k, repr(v)) for k, v in self._locals),
        repr(self._result)))

  def __str__(self):
    return ('(let {} in {})'.format(
        ','.join('{}={}'.format(k, str(v)) for k, v in self._locals),
        str(self._result)))


class Intrinsic(Value):
  """A representation of an intrinsic.

  This class does not deal with parsing intrinsic URIs and verifying their
  types, it is only a container. Parsing and type analysis are a responsibility
  or a component external to this module.
  """

  def __init__(self, uri, type_spec):
    """Creates an intrinsic.

    Args:
      uri: The URI of the intrinsic.
      type_spec: Either the types.Type that represents the type of this
        intrinsic, or something convertible to it by types.to_type().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(uri, string_types)
    if type_spec is None:
      raise TypeError(
          'Intrinsic {} cannot be created without a TFF type.'.format(uri))
    type_spec = types.to_type(type_spec)
    super(Intrinsic, self).__init__(type_spec)
    self._uri = uri

  @property
  def uri(self):
    return self._uri

  def __repr__(self):
    return 'Intrinsic(\'{}\', {})'.format(self._uri, repr(self.type_signature))

  def __str__(self):
    return self._uri


class Data(Value):
  """A representation of data (an input pipeline).

  This class does not deal with parsing data URIs and verifying correctness,
  it is only a container. Parsing and type analysis are a responsibility
  or a component external to this module.
  """

  def __init__(self, uri, type_spec):
    """Creates a representation of data.

    Args:
      uri: The URI that characterizes the data.
      type_spec: Either the types.Type that represents the type of this data,
        or something convertible to it by types.to_type().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(uri, string_types)
    if type_spec is None:
      raise TypeError(
          'Intrinsic {} cannot be created without a TFF type.'.format(uri))
    type_spec = types.to_type(type_spec)
    super(Data, self).__init__(type_spec)
    self._uri = uri

  @property
  def uri(self):
    return self._uri

  def __repr__(self):
    return 'Data(\'{}\', {})'.format(self._uri, repr(self.type_signature))

  def __str__(self):
    return self._uri


class Computation(Value):
  """A representation of a fully constructed and serialized computation."""

  def __init__(self, proto, name=None):
    """Creates a representation of a fully constructed computation.

    Args:
      proto: An instance of pb.Computation with the computation logic.
      name: An optional string name to associate with this computation, used
        only for debugging purposes. If the name is not specified (None), it
        is autogenerated as a hexadecimal string from the hash of the proto.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(proto, pb.Computation)
    if name is not None:
      py_typecheck.check_type(name, string_types)
    super(Computation, self).__init__(
        type_serialization.deserialize_type(proto.type))
    self._proto = proto
    self._name = name if name is not None else (
        '{:x}'.format(zlib.adler32(repr(self._proto)) & 0xFFFFFFFF))

  @property
  def proto(self):
    return self._proto

  def __repr__(self):
    return 'Computation({}, {})'.format(self._name, repr(self.type_signature))

  def __str__(self):
    return 'comp({})'.format(self._name)


def to_value(arg):
  """Converts the argument into an instance of Value.

  Args:
    arg: Either an instance of Value, or an argument convertible to Value.
      The argument must not be None. The types of non-Value arguments that are
      currently convertible to Value include the following:

      * Lists, tuples, anonymous tuples, named tuples, and dictionaries, all of
        which are converted into instances of Tuple.

  Returns:
    An instance of Value corresponding to the given 'arg'.

  Raises:
    TypeError: if 'arg' is of an unsupported type.
  """
  if isinstance(arg, Value):
    return arg
  elif isinstance(arg, anonymous_tuple.AnonymousTuple):
    return Tuple(anonymous_tuple.to_elements(arg))
  elif '_asdict' in vars(type(arg)):
    return to_value(arg._asdict())
  elif isinstance(arg, dict):
    return Tuple(list(arg.iteritems()))
  elif isinstance(arg, (tuple, list)):
    return Tuple(list(arg))
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a TFF value.'.format(
            py_typecheck.type_string(type(arg))))
