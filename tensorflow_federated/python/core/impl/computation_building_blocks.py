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
"""Classes representing various kinds of computations in a deserialized form."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import zlib

import six

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils


def _check_computation_oneof(computation_proto, expected_computation_oneof):
  py_typecheck.check_type(computation_proto, pb.Computation)
  computation_oneof = computation_proto.WhichOneof('computation')
  if computation_oneof != expected_computation_oneof:
    raise TypeError('Expected a {} computation, found {}.'.format(
        expected_computation_oneof, computation_oneof))


@six.add_metaclass(abc.ABCMeta)
class ComputationBuildingBlock(typed_object.TypedObject):
  """The abstract base class for abstractions in the TFF's internal language.

  Instances of this class correspond roughly one-to-one to the abstractions
  defined in the `Computation` message in TFF's `computation.proto`, and are
  intended primarily for the ease of manipulating the abstract syntax trees
  (AST) of federated computations as they are transformed by TFF's compiler
  pipeline to mold into the needs of a particular execution backend. The only
  abstraction that does not have a dedicated Python equivalent is a section
  of TensorFlow code (it's represented by `tff.framework.CompiledComputation`).
  """

  _deserializer_dict = None  # Defined at the end of this file.

  @classmethod
  def from_proto(cls, computation_proto):
    """Returns an instance of a derived class based on 'computation_proto'.

    Args:
      computation_proto: An instance of pb.Computation.

    Returns:
      An instance of a class that implements 'ComputationBuildingBlock' and
      that contains the deserialized logic from in 'computation_proto'.

    Raises:
      NotImplementedError: if computation_proto contains a kind of computation
        for which deserialization has not been implemented yet.
      ValueError: if deserialization failed due to the argument being invalid.
    """
    py_typecheck.check_type(computation_proto, pb.Computation)
    computation_oneof = computation_proto.WhichOneof('computation')
    deserializer = cls._deserializer_dict.get(computation_oneof)
    if deserializer is not None:
      deserialized = deserializer(computation_proto)
      type_spec = type_serialization.deserialize_type(computation_proto.type)
      if not type_utils.are_equivalent_types(deserialized.type_signature,
                                             type_spec):
        raise ValueError(
            'The type {} derived from the computation structure does not '
            'match the type {} declared in its signature'.format(
                str(deserialized.type_signature), str(type_spec)))
      return deserialized
    else:
      raise NotImplementedError(
          'Deserialization for computations of type {} has not been '
          'implemented yet.'.format(computation_oneof))

  def __init__(self, type_spec):
    """Constructs a computation building block with the given TFF type.

    Args:
      type_spec: An instance of types.Type, or something convertible to it via
        types.to_type().
    """
    type_signature = computation_types.to_type(type_spec)
    type_utils.check_well_formed(type_signature)
    self._type_signature = type_signature

  @property
  def type_signature(self):
    return self._type_signature

  @abc.abstractproperty
  def proto(self):
    """Returns a serialized form of this object as a pb.Computation instance."""
    raise NotImplementedError

  @abc.abstractproperty
  def tff_repr(self):
    """Returns the representation of the instance using TFF syntax."""
    pass

  # TODO(b/113112885): Add memoization after identifying a suitable externally
  # available standard library that works in Python 2/3.

  @abc.abstractmethod
  def __repr__(self):
    raise NotImplementedError

  def __str__(self):
    return self.tff_repr


class Reference(ComputationBuildingBlock):
  """A reference to a name defined earlier in TFF's internal language.

  Names are defined by lambda expressions (which have formal named parameters),
  and block structures (which can have one or more locals). The reference
  construct is used to refer to those parameters or locals by a string name.
  The usual hiding rules apply. A reference binds to the closest definition of
  the given name in the most deeply nested surrounding lambda or block.

  A concise notation for a reference to name `foo` is `foo`. For example, in
  a lambda expression `(x -> f(x))` there are two references, one to `x` that
  is defined as the formal parameter of the lambda epxression, and one to `f`
  that must have been defined somewhere in the surrounding context.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'reference')
    return cls(
        str(computation_proto.reference.name),
        type_serialization.deserialize_type(computation_proto.type))

  def __init__(self, name, type_spec, context=None):
    """Creates a reference to 'name' of type 'type_spec' in context 'context'.

    Args:
      name: The name of the referenced entity.
      type_spec: The type spec of the referenced entity.
      context: The optional context in which the referenced entity is defined.
        This class does not prescribe what Python type the 'context' needs to be
        and merely exposes it as a property (see below). The only requirement is
        that the context implements str() and repr().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(name, six.string_types)
    super(Reference, self).__init__(type_spec)
    self._name = name
    self._context = context

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        reference=pb.Reference(name=self._name))

  @property
  def name(self):
    return self._name

  @property
  def context(self):
    return self._context

  @property
  def tff_repr(self):
    if self._context is not None:
      return '{}@{}'.format(self._name, str(self._context))
    else:
      return self._name

  def __repr__(self):
    return 'Reference(\'{}\', {}{})'.format(
        self._name, repr(self.type_signature),
        ', {}'.format(repr(self._context)) if self._context is not None else '')


class Selection(ComputationBuildingBlock):
  """A selection by name or index from a tuple-typed value in TFF's language.

  The concise syntax for selections is `foo.bar` (selecting a named `bar` from
  the value of expression `foo`), and `foo[n]` (selecting element at index `n`
  from the value of `foo`).
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'selection')
    selection = ComputationBuildingBlock.from_proto(
        computation_proto.selection.source)
    selection_oneof = computation_proto.selection.WhichOneof('selection')
    if selection_oneof == 'name':
      return cls(selection, name=str(computation_proto.selection.name))
    elif selection_oneof == 'index':
      return cls(selection, index=computation_proto.selection.index)
    else:
      raise ValueError('Unknown selection type \'{}\' in {}.'.format(
          selection_oneof, str(computation_proto)))

  def __init__(self, source, name=None, index=None):
    """A selection from 'source' by a string or numeric 'name_or_index'.

    Exactly one of 'name' or 'index' must be specified (not None).

    Args:
      source: The source value to select from (an instance of
        ComputationBuildingBlock).
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
    py_typecheck.check_type(source, ComputationBuildingBlock)
    self._source = source
    source_type = self._source.type_signature
    if not isinstance(source_type, computation_types.NamedTupleType):
      raise TypeError(
          'Expected the source of selection to be a TFF named tuple, '
          'instead found it to be of type {}.'.format(str(source_type)))
    if name is not None:
      py_typecheck.check_type(name, six.string_types)
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
      elements = anonymous_tuple.to_elements(source_type)
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
  def proto(self):
    if self._name is not None:
      selection = pb.Selection(source=self._source.proto, name=self._name)
    else:
      selection = pb.Selection(source=self._source.proto, index=self._index)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        selection=selection)

  @property
  def source(self):
    return self._source

  @property
  def name(self):
    return self._name

  @property
  def index(self):
    return self._index

  @property
  def tff_repr(self):
    if self._name is not None:
      return '{}.{}'.format(str(self._source), self._name)
    else:
      return '{}[{}]'.format(str(self._source), self._index)

  def __repr__(self):
    if self._name is not None:
      return 'Selection({}, name={})'.format(
          repr(self._source), '\'{}\''.format(self._name))
    else:
      return 'Selection({}, index={})'.format(repr(self._source), self._index)


class Tuple(ComputationBuildingBlock, anonymous_tuple.AnonymousTuple):
  """A tuple with named or unnamed elements in TFF's internal language.

  The concise notation for tuples is `<name_1=value_1, ...., name_n=value_n>`
  for tuples with named elements, `<value_1, ..., value_n>` for tuples with
  unnamed elements, or a mixture of these for tuples with ome named and some
  unnamed elements, where `name_k` are the names, and `value_k` are the value
  expressions.

  For example, a lambda expression that applies `fn` to elements of 2-tuples
  pointwise could be represented as `(arg -> <fn(arg[0]),fn(arg[1])>)`.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'tuple')
    return cls([(str(e.name) if e.name else None,
                 ComputationBuildingBlock.from_proto(e.value))
                for e in computation_proto.tuple.element])

  def __init__(self, elements):
    """Constructs a tuple from the given list of elements.

    Args:
      elements: The elements of the tuple, supplied as a list of (name, value)
        pairs, where 'name' can be None in case the corresponding element is not
        named and only accessible via an index (see also AnonymousTuple).

    Raises:
      TypeError: if arguments are of the wrong types.
    """

    # Not using super() here and below, as the two base classes have different
    # signatures of their constructors, and the named tuple implementation
    # of selection interfaces should override that in the generic class 'Value'
    # to favor simplified expressions where simplification is possible.
    def _map_element(e):
      """Returns a named or unnamed element."""
      if isinstance(e, ComputationBuildingBlock):
        return (None, e)
      elif py_typecheck.is_name_value_pair(
          e, name_required=False, value_type=ComputationBuildingBlock):
        if e[0] is not None and not e[0]:
          raise ValueError('Unexpected tuple element with empty string name.')
        return (e[0], e[1])
      else:
        raise TypeError('Unexpected tuple element: {}.'.format(str(e)))

    elements = [_map_element(e) for e in elements]
    ComputationBuildingBlock.__init__(
        self,
        computation_types.NamedTupleType([
            ((e[0], e[1].type_signature) if e[0] else e[1].type_signature)
            for e in elements
        ]))
    anonymous_tuple.AnonymousTuple.__init__(self, elements)

  @property
  def proto(self):
    elements = []
    for k, v in anonymous_tuple.to_elements(self):
      if k is not None:
        element = pb.Tuple.Element(name=k, value=v.proto)
      else:
        element = pb.Tuple.Element(value=v.proto)
      elements.append(element)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        tuple=pb.Tuple(element=elements))

  @property
  def tff_repr(self):
    return anonymous_tuple.AnonymousTuple.__str__(self)

  def __repr__(self):
    return 'Tuple([{}])'.format(', '.join(
        '({}, {})'.format('\'{}\''.format(e[0]) if e[0] is not None else 'None',
                          repr(e[1]))
        for e in anonymous_tuple.to_elements(self)))


class Call(ComputationBuildingBlock):
  """A representation of a function invocation in TFF's internal language.

  The call construct takes an argument tuple with two elements, the first being
  the function to invoke (represented as a computation with a functional result
  type), and the second being the argument to feed to that function. Typically,
  the function is either a TFF instrinsic, or a lambda expression.

  The concise notation for calls is `foo(bar)`, where `foo` is the function,
  and `bar` is the argument.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'call')
    fn = ComputationBuildingBlock.from_proto(computation_proto.call.function)
    arg_proto = computation_proto.call.argument
    if arg_proto.WhichOneof('computation') is not None:
      arg = ComputationBuildingBlock.from_proto(arg_proto)
    else:
      arg = None
    return cls(fn, arg)

  def __init__(self, fn, arg=None):
    """Creates a call to 'fn' with argument 'arg'.

    Args:
      fn: A value of a functional type that represents the function to invoke.
      arg: The optional argument, present iff 'fn' expects one, of a type that
        matches the type of 'fn'.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(fn, ComputationBuildingBlock)
    if arg is not None:
      py_typecheck.check_type(arg, ComputationBuildingBlock)
    if not isinstance(fn.type_signature, computation_types.FunctionType):
      raise TypeError('Expected fn to be of a functional type, '
                      'but found that its type is {}.'.format(
                          str(fn.type_signature)))
    if fn.type_signature.parameter is not None:
      if arg is None:
        raise TypeError('The invoked function expects an argument of type {}, '
                        'but got None instead.'.format(
                            str(fn.type_signature.parameter)))
      if not type_utils.is_assignable_from(fn.type_signature.parameter,
                                           arg.type_signature):
        raise TypeError(
            'The parameter of the invoked function is expected to be of '
            'type {}, but the supplied argument is of an incompatible '
            'type {}.'.format(
                str(fn.type_signature.parameter), str(arg.type_signature)))
    elif arg is not None:
      raise TypeError(
          'The invoked function does not expect any parameters, but got '
          'an argument of type {}.'.format(py_typecheck.type_string(type(arg))))
    super(Call, self).__init__(fn.type_signature.result)
    # By now, this condition should hold, so we only double-check in debug mode.
    assert (arg is not None) == (fn.type_signature.parameter is not None)
    self._function = fn
    self._argument = arg

  @property
  def proto(self):
    if self._argument is not None:
      call = pb.Call(
          function=self._function.proto, argument=self._argument.proto)
    else:
      call = pb.Call(function=self._function.proto)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature), call=call)

  @property
  def function(self):
    return self._function

  @property
  def argument(self):
    return self._argument

  @property
  def tff_repr(self):
    if self._argument is not None:
      return '{}({})'.format(str(self._function), str(self._argument))
    else:
      return '{}()'.format(str(self._function))

  def __repr__(self):
    if self._argument is not None:
      return 'Call({}, {})'.format(repr(self._function), repr(self._argument))
    else:
      return 'Call({})'.format(repr(self._function))


class Lambda(ComputationBuildingBlock):
  """A representation of a lambda expression in TFF's internal language.

  A lambda expression consists of a string formal parameter name, and a result
  expression that can contain references by name to that formal parameter. A
  concise notation for lambdas is `(foo -> bar)`, where `foo` is the name of
  the formal parameter, and `bar` is the result expression.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'lambda')
    the_lambda = getattr(computation_proto, 'lambda')
    return cls(
        str(the_lambda.parameter_name),
        type_serialization.deserialize_type(
            computation_proto.type.function.parameter),
        ComputationBuildingBlock.from_proto(the_lambda.result))

  def __init__(self, parameter_name, parameter_type, result):
    """Creates a lambda expression.

    Args:
      parameter_name: The (string) name of the parameter accepted by the lambda.
        This name can be used by Reference() instances in the body of the lambda
        to refer to the parameter.
      parameter_type: The type of the parameter, an instance of types.Type or
        something convertible to it by types.to_type().
      result: The resulting value produced by the expression that forms the body
        of the lambda. Must be an instance of ComputationBuildingBlock.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(parameter_name, six.string_types)
    if parameter_type is None:
      raise TypeError('A lambda expression must have a valid parameter type.')
    parameter_type = computation_types.to_type(parameter_type)
    assert isinstance(parameter_type, computation_types.Type)
    py_typecheck.check_type(result, ComputationBuildingBlock)
    super(Lambda, self).__init__(
        computation_types.FunctionType(parameter_type, result.type_signature))
    self._parameter_name = parameter_name
    self._parameter_type = parameter_type
    self._result = result

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        **{
            'lambda':
                pb.Lambda(
                    parameter_name=self._parameter_name,
                    result=self._result.proto)
        })

  @property
  def parameter_name(self):
    return self._parameter_name

  @property
  def parameter_type(self):
    return self._parameter_type

  @property
  def result(self):
    return self._result

  @property
  def tff_repr(self):
    return '({} -> {})'.format(self._parameter_name, str(self._result))

  def __repr__(self):
    return ('Lambda(\'{}\', {}, {})'.format(self._parameter_name,
                                            repr(self._parameter_type),
                                            repr(self._result)))


class Block(ComputationBuildingBlock):
  """A representation of a block of code in TFF's internal language.

  A block is a syntactic structure that consists of a sequence of local name
  bindings followed by a result. The bindings are interpreted sequentially,
  with bindings later in the sequence in the scope of those listed earlier,
  and the result in the scope of the entire sequence. The usual hiding rules
  apply.

  An informal concise notation for blocks is the following, with `name_k`
  representing the names defined locally for the block, `value_k` the values
  associated with them, and `result` being the expression that reprsents the
  value of the block construct.

  ```
  let name_1=value_1, name_2=value_2, ..., name_n=value_n in result
  ```

  Blocks are technically a redundant abstraction, as they can be equally well
  represented by lambda expressions. A block of the form `let x=y in z` is
  roughly equivalent to `(x -> z)(y)`. Although redundant, blocks have a use
  as a way to reduce TFF computation ASTs to a simpler, less nested and more
  readable form, and are helpful in AST transformations as a mechanism that
  prevents possible naming conflicts.

  An example use of a block expression to flatten a nested structure below:

  ```
  z = federated_sum(federated_map(x, federated_broadcast(y)))
  ```

  An equivalent form in a more sequential notation using a block expression:
  ```
  let
    v1 = federated_broadcast(y),
    v2 = federated_map(x, v1)
  in
    federated_sum(v2)
  ```
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'block')
    return cls([(str(loc.name), ComputationBuildingBlock.from_proto(loc.value))
                for loc in computation_proto.block.local],
               ComputationBuildingBlock.from_proto(
                   computation_proto.block.result))

  def __init__(self, local_symbols, result):
    """Creates a block of TFF code.

    Args:
      local_symbols: The list of one or more local declarations, each of which
        is a 2-tuple (name, value), with 'name' being the string name of a local
        symbol being defined, and 'value' being the instance of
        ComputationBuildingBlock, the output of which will be locally bound to
        that name.
      result: An instance of ComputationBuildingBlock that computes the result.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    updated_locals = []
    for index, element in enumerate(local_symbols):
      if (not isinstance(element, tuple) or (len(element) != 2) or
          not isinstance(element[0], six.string_types)):
        raise TypeError(
            'Expected the locals to be a list of 2-element tuples with string '
            'name as their first element, but this is not the case for the '
            'local at position {} in the sequence: {}.'.format(
                index, str(element)))
      name = element[0]
      value = element[1]
      py_typecheck.check_type(value, ComputationBuildingBlock)
      updated_locals.append((name, value))
    py_typecheck.check_type(result, ComputationBuildingBlock)
    super(Block, self).__init__(result.type_signature)
    self._locals = updated_locals
    self._result = result

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        block=pb.Block(
            **{
                'local': [
                    pb.Block.Local(name=k, value=v.proto)
                    for k, v in self._locals
                ],
                'result': self._result.proto
            }))

  @property
  def locals(self):
    return list(self._locals)

  @property
  def result(self):
    return self._result

  @property
  def tff_repr(self):
    return ('(let {} in {})'.format(
        ','.join('{}={}'.format(k, str(v)) for k, v in self._locals),
        str(self._result)))

  def __repr__(self):
    return ('Block([{}], {})'.format(
        ', '.join('(\'{}\', {})'.format(k, repr(v)) for k, v in self._locals),
        repr(self._result)))


class Intrinsic(ComputationBuildingBlock):
  """A representation of an intrinsic in TFF's internal language.

  An instrinsic is a symbol known to the TFF's compiler pipeline, represended
  a a known URI. It generally appears in expressions with a concrete type,
  although all intrinsic are defined with template types. This class does not
  deal with parsing intrinsic URIs and verifying their types, it is only a
  container. Parsing and type analysis are a responsibility or the components
  that manipulate ASTs.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'intrinsic')
    return cls(computation_proto.intrinsic.uri,
               type_serialization.deserialize_type(computation_proto.type))

  def __init__(self, uri, type_spec):
    """Creates an intrinsic.

    Args:
      uri: The URI of the intrinsic.
      type_spec: Either the types.Type that represents the type of this
        intrinsic, or something convertible to it by types.to_type().

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(uri, six.string_types)
    if type_spec is None:
      raise TypeError(
          'Intrinsic {} cannot be created without a TFF type.'.format(uri))
    type_spec = computation_types.to_type(type_spec)
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(uri)
    if intrinsic_def:
      typecheck = type_utils.is_concrete_instance_of(
          type_spec, intrinsic_def.type_signature)
      if not typecheck:
        raise TypeError('Tried to construct an Intrinsic with bad type '
                        'signature; Intrinsic {} expects type signature {}, '
                        'and you tried to construct one of type {}.'.format(
                            uri, intrinsic_def.type_signature, type_spec))
    super(Intrinsic, self).__init__(type_spec)
    self._uri = uri

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        intrinsic=pb.Intrinsic(uri=self._uri))

  @property
  def uri(self):
    return self._uri

  @property
  def tff_repr(self):
    return self._uri

  def __repr__(self):
    return 'Intrinsic(\'{}\', {})'.format(self._uri, repr(self.type_signature))


class Data(ComputationBuildingBlock):
  """A representation of data (an input pipeline).

  This class does not deal with parsing data URIs and verifying correctness,
  it is only a container. Parsing and type analysis are a responsibility
  or a component external to this module.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'data')
    return cls(computation_proto.data.uri,
               type_serialization.deserialize_type(computation_proto.type))

  def __init__(self, uri, type_spec):
    """Creates a representation of data.

    Args:
      uri: The URI that characterizes the data.
      type_spec: Either the types.Type that represents the type of this data, or
        something convertible to it by types.to_type().

    Raises:
      TypeError: if the arguments are of the wrong types.
      ValueError: if the user tries to specify an empty URI.
    """
    py_typecheck.check_type(uri, six.string_types)
    if not uri:
      raise ValueError('Empty string cannot be passed as URI to Data.')
    if type_spec is None:
      raise TypeError(
          'Intrinsic {} cannot be created without a TFF type.'.format(uri))
    type_spec = computation_types.to_type(type_spec)
    super(Data, self).__init__(type_spec)
    self._uri = uri

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        data=pb.Data(uri=self._uri))

  @property
  def uri(self):
    return self._uri

  @property
  def tff_repr(self):
    return self._uri

  def __repr__(self):
    return 'Data(\'{}\', {})'.format(self._uri, repr(self.type_signature))


class CompiledComputation(ComputationBuildingBlock):
  """A representation of a fully constructed and serialized computation.

  A compile comutation is one that has not been parsed into constituents, and
  is simply represented as an embedded `Computation` protocol buffer. Whereas
  technically, any computation can be represented and passed around this way,
  this structure is generally only used to represent TensorFlow sections, for
  which otherwise there isn't any dedicated structure.
  """

  def __init__(self, proto, name=None):
    """Creates a representation of a fully constructed computation.

    Args:
      proto: An instance of pb.Computation with the computation logic.
      name: An optional string name to associate with this computation, used
        only for debugging purposes. If the name is not specified (None), it is
        autogenerated as a hexadecimal string from the hash of the proto.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(proto, pb.Computation)
    if name is not None:
      py_typecheck.check_type(name, six.string_types)
    super(CompiledComputation,
          self).__init__(type_serialization.deserialize_type(proto.type))
    self._proto = proto
    if name is not None:
      self._name = name
    else:
      self._name = '{:x}'.format(
          zlib.adler32(six.b(repr(self._proto))) & 0xFFFFFFFF)

  @property
  def proto(self):
    return self._proto

  @property
  def tff_repr(self):
    return 'comp#{}'.format(self._name)

  def __repr__(self):
    return 'CompiledComputation({}, {})'.format(self._name,
                                                repr(self.type_signature))


class Placement(ComputationBuildingBlock):
  """A representation of a placement literal in TFF's internal language.

  Currently this can only be `tff.SERVER` or `tff.CLIENTS`.
  """

  @classmethod
  def from_proto(cls, computation_proto):
    _check_computation_oneof(computation_proto, 'placement')
    py_typecheck.check_type(
        type_serialization.deserialize_type(computation_proto.type),
        computation_types.PlacementType)
    return cls(
        placement_literals.uri_to_placement_literal(
            str(computation_proto.placement.uri)))

  def __init__(self, literal):
    """Constructs a new placement instance for the given placement literal.

    Args:
      literal: The placement literal.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(literal, placement_literals.PlacementLiteral)
    super(Placement, self).__init__(computation_types.PlacementType())
    self._literal = literal

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        placement=pb.Placement(uri=self._literal.uri))

  @property
  def uri(self):
    return self._literal.uri

  @property
  def tff_repr(self):
    return str(self._literal)

  def __repr__(self):
    return 'Placement(\'{}\')'.format(self.uri)


# pylint: disable=protected-access
ComputationBuildingBlock._deserializer_dict = {
    'reference': Reference.from_proto,
    'selection': Selection.from_proto,
    'tuple': Tuple.from_proto,
    'call': Call.from_proto,
    'lambda': Lambda.from_proto,
    'block': Block.from_proto,
    'intrinsic': Intrinsic.from_proto,
    'data': Data.from_proto,
    'placement': Placement.from_proto,
    'tensorflow': CompiledComputation,
}
# pylint: enable=protected-access
