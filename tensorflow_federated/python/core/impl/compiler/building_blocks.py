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
"""A library of classes representing computations in a deserialized form."""

import abc
import enum
from typing import Any, Iterable, List, Optional, Tuple, Type
import zlib

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_serialization


def _check_computation_oneof(
    computation_proto: pb.Computation,
    expected_computation_oneof: Optional[str],
):
  """Checks that `computation_proto` is a oneof of the expected variant."""
  computation_oneof = computation_proto.WhichOneof('computation')
  if computation_oneof != expected_computation_oneof:
    raise TypeError('Expected a {} computation, found {}.'.format(
        expected_computation_oneof, computation_oneof))


class UnexpectedBlockError(TypeError):

  def __init__(self, expected: Type['ComputationBuildingBlock'],
               actual: 'ComputationBuildingBlock'):
    message = f'Expected block of kind {expected}, found block {actual}'
    super().__init__(message)
    self.actual = actual
    self.expected = expected


class ComputationBuildingBlock(typed_object.TypedObject, metaclass=abc.ABCMeta):
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
  def from_proto(
      cls: Type['ComputationBuildingBlock'],
      computation_proto: pb.Computation,
  ) -> 'ComputationBuildingBlock':
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
      if not deserialized.type_signature.is_equivalent_to(type_spec):
        raise ValueError(
            'The type {} derived from the computation structure does not '
            'match the type {} declared in its signature'.format(
                deserialized.type_signature, type_spec))
      return deserialized
    else:
      raise NotImplementedError(
          'Deserialization for computations of type {} has not been '
          'implemented yet.'.format(computation_oneof))
    return deserializer(computation_proto)

  def __init__(self, type_spec):
    """Constructs a computation building block with the given TFF type.

    Args:
      type_spec: An instance of types.Type, or something convertible to it via
        types.to_type().
    """
    type_signature = computation_types.to_type(type_spec)
    self._type_signature = type_signature

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature

  def compact_representation(self):
    """Returns the compact string representation of this building block."""
    return _string_representation(self, formatted=False)

  def formatted_representation(self):
    """Returns the formatted string representation of this building block."""
    return _string_representation(self, formatted=True)

  def structural_representation(self):
    """Returns the structural string representation of this building block."""
    return _structural_representation(self)

  def check_reference(self):
    """Check that this is a 'Reference'."""
    if not self.is_reference():
      UnexpectedBlockError(Reference, self)

  def is_reference(self):
    """Returns whether or not this block is a `Reference`."""
    return False

  def check_selection(self):
    """Check that this is a 'Selection'."""
    if not self.is_selection():
      UnexpectedBlockError(Selection, self)

  def is_selection(self):
    """Returns whether or not this block is a `Selection`."""
    return False

  def check_struct(self):
    """Check that this is a `Struct`."""
    if not self.is_struct():
      UnexpectedBlockError(Struct, self)

  def is_struct(self):
    """Returns whether or not this block is a `Struct`."""
    return False

  def check_call(self):
    """Check that this is a 'Call'."""
    if not self.is_call():
      UnexpectedBlockError(Call, self)

  def is_call(self):
    """Returns whether or not this block is a `Call`."""
    return False

  def check_lambda(self):
    """Check that this is a 'Lambda'."""
    if not self.is_lambda():
      UnexpectedBlockError(Lambda, self)

  def is_lambda(self):
    """Returns whether or not this block is a `Lambda`."""
    return False

  def check_block(self):
    """Check that this is a 'Block'."""
    if not self.is_block():
      UnexpectedBlockError(Block, self)

  def is_block(self):
    """Returns whether or not this block is a `Block`."""
    return False

  def check_intrinsic(self):
    """Check that this is an 'Intrinsic'."""
    if not self.is_intrinsic():
      UnexpectedBlockError(Intrinsic, self)

  def is_intrinsic(self):
    """Returns whether or not this block is an `Intrinsic`."""
    return False

  def check_data(self):
    """Check that this is a 'Data'."""
    if not self.is_data():
      UnexpectedBlockError(Data, self)

  def is_data(self):
    """Returns whether or not this block is a `Data`."""
    return False

  def check_compiled_computation(self):
    """Check that this is a 'CompiledComputation'."""
    if not self.is_compiled_computation():
      UnexpectedBlockError(CompiledComputation, self)

  def is_compiled_computation(self):
    """Returns whether or not this block is a `CompiledComputation`."""
    return False

  def check_placement(self):
    """Check that this is a 'Placement'."""
    if not self.is_placement():
      UnexpectedBlockError(Placement, self)

  def is_placement(self):
    """Returns whether or not this block is a `Placement`."""
    return False

  @abc.abstractproperty
  def proto(self):
    """Returns a serialized form of this object as a pb.Computation instance."""
    raise NotImplementedError

  # TODO(b/113112885): Add memoization after identifying a suitable externally
  # available standard library that works in Python 2/3.

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this computation building block."""
    raise NotImplementedError

  def __str__(self):
    """Returns a concise representation of this computation building block."""
    return self.compact_representation()


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
  def from_proto(
      cls: Type['Reference'],
      computation_proto: pb.Computation,
  ) -> 'Reference':
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
    py_typecheck.check_type(name, str)
    super().__init__(type_spec)
    self._name = name
    self._context = context

  @property
  def proto(self):
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        reference=pb.Reference(name=self._name))

  def is_reference(self):
    return True

  @property
  def name(self):
    return self._name

  @property
  def context(self):
    return self._context

  def __repr__(self):
    return 'Reference(\'{}\', {!r}{})'.format(
        self._name, self.type_signature,
        ', {!r}'.format(self._context) if self._context is not None else '')


class Selection(ComputationBuildingBlock):
  """A selection by name or index from a struct-typed value in TFF's language.

  The concise syntax for selections is `foo.bar` (selecting a named `bar` from
  the value of expression `foo`), and `foo[n]` (selecting element at index `n`
  from the value of `foo`).
  """

  @classmethod
  def from_proto(
      cls: Type['Selection'],
      computation_proto: pb.Computation,
  ) -> 'Selection':
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
          selection_oneof, computation_proto))

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
    py_typecheck.check_type(source, ComputationBuildingBlock)
    if name is None and index is None:
      raise ValueError(
          'Must define either a name or index, and neither was specified.')
    if name is not None and index is not None:
      raise ValueError(
          'Cannot simultaneously specify a name and an index, choose one.')
    source_type = source.type_signature
    if not source_type.is_struct():
      raise TypeError('Expected the source of selection to be a TFF struct, '
                      'instead found it to be of type {}.'.format(source_type))
    if name is not None:
      py_typecheck.check_type(name, str)
      if not name:
        raise ValueError('The name of the selected element cannot be empty.')
      # Normalize, in case we are dealing with a Unicode type or some such.
      name = str(name)
      if not structure.has_field(source_type, name):
        raise ValueError(
            'The name \'{}\' does not correspond to any of the names in the '
            'struct type: {}.'.format(name, structure.name_list(source_type)))
      type_signature = source_type[name]
    else:
      py_typecheck.check_type(index, int)
      length = len(source_type)
      if index < 0 or index >= length:
        raise ValueError(
            'The index \'{}\' does not fit into the valid range in the '
            'struct type: 0..{}.'.format(name, length))
      type_signature = source_type[index]
    super().__init__(type_signature)
    self._source = source
    self._name = name
    self._index = index

  @property
  def proto(self):
    if self._name is not None:
      selection = pb.Selection(source=self._source.proto, name=self._name)
    else:
      selection = pb.Selection(source=self._source.proto, index=self._index)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        selection=selection)

  def is_selection(self):
    return True

  @property
  def source(self):
    return self._source

  @property
  def name(self):
    return self._name

  @property
  def index(self):
    return self._index

  def __repr__(self):
    if self._name is not None:
      return 'Selection({!r}, name=\'{}\')'.format(self._source, self._name)
    else:
      return 'Selection({!r}, index={})'.format(self._source, self._index)


class Struct(ComputationBuildingBlock, structure.Struct):
  """A struct with named or unnamed elements in TFF's internal language.

  The concise notation for structs is `<name_1=value_1, ...., name_n=value_n>`
  for structs with named elements, `<value_1, ..., value_n>` for structs with
  unnamed elements, or a mixture of these for structs with some named and some
  unnamed elements, where `name_k` are the names, and `value_k` are the value
  expressions.

  For example, a lambda expression that applies `fn` to elements of 2-structs
  pointwise could be represented as `(arg -> <fn(arg[0]),fn(arg[1])>)`.
  """

  @classmethod
  def from_proto(
      cls: Type['Struct'],
      computation_proto: pb.Computation,
  ) -> 'Struct':
    _check_computation_oneof(computation_proto, 'struct')
    return cls([(str(e.name) if e.name else None,
                 ComputationBuildingBlock.from_proto(e.value))
                for e in computation_proto.struct.element])

  def __init__(self, elements, container_type=None):
    """Constructs a struct from the given list of elements.

    Args:
      elements: The elements of the struct, supplied as a list of (name, value)
        pairs, where 'name' can be None in case the corresponding element is not
        named and only accessible via an index (see also `structure.Struct`).
      container_type: An optional Python container type to associate with the
        struct.

    Raises:
      TypeError: if arguments are of the wrong types.
    """

    # Not using super() here and below, as the two base classes have different
    # signatures of their constructors, and the struct implementation
    # of selection interfaces should override that in the generic class 'Value'
    # to favor simplified expressions where simplification is possible.
    def _map_element(e):
      """Returns a named or unnamed element."""
      if isinstance(e, ComputationBuildingBlock):
        return (None, e)
      elif py_typecheck.is_name_value_pair(
          e, name_required=False, value_type=ComputationBuildingBlock):
        if e[0] is not None and not e[0]:
          raise ValueError('Unexpected struct element with empty string name.')
        return (e[0], e[1])
      else:
        raise TypeError('Unexpected struct element: {}.'.format(e))

    elements = [_map_element(e) for e in elements]
    element_pairs = [((e[0],
                       e[1].type_signature) if e[0] else e[1].type_signature)
                     for e in elements]

    if container_type is None:
      type_signature = computation_types.StructType(element_pairs)
    else:
      type_signature = computation_types.StructWithPythonType(
          element_pairs, container_type)
    ComputationBuildingBlock.__init__(self, type_signature)
    structure.Struct.__init__(self, elements)

  @property
  def proto(self):
    elements = []
    for k, v in structure.iter_elements(self):
      if k is not None:
        element = pb.Struct.Element(name=k, value=v.proto)
      else:
        element = pb.Struct.Element(value=v.proto)
      elements.append(element)
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        struct=pb.Struct(element=elements))

  def is_struct(self):
    return True

  def __repr__(self):

    def _element_repr(element):
      name, value = element
      name_repr = '\'{}\''.format(name) if name is not None else 'None'
      return '({}, {!r})'.format(name_repr, value)

    return 'Struct([{}])'.format(', '.join(
        _element_repr(e) for e in structure.iter_elements(self)))


class Call(ComputationBuildingBlock):
  """A representation of a function invocation in TFF's internal language.

  The call construct takes an argument struct with two elements, the first being
  the function to invoke (represented as a computation with a functional result
  type), and the second being the argument to feed to that function. Typically,
  the function is either a TFF instrinsic, or a lambda expression.

  The concise notation for calls is `foo(bar)`, where `foo` is the function,
  and `bar` is the argument.
  """

  @classmethod
  def from_proto(
      cls: Type['Call'],
      computation_proto: pb.Computation,
  ) -> 'Call':
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
    if not fn.type_signature.is_function():
      raise TypeError('Expected fn to be of a functional type, '
                      'but found that its type is {}.'.format(
                          fn.type_signature))
    if fn.type_signature.parameter is not None:
      if arg is None:
        raise TypeError('The invoked function expects an argument of type {}, '
                        'but got None instead.'.format(
                            fn.type_signature.parameter))
      if not fn.type_signature.parameter.is_assignable_from(arg.type_signature):
        raise TypeError(
            'The parameter of the invoked function is expected to be of '
            'type {}, but the supplied argument is of an incompatible '
            'type {}.'.format(fn.type_signature.parameter, arg.type_signature))
    elif arg is not None:
      raise TypeError(
          'The invoked function does not expect any parameters, but got '
          'an argument of type {}.'.format(py_typecheck.type_string(type(arg))))
    super().__init__(fn.type_signature.result)
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

  def is_call(self):
    return True

  @property
  def function(self):
    return self._function

  @property
  def argument(self):
    return self._argument

  def __repr__(self):
    if self._argument is not None:
      return 'Call({!r}, {!r})'.format(self._function, self._argument)
    else:
      return 'Call({!r})'.format(self._function)


class Lambda(ComputationBuildingBlock):
  """A representation of a lambda expression in TFF's internal language.

  A lambda expression consists of a string formal parameter name, and a result
  expression that can contain references by name to that formal parameter. A
  concise notation for lambdas is `(foo -> bar)`, where `foo` is the name of
  the formal parameter, and `bar` is the result expression.
  """

  @classmethod
  def from_proto(
      cls: Type['Lambda'],
      computation_proto: pb.Computation,
  ) -> 'Lambda':
    _check_computation_oneof(computation_proto, 'lambda')
    the_lambda = getattr(computation_proto, 'lambda')
    return cls(
        str(the_lambda.parameter_name),
        type_serialization.deserialize_type(
            computation_proto.type.function.parameter),
        ComputationBuildingBlock.from_proto(the_lambda.result))

  def __init__(
      self,
      parameter_name: Optional[str],
      parameter_type: Optional[Any],
      result: ComputationBuildingBlock,
  ):
    """Creates a lambda expression.

    Args:
      parameter_name: The (string) name of the parameter accepted by the lambda.
        This name can be used by Reference() instances in the body of the lambda
        to refer to the parameter. Note that an empty parameter name shall be
        treated as equivalent to no parameter.
      parameter_type: The type of the parameter, an instance of types.Type or
        something convertible to it by types.to_type().
      result: The resulting value produced by the expression that forms the body
        of the lambda. Must be an instance of ComputationBuildingBlock.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    if parameter_name == '':  # pylint: disable=g-explicit-bool-comparison
      parameter_name = None
    if (parameter_name is None) != (parameter_type is None):
      raise TypeError(
          'A lambda expression must have either a valid parameter name and type '
          'or both parameter name and type must be `None`. '
          '`parameter_name` was {} but `parameter_type` was {}.'.format(
              parameter_name, parameter_type))
    if parameter_name is not None:
      py_typecheck.check_type(parameter_name, str)
      parameter_type = computation_types.to_type(parameter_type)
    py_typecheck.check_type(result, ComputationBuildingBlock)
    super().__init__(
        computation_types.FunctionType(parameter_type, result.type_signature))
    self._parameter_name = parameter_name
    self._parameter_type = parameter_type
    self._result = result

  @property
  def proto(self) -> pb.Computation:
    type_signature = type_serialization.serialize_type(self.type_signature)
    fn = pb.Lambda(
        parameter_name=self._parameter_name, result=self._result.proto)
    # We are unpacking the lambda argument here because `lambda` is a reserved
    # keyword in Python, but it is also the name of the parameter for a
    # `pb.Computation`.
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
    return pb.Computation(type=type_signature, **{'lambda': fn})  # pytype: disable=wrong-keyword-args

  def is_lambda(self):
    return True

  @property
  def parameter_name(self) -> Optional[str]:
    return self._parameter_name

  @property
  def parameter_type(self) -> Optional[computation_types.Type]:
    return self._parameter_type

  @property
  def result(self) -> ComputationBuildingBlock:
    return self._result

  def __repr__(self) -> str:
    return 'Lambda(\'{}\', {!r}, {!r})'.format(self._parameter_name,
                                               self._parameter_type,
                                               self._result)


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
  def from_proto(
      cls: Type['Block'],
      computation_proto: pb.Computation,
  ) -> 'Block':
    _check_computation_oneof(computation_proto, 'block')
    return cls([(str(loc.name), ComputationBuildingBlock.from_proto(loc.value))
                for loc in computation_proto.block.local],
               ComputationBuildingBlock.from_proto(
                   computation_proto.block.result))

  def __init__(
      self,
      local_symbols: Iterable[Tuple[str, ComputationBuildingBlock]],
      result: ComputationBuildingBlock,
  ):
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
          not isinstance(element[0], str)):
        raise TypeError(
            'Expected the locals to be a list of 2-element structs with string '
            'name as their first element, but this is not the case for the '
            'local at position {} in the sequence: {}.'.format(index, element))
      name = element[0]
      value = element[1]
      py_typecheck.check_type(value, ComputationBuildingBlock)
      updated_locals.append((name, value))
    py_typecheck.check_type(result, ComputationBuildingBlock)
    super().__init__(result.type_signature)
    self._locals = updated_locals
    self._result = result

  @property
  def proto(self) -> pb.Computation:
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

  def is_block(self):
    return True

  @property
  def locals(self) -> List[Tuple[str, ComputationBuildingBlock]]:
    return list(self._locals)

  @property
  def result(self) -> ComputationBuildingBlock:
    return self._result

  def __repr__(self) -> str:
    return 'Block([{}], {!r})'.format(
        ', '.join('(\'{}\', {!r})'.format(k, v) for k, v in self._locals),
        self._result)


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
  def from_proto(
      cls: Type['Intrinsic'],
      computation_proto: pb.Computation,
  ) -> 'Intrinsic':
    _check_computation_oneof(computation_proto, 'intrinsic')
    return cls(computation_proto.intrinsic.uri,
               type_serialization.deserialize_type(computation_proto.type))

  def __init__(self, uri: str, type_signature: computation_types.Type):
    """Creates an intrinsic.

    Args:
      uri: The URI of the intrinsic.
      type_signature: A `tff.Type`, the type of the intrinsic.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(uri, str)
    py_typecheck.check_type(type_signature, computation_types.Type)
    intrinsic_def = intrinsic_defs.uri_to_intrinsic_def(uri)
    if intrinsic_def is not None:
      # Note: this is really expensive.
      if not type_analysis.is_concrete_instance_of(
          type_signature, intrinsic_def.type_signature):
        raise TypeError('Tried to construct an Intrinsic with bad type '
                        'signature; Intrinsic {} expects type signature {}, '
                        'and you tried to construct one of type {}.'.format(
                            uri, intrinsic_def.type_signature, type_signature))
    super().__init__(type_signature)
    self._uri = uri

  @property
  def proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        intrinsic=pb.Intrinsic(uri=self._uri))

  def is_intrinsic(self):
    return True

  @property
  def uri(self) -> str:
    return self._uri

  def __repr__(self) -> str:
    return 'Intrinsic(\'{}\', {!r})'.format(self._uri, self.type_signature)


class Data(ComputationBuildingBlock):
  """A representation of data (an input pipeline).

  This class does not deal with parsing data URIs and verifying correctness,
  it is only a container. Parsing and type analysis are a responsibility
  or a component external to this module.
  """

  @classmethod
  def from_proto(
      cls: Type['Data'],
      computation_proto: pb.Computation,
  ) -> 'Data':
    _check_computation_oneof(computation_proto, 'data')
    return cls(computation_proto.data.uri,
               type_serialization.deserialize_type(computation_proto.type))

  def __init__(self, uri: str, type_spec: Any):
    """Creates a representation of data.

    Args:
      uri: The URI that characterizes the data.
      type_spec: Either the types.Type that represents the type of this data, or
        something convertible to it by types.to_type().

    Raises:
      TypeError: if the arguments are of the wrong types.
      ValueError: if the user tries to specify an empty URI.
    """
    py_typecheck.check_type(uri, str)
    if not uri:
      raise ValueError('Empty string cannot be passed as URI to Data.')
    if type_spec is None:
      raise TypeError(
          'Intrinsic {} cannot be created without a TFF type.'.format(uri))
    type_spec = computation_types.to_type(type_spec)
    super().__init__(type_spec)
    self._uri = uri

  @property
  def proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        data=pb.Data(uri=self._uri))

  def is_data(self):
    return True

  @property
  def uri(self) -> str:
    return self._uri

  def __repr__(self) -> str:
    return 'Data(\'{}\', {!r})'.format(self._uri, self.type_signature)


class CompiledComputation(ComputationBuildingBlock):
  """A representation of a fully constructed and serialized computation.

  A compile comutation is one that has not been parsed into constituents, and
  is simply represented as an embedded `Computation` protocol buffer. Whereas
  technically, any computation can be represented and passed around this way,
  this structure is generally only used to represent TensorFlow sections, for
  which otherwise there isn't any dedicated structure.
  """

  def __init__(self,
               proto: pb.Computation,
               name: Optional[str] = None,
               type_signature: Optional[computation_types.Type] = None):
    """Creates a representation of a fully constructed computation.

    Args:
      proto: An instance of pb.Computation with the computation logic.
      name: An optional string name to associate with this computation, used
        only for debugging purposes. If the name is not specified (None), it is
        autogenerated as a hexadecimal string from the hash of the proto.
      type_signature: An optional type signature to associate with this
        computation rather than the serialized one.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(proto, pb.Computation)
    if name is not None:
      py_typecheck.check_type(name, str)
    if type_signature is None:
      type_signature = type_serialization.deserialize_type(proto.type)
    py_typecheck.check_type(type_signature, computation_types.Type)
    super().__init__(type_signature)
    self._proto = proto
    if name is not None:
      self._name = name
    else:
      self._name = '{:x}'.format(zlib.adler32(self._proto.SerializeToString()))

  @property
  def proto(self) -> pb.Computation:
    return self._proto

  def is_compiled_computation(self):
    return True

  @property
  def name(self) -> str:
    return self._name

  def __repr__(self) -> str:
    return 'CompiledComputation(\'{}\', {!r})'.format(self._name,
                                                      self.type_signature)


class Placement(ComputationBuildingBlock):
  """A representation of a placement literal in TFF's internal language.

  Currently this can only be `tff.SERVER` or `tff.CLIENTS`.
  """

  @classmethod
  def from_proto(
      cls: Type['Placement'],
      computation_proto: pb.Computation,
  ) -> 'Placement':
    _check_computation_oneof(computation_proto, 'placement')
    return cls(
        placement_literals.uri_to_placement_literal(
            str(computation_proto.placement.uri)))

  def __init__(self, literal: placement_literals.PlacementLiteral):
    """Constructs a new placement instance for the given placement literal.

    Args:
      literal: The placement literal.

    Raises:
      TypeError: if the arguments are of the wrong types.
    """
    py_typecheck.check_type(literal, placement_literals.PlacementLiteral)
    super().__init__(computation_types.PlacementType())
    self._literal = literal

  @property
  def proto(self) -> pb.Computation:
    return pb.Computation(
        type=type_serialization.serialize_type(self.type_signature),
        placement=pb.Placement(uri=self._literal.uri))

  def is_placement(self):
    return True

  @property
  def uri(self) -> str:
    return self._literal.uri

  def __repr__(self) -> str:
    return 'Placement(\'{}\')'.format(self.uri)


def _string_representation(
    comp: ComputationBuildingBlock,
    formatted: bool,
) -> str:
  """Returns the string representation of a `ComputationBuildingBlock`.

  This functions creates a `list` of strings representing the given `comp`;
  combines the strings in either a formatted or un-formatted representation; and
  returns the resulting string represetnation.

  Args:
    comp: An instance of a `ComputationBuildingBlock`.
    formatted: A boolean indicating if the returned string should be formatted.

  Raises:
    TypeError: If `comp` has an unepxected type.
  """
  py_typecheck.check_type(comp, ComputationBuildingBlock)

  def _join(components: Iterable[List[str]]) -> List[str]:
    """Returns a `list` of strings by combining each component in `components`.

    >>> _join([['a'], ['b'], ['c']])
    ['abc']

    >>> _join([['a', 'b', 'c'], ['d', 'e', 'f']])
    ['abcd', 'ef']

    This function is used to help track where new-lines should be inserted into
    the string representation if the lines are formatted.

    Args:
      components: A `list` where each element is a `list` of strings
        representing a part of the string of a `ComputationBuildingBlock`.
    """
    lines = ['']
    for component in components:
      lines[-1] = '{}{}'.format(lines[-1], component[0])
      lines.extend(component[1:])
    return lines

  def _indent(lines, indent_chars='  '):
    """Returns a `list` of strings indented across a slice."""
    return ['{}{}'.format(indent_chars, e) for e in lines]

  def _lines_for_named_comps(named_comps, formatted):
    """Returns a `list` of strings representing the given `named_comps`.

    Args:
      named_comps: A `list` of named comutations, each being a pair consisting
        of a name (either a string, or `None`) and a `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    lines = []
    for index, (name, comp) in enumerate(named_comps):
      if index != 0:
        if formatted:
          lines.append([',', ''])
        else:
          lines.append([','])
      element_lines = _lines_for_comp(comp, formatted)
      if name is not None:
        element_lines = _join([
            ['{}='.format(name)],
            element_lines,
        ])
      lines.append(element_lines)
    return _join(lines)

  def _lines_for_comp(comp, formatted):
    """Returns a `list` of strings representing the given `comp`.

    Args:
      comp: An instance of a `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    if comp.is_block():
      lines = []
      variables_lines = _lines_for_named_comps(comp.locals, formatted)
      if formatted:
        variables_lines = _indent(variables_lines)
        lines.extend([['(let ', ''], variables_lines, ['', ' in ']])
      else:
        lines.extend([['(let '], variables_lines, [' in ']])
      result_lines = _lines_for_comp(comp.result, formatted)
      lines.append(result_lines)
      lines.append([')'])
      return _join(lines)
    elif comp.is_reference():
      if comp.context is not None:
        return ['{}@{}'.format(comp.name, comp.context)]
      else:
        return [comp.name]
    elif comp.is_selection():
      source_lines = _lines_for_comp(comp.source, formatted)
      if comp.name is not None:
        return _join([source_lines, ['.{}'.format(comp.name)]])
      else:
        return _join([source_lines, ['[{}]'.format(comp.index)]])
    elif comp.is_call():
      function_lines = _lines_for_comp(comp.function, formatted)
      if comp.argument is not None:
        argument_lines = _lines_for_comp(comp.argument, formatted)
        return _join([function_lines, ['('], argument_lines, [')']])
      else:
        return _join([function_lines, ['()']])
    elif comp.is_compiled_computation():
      return ['comp#{}'.format(comp.name)]
    elif comp.is_data():
      return [comp.uri]
    elif comp.is_intrinsic():
      return [comp.uri]
    elif comp.is_lambda():
      result_lines = _lines_for_comp(comp.result, formatted)
      if comp.parameter_type is None:
        param_name = ''
      else:
        param_name = comp.parameter_name
      lines = [['({} -> '.format(param_name)], result_lines, [')']]
      return _join(lines)
    elif comp.is_placement():
      return [comp._literal.name]  # pylint: disable=protected-access
    elif comp.is_struct():
      if len(comp) == 0:  # pylint: disable=g-explicit-length-test
        return ['<>']
      elements = structure.to_elements(comp)
      elements_lines = _lines_for_named_comps(elements, formatted)
      if formatted:
        elements_lines = _indent(elements_lines)
        lines = [['<', ''], elements_lines, ['', '>']]
      else:
        lines = [['<'], elements_lines, ['>']]
      return _join(lines)
    else:
      raise NotImplementedError('Unexpected type found: {}.'.format(type(comp)))

  lines = _lines_for_comp(comp, formatted)
  lines = [line.rstrip() for line in lines]
  if formatted:
    return '\n'.join(lines)
  else:
    return ''.join(lines)


def _structural_representation(comp):
  """Returns the structural string representation of the given `comp`.

  This functions creates and returns a string representing the structure of the
  abstract syntax tree for the given `comp`.

  Args:
    comp: An instance of a `ComputationBuildingBlock`.

  Raises:
    TypeError: If `comp` has an unepxected type.
  """
  py_typecheck.check_type(comp, ComputationBuildingBlock)
  padding_char = ' '

  def _get_leading_padding(string):
    """Returns the length of the leading padding for the given `string`."""
    for index, character in enumerate(string):
      if character != padding_char:
        return index
    return len(string)

  def _get_trailing_padding(string):
    """Returns the length of the trailing padding for the given `string`."""
    for index, character in enumerate(reversed(string)):
      if character != padding_char:
        return index
    return len(string)

  def _pad_left(lines, total_width):
    """Pads the beginning of each line in `lines` to the given `total_width`.

    >>>_pad_left(['aa', 'bb'], 4)
    ['  aa', '  bb',]

    Args:
      lines: A `list` of strings to pad.
      total_width: The length that each line in `lines` should be padded to.

    Returns:
      A `list` of lines with padding applied.
    """

    def _pad_line_left(line, total_width):
      current_width = len(line)
      assert current_width <= total_width
      padding = total_width - current_width
      return '{}{}'.format(padding_char * padding, line)

    return [_pad_line_left(line, total_width) for line in lines]

  def _pad_right(lines, total_width):
    """Pads the end of each line in `lines` to the given `total_width`.

    >>>_pad_right(['aa', 'bb'], 4)
    ['aa  ', 'bb  ']

    Args:
      lines: A `list` of strings to pad.
      total_width: The length that each line in `lines` should be padded to.

    Returns:
      A `list` of lines with padding applied.
    """

    def _pad_line_right(line, total_width):
      current_width = len(line)
      assert current_width <= total_width
      padding = total_width - current_width
      return '{}{}'.format(line, padding_char * padding)

    return [_pad_line_right(line, total_width) for line in lines]

  class Alignment(enum.Enum):
    LEFT = 1
    RIGHT = 2

  def _concatenate(lines_1, lines_2, align):
    """Concatenates two `list`s of strings.

    Concatenates two `list`s of strings by appending one list of strings to the
    other and then aligning lines of different widths by either padding the left
    or padding the right of each line to the width of the longest line.

    >>>_concatenate(['aa', 'bb'], ['ccc'], Alignment.LEFT)
    ['aa ', 'bb ', 'ccc']

    Args:
      lines_1: A `list` of strings.
      lines_2: A `list` of strings.
      align: An enum indicating how to align lines of different widths.

    Returns:
      A `list` of lines.
    """
    lines = lines_1 + lines_2
    longest_line = max(lines, key=len)
    longest_width = len(longest_line)
    if align is Alignment.LEFT:
      return _pad_right(lines, longest_width)
    elif align is Alignment.RIGHT:
      return _pad_left(lines, longest_width)

  def _calculate_inset_from_padding(left, right, preferred_padding,
                                    minimum_content_padding):
    """Calculates the inset for the given padding.

    Note: This function is intended to only be called from `_fit_with_padding`.

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      preferred_padding: The preferred amount of non-negative padding between
        the lines in the fitted `list` of strings.
      minimum_content_padding: The minimum amount of non-negative padding
        allowed between the lines in the fitted `list` of strings.

    Returns:
      An integer.
    """
    assert preferred_padding >= 0
    assert minimum_content_padding >= 0

    trailing_padding = _get_trailing_padding(left[0])
    leading_padding = _get_leading_padding(right[0])
    inset = trailing_padding + leading_padding - preferred_padding
    for left_line, right_line in zip(left[1:], right[1:]):
      trailing_padding = _get_trailing_padding(left_line)
      leading_padding = _get_leading_padding(right_line)
      minimum_inset = trailing_padding + leading_padding - minimum_content_padding
      inset = min(inset, minimum_inset)
    return inset

  def _fit_with_inset(left, right, inset):
    r"""Concatenates the lines of two `list`s of strings.

    Note: This function is intended to only be called from `_fit_with_padding`.

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      inset: The amount of padding to remove or add when concatenating the
        lines.

    Returns:
      A `list` of lines.
    """
    lines = []
    for left_line, right_line in zip(left, right):
      if inset > 0:
        left_inset = 0
        right_inset = 0
        trailing_padding = _get_trailing_padding(left_line)
        if trailing_padding > 0:
          left_inset = min(trailing_padding, inset)
          left_line = left_line[:-left_inset]
        if inset - left_inset > 0:
          leading_padding = _get_leading_padding(right_line)
          if leading_padding > 0:
            right_inset = min(leading_padding, inset - left_inset)
            right_line = right_line[right_inset:]
      padding = abs(inset) if inset < 0 else 0
      line = ''.join([left_line, padding_char * padding, right_line])
      lines.append(line)
    left_height = len(left)
    right_height = len(right)
    if left_height > right_height:
      lines.extend(left[right_height:])
    elif right_height > left_height:
      lines.extend(right[left_height:])
    longest_line = max(lines, key=len)
    longest_width = len(longest_line)
    shortest_line = min(lines, key=len)
    shortest_width = len(shortest_line)
    if shortest_width != longest_width:
      if left_height > right_height:
        lines = _pad_right(lines, longest_width)
      else:
        lines = _pad_left(lines, longest_width)
    return lines

  def _fit_with_padding(left,
                        right,
                        preferred_padding,
                        minimum_content_padding=4):
    r"""Concatenates the lines of two `list`s of strings.

    Concatenates the lines of two `list`s of strings by appending each line
    together using a padding. The same padding is used to append each line and
    the padding is calculated starting from the `preferred_padding` without
    going below `minimum_content_padding` on any of the lines. If the two
    `list`s of strings have different lengths, padding will be applied to
    maintain the length of each string in the resulting `list` of strings.

    >>>_fit_with_padding(['aa', 'bb'], ['ccc'])
    ['aa    cccc', 'bb        ']

    >>>_fit_with_padding(['aa          ', 'bb          '], ['          ccc'])
    ['aa    cccc', 'bb        ']

    Args:
      left: A `list` of strings.
      right: A `list` of strings.
      preferred_padding: The preferred amount of non-negative padding between
        the lines in the fitted `list` of strings.
      minimum_content_padding: The minimum amount of non-negative padding
        allowed between the lines in the fitted `list` of strings.

    Returns:
      A `list` of lines.
    """
    inset = _calculate_inset_from_padding(left, right, preferred_padding,
                                          minimum_content_padding)
    return _fit_with_inset(left, right, inset)

  def _get_node_label(comp):
    """Returns a string for node in the structure of the given `comp`."""
    if comp.is_block():
      return 'Block'
    elif comp.is_call():
      return 'Call'
    elif comp.is_compiled_computation():
      return 'Compiled({})'.format(comp.name)
    elif comp.is_data():
      return comp.uri
    elif comp.is_intrinsic():
      return comp.uri
    elif comp.is_lambda():
      return 'Lambda({})'.format(comp.parameter_name)
    elif comp.is_reference():
      return 'Ref({})'.format(comp.name)
    elif comp.is_placement():
      return 'Placement'
    elif comp.is_selection():
      key = comp.name if comp.name is not None else comp.index
      return 'Sel({})'.format(key)
    elif comp.is_struct():
      return 'Struct'
    else:
      raise TypeError('Unexpected type found: {}.'.format(type(comp)))

  def _lines_for_named_comps(named_comps):
    """Returns a `list` of strings representing the given `named_comps`.

    Args:
      named_comps: A `list` of named comutations, each being a pair consisting
        of a name (either a string, or `None`) and a `ComputationBuildingBlock`.
    """
    lines = ['[']
    for index, (name, comp) in enumerate(named_comps):
      comp_lines = _lines_for_comp(comp)
      if name is not None:
        label = '{}='.format(name)
        comp_lines = _fit_with_padding([label], comp_lines, 0, 0)
      if index == 0:
        lines = _fit_with_padding(lines, comp_lines, 0, 0)
      else:
        lines = _fit_with_padding(lines, [','], 0, 0)
        lines = _fit_with_padding(lines, comp_lines, 1)
    lines = _fit_with_padding(lines, [']'], 0, 0)
    return lines

  def _lines_for_comp(comp):
    """Returns a `list` of strings representing the given `comp`.

    Args:
      comp: An instance of a `ComputationBuildingBlock`.
    """
    node_label = _get_node_label(comp)

    if (comp.is_compiled_computation() or comp.is_data() or
        comp.is_intrinsic() or comp.is_placement() or comp.is_reference()):
      return [node_label]
    elif comp.is_block():
      variables_lines = _lines_for_named_comps(comp.locals)
      variables_width = len(variables_lines[0])
      variables_trailing_padding = _get_trailing_padding(variables_lines[0])
      leading_padding = variables_width - variables_trailing_padding
      edge_line = '{}/'.format(padding_char * leading_padding)
      variables_lines = _concatenate([edge_line], variables_lines,
                                     Alignment.LEFT)

      result_lines = _lines_for_comp(comp.result)
      result_width = len(result_lines[0])
      leading_padding = _get_leading_padding(result_lines[0]) - 1
      trailing_padding = result_width - leading_padding - 1
      edge_line = '\\{}'.format(padding_char * trailing_padding)
      result_lines = _concatenate([edge_line], result_lines, Alignment.RIGHT)

      preferred_padding = len(node_label)
      lines = _fit_with_padding(variables_lines, result_lines,
                                preferred_padding)
      leading_padding = _get_leading_padding(lines[0]) + 1
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      return _concatenate([node_line], lines, Alignment.LEFT)
    elif comp.is_call():
      function_lines = _lines_for_comp(comp.function)
      function_width = len(function_lines[0])
      function_trailing_padding = _get_trailing_padding(function_lines[0])
      leading_padding = function_width - function_trailing_padding
      edge_line = '{}/'.format(padding_char * leading_padding)
      function_lines = _concatenate([edge_line], function_lines, Alignment.LEFT)

      if comp.argument is not None:
        argument_lines = _lines_for_comp(comp.argument)
        argument_width = len(argument_lines[0])
        leading_padding = _get_leading_padding(argument_lines[0]) - 1
        trailing_padding = argument_width - leading_padding - 1
        edge_line = '\\{}'.format(padding_char * trailing_padding)
        argument_lines = _concatenate([edge_line], argument_lines,
                                      Alignment.RIGHT)

        preferred_padding = len(node_label)
        lines = _fit_with_padding(function_lines, argument_lines,
                                  preferred_padding)
      else:
        lines = function_lines
      leading_padding = _get_leading_padding(lines[0]) + 1
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      return _concatenate([node_line], lines, Alignment.LEFT)
    elif comp.is_lambda():
      result_lines = _lines_for_comp(comp.result)
      leading_padding = _get_leading_padding(result_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], result_lines, Alignment.LEFT)
    elif comp.is_selection():
      source_lines = _lines_for_comp(comp.source)
      leading_padding = _get_leading_padding(source_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], source_lines, Alignment.LEFT)
    elif comp.is_struct():
      elements = structure.to_elements(comp)
      elements_lines = _lines_for_named_comps(elements)
      leading_padding = _get_leading_padding(elements_lines[0])
      node_line = '{}{}'.format(padding_char * leading_padding, node_label)
      edge_line = '{}|'.format(padding_char * leading_padding)
      return _concatenate([node_line, edge_line], elements_lines,
                          Alignment.LEFT)
    else:
      raise NotImplementedError('Unexpected type found: {}.'.format(type(comp)))

  lines = _lines_for_comp(comp)
  lines = [line.rstrip() for line in lines]
  return '\n'.join(lines)


# pylint: disable=protected-access
ComputationBuildingBlock._deserializer_dict = {
    'reference': Reference.from_proto,
    'selection': Selection.from_proto,
    'struct': Struct.from_proto,
    'call': Call.from_proto,
    'lambda': Lambda.from_proto,
    'block': Block.from_proto,
    'intrinsic': Intrinsic.from_proto,
    'data': Data.from_proto,
    'placement': Placement.from_proto,
    'tensorflow': CompiledComputation,
}
# pylint: enable=protected-access
