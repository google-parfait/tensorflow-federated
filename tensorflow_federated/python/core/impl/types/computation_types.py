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
"""Defines functions and classes for building and manipulating TFF types."""

import abc
import atexit
import collections
from collections.abc import Hashable, Iterable, Iterator, Mapping
import difflib
import enum
from typing import Optional, TypeVar
import weakref

import attr
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.tensorflow_libs import tensor_utils

T = TypeVar('T')


class UnexpectedTypeError(TypeError):

  def __init__(self, expected: type['Type'], actual: 'Type'):
    message = f'Expected type of kind {expected}, found type {actual}'
    super().__init__(message)
    self.actual = actual
    self.expected = expected


# Prevent wrapping on a 100-character terminal.
MAX_LINE_LEN = 100


@enum.unique
class TypeRelation(enum.Enum):
  EQUIVALENT = 'equivalent'
  IDENTICAL = 'identical'
  ASSIGNABLE = 'assignable'


def type_mismatch_error_message(
    first: 'Type',
    second: 'Type',
    relation: TypeRelation,
    second_is_expected: bool = False,
) -> str:
  """Returns an error message describing the mismatch between two types."""
  maybe_expected = 'expected ' if second_is_expected else ''
  first_str = first.compact_representation()
  second_str = second.compact_representation()
  diff = None
  if first_str == second_str:
    # The two only differ in container types or some other property not
    # visible via the compact representation, so show `repr` instead.
    # No diff is used because `repr` prints to a single line.
    first_str = repr(first)
    second_str = repr(second)
    diff = None
  elif len(first_str) > MAX_LINE_LEN or len(second_str) > MAX_LINE_LEN:
    # The types are large structures, and so the formatted representation is
    # used and a summary diff is added. The logic here is that large types
    # may be easier to diff visually with a more structured representation,
    # and logical line breaks are required to make diff output useful.
    first_str = first.formatted_representation()
    second_str = second.formatted_representation()
    split_first = first_str.split('\n')
    split_second = second_str.split('\n')
    diff = '\n'.join(difflib.unified_diff(split_first, split_second))
  message = [
      'Type',
      f'`{first_str}`',
      f'is not {relation.value} to {maybe_expected}type',
      f'`{second_str}`',
  ]
  if diff:
    message += [f'\nDiff:\n{diff}']
  single_line = ' '.join(message)
  if len(single_line) > MAX_LINE_LEN or '\n' in single_line:
    return '\n'.join(message)
  else:
    return single_line


class TypeNotAssignableError(TypeError):

  def __init__(self, source_type, target_type):
    self.message = type_mismatch_error_message(
        source_type, target_type, TypeRelation.ASSIGNABLE
    )
    super().__init__(self.message)
    self.source_type = source_type
    self.target_type = target_type


class TypesNotEquivalentError(TypeError):

  def __init__(self, first_type, second_type):
    self.message = type_mismatch_error_message(
        first_type, second_type, TypeRelation.EQUIVALENT
    )
    super().__init__(self.message)
    self.first_type = first_type
    self.second_type = second_type


class TypesNotIdenticalError(TypeError):

  def __init__(self, first_type, second_type):
    self.message = type_mismatch_error_message(
        first_type, second_type, TypeRelation.IDENTICAL
    )
    super().__init__(self.message)
    self.first_type = first_type
    self.second_type = second_type


class Type(metaclass=abc.ABCMeta):
  """An abstract interface for all classes that represent TFF types."""

  def compact_representation(self) -> str:
    """Returns the compact string representation of this type."""
    return _string_representation(self, formatted=False)

  def formatted_representation(self) -> str:
    """Returns the formatted string representation of this type."""
    return _string_representation(self, formatted=True)

  @abc.abstractmethod
  def children(self):
    """Returns a generator yielding immediate child types."""
    raise NotImplementedError

  def check_abstract(self) -> None:
    """Check that this is a `tff.AbstractType`."""
    if not self.is_abstract():
      raise UnexpectedTypeError(AbstractType, self)

  def is_abstract(self) -> bool:
    """Returns whether or not this type is a `tff.AbstractType`."""
    return False

  def check_federated(self) -> None:
    """Check that this is a `tff.FederatedType`."""
    if not self.is_federated():
      raise UnexpectedTypeError(FederatedType, self)

  def is_federated(self) -> bool:
    """Returns whether or not this type is a `tff.FederatedType`."""
    return False

  def check_function(self) -> None:
    """Check that this is a `tff.FunctionType`."""
    if not self.is_function():
      raise UnexpectedTypeError(FunctionType, self)

  def is_function(self) -> bool:
    """Returns whether or not this type is a `tff.FunctionType`."""
    return False

  def check_placement(self) -> None:
    """Check that this is a `tff.PlacementType`."""
    if not self.is_placement():
      raise UnexpectedTypeError(PlacementType, self)

  def is_placement(self) -> bool:
    """Returns whether or not this type is a `tff.PlacementType`."""
    return False

  def check_sequence(self) -> None:
    """Check that this is a `tff.SequenceType`."""
    if not self.is_sequence():
      raise UnexpectedTypeError(SequenceType, self)

  def is_sequence(self) -> bool:
    """Returns whether or not this type is a `tff.SequenceType`."""
    return False

  def check_struct(self) -> None:
    """Check that this is a `tff.StructType`."""
    if not self.is_struct():
      raise UnexpectedTypeError(StructType, self)

  def is_struct(self) -> bool:
    """Returns whether or not this type is a `tff.StructType`."""
    return False

  def check_struct_with_python(self) -> None:
    """Check that this is a `tff.StructWithPythonType`."""
    if not self.is_struct_with_python():
      raise UnexpectedTypeError(StructWithPythonType, self)

  def is_struct_with_python(self) -> bool:
    """Returns whether or not this type is a `tff.StructWithPythonType`."""
    return False

  def check_tensor(self) -> None:
    """Check that this is a `tff.TensorType`."""
    if not self.is_tensor():
      raise UnexpectedTypeError(TensorType, self)

  def is_tensor(self) -> bool:
    """Returns whether or not this type is a `tff.TensorType`."""
    return False

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this type."""
    raise NotImplementedError

  def __str__(self):
    """Returns a concise representation of this type."""
    return self.compact_representation()

  @abc.abstractmethod
  def __hash__(self):
    """Produces a hash value for this type."""
    raise NotImplementedError

  @abc.abstractmethod
  def __eq__(self, other):
    """Determines whether two type definitions are identical.

    Note that this notion of equality is stronger than equivalence. Two types
    with equivalent definitions may not be identical, e.g., if they represent
    templates with differently named type variables in their definitions.

    Args:
      other: The other type to compare against.

    Returns:
      `True` iff type definitions are syntatically identical (as defined above),
      or `False` otherwise.

    Raises:
      NotImplementedError: If not implemented in the derived class.
    """
    raise NotImplementedError

  def __ne__(self, other):
    return not self == other

  def check_assignable_from(self, source_type: 'Type') -> None:
    """Raises if values of `source_type` cannot be cast to this type."""
    if not self.is_assignable_from(source_type):
      raise TypeNotAssignableError(source_type=source_type, target_type=self)

  @abc.abstractmethod
  def is_assignable_from(self, source_type: 'Type') -> bool:
    """Returns whether values of `source_type` can be cast to this type."""
    raise NotImplementedError

  def check_equivalent_to(self, other: 'Type') -> None:
    """Raises if values of 'other' cannot be cast to and from this type."""
    if not self.is_equivalent_to(other):
      raise TypesNotEquivalentError(self, other)

  def is_equivalent_to(self, other: 'Type') -> bool:
    """Returns whether values of `other` can be cast to and from this type."""
    return self.is_assignable_from(other) and other.is_assignable_from(self)

  def check_identical_to(self, other: 'Type') -> None:
    """Raises if `other` and `Type` are not exactly identical."""
    if not self.is_identical_to(other):
      raise TypesNotIdenticalError(self, other)

  def is_identical_to(self, other: 'Type') -> bool:
    """Returns whether or not `self` and `other` are exactly identical."""
    return self == other


class _Intern(abc.ABCMeta):
  """A metaclass which interns instances.

  This is used to create classes where the following predicate holds:
  `MyClass(some_args) is MyClass(some_args)`

  That is, objects of the class with the same constructor parameters result
  in values with the same object identity. This can make comparison of deep
  structures much cheaper, since a shallow equality check can short-circuit
  comparison.

  Classes which set `_Intern` as a metaclass must have a
  `_hashable_from_init_args` classmethod which defines exactly the parameters
  passed to the `__init__` method. If one of the parameters passed to the
  `_Intern.__call__` is an iterator it will be converted to a list before
  `_hashable_from_init_args` and `__init__` are called.

  Note: also that this metaclass must only be used with *immutable* values, as
  mutation would cause all similarly-constructed instances to be mutated
  together.

  Inherits from `abc.ABCMeta` to prevent subclass conflicts.
  """

  @classmethod
  def _hashable_from_init_args(mcs, *args, **kwargs) -> Hashable:
    raise NotImplementedError

  def __call__(cls, *args, **kwargs):

    # Convert all `Iterator`s in both `args` and `kwargs` to `list`s so they can
    # be used in both `_hashable_from_init_args` and `__init__`.
    def _normalize(obj):
      if isinstance(obj, Iterator):
        return list(obj)
      else:
        return obj

    args = [_normalize(x) for x in args]
    kwargs = {k: _normalize(v) for k, v in kwargs.items()}

    # Salt the key with `cls` to account for two different classes that return
    # the same result from `_hashable_from_init_args`.
    key = (cls, cls._hashable_from_init_args(*args, **kwargs))
    intern_pool = _intern_pool[cls]
    instance = intern_pool.get(key, None)
    if instance is None:
      instance = super().__call__(*args, **kwargs)
      intern_pool[key] = instance
    return instance


# A per-`typing.Type` map from `__init__` arguments to object instances.
#
# This is used by the `_Intern` metaclass to allow reuse of object instances
# when new objects are requested with the same `__init__` arguments as
# existing object instances.
#
# Implementation note: this double-map is used rather than a single map
# stored as a field of each class because some class objects themselves would
# begin destruction before the map fields of other classes, causing errors
# during destruction.
_intern_pool: dict[type[Type], dict[Hashable, Type]] = collections.defaultdict(
    dict
)


def _clear_intern_pool() -> None:
  # We must clear our `WeakKeyValueDictionary`s at the end of the program to
  # prevent Python from deleting the standard library out from under us before
  # removing the  entries from the dictionary. Yes, this is cursed.
  #
  # If this isn't done, Python will call `__eq__` on our types after
  # `abc.ABCMeta` has already been deleted from the world, resulting in
  # exceptions after main.
  global _intern_pool
  _intern_pool = None


atexit.register(_clear_intern_pool)


def _is_dtype_spec(dtype: object) -> bool:
  """Determines whether `dtype` is a representation of a TF or Numpy dtype.

  Args:
    dtype: The representation to check.

  Returns:
    Boolean result indicating whether `dtype` is a Numpy or TF dtype.
  """
  return (
      isinstance(dtype, tf.dtypes.DType)
      or isinstance(dtype, type)
      and issubclass(dtype, np.number)
      or isinstance(dtype, np.dtype)
  )


class TensorType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing types of tensors in TFF."""

  @classmethod
  def _hashable_from_init_args(
      cls, dtype: object, shape: Optional[object] = None
  ) -> Hashable:
    if not isinstance(dtype, tf.dtypes.DType):
      dtype = tf.dtypes.as_dtype(dtype)
    if shape is None:
      shape = tf.TensorShape([])
    elif not isinstance(shape, tf.TensorShape):
      shape = tf.TensorShape(shape)
    return (dtype, shape)

  def __init__(self, dtype: object, shape: Optional[object] = None):
    """Constructs a new instance from the given `dtype` and `shape`.

    Args:
      dtype: An instance of `tf.dtypes.DType` or one of the Numpy numeric types.
      shape: An optional instance of `tf.TensorShape` or an argument that can be
        passed to its constructor (such as a `list` or a `tuple`). `None` yields
        the default scalar shape.

    Raises:
      TypeError: if arguments are of the wrong types.
    """
    if not _is_dtype_spec(dtype):
      raise TypeError('Unrecognized dtype {}.'.format(str(dtype)))

    if not isinstance(dtype, tf.dtypes.DType):
      dtype = tf.dtypes.as_dtype(dtype)
    self._dtype = dtype
    if shape is None:
      shape = tf.TensorShape([])
    elif not isinstance(shape, tf.TensorShape):
      shape = tf.TensorShape(shape)
    self._shape = shape
    _check_well_formed(self)

  def children(self):
    return iter(())

  def is_tensor(self) -> bool:
    return True

  @property
  def dtype(self) -> tf.dtypes.DType:
    return self._dtype

  @property
  def shape(self) -> tf.TensorShape:
    return self._shape

  def __repr__(self):
    if self._shape.ndims is None:
      return 'TensorType({!r}, shape=None)'.format(self._dtype)
    elif self._shape.ndims > 0:
      values = self._shape.as_list()
      return 'TensorType({!r}, {!r})'.format(self._dtype, values)
    else:
      return 'TensorType({!r})'.format(self._dtype)

  def __hash__(self):
    return hash((self._dtype, self._shape))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, TensorType)
        and self._dtype == other.dtype
        and tensor_utils.same_shape(self._shape, other.shape)
    )

  def is_assignable_from(self, source_type: 'Type') -> bool:
    if self is source_type:
      return True
    if (
        not isinstance(source_type, TensorType)
        or self.dtype != source_type.dtype
    ):
      return False
    target_shape = self.shape
    source_shape = source_type.shape

    # TensorShape's `is_compatible_with` relation is nontransitive
    # a property TFF does not desire in its subtyping relation. So we implement
    # our own comparison between shapes below.
    if target_shape.rank is None:
      return True
    elif source_shape.rank is None:
      return False

    # The as_list call here is safe, as both TensorShapes have known rank.
    target_shape_list = target_shape.as_list()
    source_shape_list = source_shape.as_list()
    if len(target_shape_list) != len(source_shape_list):
      return False

    def _dimension_is_assignable_from(target_dim, source_dim):
      return (target_dim is None) or (target_dim == source_dim)

    return all(
        _dimension_is_assignable_from(target_shape_elem, source_shape_elem)
        for target_shape_elem, source_shape_elem in zip(
            target_shape_list, source_shape_list
        )
    )


def _format_struct_type_members(struct_type: 'StructType') -> str:
  def _element_repr(element):
    name, value = element
    if name is not None:
      return "('{}', {!r})".format(name, value)
    return repr(value)

  return ', '.join(
      _element_repr(e) for e in structure.iter_elements(struct_type)
  )


def _to_named_types(
    elements: Iterable[object],
) -> Iterable[tuple[Optional[str], Type]]:
  """Creates an `Iterable` of optionally named types from `elements`.

  This function creates an `Iterable` of optionally named types by iterating
  over `elements` and normalizing each element.

  If `elements` is an `Iterable` with named elements (e.g. `Mapping` or
  `NamedTuple`), the normalize element will have a name equal to the name of the
  element and a value equal to the value of the element convereted to a type
  using `to_type`.

  If `elements` is an `Iterable` with unnamed elements (e.g. list), the
  normalized element will have a name of `None` and a value equal to the element
  convereted to a type using `to_type`.

  Note: This function treats a single element being passed in as `elements` as
  if it were an iterable of that element.

  Args:
    elements: An iterable of named or unamed objects to convert to `tff.Types`.
      See `tff.types.to_type` for more infomration.

  Returns:
    An `Iterable` where each each element is `tuple[Optional[str], Type]`.
  """

  if py_typecheck.is_name_value_pair(elements, name_required=False):
    elements = [elements]
  elif py_typecheck.is_named_tuple(elements):
    elements = elements._asdict().items()  # pytype: disable=attribute-error
  elif isinstance(elements, Mapping):
    elements = elements.items()

  def _to_named_value_pair(element: object) -> tuple[Optional[str], Type]:
    if py_typecheck.is_name_value_pair(element, name_required=False):
      name, value = element  # pytype: disable=attribute-error
    else:
      name = None
      value = element
    value = to_type(value)
    return (name, value)

  return [_to_named_value_pair(x) for x in elements]


class StructType(structure.Struct, Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing structural types in TFF.

  Elements initialized by name can be accessed as `foo.name`, and otherwise by
  index, `foo[index]`.
  """

  @classmethod
  def _hashable_from_init_args(
      cls,
      elements: Iterable[object],
      *,
      convert: bool = True,
      enable_well_formed_check: bool = True,
  ) -> Hashable:
    del enable_well_formed_check  # Unused.

    if convert:
      elements = _to_named_types(elements)
    return (tuple(elements), convert)

  def __init__(
      self,
      elements: Iterable[object],
      *,
      convert: bool = True,
      enable_well_formed_check: bool = True,
  ):
    """Constructs a new instance from the given element types.

    Args:
      elements: An iterable of element specifications. Each element
        specification is either a type spec (an instance of `tff.Type` or
        something convertible to it via `tff.types.to_type`) for the element, or
        a (name, spec) for elements that have defined names. Alternatively, one
        can supply here an instance of `collections.OrderedDict` mapping element
        names to their types (or things that are convertible to types).
      convert: A flag to determine if the elements should be converted using
        `tff.types.to_type` or not.
      enable_well_formed_check: This flag exists only so that
        `StructWithPythonType` can disable the well-formedness check, as the
        type won't be well-formed until the subclass has finished its own
        initialization.
    """
    py_typecheck.check_type(elements, Iterable)
    if convert:
      elements = _to_named_types(elements)
    structure.Struct.__init__(self, elements)
    if enable_well_formed_check:
      _check_well_formed(self)

  def children(self):
    return (element for _, element in structure.iter_elements(self))

  @property
  def python_container(self) -> Optional[type[object]]:
    return None

  def is_struct(self) -> bool:
    return True

  def __repr__(self):
    members = _format_struct_type_members(self)
    return f'StructType([{members}])'

  def __hash__(self):
    # Salt to avoid overlap.
    return hash((structure.Struct.__hash__(self), 'NTT'))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, StructType) and structure.Struct.__eq__(self, other)
    )

  def is_assignable_from(self, source_type: 'Type') -> bool:
    if self is source_type:
      return True
    if not isinstance(source_type, StructType):
      return False
    target_elements = structure.to_elements(self)
    source_elements = structure.to_elements(source_type)
    return (len(target_elements) == len(source_elements)) and all(
        (
            (source_elements[k][0] in [target_elements[k][0], None])
            and target_elements[k][1].is_assignable_from(source_elements[k][1])
        )
        for k in range(len(target_elements))
    )


class StructWithPythonType(StructType, metaclass=_Intern):
  """A representation of a structure paired with a Python container type."""

  @classmethod
  def _hashable_from_init_args(
      cls, elements: Iterable[object], container_type: type[object]
  ) -> Hashable:
    named_types = _to_named_types(elements)
    return (tuple(named_types), container_type)

  def __init__(self, elements: Iterable[object], container_type: type[object]):
    # We don't want to check our type for well-formedness until after we've
    # set `_container_type`.
    super().__init__(elements, enable_well_formed_check=False)
    self._container_type = container_type
    _check_well_formed(self)

  def is_struct_with_python(self) -> bool:
    return True

  @property
  def python_container(self) -> type[object]:
    return self._container_type

  def __repr__(self):
    members = _format_struct_type_members(self)
    return 'StructType([{}]) as {}'.format(
        members, self._container_type.__name__
    )

  def __hash__(self):
    # Salt to avoid overlap.
    return hash((structure.Struct.__hash__(self), 'NTTWPCT'))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, StructWithPythonType)
        and (self._container_type == other._container_type)
        and structure.Struct.__eq__(self, other)
    )


class SequenceType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing types of sequences in TFF.

  IMPORTANT: since `SequenceType` is frequently backed by `tf.data.Dataset`
  which converts `list` to `tuple`, any `SequenceType` constructed with
  `StructWithPythonType` elements will convert any `list` python container type
  to `tuple` python container types for interoperability.
  """

  @classmethod
  def _hashable_from_init_args(cls, element: object) -> Hashable:
    element = to_type(element)
    return (element,)

  def __init__(self, element: object):
    """Constructs a new instance from the given `element` type.

    Args:
      element: A specification of the element type, either an instance of
        `tff.Type` or something convertible to it by `tff.types.to_type`.
    """

    def convert_struct_with_list_to_struct_with_tuple(type_spec: T) -> T:
      """Convert any StructWithPythonType using lists to use tuples."""
      # We ignore non-struct, non-tensor types, these are not well formed types
      # for sequence elements.
      if not type_spec.is_struct():
        return type_spec
      elements = [
          (name, convert_struct_with_list_to_struct_with_tuple(value))
          for name, value in structure.iter_elements(type_spec)  # pytype: disable=wrong-arg-types
      ]
      if not isinstance(type_spec, StructWithPythonType):
        return StructType(elements=elements)
      container_cls = type_spec.python_container
      return StructWithPythonType(
          elements=elements,
          container_type=tuple if container_cls is list else container_cls,
      )

    type_spec = to_type(element)
    self._element = convert_struct_with_list_to_struct_with_tuple(type_spec)
    _check_well_formed(self)

  def children(self):
    yield self._element

  def is_sequence(self) -> bool:
    return True

  @property
  def element(self) -> Type:
    return self._element

  def __repr__(self):
    return 'SequenceType({!r})'.format(self._element)

  def __hash__(self):
    return hash(self._element)

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, SequenceType) and self._element == other.element
    )

  def is_assignable_from(self, source_type: 'Type') -> bool:
    if self is source_type:
      return True
    return isinstance(
        source_type, SequenceType
    ) and self.element.is_assignable_from(source_type.element)


class FunctionType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing functional types in TFF."""

  @classmethod
  def _hashable_from_init_args(
      cls, parameter: Optional[object], result: object
  ) -> Hashable:
    if parameter is not None:
      parameter = to_type(parameter)
    result = to_type(result)
    return (parameter, result)

  def __init__(self, parameter: Optional[object], result: object):
    """Constructs a new instance from the given `parameter` and `result` types.

    Args:
      parameter: A specification of the parameter type, either an instance of
        `tff.Type` or something convertible to it by `tff.types.to_type`.
        Multiple input arguments can be specified as a single `tff.StructType`.
      result: A specification of the result type, either an instance of
        `tff.Type` or something convertible to it by `tff.types.to_type`.
    """
    if parameter is not None:
      parameter = to_type(parameter)
    self._parameter = parameter
    self._result = to_type(result)
    _check_well_formed(self)

  def children(self):
    if self._parameter is not None:
      yield self._parameter
    yield self._result

  def is_function(self) -> bool:
    return True

  @property
  def parameter(self) -> Optional[Type]:
    return self._parameter

  @property
  def result(self) -> Type:
    return self._result

  def __repr__(self):
    return 'FunctionType({!r}, {!r})'.format(self._parameter, self._result)

  def __hash__(self):
    return hash((self._parameter, self._result))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, FunctionType)
        and self._parameter == other.parameter
        and self._result == other.result
    )

  def is_assignable_from(self, source_type: 'Type') -> bool:
    if self is source_type:
      return True
    if not isinstance(source_type, FunctionType):
      return False
    if (self.parameter is None) != (source_type.parameter is None):
      return False
    # Note that function parameters are contravariant, so we invert the check.
    if (
        self.parameter is not None
        and not source_type.parameter.is_assignable_from(self.parameter)
    ):
      return False
    return self.result.is_assignable_from(source_type.result)


class AbstractType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing abstract types in TFF."""

  @classmethod
  def _hashable_from_init_args(cls, label: str) -> Hashable:
    return (label,)

  def __init__(self, label: str):
    """Constructs a new instance from the given string `label`.

    Args:
      label: A string label of an abstract type. All occurences of the label
        within a computation's type signature refer to the same concrete type.
    """
    py_typecheck.check_type(label, str)
    self._label = label
    _check_well_formed(self)

  def children(self):
    return iter(())

  def is_abstract(self) -> bool:
    return True

  @property
  def label(self) -> str:
    return self._label

  def __repr__(self):
    return "AbstractType('{}')".format(self._label)

  def __hash__(self):
    return hash(self._label)

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, AbstractType) and self._label == other.label
    )

  def is_assignable_from(self, source_type: 'Type') -> bool:
    del source_type  # Unused.
    # TODO(b/113112108): Revise this to extend the relation of assignability to
    # abstract types.
    raise TypeError('Abstract types are not comparable.')


class PlacementType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing the placement type in TFF.

  There is only one placement type, a TFF built-in, just as there is only one
  `int` or `str` type in Python. All instances of this class represent the same
  built-in TFF placement type.
  """

  @classmethod
  def _hashable_from_init_args(cls) -> Hashable:
    return ()

  def __init__(self):
    _check_well_formed(self)

  def children(self):
    return iter(())

  def is_placement(self) -> bool:
    return True

  def __repr__(self):
    return 'PlacementType()'

  def __hash__(self):
    return 0

  def __eq__(self, other):
    return (self is other) or isinstance(other, PlacementType)

  def is_assignable_from(self, source_type: 'Type') -> bool:
    if self is source_type:
      return True
    return isinstance(source_type, PlacementType)


class FederatedType(Type, metaclass=_Intern):
  """An implementation of `tff.Type` representing federated types in TFF."""

  @classmethod
  def _hashable_from_init_args(
      cls,
      member: object,
      placement: placements.PlacementLiteral,
      all_equal: Optional[bool] = None,
  ) -> Hashable:
    member = to_type(member)
    return (member, placement, all_equal)

  def __init__(
      self,
      member: object,
      placement: placements.PlacementLiteral,
      all_equal: Optional[bool] = None,
  ):
    """Constructs a new federated type instance.

    Args:
      member: An instance of `tff.Type` or something convertible to it, that
        represents the type of the member components of each value of this
        federated type.
      placement: The specification of placement that the member components of
        this federated type are hosted on. Must be either a placement literal
        such as `tff.SERVER` or `tff.CLIENTS` to refer to a globally defined
        placement, or a placement label to refer to a placement defined in other
        parts of a type signature. Specifying placement labels is not
        implemented yet.
      all_equal: A `bool` value that indicates whether all members of the
        federated type are equal (`True`), or are allowed to differ (`False`).
        If `all_equal` is `None`, the value is selected as the default for the
        placement, e.g., `True` for `tff.SERVER` and `False` for `tff.CLIENTS`.
    """
    self._member = to_type(member)
    self._placement = placement
    if all_equal is None:
      all_equal = placement.default_all_equal
    self._all_equal = all_equal
    _check_well_formed(self)

  # TODO(b/113112108): Extend this to support federated types parameterized
  # by abstract placement labels, such as those used in generic types of
  # federated operators.

  def children(self):
    yield self._member

  def is_federated(self) -> bool:
    return True

  @property
  def member(self) -> Type:
    return self._member

  @property
  def placement(self) -> placements.PlacementLiteral:
    return self._placement

  @property
  def all_equal(self) -> bool:
    return self._all_equal

  def __repr__(self):
    return 'FederatedType({!r}, {!r}, {!r})'.format(
        self._member, self._placement, self._all_equal
    )

  def __hash__(self):
    return hash((self._member, self._placement, self._all_equal))

  def __eq__(self, other):
    return (self is other) or (
        isinstance(other, FederatedType)
        and self._member == other.member
        and self._placement == other.placement
        and self._all_equal == other.all_equal
    )

  def is_assignable_from(self, source_type: 'Type') -> bool:
    if self is source_type:
      return True
    return (
        isinstance(source_type, FederatedType)
        and self.member.is_assignable_from(source_type.member)
        and (not self.all_equal or source_type.all_equal)
        and self.placement is source_type.placement
    )


def at_server(type_spec: object) -> FederatedType:
  """Constructs a federated type of the form `T@SERVER`.

  Args:
    type_spec: An instance of `tff.Type` or something convertible to it.

  Returns:
    The type of the form `T@SERVER`, where `T` is the `type_spec`.
  """
  return FederatedType(type_spec, placements.SERVER, all_equal=True)


def at_clients(type_spec: object, all_equal: bool = False) -> FederatedType:
  """Constructs a federated type of the form `{T}@CLIENTS`.

  Args:
    type_spec: An instance of `tff.Type` or something convertible to it.
    all_equal: The `all_equal` bit, `False` by default.

  Returns:
    The type of the form `{T}@CLIENTS` (by default) or `T@CLIENTS` (if specified
    by setting the `all_equal` bit), where `T` is the `type_spec`.
  """
  return FederatedType(type_spec, placements.CLIENTS, all_equal=all_equal)


def to_type(obj: object) -> Type:
  """Converts the argument into an instance of `tff.Type`.

  Examples of arguments convertible to tensor types:

  ```python
  tf.int32
  (tf.int32, [10])
  (tf.int32, [None])
  np.int32
  ```

  Examples of arguments convertible to flat named tuple types:

  ```python
  [tf.int32, tf.bool]
  (tf.int32, tf.bool)
  [('a', tf.int32), ('b', tf.bool)]
  ('a', tf.int32)
  collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
  ```

  Examples of arguments convertible to nested named tuple types:

  ```python
  (tf.int32, (tf.float32, tf.bool))
  (tf.int32, (('x', tf.float32), tf.bool))
  ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
  ```

  `attr.s` class instances can also be used to describe TFF types by populating
  the fields with the corresponding types:

  ```python
  @attr.s(auto_attribs=True)
  class MyDataClass:
    int_scalar: tf.Tensor
    string_array: tf.Tensor

    @classmethod
    def tff_type(cls) -> tff.Type:
      return tff.types.to_type(cls(
        int_scalar=tf.int32,
        string_array=tf.TensorSpec(dtype=tf.string, shape=[3]),
      ))

  @tff.tf_computation(MyDataClass.tff_type())
  def work(my_data):
    assert isinstance(my_data, MyDataClass)
    ...
  ```

  Args:
    obj: Either an instance of `tff.Type`, or an argument convertible to
      `tff.Type`.

  Returns:
    An instance of `tff.Type` corresponding to the given `obj`.
  """
  # TODO(b/113112108): Add multiple examples of valid type specs here in the
  # comments, in addition to the unit test.
  if isinstance(obj, Type):
    return obj
  elif _is_dtype_spec(obj):
    return TensorType(obj)
  elif isinstance(obj, tf.TensorSpec):
    return TensorType(obj.dtype, obj.shape)
  elif isinstance(obj, tf.data.DatasetSpec):
    return SequenceType(element=to_type(obj.element_spec))
  elif (
      isinstance(obj, tuple)
      and (len(obj) == 2)
      and _is_dtype_spec(obj[0])
      and (
          isinstance(obj[1], tf.TensorShape)
          or (
              isinstance(obj[1], (list, tuple))
              and all((isinstance(x, int) or x is None) for x in obj[1])
          )
      )
  ):
    # We found a 2-element tuple of the form (dtype, shape), where dtype is an
    # instance of tf.dtypes.DType, and shape is either an instance of
    # tf.TensorShape, or a list, or a tuple that can be fed as argument into a
    # tf.TensorShape. We thus convert this into a TensorType.
    return TensorType(obj[0], obj[1])
  elif isinstance(obj, (list, tuple)):
    if any(py_typecheck.is_name_value_pair(e) for e in obj):
      # The sequence has a (name, value) elements, the whole sequence is most
      # likely intended to be a `Struct`, do not store the Python
      # container.
      return StructType(obj)
    else:
      return StructWithPythonType(obj, type(obj))
  elif isinstance(obj, collections.OrderedDict):
    return StructWithPythonType(obj, type(obj))
  elif py_typecheck.is_attrs(obj):
    return _to_type_from_attrs(obj)
  elif isinstance(obj, Mapping):
    # This is an unsupported mapping, likely a `dict`. StructType adds an
    # ordering, which the original container did not have.
    raise TypeError(
        'Unsupported mapping type {}. Use collections.OrderedDict for '
        'mappings.'.format(py_typecheck.type_string(type(obj)))
    )
  elif isinstance(obj, structure.Struct):
    return StructType(structure.to_elements(obj))
  elif isinstance(obj, tf.RaggedTensorSpec):
    if obj.flat_values_spec is not None:
      flat_values_type = to_type(obj.flat_values_spec)
    else:
      # We could provide a more specific shape here if `obj.shape is not None`:
      # `flat_values_shape = [None] + obj.shape[obj.ragged_rank + 1:]`
      # However, we can't go back from this type into a `tf.RaggedTensorSpec`,
      # meaning that round-tripping a `tf.RaggedTensorSpec` through
      # `type_conversions.type_to_tf_structure(to_type(obj))`
      # would *not* be a no-op: it would clear away the extra shape information,
      # leading to compilation errors. This round-trip is tested in
      # `type_conversions_test.py` to ensure correctness.
      flat_values_shape = tf.TensorShape(None)
      flat_values_type = TensorType(obj.dtype, flat_values_shape)
    nested_row_splits_type = StructWithPythonType(
        ([(None, TensorType(obj.row_splits_dtype, [None]))] * obj.ragged_rank),
        tuple,
    )
    return StructWithPythonType(
        [
            ('flat_values', flat_values_type),
            ('nested_row_splits', nested_row_splits_type),
        ],
        tf.RaggedTensor,
    )
  elif isinstance(obj, tf.SparseTensorSpec):
    dtype = obj.dtype
    shape = obj.shape
    unknown_num_values = None
    rank = None if shape is None else shape.rank
    return StructWithPythonType(
        [
            ('indices', TensorType(tf.int64, [unknown_num_values, rank])),
            ('values', TensorType(dtype, [unknown_num_values])),
            ('dense_shape', TensorType(tf.int64, [rank])),
        ],
        tf.SparseTensor,
    )
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a type spec.'.format(
            py_typecheck.type_string(type(obj))
        )
    )


def _to_type_from_attrs(spec) -> StructWithPythonType:
  """Converts an `attr.s` class or instance to a `tff.Type`."""
  if isinstance(spec, type):
    # attrs class type
    raise TypeError(
        'Converting `attr` classes to a federated type is no longer supported. '
        'Either populate an instance of the `attr.s` class with the '
        'appropriate field types, or use one of the other forms described in '
        '`tff.types.to_type()` instead.'
    )
  else:
    # attrs class instance, inspect the field values for instances convertible
    # to types.
    elements = attr.asdict(
        spec, dict_factory=collections.OrderedDict, recurse=False
    )
    the_type = type(spec)

  return StructWithPythonType(elements, the_type)


@attr.s(auto_attribs=True, slots=True)
class _PossiblyDisallowedChildren:
  """A set of possibly disallowed types contained within a type.

  These kinds of types may be banned in one or more contexts, so
  `_possibly_disallowed_members` and its cache record which of these type kinds
  appears inside a type, allowing for quick well-formedness checks.
  """

  federated: Optional[Type]
  function: Optional[Type]
  sequence: Optional[Type]


# Manual cache used rather than `cachetools.cached` due to incompatibility
# with `WeakKeyDictionary`. We want to use a `WeakKeyDictionary` so that
# cache entries are destroyed once the types they index no longer exist.
_possibly_disallowed_children_cache = weakref.WeakKeyDictionary({})


def _clear_disallowed_cache():
  # We must clear our `WeakKeyValueDictionary`s at the end of the program to
  # prevent Python from deleting the standard library out from under us before
  # removing the  entries from the dictionary. Yes, this is cursed.
  #
  # If this isn't done, Python will call `__eq__` on our types after
  # `abc.ABCMeta` has already been deleted from the world, resulting in
  # exceptions after main.
  global _possibly_disallowed_children_cache
  _possibly_disallowed_children_cache = None


atexit.register(_clear_disallowed_cache)


def _possibly_disallowed_children(
    type_signature: Type,
) -> _PossiblyDisallowedChildren:
  """Returns possibly disallowed child types appearing in `type_signature`."""
  cached = _possibly_disallowed_children_cache.get(type_signature, None)  # pytype: disable=attribute-error
  if cached:
    return cached
  disallowed = _PossiblyDisallowedChildren(None, None, None)
  for child_type in type_signature.children():
    if child_type is None:
      raise ValueError(type_signature)
    if child_type.is_federated():
      disallowed = attr.evolve(disallowed, federated=child_type)
    elif child_type.is_function():
      disallowed = attr.evolve(disallowed, function=child_type)
    elif child_type.is_sequence():
      disallowed = attr.evolve(disallowed, sequence=child_type)
    from_grandchildren = _possibly_disallowed_children(child_type)
    disallowed = _PossiblyDisallowedChildren(
        federated=disallowed.federated or from_grandchildren.federated,
        function=disallowed.function or from_grandchildren.function,
        sequence=disallowed.sequence or from_grandchildren.sequence,
    )
  _possibly_disallowed_children_cache[type_signature] = (
      disallowed  # pytype: disable=unsupported-operands
  )
  return disallowed


_FEDERATED_TYPES = 'federated types (types placed @CLIENT or @SERVER)'
_FUNCTION_TYPES = 'function types'
_SEQUENCE_TYPES = 'sequence types'


def _check_well_formed(type_signature: Type):
  """Checks `type_signature`'s validity. Assumes that child types are valid."""

  def _check_disallowed(disallowed_type, disallowed_kind, context):
    if disallowed_type is None:
      return
    raise TypeError(
        f'{disallowed_type} has been encountered in the type {type_signature}. '
        f'{disallowed_kind} are disallowed inside of {context}.'
    )

  children = _possibly_disallowed_children(type_signature)

  if type_signature.is_federated():
    # Federated types cannot have federated or functional children.
    for child_type, kind in (
        (children.federated, _FEDERATED_TYPES),
        (children.function, _FUNCTION_TYPES),
    ):
      _check_disallowed(child_type, kind, _FEDERATED_TYPES)
  elif type_signature.is_sequence():
    # Sequence types cannot have federated, functional, or sequence children.
    for child_type, kind in (
        (children.federated, _FEDERATED_TYPES),
        (children.function, _FUNCTION_TYPES),
        (children.sequence, _SEQUENCE_TYPES),
    ):
      _check_disallowed(child_type, kind, _SEQUENCE_TYPES)


def _string_representation(type_spec, formatted: bool) -> str:
  """Returns the string representation of a TFF `Type`.

  This function creates a `list` of strings representing the given `type_spec`;
  combines the strings in either a formatted or un-formatted representation; and
  returns the resulting string representation.

  Args:
    type_spec: An instance of a TFF `Type`.
    formatted: A boolean indicating if the returned string should be formatted.

  Raises:
    TypeError: If `type_spec` has an unexpected type.
  """
  py_typecheck.check_type(type_spec, Type)

  def _combine(components):
    """Returns a `list` of strings by combining `components`.

    This function creates and returns a `list` of strings by combining a `list`
    of `components`. Each `component` is a `list` of strings representing a part
    of the string of a TFF `Type`. The `components` are combined by iteratively
    **appending** the last element of the result with the first element of the
    `component` and then **extending** the result with remaining elements of the
    `component`.

    For example:

    >>> _combine([['a'], ['b'], ['c']])
    ['abc']

    >>> _combine([['a', 'b', 'c'], ['d', 'e', 'f']])
    ['abcd', 'ef']

    This function is used to help track where new-lines should be inserted into
    the string representation if the lines are formatted.

    Args:
      components: A `list` where each element is a `list` of strings
        representing a part of the string of a TFF `Type`.
    """
    lines = ['']
    for component in components:
      lines[-1] = '{}{}'.format(lines[-1], component[0])
      lines.extend(component[1:])
    return lines

  def _indent(lines, indent_chars='  '):
    """Returns an indented `list` of strings."""
    return ['{}{}'.format(indent_chars, e) for e in lines]

  def _lines_for_named_types(named_type_specs, formatted):
    """Returns a `list` of strings representing the given `named_type_specs`.

    Args:
      named_type_specs: A `list` of named comutations, each being a pair
        consisting of a name (either a string, or `None`) and a
        `ComputationBuildingBlock`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    lines = []
    for index, (name, type_spec) in enumerate(named_type_specs):
      if index != 0:
        if formatted:
          lines.append([',', ''])
        else:
          lines.append([','])
      element_lines = _lines_for_type(type_spec, formatted)
      if name is not None:
        element_lines = _combine([
            ['{}='.format(name)],
            element_lines,
        ])
      lines.append(element_lines)
    return _combine(lines)

  def _lines_for_type(type_spec, formatted):
    """Returns a `list` of strings representing the given `type_spec`.

    Args:
      type_spec: An instance of a TFF `Type`.
      formatted: A boolean indicating if the returned string should be
        formatted.
    """
    if type_spec.is_abstract():
      return [type_spec.label]
    elif type_spec.is_federated():
      member_lines = _lines_for_type(type_spec.member, formatted)
      placement_line = '@{}'.format(type_spec.placement)
      if type_spec.all_equal:
        return _combine([member_lines, [placement_line]])
      else:
        return _combine([['{'], member_lines, ['}'], [placement_line]])
    elif type_spec.is_function():
      if type_spec.parameter is not None:
        parameter_lines = _lines_for_type(type_spec.parameter, formatted)
      else:
        parameter_lines = ['']
      result_lines = _lines_for_type(type_spec.result, formatted)
      return _combine([['('], parameter_lines, [' -> '], result_lines, [')']])
    elif type_spec.is_struct():
      if not type_spec:
        return ['<>']
      elements = structure.to_elements(type_spec)
      elements_lines = _lines_for_named_types(elements, formatted)
      if formatted:
        elements_lines = _indent(elements_lines)
        lines = [['<', ''], elements_lines, ['', '>']]
      else:
        lines = [['<'], elements_lines, ['>']]
      return _combine(lines)
    elif type_spec.is_placement():
      return ['placement']
    elif type_spec.is_sequence():
      element_lines = _lines_for_type(type_spec.element, formatted)
      return _combine([element_lines, ['*']])
    elif type_spec.is_tensor():
      if type_spec.shape.ndims is None:
        return ['{!r}(shape=None)'.format(type_spec.dtype.name)]
      elif type_spec.shape.ndims > 0:

        def _value_string(value):
          return str(value) if value is not None else '?'

        value_strings = [_value_string(e.value) for e in type_spec.shape.dims]
        values_strings = ','.join(value_strings)
        return ['{}[{}]'.format(type_spec.dtype.name, values_strings)]
      else:
        return [type_spec.dtype.name]
    else:
      raise NotImplementedError(
          'Unexpected type found: {}.'.format(type(type_spec))
      )

  lines = _lines_for_type(type_spec, formatted)
  lines = [line.rstrip() for line in lines]
  if formatted:
    return '\n'.join(lines)
  else:
    return ''.join(lines)
