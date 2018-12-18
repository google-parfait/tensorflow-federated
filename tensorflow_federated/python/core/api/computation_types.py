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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

# Dependency imports

import six
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import placement_literals


@six.add_metaclass(abc.ABCMeta)
class Type(object):
  """An abstract interface for all classes that represent TFF types."""

  @abc.abstractmethod
  def is_assignable_from(self, other):
    """Determines whether this TFF type is assignable from another TFF type.

    Args:
      other: Another type, an instance of `Type`.

    Returns:
      `True` if self is assignable from other, `False` otherwise.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this type."""
    raise NotImplementedError

  @abc.abstractmethod
  def __str__(self):
    """Returns a concise representation of this type."""
    raise NotImplementedError

  # Types are partially ordered by the relation of assignability: if type T is
  # assignable from type T', so T' is more narrowly specialized than T, we say
  # that T >= T'.

  def __ge__(self, other):
    return self.is_assignable_from(other)

  def __le__(self, other):
    return other.__ge__(self)

  def __eq__(self, other):
    return self.__ge__(other) and self.__le__(other)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __gt__(self, other):
    return self.__ge__(other) and not self.__le__(other)

  def __lt__(self, other):
    return self.__le__(other) and not self.__ge__(other)


class TensorType(Type):
  """An implementation of `Type` for representing types of tensors in TFF."""

  def __init__(self, dtype, shape=None):
    """Constructs a new instance from the given dtype and shape.

    Args:
      dtype: An instance of `tf.DType`.
      shape: An optional instance of `tf.TensorShape` or an argument that can be
        passed to its constructor (such as a `list` or a `tuple`), or `None` for
        the default scalar shape. Unspecified shapes are not supported.

    Raises:
      TypeError: if arguments are of the wrong types.
    """
    py_typecheck.check_type(dtype, tf.DType)
    self._dtype = dtype
    if shape is None:
      self._shape = tf.TensorShape([])
    elif isinstance(shape, tf.TensorShape):
      self._shape = shape
    else:
      self._shape = tf.TensorShape(shape)

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)

    def _shape_is_assignable_from(x, y):

      def _dimension_is_assignable_from(x, y):
        # Either the first dimension is undefined or it has the same size as
        # the second.
        return (x.value is None) or (x.value == y.value)

      # Shapes must have equal ranks, and all dimensions in first have to be
      # assignable from the corresponding dimensions in the second.
      return ((x.ndims == y.ndims) and ((x.dims is None) or all(
          _dimension_is_assignable_from(x.dims[k], y.dims[k])
          for k in range(x.ndims))))

    return (isinstance(other, TensorType) and (self.dtype == other.dtype) and
            _shape_is_assignable_from(self.shape, other.shape))

  def __repr__(self):
    if self._shape.ndims > 0:
      values = repr([dim.value for dim in self._shape.dims])
      return 'TensorType({}, {})'.format(repr(self._dtype), values)
    else:
      return 'TensorType({})'.format(repr(self._dtype))

  def __str__(self):
    if self._shape.ndims > 0:
      values = [
          str(dim.value) if dim.value is not None else '?'
          for dim in self._shape.dims
      ]
      return '{}[{}]'.format(self._dtype.name, ','.join(values))
    else:
      return self._dtype.name


class NamedTupleType(Type):
  """An implementation of `Type` for representing named tuple types in TFF."""

  def __init__(self, elements):
    """Constructs a new instance from the given element types.

    Args:
      elements: A list of element specifications. Each element specification is
        either a type spec (an instance of `Type` or something convertible to it
        via `to_type()`) for the element, or a pair (name, spec) for elements
        that have defined names. Alternatively, one can supply here an instance
        of `collections.OrderedDict` mapping element names to their types (or
        things that are convertible to types).

    Raises:
      TypeError: if the arguments are of the wrong types.
      ValueError: if the named tuple contains no elements.
    """
    py_typecheck.check_type(elements, (list, tuple, collections.OrderedDict))
    if '_asdict' in vars(type(elements)):
      elements = elements._asdict()
    if isinstance(elements, collections.OrderedDict):
      elements = elements.items()
    if not elements:
      raise ValueError('A named tuple must contain at least one element.')

    def _is_named_element(e):
      return (isinstance(e, tuple) and (len(e) == 2) and
              isinstance(e[0], six.string_types))

    def _map_element(e):
      if isinstance(e, Type):
        return (None, e)
      elif _is_named_element(e):
        return (e[0], to_type(e[1]))
      else:
        return (None, to_type(e))

    if _is_named_element(elements):
      self._elements = [(elements[0], to_type(elements[1]))]
    else:
      self._elements = [_map_element(e) for e in elements]

  @property
  def elements(self):
    # Shallow copy to prevent accidental modification by the caller.
    return list(self._elements)

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)
    if not isinstance(other, NamedTupleType):
      return False
    other_elements = other.elements
    return ((len(self._elements) == len(other_elements)) and all(
        ((self._elements[k][0] in [other_elements[k][0], None]) and
         (self._elements[k][1].is_assignable_from(other_elements[k][1])))
        for k in range(len(self._elements))))

  def __repr__(self):

    def _element_repr(e):
      if e[0] is not None:
        return '(\'{}\', {})'.format(e[0], repr(e[1]))
      else:
        return repr(e[1])

    return 'NamedTupleType([{}])'.format(', '.join(
        [_element_repr(e) for e in self._elements]))

  def __str__(self):

    def _element_str(name, value):
      return '{}={}'.format(name, str(value)) if name else str(value)

    return ('<{}>'.format(','.join(
        [_element_str(name, value) for name, value in self._elements])))


class SequenceType(Type):
  """An implementation of `Type` for representing types of sequences in TFF."""

  def __init__(self, element):
    """Constructs a new instance from the given element type.

    Args:
      element: A specification of the element type, either an instance of `Type`
        or something convertible to it by `to_type()`.
    """
    self._element = to_type(element)

  @property
  def element(self):
    return self._element

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)
    return (isinstance(other, SequenceType) and
            self.element.is_assignable_from(other.element))

  def __repr__(self):
    return 'SequenceType({})'.format(repr(self._element))

  def __str__(self):
    return '{}*'.format(str(self._element))


class FunctionType(Type):
  """An implementation of `Type` for representing functional types in TFF."""

  def __init__(self, parameter, result):
    """Constructs a new instance from the given parameter and result types.

    Args:
      parameter: A specification of the parameter type, either an instance of
        `Type` or something convertible to it by `to_type()`.
      result: A specification of the result type, either an instance of `Type`
        or something convertible to it by `to_type()`.
    """
    self._parameter = to_type(parameter)
    self._result = to_type(result)

  @property
  def parameter(self):
    return self._parameter

  @property
  def result(self):
    return self._result

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)
    return (isinstance(other, FunctionType) and
            (((other.parameter is None) and (self.parameter is None)) or
             ((other.parameter is not None) and (self.parameter is not None) and
              other.parameter.is_assignable_from(self.parameter)) and
             self.result.is_assignable_from(other.result)))

  def __repr__(self):
    return 'FunctionType({}, {})'.format(
        repr(self._parameter), repr(self._result))

  def __str__(self):
    return '({} -> {})'.format(
        str(self._parameter) if self._parameter is not None else '',
        str(self._result))


class AbstractType(Type):
  """An implementation of `Type` for representing abstract types in TFF."""

  def __init__(self, label):
    """Constructs a new instance from the given string label.

    Args:
      label: A string label of an abstract type. All occurences of the label
        within a computation's type signature refer to the same concrete type.
    """
    py_typecheck.check_type(label, six.string_types)
    self._label = str(label)

  @property
  def label(self):
    return self._label

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)

    # TODO(b/113112108): Revise this to extend the relation of assignability to
    # abstract types.
    raise ValueError('Abstract types are not comparable.')

  def __repr__(self):
    return 'AbstractType(\'{}\')'.format(self._label)

  def __str__(self):
    return self._label


class PlacementType(Type):
  """An implementation of `Type` for representing the placement type in TFF.

  There is only one placement type, a TFF built-in, just as there is only one
  `int` or `str` type in Python. All instances of this class represent the same
  built-in TFF placement type.
  """

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)
    return isinstance(other, PlacementType)

  def __repr__(self):
    return 'PlacementType()'

  def __str__(self):
    return 'placement'


class FederatedType(Type):
  """An implementation of `Type` for representing federated types in TFF."""

  def __init__(self, member, placement, all_equal=False):
    """Constructs a new federated type instance.

    Args:
      member: An instance of `Type` (or something convertible to it) that
        represents the type of the member components of each value of this
        federated type.
      placement: The specification of placement that the member components of
        this federated type are hosted on. Must be either a placement literal
        such as `SERVER` or `CLIENTS` to refer to a globally defined placement,
        or a placement label to refer to a placement defined in other parts of a
        type signature. Specifying placement labels is not implemented yet.
      all_equal: A `bool` value that indicates whether all members of the
        federated type are equal (`True`), or are allowed to differ (`False`).
    """
    if not isinstance(placement, placement_literals.PlacementLiteral):
      raise NotImplementedError(
          'At the moment, only specifying placement literals is implemented.')
    py_typecheck.check_type(all_equal, bool)
    self._member = to_type(member)
    self._placement = placement
    self._all_equal = all_equal

  # TODO(b/113112108): Extend this to support federated types parameterized
  # by abstract placement labels, such as those used in generic types of
  # federated operators.

  @property
  def member(self):
    return self._member

  @property
  def placement(self):
    return self._placement

  @property
  def all_equal(self):
    return self._all_equal

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)
    if (not isinstance(other, FederatedType) or
        not self._member.is_assignable_from(other.member) or
        self._all_equal and not other.all_equal):
      return False
    for val in [self, other]:
      py_typecheck.check_type(val.placement,
                              placement_literals.PlacementLiteral)
    return self.placement is other.placement

  def __repr__(self):
    return 'FederatedType({}, {}, {})'.format(
        repr(self._member), repr(self._placement), repr(self._all_equal))

  def __str__(self):
    if self._all_equal:
      return '{}@{}'.format(str(self._member), str(self._placement))
    else:
      return '{{{}}}@{}'.format(str(self._member), str(self._placement))


def to_type(spec):
  # pyformat: disable
  """Converts the argument into an instance of `Type`.

  Args:
    spec: Either an instance of `Type`, or an argument convertible to Type.
      Assorted examples of type specifications are included below.

      Examples of arguments convertible to tensor types:

      ```
      tf.int32
      (tf.int32, [10])
      (tf.int32, [None])
      ```

      Examples of arguments convertible to flat named tuple types:

      ```
      [tf.int32, tf.bool]
      (tf.int32, tf.bool)
      [('a', tf.int32), ('b', tf.bool)]
      ('a', tf.int32)
      collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])
      ```

      Examples of arguments convertible to nested named tuple types:

      ```
      (tf.int32, (tf.float32, tf.bool))
      (tf.int32, (('x', tf.float32), tf.bool))
      ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))
      ```

  Returns:
    An instance of `Type` corresponding to the given spec.
  """
  # pyformat: enable
  # TODO(b/113112108): Add multiple examples of valid type specs here in the
  # comments, in addition to the unit test.

  if isinstance(spec, Type) or spec is None:
    return spec
  elif isinstance(spec, tf.DType):
    return TensorType(spec)
  elif (isinstance(spec, tuple) and (len(spec) == 2) and
        isinstance(spec[0], tf.DType) and
        (isinstance(spec[1], tf.TensorShape) or
         (isinstance(spec[1], (list, tuple)) and all(
             (isinstance(x, int) or x is None) for x in spec[1])))):
    # We found a 2-element tuple of the form (dtype, shape), where dtype is an
    # instance of tf.DType, and shape is either an instance of tf.TensorShape,
    # or a list, or a tuple that can be fed as argument into a tf.TensorShape.
    # We thus convert this into a TensorType.
    return TensorType(spec[0], spec[1])
  elif isinstance(spec, (list, tuple, collections.OrderedDict)):
    return NamedTupleType(spec)
  else:
    raise TypeError(
        'Unable to interpret an argument of type {} as a type spec.'.format(
            py_typecheck.type_string(type(spec))))
