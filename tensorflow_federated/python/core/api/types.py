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
from six import string_types
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


class Type(object):
  """An abstract interface for all classes that represent TFF types."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def is_assignable_from(self, other):
    """Determines whether this TFF type is assignable from another TFF type.

    Args:
      other: Another type, an instance of Type.

    Returns:
      True if self is assignable from other, False otherwise.
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
  """An implementation of Type for representing types of tensors in TFF."""

  def __init__(self, dtype, shape=None):
    """Constructs a new instance from the given dtype and shape.

    Args:
      dtype: An instance of tf.DType.
      shape: An optional instance of tf.TensorShape or an argument that can be
        passed to its constructor (such as a list or a tuple), or None for the
        default scalar shape. Unspecified shapes are not supported.

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
          for k in xrange(x.ndims))))
    return (isinstance(other, TensorType) and
            (self.dtype == other.dtype) and
            _shape_is_assignable_from(self.shape, other.shape))

  def __repr__(self):
    return (
        'TensorType({}, {})'.format(
            repr(self._dtype), repr([dim.value for dim in self._shape.dims]))
        if self._shape.ndims > 0
        else 'TensorType({})'.format(repr(self._dtype)))

  def __str__(self):
    return (
        '{}[{}]'.format(
            self._dtype.name,
            ','.join((str(dim.value) if dim.value else '?')
                     for dim in self._shape.dims))
        if self._shape.ndims > 0
        else self._dtype.name)


class NamedTupleType(Type):
  """An implementation of Type for representing types of named tuples in TFF."""

  def __init__(self, elements):
    """Constructs a new instance from the given element types.

    Args:
      elements: A list of element specifications. Each element specification
        is either a type spec (an instance of Type or something convertible to
        it via to_type() below) for the element, or a pair (name, spec) for
        elements that have defined names. Alternatively, one can supply here
        an instance of collections.OrderedDict mapping element names to their
        types (or things that are convertible to types).

    Raises:
      TypeError: if the arguments are of the wrong types.
      ValueError: if the named tuple contains no elements.
    """
    py_typecheck.check_type(elements, (list, tuple, collections.OrderedDict))
    if isinstance(elements, collections.OrderedDict):
      elements = elements.items()
    if not elements:
      raise ValueError('A named tuple must contain at least one element.')
    def _is_named_element(e):
      return (isinstance(e, tuple) and (len(e) == 2) and
              isinstance(e[0], string_types))
    def _map_element(e):
      return ((None, e) if isinstance(e, Type)
              else (
                  (e[0], to_type(e[1])) if _is_named_element(e)
                  else (
                      (None, to_type(e)))))
    self._elements = (
        [(elements[0], to_type(elements[1]))] if _is_named_element(elements)
        else [_map_element(e) for e in elements])

  @property
  def elements(self):
    # Shallow copy to prevent accidental modification by the caller.
    return list(self._elements)

  def is_assignable_from(self, other):
    py_typecheck.check_type(other, Type)
    if not isinstance(other, NamedTupleType):
      return False
    other_elements = other.elements
    return (
        (len(self._elements) == len(other_elements)) and all(
            ((self._elements[k][0] in [other_elements[k][0], None]) and
             (self._elements[k][1].is_assignable_from(other_elements[k][1])))
            for k in xrange(len(self._elements))))

  def __repr__(self):
    return (
        'NamedTupleType([{}])'.format(', '.join(
            '(\'{}\', {})'.format(e[0], repr(e[1])) if e[0] else repr(e[1])
            for e in self._elements)))

  def __str__(self):
    return ('<{}>'.format(','.join(
        ('{}={}'.format(name, str(value)) if name else str(value))
        for name, value in self._elements)))


class SequenceType(Type):
  """An implementation of Type for representing types of sequences in TFF."""

  def __init__(self, element):
    """Constructs a new instance from the given element type.

    Args:
      element: A specification of the element type, either an instance of Type
        or something convertible to it by to_type().
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
  """An implementation of Type for representing functional types in TFF."""

  def __init__(self, parameter, result):
    """Constructs a new instance from the given parameter and result types.

    Args:
      parameter: A specification of the parameter type, either an instance of
        Type or something convertible to it by to_type().
      result: A specification of the result type, either an instance of
        Type or something convertible to it by to_type().
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
        str(self._parameter) if self._parameter else '', str(self._result))


class AbstractType(Type):
  """An implementation of Type for representing abstract types in TFF."""

  def __init__(self, label):
    """Constructs a new instance from the given string label.

    Args:
      label: A string label of an abstract type. All occurences of the label
        within a computation's type signature refer to the same concrete type.
    """
    py_typecheck.check_type(label, string_types)
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


# TODO(b/113112108): Define the representations of all the remaining TFF types.


def to_type(spec):
  """Converts the argument into an instance of Type.

  Args:
    spec: Either an instance of Type, or an argument convertible to Type.
      Assorted examples of type specifications are included below.

      Examples of arguments convertible to tensor types:

        tf.int32
        (tf.int32, [10])
        (tf.int32, [None])

      Examples of arguments convertible to flat named tuple types:

        [tf.int32, tf.bool]
        (tf.int32, tf.bool)
        [('a', tf.int32), ('b', tf.bool)]
        ('a', tf.int32)
        collections.OrderedDict([('a', tf.int32), ('b', tf.bool)])

      Examples of arguments convertible to nested named tuple types:

        (tf.int32, (tf.float32, tf.bool))
        (tf.int32, (('x', tf.float32), tf.bool))
        ((tf.int32, [1]), (('x', (tf.float32, [2])), (tf.bool, [3])))

  Returns:
    An instance of tb.Type corresponding to the given spec.
  """
  # TODO(b/113112108): Add multiple examples of valid type specs here in the
  # comments, in addition to the unit test.

  if isinstance(spec, Type) or spec is None:
    return spec
  elif isinstance(spec, tf.DType):
    return TensorType(spec)
  elif (isinstance(spec, tuple) and
        (len(spec) == 2) and
        isinstance(spec[0], tf.DType) and
        (isinstance(spec[1], tf.TensorShape) or
         (isinstance(spec[1], (list, tuple)) and
          all((isinstance(x, int) or x is None) for x in spec[1])))):
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
