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
"""Defines functions and classes for building and manipulating TFF types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.tensorflow_libs import tensor_utils


@six.add_metaclass(abc.ABCMeta)
class Type(object):
  """An abstract interface for all classes that represent TFF types."""

  @abc.abstractmethod
  def __repr__(self):
    """Returns a full-form representation of this type."""
    raise NotImplementedError

  @abc.abstractmethod
  def __str__(self):
    """Returns a concise representation of this type."""
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


class TensorType(Type):
  """An implementation of `tff.Type` representing types of tensors in TFF."""

  def __init__(self, dtype, shape=None):
    """Constructs a new instance from the given `dtype` and `shape`.

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
    # TODO(b/123764922): If we are passed a shape of `TensorShape(None)`, which
    # does happen along some codepaths, we have slightly violated the
    # assumptions of `TensorType` (see the special casing in `repr` and `str`
    # below. This is related to compatibility checking,
    # and there are a few options. For now, simply adding a case in
    # `impl/type_utils.is_assignable_from` to catch. We could alternatively
    # treat this case the same as if we have been passed a shape of `None` in
    # this constructor.
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

  def __repr__(self):
    if self._shape.ndims is None:
      return 'TensorType({}, {})'.format(repr(self._dtype), None)
    elif self._shape.ndims > 0:
      values = repr([dim.value for dim in self._shape.dims])
      return 'TensorType({}, {})'.format(repr(self._dtype), values)
    else:
      return 'TensorType({})'.format(repr(self._dtype))

  def __str__(self):
    if self._shape.ndims is None:
      return '{}[{}]'.format(repr(self._dtype), None)
    elif self._shape.ndims > 0:
      values = [
          str(dim.value) if dim.value is not None else '?'
          for dim in self._shape.dims
      ]
      return '{}[{}]'.format(self._dtype.name, ','.join(values))
    else:
      return self._dtype.name

  def __eq__(self, other):
    return (isinstance(other, TensorType) and self._dtype == other.dtype and
            tensor_utils.same_shape(self._shape, other.shape))


class NamedTupleType(anonymous_tuple.AnonymousTuple, Type):
  """An implementation of `tff.Type` representing named tuple types in TFF."""

  def __init__(self, elements):
    """Constructs a new instance from the given element types.

    Args:
      elements: Element specifications, in the format of a `list`, `tuple`, or
        `collections.OrderedDict`. Each element specification is either a type
        spec (an instance of `tff.Type` or something convertible to it via
        `tff.to_type`) for the element, or a (name, spec) for elements that have
        defined names. Alternatively, one can supply here an instance of
        `collections.OrderedDict` mapping element names to their types (or
        things that are convertible to types).
    """
    py_typecheck.check_type(elements, (list, tuple, collections.OrderedDict))
    if py_typecheck.is_named_tuple(elements):
      elements = elements._asdict()
    if isinstance(elements, collections.OrderedDict):
      elements = list(elements.items())

    def _is_full_element_spec(e):
      return (isinstance(e, tuple) and (len(e) == 2) and
              (e[0] is None or isinstance(e[0], six.string_types)))

    def _map_element(e):
      if isinstance(e, Type):
        return (None, e)
      elif _is_full_element_spec(e):
        return (e[0], to_type(e[1]))
      else:
        return (None, to_type(e))

    if _is_full_element_spec(elements):
      elements = [(elements[0], to_type(elements[1]))]
    else:
      elements = [_map_element(e) for e in elements]

    anonymous_tuple.AnonymousTuple.__init__(self, elements)

  def __repr__(self):

    def _element_repr(e):
      if e[0] is not None:
        return '(\'{}\', {})'.format(e[0], repr(e[1]))
      else:
        return repr(e[1])

    return 'NamedTupleType([{}])'.format(', '.join(
        [_element_repr(e) for e in anonymous_tuple.to_elements(self)]))

  def __str__(self):

    def _element_str(name, value):
      return '{}={}'.format(name, str(value)) if name else str(value)

    return ('<{}>'.format(','.join([
        _element_str(name, value)
        for name, value in anonymous_tuple.to_elements(self)
    ])))

  def __eq__(self, other):
    return (isinstance(other, NamedTupleType) and
            super(NamedTupleType, self).__eq__(other))


class SequenceType(Type):
  """An implementation of `tff.Type` representing types of sequences in TFF."""

  def __init__(self, element):
    """Constructs a new instance from the given `element` type.

    Args:
      element: A specification of the element type, either an instance of
        `tff.Type` or something convertible to it by `tff.to_type`.
    """
    self._element = to_type(element)

  @property
  def element(self):
    return self._element

  def __repr__(self):
    return 'SequenceType({})'.format(repr(self._element))

  def __str__(self):
    return '{}*'.format(str(self._element))

  def __eq__(self, other):
    return isinstance(other, SequenceType) and self._element == other.element


class FunctionType(Type):
  """An implementation of `tff.Type` representing functional types in TFF."""

  def __init__(self, parameter, result):
    """Constructs a new instance from the given `parameter` and `result` types.

    Args:
      parameter: A specification of the parameter type, either an instance of
        `tff.Type` or something convertible to it by `tff.to_type`.
      result: A specification of the result type, either an instance of
        `tff.Type` or something convertible to it by `tff.to_type`.
    """
    self._parameter = to_type(parameter)
    self._result = to_type(result)

  @property
  def parameter(self):
    return self._parameter

  @property
  def result(self):
    return self._result

  def __repr__(self):
    return 'FunctionType({}, {})'.format(
        repr(self._parameter), repr(self._result))

  def __str__(self):
    return '({} -> {})'.format(
        str(self._parameter) if self._parameter is not None else '',
        str(self._result))

  def __eq__(self, other):
    return (isinstance(other, FunctionType) and
            self._parameter == other.parameter and self._result == other.result)


class AbstractType(Type):
  """An implementation of `tff.Type` representing abstract types in TFF."""

  def __init__(self, label):
    """Constructs a new instance from the given string `label`.

    Args:
      label: A string label of an abstract type. All occurences of the label
        within a computation's type signature refer to the same concrete type.
    """
    py_typecheck.check_type(label, six.string_types)
    self._label = str(label)

  @property
  def label(self):
    return self._label

  def __repr__(self):
    return 'AbstractType(\'{}\')'.format(self._label)

  def __str__(self):
    return self._label

  def __eq__(self, other):
    return isinstance(other, AbstractType) and self._label == other.label


class PlacementType(Type):
  """An implementation of `tff.Type` representing the placement type in TFF.

  There is only one placement type, a TFF built-in, just as there is only one
  `int` or `str` type in Python. All instances of this class represent the same
  built-in TFF placement type.
  """

  def __repr__(self):
    return 'PlacementType()'

  def __str__(self):
    return 'placement'

  def __eq__(self, other):
    return isinstance(other, PlacementType)


class FederatedType(Type):
  """An implementation of `tff.Type` representing federated types in TFF."""

  def __init__(self, member, placement, all_equal=False):
    """Constructs a new federated type instance.

    Args:
      member: An instance of `tff.Type` (or something convertible to it) that
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

  def __repr__(self):
    return 'FederatedType({}, {}, {})'.format(
        repr(self._member), repr(self._placement), repr(self._all_equal))

  def __str__(self):
    if self._all_equal:
      return '{}@{}'.format(str(self._member), str(self._placement))
    else:
      return '{{{}}}@{}'.format(str(self._member), str(self._placement))

  def __eq__(self, other):
    return (isinstance(other, FederatedType) and
            self._member == other.member and
            self._placement == other.placement and
            self._all_equal == other.all_equal)


def to_type(spec):
  """Converts the argument into an instance of `tff.Type`.

  Examples of arguments convertible to tensor types:

  ```python
  tf.int32
  (tf.int32, [10])
  (tf.int32, [None])
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

  Args:
    spec: Either an instance of `tff.Type`, or an argument convertible to
      `tff.Type`.

  Returns:
    An instance of `tff.Type` corresponding to the given `spec`.
  """
  # TODO(b/113112108): Add multiple examples of valid type specs here in the
  # comments, in addition to the unit test.

  if isinstance(spec, Type) or spec is None:
    return spec
  elif isinstance(spec, tf.DType):
    return TensorType(spec)
  elif isinstance(spec, tf.TensorSpec):
    return TensorType(spec.dtype, spec.shape)
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
