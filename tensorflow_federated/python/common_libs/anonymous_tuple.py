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
"""Anonymous named tuples to represent generic tuple values in computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import attr
import six
from six.moves import zip
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


class AnonymousTuple(object):
  """Represents an anonymous named tuple.

  Anonymous tuples are similar to named tuples, in that their elements can be
  accessed by name or by index, but unlike `collections.namedtuple`, they can
  be instantiated without having to explicitly construct a new class for each
  instance, which incurs unnecessary overhead. Anonymous tuples are thus
  related to `collections.namedtuples` much in the same way anonymous lambdas
  are related to named functions explicitly declared with `def`. One of the
  intended uses of annoymous tuples is to represent structured parameters in
  computations defined as Python functions or TF defuns.

  Example:

  ```python
  x = AnonymousTuple([('foo', 10), (None, 20), ('bar', 30)])

  len(x) == 3
  x[0] == 10
  x[1] == 20
  x[2] == 30
  list(iter(x)) == [10, 20, 30]
  sorted(dir(x)) == ['bar', 'foo']
  x.foo == 10
  x.bar == 30
  ```

  Note that in general, naming the members of these tuples is optional. Thus,
  an `AnonymousTuple` can be used just like an ordinary positional tuple.

  Also note that the user will not be creating such tuples. They are a hidden
  part of the impementation designed to work together with function decorators.
  """
  __slots__ = ('_hash', '_element_array', '_name_to_index')

  # TODO(b/113112108): Define more magic methods for convenience in handling
  # anonymous tuples. Possibly move out to a more generic location or replace
  # with pre-existing type if a sufficiently widely used one can be found.
  def __init__(self, elements):
    """Constructs a new anonymous named tuple with the given elements.

    Args:
      elements: A list of element specifications, each being a pair consisting
        of the element name (either a string, or None), and the element value.
        The order is significant.

    Raises:
      TypeError: if the `elements` are not a list, or if any of the items on
        the list is not a pair with a string at the first position.
    """
    py_typecheck.check_type(elements, list)
    for e in elements:
      if not (isinstance(e, tuple) and (len(e) == 2) and
              (e[0] is None or isinstance(e[0], six.string_types))):
        raise TypeError(
            'Expected every item on the list to be a pair in which the first '
            'element is a string, found {}.'.format(repr(e)))

    self._element_array = tuple(e[1] for e in elements)
    self._name_to_index = {}
    for idx, e in enumerate(elements):
      name = e[0]
      if name is None:
        continue
      if name in self._name_to_index:
        raise ValueError('AnonymousTuple does not support duplicated '
                         'names, but found ' + str([e[0] for e in elements]))
      self._name_to_index[name] = idx
    self._hash = None

  def __len__(self):
    return len(self._element_array)

  def __iter__(self):
    return iter(self._element_array)

  def __dir__(self):
    return list(self._name_to_index.keys())

  def __getitem__(self, key):
    py_typecheck.check_type(key, (int, slice))
    if isinstance(key, int):
      if key < 0 or key >= len(self._element_array):
        raise IndexError(
            'Element index {} is out of range, tuple has {} elements.'.format(
                str(key), str(len(self._element_array))))
    return self._element_array[key]

  def __getattr__(self, name):
    if name not in self._name_to_index:
      raise AttributeError(
          'The tuple does not have a member "{}". Members (only first 10): {}'
          .format(name,
                  list(self._name_to_index.keys())[:10]))
    return self._element_array[self._name_to_index[name]]

  def __eq__(self, other):
    # pylint: disable=protected-access
    return (isinstance(other, AnonymousTuple) and
            (self._element_array == other._element_array) and
            (self._name_to_index == other._name_to_index))
    # pylint: enable=protected-access

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return 'AnonymousTuple([{}])'.format(', '.join(
        '({}, {})'.format(e[0], repr(e[1])) for e in to_elements(self)))

  def __str__(self):
    return '<{}>'.format(','.join(
        ('{}={}'.format(e[0], str(e[1])) if e[0] is not None else str(e[1]))
        for e in to_elements(self)))

  def __hash__(self):
    if self._hash is None:
      self._hash = hash((
          'anonymous_tuple',  # salting to avoid type mismatch.
          self._element_array,
          tuple(self._name_to_index.items())))
    return self._hash


def to_elements(an_anonymous_tuple):
  """Retrieves the list of (name, value) pairs from an anonymous tuple.

  Modeled as a module function rather than a method of `AnonymousTuple` to avoid
  naming conflicts with the tuple attributes, and so as not to expose the user
  to this implementation-oriented functionality.

  Args:
    an_anonymous_tuple: An instance of `AnonymousTuple`.

  Returns:
    The list of (name, value) pairs in which names can be None. Identical to
    the format that's accepted by the tuple constructor.

  Raises:
    TypeError: if the argument is not an `AnonymousTuple`.
  """
  py_typecheck.check_type(an_anonymous_tuple, AnonymousTuple)
  # pylint: disable=protected-access
  index_to_name = {
      idx: name
      for name, idx in six.iteritems(an_anonymous_tuple._name_to_index)
  }
  return [(index_to_name.get(idx), val)
          for idx, val in enumerate(an_anonymous_tuple._element_array)]
  # pylint: enable=protected-access


def to_odict(anon_tuple):
  """Returns anon_tuple as an `OrderedDict`, if possible.

  Args:
    anon_tuple: An `AnonymousTuple`.

  Raises:
    ValueError: If the anonymous tuple contains unnamed elements.

  Returns:
    An `OrderedDict`.
  """
  py_typecheck.check_type(anon_tuple, AnonymousTuple)
  elements = to_elements(anon_tuple)
  for name, _ in elements:
    if name is None:
      raise ValueError('Can\'t convert an AnonymousTuple with unnamed '
                       'entries to an OrderedDict')
  return collections.OrderedDict(elements)


def flatten(structure):
  """Returns a list of values in a possibly recursively nested tuple.

  N.B. This implementation is not compatible with the approach of
  `tf.contrib.nest.flatten`, which enforces lexical order for `OrderedDict`s.

  Args:
    structure: An anonymous tuple, possibly recursively nested, or a non-tuple
      element that is packed as a singleton list.

  Returns:
    The list of values in the tuple (or the singleton argument if not a tuple).
  """
  if not isinstance(structure, AnonymousTuple):
    return tf.nest.flatten(structure)
  else:
    result = []
    for _, v in to_elements(structure):
      result.extend(flatten(v))
    return result


def pack_sequence_as(structure, flat_sequence):
  """Returns a list of values in a possibly recursively nested tuple.

  Args:
    structure: An anonymous tuple, possibly recursively nested, or a non-tuple
      argument to match a singleton sequence.
    flat_sequence: A flat Python list of values.

  Returns:
    An anonymous tuple nested the same way as `structure`, but with leaves
    replaced with `flat_sequence` such that when flatten, it yields a list
    with the same contents as `flat_sequence`.
  """
  py_typecheck.check_type(flat_sequence, list)

  def _pack(structure, flat_sequence, position):
    if not isinstance(structure, AnonymousTuple):
      return flat_sequence[position], position + 1
    else:
      elements = []
      for k, v in to_elements(structure):
        packed_v, position = _pack(v, flat_sequence, position)
        elements.append((k, packed_v))
      return AnonymousTuple(elements), position

  result, _ = _pack(structure, flat_sequence, 0)
  return result


def is_same_structure(a, b):
  """Compares whether `a` and `b` have the same nested structure.

  This method is analogous to `tf.contrib.framework.nest.assert_same_structure`,
  but returns a boolean rather than throwing an exception.

  Args:
    a: an `AnonymousTuple` object.
    b: an `AnonymousTuple` object.

  Returns:
    True iff `a` and `b` have the same nested structure.

  Raises:
    TypeError: if `a` or `b` are not of type AnonymousTuple.
  """
  py_typecheck.check_type(a, AnonymousTuple)
  py_typecheck.check_type(b, AnonymousTuple)
  elems_a = to_elements(a)
  elems_b = to_elements(b)
  if len(elems_a) != len(elems_b):
    return False
  for elem_a, elem_b in zip(elems_a, elems_b):
    val_a = elem_a[1]
    val_b = elem_b[1]
    if elem_a[0] != elem_b[0]:
      return False
    if isinstance(val_a, AnonymousTuple) and isinstance(val_b, AnonymousTuple):
      return is_same_structure(val_a, val_b)
    elif isinstance(val_a, AnonymousTuple) or isinstance(val_b, AnonymousTuple):
      return False
    else:
      try:
        tf.nest.assert_same_structure(val_a, val_b, check_types=True)
      except (ValueError, TypeError):
        return False
  return True


def map_structure(fn, *structure):
  """Applies `fn` to each entry in `structure` and returns a new structure.

  This is a special implementation of `tf.contrib.framework.nest.map_structure`
  that works for `AnonymousTuple`.

  Args:
    fn: a callable that accepts as many arguments as there are structures.
    *structure: a scalar, tuple, or list of constructed scalars and/or
      tuples/lists, or scalars. Note: numpy arrays are considered scalars.

  Returns:
    A new structure with the same arity as `structure` and same type as
    `structure[0]`, whose values correspond to `fn(x[0], x[1], ...)` where
    `x[i]` is a value in the corresponding location in `structure[i]`.

  Raises:
    TypeError: if `fn` is not a callable, or *structure contains types other
      than AnonymousTuple.
    ValueError: if `*structure` is empty.
  """
  py_typecheck.check_callable(fn)
  if not structure:
    raise ValueError('Must provide at least one structure')

  py_typecheck.check_type(structure[0], AnonymousTuple)
  for i, other in enumerate(structure[1:]):
    if not is_same_structure(structure[0], other):
      raise TypeError('Structure at position {} is not the same '
                      'structure'.format(i))

  flat_structure = [flatten(s) for s in structure]
  entries = zip(*flat_structure)
  s = [fn(*x) for x in entries]

  return pack_sequence_as(structure[0], s)


def from_container(value, recursive=False):
  """Creates an instance of `AnonymousTuple` from a Python container.

  By default, this conversion is only performed at the top level for Python
  dictionaries, `collections.OrderedDict`s, `namedtuple`s, `list`s and
  `tuple`s. Elements of these structures are not recursively converted.

  Args:
    value: The Python container to convert.
    recursive: Whether to convert elements recursively (`False` by default).

  Returns:
    The corresponding instance of `AnonymousTuple`.

  Raises:
    TypeError: If the `value` is not of one of the supported container types.
  """

  def _convert(value, recursive, must_be_container=False):
    """The actual conversion function.

    Args:
      value: Same as in `from_container`.
      recursive: Same as in `from_container`.
      must_be_container: When set to `True`, causes an exception to be raised if
        `value` is not a container.

    Returns:
      The result of conversion.

    Raises:
      TypeError: If `value` is not a container and `must_be_container` has
        been set to `True`.
    """
    if isinstance(value, AnonymousTuple):
      if recursive:
        return AnonymousTuple([
            (k, _convert(v, True)) for k, v in to_elements(value)
        ])
      else:
        return value
    elif py_typecheck.is_attrs(value):
      return _convert(
          attr.asdict(
              value, dict_factory=collections.OrderedDict, recurse=False),
          recursive, must_be_container)
    elif py_typecheck.is_named_tuple(value):
      return _convert(value._asdict(), recursive, must_be_container)
    elif isinstance(value, collections.OrderedDict):
      items = six.iteritems(value)
      if recursive:
        return AnonymousTuple([(k, _convert(v, True)) for k, v in items])
      else:
        return AnonymousTuple(list(items))
    elif isinstance(value, dict):
      items = sorted(list(six.iteritems(value)))
      if recursive:
        return AnonymousTuple([(k, _convert(v, True)) for k, v in items])
      else:
        return AnonymousTuple(items)
    elif isinstance(value, (tuple, list)):
      if recursive:
        return AnonymousTuple([(None, _convert(v, True)) for v in value])
      else:
        return AnonymousTuple([(None, v) for v in value])
    elif must_be_container:
      raise TypeError('Unable to convert a Python object of type {} into '
                      'an `AnonymousTuple`.'.format(
                          py_typecheck.type_string(type(value))))
    else:
      return value

  return _convert(value, recursive, must_be_container=True)
