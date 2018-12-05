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

# Dependency imports
import six

from tensorflow_federated.python.common_libs import py_typecheck


class AnonymousTuple(object):
  """Represents an anonymous named tuple.

  Anonymous tuples are similar to named tuples, in that their elements can be
  accessed by name or by index, but unlike collections.namedtuple, they can
  be instantiated without having to explicitly construct a new class for each
  instance, which incurs unnecessary overhead. Anonymous tuples are thus
  related to collections.namedtuples much in the same way anonymous lambdas
  are related to named functions explicitly declared with 'def()'. One of the
  intended uses of annoymous tuples is to represent structured parameters in
  computations defined as Python functions or TF defuns.

  Example:

    x = AnonymousTuple([('foo', 10), (None, 20), ('bar', 30)])

    len(x) == 3
    x[0] == 10
    x[1] == 20
    x[2] == 30
    list(iter(x)) == [10, 20, 30]
    sorted(dir(x)) == ['bar', 'foo']
    x.foo == 10
    x.bar == 30

  Note that in general, naming the members of these tuples is optional. Thus,
  an AnonymousTuple can be used just like an ordinary 'positional' tuple.

  Also note that the user will not be creating such tuples. They are a hidden
  part of the impementation designed to work together with function decorators.
  """

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
      TypeError: if the 'elements' are not a list, or if any of the items on
        the list is not a pair with a string at the first position.
    """
    py_typecheck.check_type(elements, list)
    for e in elements:
      if not (isinstance(e, tuple) and (len(e) == 2) and
              (e[0] is None or isinstance(e[0], six.string_types))):
        raise TypeError(
            'Expected every item on the list to be a pair in which the first '
            'element is a string, found {}.'.format(repr(e)))
    self._element_array = [e[1] for e in elements]
    self._name_to_index = collections.OrderedDict([
        (e[0], idx) for idx, e in enumerate(elements) if e[0] is not None])

  def __len__(self):
    return len(self._element_array)

  def __iter__(self):
    return iter(self._element_array)

  def __dir__(self):
    return self._name_to_index.keys()

  def __getitem__(self, key):
    py_typecheck.check_type(key, int)
    if key < 0 or key >= len(self._element_array):
      raise IndexError(
          'Element index {} is out of range, tuple has {} elements.'.format(
              str(key), str(len(self._element_array))))
    return self._element_array[key]

  def __getattr__(self, name):
    if name not in self._name_to_index:
      raise AttributeError(
          'The tuple does not have a member "{}".'.format(name))
    return self._element_array[self._name_to_index[name]]

  def __eq__(self, other):
    # pylint: disable=protected-access
    return (isinstance(other, AnonymousTuple) and
            (self._element_array == other._element_array) and
            (self._name_to_index == other._name_to_index))

  def __ne__(self, other):
    return not self == other

  def __repr__(self):
    return 'AnonymousTuple([{}])'.format(', '.join(
        '({}, {})'.format(e[0], repr(e[1])) for e in to_elements(self)))

  def __str__(self):
    return '<{}>'.format(','.join(
        ('{}={}'.format(e[0], str(e[1])) if e[0] else str(e[1]))
        for e in to_elements(self)))


def to_elements(an_anonymous_tuple):
  """Retrieves the list of (name, value) pairs from an anonymous tuple.

  Modeled as a module function rather than a method of AnonymousTuple to avoid
  naming conflicts with the tuple attributes, and so as not to expose the user
  to this implementation-oriented functionality.

  Args:
    an_anonymous_tuple: An instance of AnonymousTuple.

  Returns:
    The list of (name, value) pairs in which names can be None. Identical to
    the format that's accepted by the tuple constructor.

  Raises:
    TypeError: if the argument is not an AnonymousTuple.
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
