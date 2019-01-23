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
"""Utility functions for checking Python types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from six.moves import builtins


def check_type(target, type_spec, label=None):
  """Checks that `target` is of Python type or types `type_spec`.

  Args:
    target: An object, the Python type of which to check.
    type_spec: Either a Python type, or a tuple of Python types; the same as
      what's accepted by isinstance.
    label: An optional label associated with the target, used to create a more
      human-readable error message.

  Returns:
    The target.

  Raises:
    TypeError: when the target is not of one of the types in `type_spec`.
  """
  _check_is_type_spec(type_spec)
  if not isinstance(target, type_spec):
    raise TypeError('Expected {}{}, found {}.'.format(
        '{} to be of type '.format(label) if label is not None else '',
        type_string(type_spec), type_string(type(target))))
  return target


def check_callable(target, label=None):
  """Checks target is callable and then returns it."""
  if not callable(target):
    raise TypeError('Expected {} callable, found non-callable {}.'.format(
        '{} to be'.format(label) if label is not None else 'a',
        type_string(type(target))))
  return target


def type_string(type_spec):
  """Creates a string representation of `type_spec` for error reporting.

  Args:
    type_spec: Either a Python type, or a tuple of Python types; the same as
      what's accepted by isinstance.

  Returns:
    A string representation for use in error reporting.

  Raises:
    TypeError: if the `type_spec` is not of the right type.
  """
  _check_is_type_spec(type_spec)
  if isinstance(type_spec, type):
    if type_spec.__module__ == builtins.__name__:
      return type_spec.__name__
    else:
      return '{}.{}'.format(type_spec.__module__, type_spec.__name__)
  else:
    assert isinstance(type_spec, (tuple, list))
    type_names = [type_string(x) for x in type_spec]
    if len(type_names) == 1:
      return type_names[0]
    elif len(type_names) == 2:
      return '{} or {}'.format(*type_names)
    else:
      return ', '.join(type_names[0:-1] + ['or {}'.format(type_names[-1])])


def _check_is_type_spec(type_spec):
  """Determines if `type_spec` is a valid type specification.

  Args:
    type_spec: Either a Python type, or a tuple of Python types; the same as
      what's accepted by isinstance.

  Raises:
    TypeError: if `type_spec` is not as defined above.
  """
  if isinstance(type_spec, type):
    return
  if (isinstance(type_spec, (tuple, list)) and type_spec and
      all(isinstance(x, type) for x in type_spec)):
    return
  raise TypeError(
      'Expected a type, or a tuple or list of types, found {}.'.format(
          type_string(type(type_spec))))


def is_named_tuple(value):
  """Determines whether `value` can be considered a `collections.namedtuple`.

  As `collections.namedtuple` creates a new class with no common a base for each
  named tuple, there is no simple way to check the type with `isintance(T)`.
  Instead, this method looks to see if `value` has an `_asdict` attribute (which
  all namedtuple subclasses support).

  Args:
    value: an instance of a Python class.

  Returns:
    True iff `value` can be considered a named tuple.
  """
  cls = type(value)
  if '_asdict' in vars(cls):
    return True
  parent_classes = inspect.getmro(cls)[1:]
  for p in parent_classes:
    if '_asdict' in vars(p):
      return True
  return False
