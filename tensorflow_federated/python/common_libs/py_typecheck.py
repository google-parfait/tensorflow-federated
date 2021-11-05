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

import builtins
import collections

import attr


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
  if not isinstance(target, type_spec):
    raise TypeError('Expected {}{}, found {}.'.format(
        '{} to be of type '.format(label) if label is not None else '',
        type_string(type_spec), type_string(type(target))))
  return target


def check_none(target, label=None):
  """Checks that `target` is `None`.

  Args:
    target: An object to check.
    label: An optional label associated with the target, used to create a more
      human-readable error message.

  Raises:
    TypeError: when the target is not `None`.
  """
  if target is not None:
    raise TypeError('Expected {} to be `None`, found {}.'.format(
        label if label is not None else 'the argument',
        type_string(type(target))))


def check_not_none(target, label=None):
  """Checks that `target` is not `None`.

  Args:
    target: An object to check.
    label: An optional label associated with the target, used to create a more
      human-readable error message.

  Raises:
    TypeError: when the target is `None`.
  """
  if target is None:
    raise TypeError('Expected {} to not be `None`.'.format(
        label if label is not None else 'the argument'))


def check_subclass(target_class, parent_class):
  """Tests that `target_class` subclasses `parent_class`, that both are classes.

  Args:
    target_class: A class to check for inheritence from `parent_class`.
    parent_class: A python class or tuple of classes.

  Returns:
    The `target_class`, unchanged.

  Raises:
    TypeError if the `target_class` doesn't subclass a class in `parent_class`.
  """
  if not issubclass(target_class, parent_class):
    raise TypeError('Expected {} to subclass {}, but it does not.'.format(
        target_class, parent_class))
  return target_class


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


def is_attrs(value):
  """Determines whether `value` is an attrs decorated class or instance of."""
  return attr.has(value)


def is_named_tuple(value):
  """Determines whether `value` can be considered a `collections.namedtuple`.

  As `collections.namedtuple` creates a new class with no common a base for each
  named tuple, there is no simple way to check the type with `isintance(T)`.
  Instead, this method looks to see if `value` has an `_fields` attribute (which
  all namedtuple subclasses support).

  Args:
    value: an instance of a Python class or a Python type object.

  Returns:
    True iff `value` can be considered an instance or type of
    `collections.namedtuple`.
  """
  if isinstance(value, type):
    return issubclass(value, tuple) and hasattr(value, '_fields')
  else:
    return is_named_tuple(type(value))


def is_name_value_pair(element, name_required=True, value_type=None):
  """Determines whether `element` can be considered a name and value pair.

  In TFF a named field is a `collection.Sequence` of two elements, a `name`
  (which can optionally be `None`) and a `value`.

  Args:
    element: The Python object to test.
    name_required: Optional, a boolean specifying if the `name` of the pair can
      be `None`.
    value_type: Optional, either a Python type, or a tuple of Python types; the
      same as what's accepted by isinstance.

  Returns:
    `True` if `element` is a named tuple element, otherwise `False`.
  """
  if not isinstance(element, collections.abc.Sequence) or len(element) != 2:
    return False
  if ((name_required or element[0] is not None) and
      not isinstance(element[0], str)):
    return False
  if value_type is not None and not isinstance(element[1], value_type):
    return False
  return True


def check_len(target, length):
  """Checks that the length of `target` is equal to `length`.

  Args:
    target: The object to check.
    length: The expected length.

  Raises:
    ValueError: If the lengths do not match.
  """
  if len(target) != length:
    raise ValueError(
        'Expected an argument of length {}, got one of length {} ({}).'.format(
            length, len(target), target))


def check_non_negative_float(target, label=None):
  """Checks that `target` is non-negative float.

  Args:
    target: An object to check.
    label: An optional label associated with the target, used to create a more
      human-readable error message.

  Raises:
    TypeError: When the target is not `float`.
    ValueError: When the target is a `float` with negative value.
  """
  check_type(target, float)
  if target < 0.0:
    raise ValueError(f'Provided {label if label else "target"} must be '
                     f'non-negative, found {target}')
