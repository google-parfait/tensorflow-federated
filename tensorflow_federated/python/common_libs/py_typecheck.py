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
from collections.abc import Sequence
import sys
import typing
from typing import Optional, Protocol, TypeVar, Union

import attrs
from typing_extensions import TypeGuard


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
    raise TypeError(
        'Expected {}{}, found {}.'.format(
            '{} to be of type '.format(label) if label is not None else '',
            type_string(type_spec),
            type_string(type(target)),
        )
    )
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


def check_attrs(value):
  """Checks that `value` is an attrs decorated class or an instance thereof."""
  if not attrs.has(type(value)):
    raise TypeError(
        'Expected an instance of an attrs decorated class, or an '
        'attrs-decorated class type; found a value of type '
        f'{type(value)}'
    )


@typing.runtime_checkable
class SupportsNamedTuple(Protocol):
  """A `typing.Protocol` with two abstract method `_fields` and `_asdict`."""

  @property
  def _fields(self) -> tuple[str, ...]:
    ...

  def _asdict(self) -> dict[str, object]:
    ...


_NT = TypeVar('_NT', bound=Optional[str])
_VT = TypeVar('_VT', bound=object)


def is_name_value_pair(
    obj: object,
    name_type: type[_NT] = Optional[str],
    value_type: type[_VT] = object,
) -> TypeGuard[tuple[_NT, _VT]]:
  """Returns `True` if `obj` is a name value pair, otherwise `False`.

  In TFF, a name value pair (or named field) is a `collection.abc.Sequence` of
  exactly two elements, a `name` (which can be `None`) and a `value`.

  Args:
    obj: The object to test.
    name_type: The type of the name.
    value_type: The type of the value.
  """
  if not isinstance(obj, Sequence) or len(obj) != 2:
    return False
  name, value = obj

  # Before Python 3.10, you could not pass a `Union Type` to isinstance, see
  # https://docs.python.org/3/library/functions.html#isinstance.
  if sys.version_info < (3, 10):

    def _unpack_type(x):
      origin = typing.get_origin(x)
      if origin is Union:
        return typing.get_args(name_type)
      elif origin is not None:
        return origin
      else:
        return x

    name_type = _unpack_type(name_type)
    value_type = _unpack_type(value_type)

  return isinstance(name, name_type) and isinstance(value, value_type)
