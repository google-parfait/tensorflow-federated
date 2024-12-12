# Copyright 2021, The TensorFlow Federated Authors.
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
"""Utilities for testing the program library."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import contextlib
import functools
import struct
import sys
from typing import NamedTuple, Optional, TypeVar, Union
import warnings

import attrs
import federated_language
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck


T = TypeVar('T')


class TestMaterializableValueReference(
    federated_language.program.MaterializableValueReference
):
  """A test implementation of `federated_language.program.MaterializableValueReference`."""

  def __init__(self, value: federated_language.program.MaterializedValue):
    self._value = value

    if isinstance(value, bool):
      self._type_signature = federated_language.TensorType(np.bool_)
    elif isinstance(value, int):
      self._type_signature = federated_language.TensorType(np.int32)
    elif isinstance(value, str):
      self._type_signature = federated_language.TensorType(np.str_)
    elif isinstance(value, list):
      self._type_signature = federated_language.SequenceType(np.int32)
    else:
      raise NotImplementedError(f'Unexpected type found: {type(value)}.')

  @property
  def type_signature(
      self,
  ) -> federated_language.program.MaterializableTypeSignature:
    return self._type_signature

  async def get_value(self) -> federated_language.program.MaterializedValue:
    return self._value

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, TestMaterializableValueReference):
      return NotImplemented
    if self._type_signature != other._type_signature:
      return False
    if isinstance(self._type_signature, federated_language.SequenceType):
      return list(self._value) == list(other._value)
    else:
      return self._value == other._value


class TestSerializable(federated_language.Serializable):
  """A test implementation of `tff.Serializable`."""

  def __init__(self, a: int, b: int) -> None:
    self._a = a
    self._b = b

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'TestSerializable':
    a, b = struct.unpack('!ii', buffer)
    return TestSerializable(a, b)

  def to_bytes(self) -> bytes:
    return struct.pack('!ii', self._a, self._b)

  def __eq__(self, other) -> bool:
    if self is other:
      return True
    elif not isinstance(other, TestSerializable):
      return NotImplemented
    return (self._a, self._b) == (other._a, other._b)

  def __repr__(self):
    return f'{self.__class__.__name__}(a={self._a}, b={self._b})'


@attrs.define
class TestAttrs:
  a: int
  b: int


class TestNamedTuple1(NamedTuple):
  a: bool
  b: int
  c: str
  d: federated_language.program.MaterializableValueReference
  e: Optional[TestSerializable]


class TestNamedTuple2(NamedTuple):
  a: int


class TestNamedTuple3(NamedTuple):
  x: TestNamedTuple1
  y: TestNamedTuple2


def to_python(value: object) -> object:
  """Returns a Python representation of `value`."""

  def _fn(obj):
    if isinstance(obj, tf.Tensor) and not tf.is_symbolic_tensor(obj):
      obj = obj.numpy()
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      else:
        return obj
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return None

  return tree.traverse(_fn, value, top_down=False)


@contextlib.contextmanager
def assert_not_warns(
    category: type[Warning],
) -> Iterator[Iterable[warnings.WarningMessage]]:
  """Yields a context manager used to assert a warning is not triggered."""

  # The `__warningregistry__`'s need to be in a pristine state for tests to
  # work properly. This code replicates the standard library implementation of
  # `TestCase.assertWarns`. See
  # https://github.com/python/cpython/blob/main/Lib/unittest/case.py for more
  # information.
  for v in list(sys.modules.values()):
    if getattr(v, '__warningregistry__', None):
      v.__warningregistry__ = {}

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always', category=category)
    yield w
    for warning in w:
      if issubclass(warning.category, category):
        raise AssertionError(f'Warned `{category.__name__}` unexpectedly.')


def assert_same_key_order(a: object, b: object) -> None:
  """Asserts that two structures contain the same order for keys."""

  def _get_item(
      structure: Union[Sequence[T], Mapping[str, T]], key: Union[str, int]
  ) -> T:
    if isinstance(structure, py_typecheck.SupportsNamedTuple):
      return getattr(structure, key)
    else:
      return structure[key]

  def _fn(path: tuple[Union[str, int], ...], obj: object) -> None:
    if isinstance(obj, Mapping):
      other = functools.reduce(_get_item, path, b)
      if not isinstance(other, Mapping):
        raise AssertionError(
            f'Expected `other` to be a `Mapping` type, found {type(other)}.'
        )
      first_keys = list(obj.keys())
      second_keys = list(other.keys())
      if first_keys != second_keys:
        raise AssertionError(
            'Expected the order of the keys in the structures to match,'
            f' {first_keys} != {second_keys}.'
        )
    return None

  tree.traverse_with_path(_fn, a)
