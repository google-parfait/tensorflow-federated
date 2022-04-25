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
"""Defines abstract interfaces representing references to values.

These abstract interfaces provide the capability to handle values without
requiring them to be materialized as Python objects. Instances of these
abstract interfaces represent values of type `tff.TensorType` and can be placed
on the server, elements of structures that are placed on the server, or
unplaced.
"""

import asyncio
import functools
from typing import Any, Callable, Coroutine

import attr
import tensorflow as tf

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import value_reference


# TODO(b/205140778) After Python 3.8 use `unittest.IsolatedAsyncioTestCase`.
def run_sync(
    fn: Callable[..., Coroutine[Any, None, Any]]) -> Callable[..., Any]:
  """A decorator for running tests synchronously."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
      return loop.run_until_complete(fn(*args, **kwargs))
    finally:
      asyncio.set_event_loop(None)
      loop.close()

  return wrapper


@attr.s
class TestAttrObject1():
  a = attr.ib()
  b = attr.ib()


@attr.s
class TestAttrObject2():
  a = attr.ib()


class TestMaterializableValueReference(
    value_reference.MaterializableValueReference):
  """A test implementation of `tff.program.MaterializableValueReference`."""

  def __init__(self, value: value_reference.MaterializablePythonType):
    if isinstance(value, int):
      self._type_signature = computation_types.TensorType(tf.int32)
    elif isinstance(value, bool):
      self._type_signature = computation_types.TensorType(tf.bool)
    elif isinstance(value, str):
      self._type_signature = computation_types.TensorType(tf.string)
    elif isinstance(value, tf.data.Dataset):
      self._type_signature = computation_types.SequenceType(tf.int32)
    else:
      raise NotImplementedError(f'Unexpected type found: {type(value)}.')
    self._value = value

  @property
  def type_signature(self) -> value_reference.MaterializableTffType:
    return self._type_signature

  async def get_value(self) -> value_reference.MaterializablePythonType:
    return self._value

  def __eq__(self, other: Any) -> bool:
    if self is other:
      return True
    elif not isinstance(other, TestMaterializableValueReference):
      return NotImplemented
    if self._type_signature != other._type_signature:
      return False
    if self._type_signature.is_sequence():
      return list(self._value) == list(other._value)
    else:
      return self._value == other._value
