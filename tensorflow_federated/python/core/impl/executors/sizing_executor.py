# Copyright 2019, The TensorFlow Federated Authors.
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
"""Utils for testing executors."""

import collections
from typing import List, Tuple

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import typed_object
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_value_base

# This type is used to pass around information related to tensors. The int
# describes the number of elements of a tensor, the DType is the same tensor's
# DType.
SizeAndDTypes = List[Tuple[int, tf.DType]]


def get_type_information(value, type_spec):
  """Gets size and dtype of the type_spec as a list.

  This function only considers type_specs which are of TensorType or
  StructType. Other types are considered not to have an appropriate metric
  for number of elements. In that case, an empty list is returned.

  Args:
    value: This value's size and type will be evaluated.
    type_spec: The corresonding tff.Type of the value.

  Returns:
    A list where each element is a list of two elements: [num_elems, dtype].
  """

  # type_spec may be None if value is a TypedObject, so manually access it.
  if isinstance(value, typed_object.TypedObject):
    type_spec = value.type_signature

  # If the type_spec is a TensorType then we can calculate the size now.
  if type_spec.is_tensor():

    # If the type is a string, then the number of elements is the sum of all
    # the lengths of strings.
    if type_spec.dtype == tf.string:
      value = tf.constant(value, tf.string)
      lengths = tf.map_fn(tf.strings.length, value, dtype=tf.int32)
      num_elements = tf.reduce_sum(lengths)
    else:
      num_elements = type_spec.shape.num_elements()
    return [[num_elements, type_spec.dtype]]

  # If the type is a StructType we can continue traversing the type_spec.
  elif type_spec.is_struct():
    type_info = []
    if isinstance(value, collections.OrderedDict):
      value = structure.from_container(value, recursive=False)
    assert isinstance(value, (structure.Struct, list, tuple))
    for nested_value, nested_type in zip(value, type_spec):
      type_info += get_type_information(nested_value, nested_type)
    return type_info
  else:
    return []


class SizingExecutor(executor_base.Executor):
  """Executor to track size of data passed through."""

  def __init__(self, target):
    """Creates a new instance of a sizing executor.

    The sizing executor tracks the total size of tensors passed through via
    `create_value`. Generally, a `SizingExecutor` is intended to be placed
    directly on top of each client executor in a federated stack.

    For a `SizingExecutor` which is placed directly on top of a client stack,
    the `broadcast_history` and `aggregate_history` will track all tensors
    passed from the Python runtime to the clients and back, respectively.

    This notion of "broadcast" and "aggregate" is somewhat specific to the TFF
    runtime, and does not map exactly to the invocations of
    `federated_broadcast` or `federated_aggregate` in the computation the TFF
    runtime is currently interpreting. Each invocation of these intrinsics will
    add data to the `broadcast_history` or `aggregate_history` as appropriate,
    but these histories will additionally capture things like materializing
    CLIENT-placed tensors back into the Python context for inspection.

    Args:
      target: An instance of `executor_base.Executor`.
    """
    py_typecheck.check_type(target, executor_base.Executor)
    self._target = target
    self._broadcast_history = []
    self._aggregate_history = []

  @property
  def aggregate_history(self) -> SizeAndDTypes:
    return self._aggregate_history

  @property
  def broadcast_history(self) -> SizeAndDTypes:
    return self._broadcast_history

  async def create_value(self, value, type_spec=None):
    target_val = await self._target.create_value(value, type_spec)
    wrapped_val = SizingExecutorValue(self, target_val)
    self._broadcast_history.extend(get_type_information(value, type_spec))
    return wrapped_val

  async def create_call(self, comp, arg=None):
    if arg is not None:
      target_val = await self._target.create_call(comp.value, arg.value)
      wrapped_val = SizingExecutorValue(self, target_val)
      return wrapped_val
    else:
      target_val = await self._target.create_call(comp.value)
      wrapped_val = SizingExecutorValue(self, target_val)
      return wrapped_val

  async def create_struct(self, elements):
    target_val = await self._target.create_struct(
        structure.map_structure(lambda x: x.value, elements))
    wrapped_val = SizingExecutorValue(self, target_val)
    return wrapped_val

  async def create_selection(self, source, index=None, name=None):
    target_val = await self._target.create_selection(
        source.value, index=index, name=name)
    wrapped_val = SizingExecutorValue(self, target_val)
    return wrapped_val

  def close(self):
    self._target.close()


class SizingExecutorValue(executor_value_base.ExecutorValue):
  """A value managed by `SizingExecutor`."""

  def __init__(self, owner, value):
    """Creates an instance of a value in the sizing executor.

    Args:
      owner: An instance of `SizingExecutor`.
      value: An embedded value from the target executor.
    """
    py_typecheck.check_type(owner, SizingExecutor)
    py_typecheck.check_type(value, executor_value_base.ExecutorValue)
    self._owner = owner
    self._value = value

  @property
  def value(self):
    return self._value

  @property
  def type_signature(self):
    return self._value.type_signature

  async def compute(self):
    result = await self._value.compute()
    self._owner.aggregate_history.extend(
        get_type_information(result, self._value.type_signature))
    return result
