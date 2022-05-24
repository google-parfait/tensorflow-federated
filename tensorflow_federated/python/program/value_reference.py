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
"""Defines abstract interfaces for representing references to values.

These abstract interfaces provide the capability to handle values without
requiring them to be materialized as Python objects. Instances of these
abstract interfaces represent values of type `tff.TensorType` and can be placed
on the server, elements of structures that are placed on the server, or
unplaced.
"""

import abc
import asyncio
from typing import Any, Iterable, Union

import numpy as np
import tree

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import typed_object

MaterializableTffType = Union[computation_types.TensorType,
                              computation_types.SequenceType,]

MaterializablePythonType = Union[np.generic, np.ndarray,
                                 Iterable[Union[np.generic, np.ndarray]],]


class MaterializableValueReference(
    typed_object.TypedObject, metaclass=abc.ABCMeta):
  """An abstract interface representing references to server-placed values."""

  @property
  @abc.abstractmethod
  def type_signature(self) -> MaterializableTffType:
    """The `tff.Type` of this object."""
    raise NotImplementedError

  @abc.abstractmethod
  async def get_value(self) -> MaterializablePythonType:
    """Returns the referenced value.

    The Python type of the referenced value depends on the `type_signature`:

    | TFF Type           | Python Type                                |
    | ------------------ | ------------------------------------------ |
    | `tff.TensorType`   | `np.generic` or `np.ndarray`               |
    | `tff.SequenceType` | `Iterable` of `np.generic` or `np.ndarray` |
    """
    raise NotImplementedError


async def materialize_value(value: Any) -> Any:
  """Returns a structure of materialized values.

  Args:
    value: A materialized value, a value reference, or structure materialized
      values and value references to materialize.
  """

  async def _materialize(value: Any) -> Any:
    if isinstance(value, MaterializableValueReference):
      return await value.get_value()
    else:
      return value

  flattened = tree.flatten(value)
  flattened = await asyncio.gather(*[_materialize(v) for v in flattened])
  return tree.unflatten_as(value, flattened)
