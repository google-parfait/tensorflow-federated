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

from typing import Optional, Union

import attr
import numpy as np

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import value_reference


@attr.s
class TestAttrObject1():
  a = attr.ib()
  b = attr.ib()


@attr.s
class TestAttrObject2():
  a = attr.ib()


class TestServerArrayReference(value_reference.ServerArrayReference):
  """A test implementation of `tff.program.ServerArrayReference`."""

  def __init__(self,
               value: Union[np.generic, np.ndarray],
               type_signature: Optional[computation_types.Type] = None):
    self._value = value
    self._type_signature = type_signature

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature

  def get_value(self) -> Union[np.generic, np.ndarray]:
    return self._value
