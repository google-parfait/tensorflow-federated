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

import abc
from typing import Union

import numpy as np

from tensorflow_federated.python.core.impl.types import typed_object


class ServerArrayReference(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """An abstract interface representing references to server placed values."""

  @abc.abstractmethod
  def get_value(self) -> Union[np.generic, np.ndarray]:
    """Returns the referenced value as a numpy scalar or array."""
    raise NotImplementedError
