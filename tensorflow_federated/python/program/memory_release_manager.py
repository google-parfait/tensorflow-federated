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
"""Utilities for releasing values from a federated program to memory."""

import collections
from typing import Any, OrderedDict, Hashable

from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import value_reference


class MemoryReleaseManager(release_manager.ReleaseManager):
  """A `tff.program.ReleaseManager` that releases values to memory.

  A `tff.program.MemoryReleaseManager` is a utility for releasing values from a
  federated program to memory and is used to release values from platform
  storage to customer storage in a federated program.

  Values are released to memory as Python objects. When the value is released,
  if the value is a value reference or a structure containing value references,
  each value reference is materialized.
  """

  def __init__(self):
    """Returns an initialized `tff.program.MemoryReleaseManager`."""
    self._values = collections.OrderedDict()

  def release(self, value: Any, key: Hashable):
    """Releases `value` from a federated program.

    Args:
      value: A materialized value, a value reference, or a structure of
        materialized values and value references representing the value to
        release.
      key: A hashable value used to reference the released `value`.
    """
    materialized_value = value_reference.materialize_value(value)
    self._values[key] = materialized_value

  def values(self) -> OrderedDict[Hashable, Any]:
    """Returns a dict of all keys and released values."""
    return self._values.copy()
