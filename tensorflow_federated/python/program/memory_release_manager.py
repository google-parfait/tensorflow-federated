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
from collections.abc import Hashable

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import value_reference


class MemoryReleaseManager(
    release_manager.ReleaseManager[
        release_manager.ReleasableStructure, Hashable
    ]
):
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

  async def release(
      self, value: release_manager.ReleasableStructure, key: Hashable
  ) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      key: A hashable value used to reference the released `value`.
    """
    py_typecheck.check_type(key, Hashable)

    materialized_value = await value_reference.materialize_value(value)
    self._values[key] = materialized_value

  def remove_all(self) -> None:
    """Removes all program states."""
    self._values = collections.OrderedDict()

  def values(
      self,
  ) -> collections.OrderedDict[Hashable, release_manager.ReleasableStructure]:
    """Returns an `collections.OrderedDict` of all keys and released values."""
    return self._values.copy()
