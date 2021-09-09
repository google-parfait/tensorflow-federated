# Copyright 2021, Google LLC.
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
"""Utilities for saving and loading intermediate program state."""

import abc
from typing import Any, List, Optional, Tuple


class VersionError(Exception):
  pass


class ProgramStateManager(metaclass=abc.ABCMeta):
  """An abstract interface for `ProgramStateManager`s.

  A `ProgramStateManager` is a utility to saving and loading intermediate
  program state that can be used for fault tolerance. The structure or type of
  the program state that is saved is unknown at construction time and can change
  as the program runs.
  """

  @abc.abstractmethod
  def versions(self) -> Optional[List[int]]:
    """Returns a list of saved versions or `None`."""
    raise NotImplementedError

  @abc.abstractmethod
  def save(self, program_state: Any, version: int):
    """Saves `program_state` for the given `version`.

    Args:
      program_state: The program state to save.
      version: A monotonically increasing integer representing the version of
        the `program_state`.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def load(self, version: int) -> Any:
    """Returns the saved program state for the given `version`.

    Args:
      version: A monotonically increasing integer representing the version of
        the `program_state`.

    Raises:
      VersionError: If there is no program state for the given `version`.
    """
    raise NotImplementedError

  def load_latest(self) -> Optional[Tuple[Any, int]]:
    """Returns the latest saved program state and version or (`None`, 0).

    Returns:
      A tuple of the latest saved (program state, version) or (`None`, 0) if
      there is no latest saved program state.
    """
    versions = self.versions()
    if versions is None:
      return None, 0
    latest_version = max(versions)
    try:
      return self.load(latest_version), latest_version
    except VersionError:
      return None, 0
