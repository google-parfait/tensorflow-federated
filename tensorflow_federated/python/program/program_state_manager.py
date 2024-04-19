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
"""Utilities for saving and loading program state in a federated program."""

import abc
from typing import Generic, Optional, Union, TypeVar

from tensorflow_federated.python.common_libs import serializable
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference

# ProgramStateManager's may save any value in addition to materializable values.
ProgramStateValue = Union[
    object,
    value_reference.MaterializableValue,
    # `tff.Serializable` objects are intended to be impemented by platform
    # authors and used by program logic authors; it should not be common for
    # program logic authors to implement `tff.Serializable` objects.
    serializable.Serializable,
]
ProgramStateStructure = TypeVar(
    'ProgramStateStructure',
    bound=structure_utils.Structure[ProgramStateValue],
)


class ProgramStateExistsError(Exception):
  """Raised when the program state already exists."""

  def __init__(self, *, version: int, path: str):
    super().__init__(
        f'Program state already exists for version [{version}] at path'
        f' [{path}].'
    )


class ProgramStateNotFoundError(Exception):
  """Raised when the program state cannot be found."""

  def __init__(self, version: int):
    super().__init__(f'No program state found for version: {version}.')


class ProgramStateManager(abc.ABC, Generic[ProgramStateStructure]):
  """An interface for saving and loading program state in a federated program.

  A `tff.program.ProgramStateManager` is used to implement fault tolerance in a
  federated program. The structure or type of the program state that is saved is
  unknown at construction time and can change as the program runs.
  """

  @abc.abstractmethod
  async def get_versions(self) -> Optional[list[int]]:
    """Returns a list of saved versions or `None`.

    Returns:
      A list of saved versions or `None` if there is no saved program state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def load(
      self, version: int, structure: ProgramStateStructure
  ) -> ProgramStateStructure:
    """Returns the saved program state for the given `version`.

    Args:
      version: A integer representing the version of a saved program state.
      structure: The structure of the saved program state for the given
        `version` used to support serialization and deserailization of
        user-defined classes in the structure.

    Raises:
      ProgramStateNotFoundError: If there is no program state for the given
        `version`.
    """
    raise NotImplementedError

  async def load_latest(
      self, structure: ProgramStateStructure
  ) -> tuple[ProgramStateStructure, int]:
    """Returns the latest saved program state and version or (`None`, 0).

    Args:
      structure: The structure of the saved program state for the given
        `version` used to support serialization and deserailization of
        user-defined classes in the structure.

    Returns:
      A tuple of the latest saved (program state, version) or (`None`, 0) if
      there is no latest saved program state.
    """
    versions = await self.get_versions()
    if versions is None:
      return None, 0
    latest_version = max(versions)
    try:
      return await self.load(latest_version, structure), latest_version
    except ProgramStateNotFoundError:
      return None, 0

  @abc.abstractmethod
  async def save(
      self, program_state: ProgramStateStructure, version: int
  ) -> None:
    """Saves `program_state` for the given `version`.

    Args:
      program_state: A `tff.program.ProgramStateStructure` to save.
      version: A strictly increasing integer representing the version of a saved
        `program_state`.

    Raises:
      ProgramStateExistsError: If there is already program state for the given
        `version`.
    """
    raise NotImplementedError
