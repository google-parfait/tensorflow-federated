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
"""Utilities for saving and loading program state to a file system.

Note: This library uses `tf.io.gfile` to perform file system operations, this
means that this library:

  * supports all the file systems supported by `tf.io.gfile`
  * encodes files in the same way as `tf.io.gfile`
"""

import asyncio
import os
import os.path
from typing import Optional, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import program_state_manager
from tensorflow_federated.python.program import value_reference


class FileProgramStateManager(
    program_state_manager.ProgramStateManager[
        program_state_manager.ProgramStateStructure
    ]
):
  """A `tff.program.ProgramStateManager` that is backed by a file system.

  A `tff.program.FileProgramStateManager` is a utility for saving and loading
  program state to a file system in a federated program and is used to implement
  fault tolerance. In particular, it is intended to only restart the same
  simulation and run with the same version of TensorFlow Federated.

  Program state is saved to the file system using the SavedModel (see
  `tf.saved_model`) format. When the program state is saved, if the program
  state is a value reference or a structure containing value references, each
  value reference is materialized. The program state is then flattened and saved
  using the SavedModel format. The structure of the program state is discarded,
  but is required to load the program state.

  Note: The SavedModel format can only contain values that can be converted to a
  `tf.Tensor` (see `tf.convert_to_tensor`), releasing any other values will
  result in an error.

  See https://www.tensorflow.org/guide/saved_model for more information about
  the SavedModel format.
  """

  def __init__(
      self,
      root_dir: Union[str, os.PathLike[str]],
      prefix: str = 'program_state_',
      keep_total: int = 5,
      keep_first: bool = True,
  ):
    """Returns an initialized `tff.program.ProgramStateManager`.

    Args:
      root_dir: A path on the file system to save program state. If this path
        does not exist it will be created.
      prefix: A string to use as the prefix for filenames.
      keep_total: An integer representing the total number of program states to
        keep. If the value is zero or smaller, all program states will be kept.
      keep_first: A boolean indicating if the first program state should be
        kept, irrespective of whether it is the oldest program state or not.
        This is desirable in settings where you would like to ensure full
        reproducibility of the simulation, especially in settings where model
        weights or optimizer states are initialized randomly. By loading from
        the initial program state, one can avoid re-initializing and obtaining
        different results.

    Raises:
      ValueError: If `root_dir` is an empty string.
    """
    py_typecheck.check_type(root_dir, (str, os.PathLike))
    if not root_dir:
      raise ValueError('Expected `root_dir` to not be an empty string.')
    py_typecheck.check_type(prefix, str)
    py_typecheck.check_type(keep_total, int)
    py_typecheck.check_type(keep_first, bool)

    if not tf.io.gfile.exists(root_dir):
      tf.io.gfile.makedirs(root_dir)
    self._root_dir = root_dir
    self._prefix = prefix
    self._keep_total = keep_total
    self._keep_first = keep_first

  async def get_versions(self) -> Optional[list[int]]:
    """Returns a list of saved versions or `None`.

    Returns:
      A list of saved versions or `None` if there is no saved program state.
    """
    if not await file_utils.exists(self._root_dir):
      return None
    versions = []
    # Due to tensorflow/issues/19378, we cannot use `tf.io.gfile.glob` here
    # because it returns directory contents recursively on Windows.
    entries = await file_utils.listdir(self._root_dir)
    for entry in entries:
      if entry.startswith(self._prefix):
        version = self._get_version_for_path(entry)
        if version is not None:
          versions.append(version)
    if not versions:
      return None
    return sorted(versions)

  def _get_version_for_path(
      self, path: Union[str, os.PathLike[str]]
  ) -> Optional[int]:
    """Returns the version for the given `path` or `None`.

    This method does not assert that the given `path` or the returned version
    represent saved program state.

    Args:
      path: The path to extract the version from.
    """
    py_typecheck.check_type(path, (str, os.PathLike))

    basename = os.path.basename(path)
    if basename.startswith(self._prefix):
      version = basename[len(self._prefix) :]
    else:
      version = None
    try:
      return int(version)
    except (TypeError, ValueError):
      return None

  def _get_path_for_version(self, version: int) -> str:
    """Returns the path for the given `version`.

    This method does not assert that the given `version` or the returned path
    represent saved program state.

    Args:
      version: The version used to construct the path.
    """
    py_typecheck.check_type(version, (int, np.integer))

    basename = f'{self._prefix}{version}'
    return os.path.join(self._root_dir, basename)

  async def load(
      self, version: int, structure: program_state_manager.ProgramStateStructure
  ) -> program_state_manager.ProgramStateStructure:
    """Returns the program state for the given `version`.

    Args:
      version: A integer representing the version of a saved program state.
      structure: The structure of the saved program state for the given
        `version` used to support serialization and deserailization of
        user-defined classes in the structure.

    Raises:
      ProgramStateManagerStateNotFoundError: If there is no program state for
        the given `version`.
      ProgramStateManagerStructureError: If `structure` does not match the value
        loaded for the given `version`.
    """
    py_typecheck.check_type(version, int)

    path = self._get_path_for_version(version)
    if not await file_utils.exists(path):
      raise program_state_manager.ProgramStateManagerStateNotFoundError(
          f'No program state found for version: {version}'
      )
    flattened_state = await file_utils.read_saved_model(path)
    try:
      program_state = tree.unflatten_as(structure, flattened_state)
    except ValueError as e:
      raise program_state_manager.ProgramStateManagerStructureError(
          f'The structure of type {type(structure)}:\n'
          f'{structure}\n'
          f'does not match the value of type {type(flattened_state)}:\n'
          f'{flattened_state}\n'
      ) from e

    def _normalize(
        value: program_state_manager.ProgramStateStructure,
    ) -> program_state_manager.ProgramStateStructure:
      """Returns a normalize the value.

      Because this implementation saves program state to the file system using
      the SavedModel format, when the program state is loaded, the values will
      be TF-native types. This function normalizes those values as numpy values
      so that when program state is loaded, those values can be used more
      naturally.

      Args:
        value: The value to normalize.
      """
      if tf.is_tensor(value):
        return value.numpy()
      return value

    normalized_value = tree.map_structure(_normalize, program_state)
    logging.info('Program state loaded: %s', path)
    return normalized_value

  async def _remove(self, version: int) -> None:
    """Removes program state for the given `version`."""
    py_typecheck.check_type(version, (int, np.integer))

    path = self._get_path_for_version(version)
    if await file_utils.exists(path):
      await file_utils.rmtree(path)
      logging.info('Program state removed: %s', path)

  async def _remove_old_program_state(self) -> None:
    """Removes old program state."""
    if self._keep_total <= 0:
      return
    versions = await self.get_versions()
    if versions is not None and len(versions) > self._keep_total:
      start = 1 if self._keep_first else 0
      stop = start - self._keep_total
      await asyncio.gather(*[self._remove(v) for v in versions[start:stop]])

  async def remove_all(self) -> None:
    """Removes all program states."""
    versions = await self.get_versions()
    if versions is not None:
      await asyncio.gather(*[self._remove(v) for v in versions])

  async def save(
      self,
      program_state: program_state_manager.ProgramStateStructure,
      version: int,
  ) -> None:
    """Saves `program_state` for the given `version`.

    Args:
      program_state: A `tff.program.ProgramStateStructure` to save.
      version: A strictly increasing integer representing the version of a saved
        `program_state`.

    Raises:
      ProgramStateManagerStateAlreadyExistsError: If there is already program
        state for the given `version`.
    """
    py_typecheck.check_type(version, (int, np.integer))

    path = self._get_path_for_version(version)
    if await file_utils.exists(path):
      raise program_state_manager.ProgramStateManagerStateAlreadyExistsError(
          f'Program state already exists for version: {version}'
      )
    materialized_state = await value_reference.materialize_value(program_state)
    flattened_state = tree.flatten(materialized_state)
    await file_utils.write_saved_model(flattened_state, path)
    logging.info('Program state saved: %s', path)
    await self._remove_old_program_state()
