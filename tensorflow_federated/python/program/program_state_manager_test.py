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

from typing import Optional
import unittest
from unittest import mock

from absl.testing import absltest

from tensorflow_federated.python.program import program_state_manager


class _TestProgramStateManager(
    program_state_manager.ProgramStateManager[
        program_state_manager.ProgramStateStructure
    ]
):
  """A test implementation of `tff.program.ProgramStateManager`.

  A `tff.program.ProgramStateManager` cannot be constructed directly because it
  has abstract methods, this implementation exists to make it possible to
  construct instances of `tff.program.ProgramStateManager` that can used as
  stubs or mocked.
  """

  async def get_versions(self) -> Optional[list[int]]:
    raise NotImplementedError

  async def load(
      self, version: int, structure: program_state_manager.ProgramStateStructure
  ) -> program_state_manager.ProgramStateStructure:
    del version, structure  # Unused.
    raise NotImplementedError

  async def save(
      self,
      program_state: program_state_manager.ProgramStateStructure,
      version: int,
  ) -> None:
    del program_state, version  # Unused.
    raise NotImplementedError


class ProgramStateManagerTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_load_latest_with_saved_program_state(self):
    program_state_mngr = _TestProgramStateManager()
    program_state_mngr.get_versions = mock.AsyncMock(return_value=[1, 2, 3])
    program_state_mngr.load = mock.AsyncMock(return_value='test3')
    structure = 'test'

    (program_state, version) = await program_state_mngr.load_latest(structure)

    program_state_mngr.get_versions.assert_called_once_with()
    program_state_mngr.load.assert_called_once_with(3, structure)
    self.assertEqual(program_state, 'test3')
    self.assertEqual(version, 3)

  async def test_load_latest_with_no_saved_program_state(self):
    program_state_mngr = _TestProgramStateManager()
    program_state_mngr.get_versions = mock.AsyncMock(return_value=None)
    program_state_mngr.load = mock.AsyncMock()
    structure = 'test'

    (program_state, version) = await program_state_mngr.load_latest(structure)

    program_state_mngr.get_versions.assert_called_once_with()
    program_state_mngr.load.assert_not_called()
    self.assertIsNone(program_state)
    self.assertEqual(version, 0)

  async def test_load_latest_with_load_failure(self):
    program_state_mngr = _TestProgramStateManager()
    program_state_mngr.get_versions = mock.AsyncMock(return_value=[1, 2, 3])
    program_state_mngr.load = mock.AsyncMock(
        side_effect=program_state_manager.ProgramStateNotFoundError(version=0)
    )
    structure = 'test'

    (program_state, version) = await program_state_mngr.load_latest(structure)

    program_state_mngr.get_versions.assert_called_once_with()
    program_state_mngr.load.assert_called_once_with(3, structure)
    self.assertIsNone(program_state)
    self.assertEqual(version, 0)


if __name__ == '__main__':
  absltest.main()
