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

from typing import Any, List, Mapping, Optional
from unittest import mock

from absl.testing import absltest

from tensorflow_federated.python.program import program_state_manager


class _TestProgramStateManager(program_state_manager.ProgramStateManager):

  def __init__(self, values: Optional[Mapping[int, Any]] = None):
    self._values = values

  def versions(self) -> Optional[List[int]]:
    if self._values is None:
      return None
    return self._values.keys()

  def save(self, program_state: Any, version: int):
    del program_state, version  # Unused.

  def load(self, version: int, structure: Any) -> Any:
    del structure  # Unused.
    if self._values is None or version not in self._values:
      raise program_state_manager.ProgramStateManagerStateNotFoundError()
    return self._values[version]


class ProgramStateManagerTest(absltest.TestCase):

  def test_load_latest_with_saved_program_state(self):
    values = {x: f'test{x}' for x in range(5)}
    structure = values[0]
    program_state_mngr = _TestProgramStateManager(values)

    (program_state, version) = program_state_mngr.load_latest(structure)

    self.assertEqual(program_state, 'test4')
    self.assertEqual(version, 4)

  def test_load_latest_with_no_saved_program_state(self):
    structure = None
    program_state_mngr = _TestProgramStateManager()

    (program_state, version) = program_state_mngr.load_latest(structure)

    self.assertIsNone(program_state)
    self.assertEqual(version, 0)

  def test_load_latest_with_load_failure(self):
    values = {x: f'test{x}' for x in range(5)}
    structure = values[0]
    program_state_mngr = _TestProgramStateManager(values)
    program_state_mngr.load = mock.MagicMock(
        side_effect=program_state_manager.ProgramStateManagerStateNotFoundError)

    (program_state, version) = program_state_mngr.load_latest(structure)

    self.assertIsNone(program_state)
    self.assertEqual(version, 0)


if __name__ == '__main__':
  absltest.main()
