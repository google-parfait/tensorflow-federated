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

from typing import Any, List, Optional

from absl.testing import absltest

from tensorflow_federated.python.simulation import program_state_manager


class ProgramStateManagerTest(absltest.TestCase):

  def test_load_latest_with_saved_program_state(self):

    class _TestProgramStateManager(program_state_manager.ProgramStateManager):

      def versions(self) -> Optional[List[int]]:
        return [0, 1, 2, 3, 4]

      def save(self, program_state: Any, version: int):
        del program_state, version  # Unused.

      def load(self, version: int) -> Any:
        return 'test'

    program_state_mngr = _TestProgramStateManager()

    (program_state, version) = program_state_mngr.load_latest()

    self.assertEqual(program_state, 'test')
    self.assertEqual(version, 4)

  def test_load_latest_with_no_saved_program_state(self):

    class _TestProgramStateManager(program_state_manager.ProgramStateManager):

      def versions(self) -> Optional[List[int]]:
        return None

      def save(self, program_state: Any, version: int):
        del program_state, version  # Unused.

      def load(self, version: int) -> Any:
        del version  # Unused.
        raise program_state_manager.VersionNotFoundError()

    program_state_mngr = _TestProgramStateManager()

    (program_state, version) = program_state_mngr.load_latest()

    self.assertIsNone(program_state)
    self.assertEqual(version, 0)

  def test_load_latest_with_load_data_failure(self):

    class _TestProgramStateManager(program_state_manager.ProgramStateManager):

      def versions(self) -> Optional[List[int]]:
        return [0, 1, 2, 3, 4]

      def save(self, program_state: Any, version: int):
        del program_state, version  # Unused.

      def load(self, version: int) -> Any:
        del version  # Unused.
        raise program_state_manager.VersionNotFoundError()

    program_state_mngr = _TestProgramStateManager()

    (program_state, version) = program_state_mngr.load_latest()

    self.assertIsNone(program_state)
    self.assertEqual(version, 0)


if __name__ == '__main__':
  absltest.main()
