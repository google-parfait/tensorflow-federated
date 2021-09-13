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

import collections
import os
import os.path

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.simulation import file_program_state_manager
from tensorflow_federated.python.simulation import program_state_manager


def _create_test_program_state(value: int = 0):
  return collections.OrderedDict([
      ('a', {
          'b': tf.constant(value),
          'c': tf.constant(value),
      }),
  ])


class FileProgramStateManagerVersionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('one', 1),
      ('two', 2),
      ('three', 3),
      ('ten', 10),
  )
  def test_returns_versions_with_saved_program_state(self, count):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir, keep_total=0)
    structure = _create_test_program_state()
    program_state_mngr.set_structure(structure)
    for i in range(count):
      test_program_state = _create_test_program_state(i)
      program_state_mngr.save(test_program_state, i * 10)

    actual_versions = program_state_mngr.versions()

    expected_versions = [i * 10 for i in range(count)]
    self.assertEqual(actual_versions, expected_versions)

  def test_returns_none_with_no_saved_program_state(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)
    structure = _create_test_program_state()
    program_state_mngr.set_structure(structure)

    actual_versions = program_state_mngr.versions()

    self.assertIsNone(actual_versions)


class FileProgramStateManagerLoadTest(parameterized.TestCase):

  def test_returns_program_state_with_one_saved_and_version_1(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)
    structure = _create_test_program_state()
    program_state_mngr.set_structure(structure)
    test_program_state_1 = _create_test_program_state(1)
    program_state_mngr.save(test_program_state_1, 1)

    actual_program_state = program_state_mngr.load(1)

    self.assertEqual(actual_program_state, test_program_state_1)

  @parameterized.named_parameters(
      ('one', 1),
      ('two', 2),
      ('three', 3),
  )
  def test_returns_program_state_with_three_saved_and_version(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)
    structure = _create_test_program_state()
    program_state_mngr.set_structure(structure)
    test_program_state_1 = _create_test_program_state(1)
    program_state_mngr.save(test_program_state_1, 1)
    test_program_state_2 = _create_test_program_state(2)
    program_state_mngr.save(test_program_state_2, 2)
    test_program_state_3 = _create_test_program_state(3)
    program_state_mngr.save(test_program_state_3, 3)

    actual_program_state = program_state_mngr.load(version)

    expected_program_state = _create_test_program_state(version)
    self.assertEqual(actual_program_state, expected_program_state)

  def test_raises_version_error_with_no_saved_program_states(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)
    structure = _create_test_program_state()
    program_state_mngr.set_structure(structure)

    with self.assertRaises(program_state_manager.VersionError):
      _ = program_state_mngr.load(0)

  def test_raises_version_error_with_unknown_version(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)
    structure = _create_test_program_state()
    program_state_mngr.set_structure(structure)
    test_program_state_1 = _create_test_program_state(1)
    program_state_mngr.save(test_program_state_1, 1)

    with self.assertRaises(program_state_manager.VersionError):
      _ = program_state_mngr.load(10)

  def test_raises_value_error_with_no_structure(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)
    structure = None
    program_state_mngr.set_structure(structure)
    test_program_state_1 = _create_test_program_state(1)
    program_state_mngr.save(test_program_state_1, 1)

    with self.assertRaises(ValueError):
      _ = program_state_mngr.load(1)


class FileProgramStateManagerSaveTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('one', 1),
      ('two', 2),
      ('three', 3),
  )
  def test_saves_program_state(self, count):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)

    for i in range(count):
      test_program_state = _create_test_program_state(i)
      program_state_mngr.save(test_program_state, i * 10)

    actual_dirs = os.listdir(temp_dir)
    expected_dirs = ['program_state_{}'.format(i * 10) for i in range(count)]
    self.assertCountEqual(actual_dirs, expected_dirs)

  def test_removes_oldest_with_keep_first_true(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir, keep_total=3, keep_first=True)

    for i in range(5):
      test_program_state = _create_test_program_state(i)
      program_state_mngr.save(test_program_state, i * 10)

    self.assertCountEqual(
        os.listdir(temp_dir), [
            'program_state_0',
            'program_state_30',
            'program_state_40',
        ])

  def test_removes_oldest_with_keep_first_false(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir, keep_total=3, keep_first=False)

    for i in range(5):
      test_program_state = _create_test_program_state(i)
      program_state_mngr.save(test_program_state, i * 10)

    self.assertCountEqual(
        os.listdir(temp_dir), [
            'program_state_20',
            'program_state_30',
            'program_state_40',
        ])

  def test_raises_already_exists_error_with_existing_version(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        temp_dir)

    test_program_state_1 = _create_test_program_state(1)
    program_state_mngr.save(test_program_state_1, 1)

    with self.assertRaises(tf.errors.AlreadyExistsError):
      program_state_mngr.save(test_program_state_1, 1)


if __name__ == '__main__':
  absltest.main()
