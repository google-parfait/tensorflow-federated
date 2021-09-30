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

import os
import os.path
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.program import file_program_state_manager
from tensorflow_federated.python.program import program_state_manager


class FileProgramStateManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_dir = self.create_tempdir()
    root_dir = os.path.join(temp_dir, 'test')
    self.assertFalse(os.path.exists(root_dir))

    file_program_state_manager.FileProgramStateManager(root_dir=root_dir)

    self.assertTrue(os.path.exists(root_dir))

  def test_does_not_raise_type_error_with_root_dir_str(self):
    try:
      file_program_state_manager.FileProgramStateManager(
          root_dir='/tmp', prefix='a_', keep_total=5, keep_first=True)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_does_not_raise_type_error_with_root_dir_path_like(self):
    temp_dir = self.create_tempdir()

    try:
      file_program_state_manager.FileProgramStateManager(
          root_dir=temp_dir, prefix='a_', keep_total=5, keep_first=True)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_root_dir(self, root_dir):
    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(root_dir=root_dir)

  def test_raises_value_error_with_root_dir_empty(self):
    with self.assertRaises(ValueError):
      file_program_state_manager.FileProgramStateManager(root_dir='')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_prefix(self, prefix):
    temp_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir=temp_dir, prefix=prefix)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_keep_total(self, keep_total):
    temp_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir=temp_dir, keep_total=keep_total)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_keep_first(self, keep_first):
    temp_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir=temp_dir, keep_first=keep_first)


class FileProgramStateManagerVersionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  def test_returns_versions_with_saved_program_state(self, count):
    temp_dir = self.create_tempdir()
    for version in range(count):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=0)

    actual_versions = program_state_mngr.versions()

    expected_versions = list(range(count))
    self.assertEqual(actual_versions, expected_versions)

  def test_returns_versions_with_saved_program_state_and_files(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
      tempfile.mkstemp(prefix=os.path.join(temp_dir, 'b_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=0)

    actual_versions = program_state_mngr.versions()

    expected_versions = list(range(10))
    self.assertEqual(actual_versions, expected_versions)

  def test_returns_versions_with_saved_program_state_and_prefixed_files(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
      tempfile.mkstemp(prefix=os.path.join(temp_dir, 'a_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=0)

    actual_versions = program_state_mngr.versions()

    expected_versions = list(range(10))
    self.assertEqual(actual_versions, expected_versions)

  def test_returns_none_if_root_dir_does_not_exist(self):
    temp_dir = self.create_tempdir()
    os.rmdir(temp_dir)
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    versions = program_state_mngr.versions()

    self.assertIsNone(versions)

  def test_returns_none_with_no_files(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    versions = program_state_mngr.versions()

    self.assertIsNone(versions)

  def test_returns_none_with_no_saved_program_state(self):
    temp_dir = self.create_tempdir()
    for _ in range(10):
      tempfile.mkstemp(prefix=os.path.join(temp_dir, 'a_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    versions = program_state_mngr.versions()

    self.assertIsNone(versions)


class FileProgramStateManagerGetVersionForPathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('normal', '/tmp/a_123'),
      ('no_root_dir', 'a_123'),
      ('top_level', '/a_123'),
  )
  def test_returns_version_with_path(self, path):
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir='/tmp', prefix='a_')

    version = program_state_mngr._get_version_for_path(path)

    self.assertEqual(version, 123)

  @parameterized.named_parameters(
      ('wrong_prefix', '/tmp/b_123'),
      ('no_version', '/tmp/a_'),
      ('not_version', '/tmp/a_abc'),
  )
  def test_returns_none_with_path(self, path):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    version = program_state_mngr._get_version_for_path(path)

    self.assertIsNone(version)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_path(self, path):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      program_state_mngr._get_version_for_path(path)


class FileProgramStateManagerGetPathForVersionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('normal', '/tmp', 'a_', 123, '/tmp/a_123'),
      ('trailing_slash', '/tmp/', 'a_', 123, '/tmp/a_123'),
      ('no_prefix', '/tmp', '', 123, '/tmp/123'),
  )
  def test_returns_version_with_root_dir_and_prefix(self, root_dir, prefix,
                                                    version, expected_path):
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix=prefix)

    actual_path = program_state_mngr._get_path_for_version(version)

    self.assertEqual(actual_path, expected_path)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_version(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      program_state_mngr._get_path_for_version(version)


class FileProgramStateManagerLoadTest(parameterized.TestCase):

  def test_returns_program_state_with_one_save(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.set_structure('state')
    program_state_mngr.save('state_1', 1)

    actual_program_state = program_state_mngr.load(1)

    self.assertEqual(actual_program_state, 'state_1')

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  def test_returns_program_state_with_three_saves(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.set_structure('state')
    for i in range(3):
      program_state_mngr.save(f'state_{i}', i)

    actual_program_state = program_state_mngr.load(version)

    expected_program_state = f'state_{version}'
    self.assertEqual(actual_program_state, expected_program_state)

  def test_raises_version_not_found_error_with_no_saved_program_state(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.set_structure('state')

    with self.assertRaises(
        program_state_manager.ProgramStateManagerVersionNotFoundError):
      _ = program_state_mngr.load(0)

  def test_raises_version_not_found_error_with_unknown_version(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.set_structure('state')
    program_state_mngr.save('state_1', 1)

    with self.assertRaises(
        program_state_manager.ProgramStateManagerVersionNotFoundError):
      program_state_mngr.load(10)

  def test_raises_structure_error_with_structure_none(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.set_structure(None)
    program_state_mngr.save('state_1', 1)

    with self.assertRaises(
        file_program_state_manager.FileProgramStateManagerStructureError):
      program_state_mngr.load(1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_version(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      program_state_mngr.load(version)


class FileProgramStateManagerSaveTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  def test_saves_program_state(self, count):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with mock.patch.object(
        program_state_mngr,
        '_remove_old_program_state') as mock_remove_old_program_state:
      for i in range(count):
        program_state_mngr.save(f'state_{i}', i)
      self.assertEqual(mock_remove_old_program_state.call_count, count)

    actual_dirs = os.listdir(temp_dir)
    expected_dirs = [f'a_{i}' for i in range(count)]
    self.assertCountEqual(actual_dirs, expected_dirs)

  def test_raises_version_already_exists_error_with_existing_version(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr.save('state_1', 1)

    with self.assertRaises(
        program_state_manager.ProgramStateManagerVersionAlreadyExistsError):
      program_state_mngr.save('state_1', 1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_version(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      program_state_mngr.save('state', version)


class FileProgramStateManagerRemoveTest(parameterized.TestCase):

  def test_removes_saved_program_state_with_one_save(self):
    temp_dir = self.create_tempdir()
    os.mkdir(os.path.join(temp_dir, 'a_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr._remove(1)

    self.assertCountEqual(os.listdir(temp_dir), [])

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  def test_removes_program_state_with_three_saves(self, version):
    temp_dir = self.create_tempdir()
    for version in range(3):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr._remove(version)

    expected_dirs = ['a_0', 'a_1', 'a_2']
    expected_dirs.remove(f'a_{version}')
    self.assertCountEqual(os.listdir(temp_dir), expected_dirs)

  def test_noops_with_unknown_version(self):
    temp_dir = self.create_tempdir()
    os.mkdir(os.path.join(temp_dir, 'a_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr._remove(10)

    self.assertCountEqual(os.listdir(temp_dir), ['a_1'])

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_version(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      program_state_mngr._remove(version)


class FileProgramStateManagerRemoveOldProgramStateTest(absltest.TestCase):

  def test_does_not_remove_saved_program_state_with_keep_total_0(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=0)

    program_state_mngr._remove_old_program_state()

    actual_dirs = os.listdir(temp_dir)
    expected_dirs = [f'a_{i}' for i in range(10)]
    self.assertCountEqual(actual_dirs, expected_dirs)

  def test_removes_oldest_with_keep_first_true(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=3, keep_first=True)

    program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(temp_dir), ['a_0', 'a_8', 'a_9'])

  def test_removes_oldest_with_keep_first_false(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=3, keep_first=False)

    program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(temp_dir), ['a_7', 'a_8', 'a_9'])

if __name__ == '__main__':
  absltest.main()
