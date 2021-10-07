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

import os.path
import shutil
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.program import file_release_manager
from tensorflow_federated.python.program import file_utils


class SavedModelFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_dir = self.create_tempdir()
    shutil.rmtree(temp_dir)
    self.assertFalse(os.path.exists(temp_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir=temp_dir)

    self.assertTrue(os.path.exists(temp_dir))

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_root_dir(self, root_dir):
    with self.assertRaises(TypeError):
      file_release_manager.SavedModelFileReleaseManager(root_dir=root_dir)

  def test_raises_value_error_with_root_dir_empty(self):
    with self.assertRaises(ValueError):
      file_release_manager.SavedModelFileReleaseManager(root_dir='')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_prefix(self, prefix):
    temp_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_release_manager.SavedModelFileReleaseManager(
          root_dir=temp_dir, prefix=prefix)


class SavedModelFileReleaseManagerGetPathForKeyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', '/tmp', 'a_', 123, '/tmp/a_123'),
      ('trailing_slash', '/tmp/', 'a_', 123, '/tmp/a_123'),
      ('no_prefix', '/tmp', '', 123, '/tmp/123'),
  )
  def test_returns_version_with_root_dir_and_prefix(self, root_dir, prefix, key,
                                                    expected_path):
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=root_dir, prefix=prefix)

    actual_path = release_mngr._get_path_for_key(key)

    self.assertEqual(actual_path, expected_path)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=temp_dir)

    with self.assertRaises(TypeError):
      release_mngr._get_path_for_key(key)


class SavedModelFileReleaseManagerReleaseTest(parameterized.TestCase):

  def test_writes_value(self):
    temp_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=temp_dir)

    with mock.patch.object(file_utils,
                           'write_saved_model') as mock_write_saved_model:
      release_mngr.release(1, 1)
      expected_path = release_mngr._get_path_for_key(1)
      mock_write_saved_model.assert_called_once_with(
          mock.ANY, expected_path, overwrite=True)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=temp_dir)

    with self.assertRaises(TypeError):
      release_mngr.release(1, key)


if __name__ == '__main__':
  absltest.main()
