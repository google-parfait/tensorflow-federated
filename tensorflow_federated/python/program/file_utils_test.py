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
import shutil

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.program import file_utils


class ReadSavedModelTest(parameterized.TestCase):

  def test_returns_value(self):
    temp_dir = self.create_tempdir()
    module = file_utils._ValueModule(1)
    tf.saved_model.save(module, temp_dir.full_path, signatures={})

    actual_value = file_utils.read_saved_model(temp_dir)

    self.assertEqual(actual_value, 1)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('list', []),
  )
  def test_raises_type_error_with_path(self, path):
    with self.assertRaises(TypeError):
      file_utils.read_saved_model(path)


class WriteSavedModelTest(parameterized.TestCase):

  def test_writes_to_new_file(self):
    temp_dir = self.create_tempdir()
    shutil.rmtree(temp_dir)
    value = 1
    self.assertFalse(os.path.exists(temp_dir))

    file_utils.write_saved_model(value, temp_dir)

    self.assertTrue(os.path.exists(temp_dir))
    module = tf.saved_model.load(temp_dir.full_path)
    actual_value = module()
    self.assertEqual(actual_value, 1)

  def test_writes_to_existing_file(self):
    temp_dir = self.create_tempdir()
    value = 1
    overwrite = True
    self.assertTrue(os.path.exists(temp_dir))

    file_utils.write_saved_model(value, temp_dir, overwrite)

    self.assertTrue(os.path.exists(temp_dir))
    module = tf.saved_model.load(temp_dir.full_path)
    actual_value = module()
    self.assertEqual(actual_value, 1)

  def test_raises_file_already_exists_error_with_existing_file(self):
    temp_dir = self.create_tempdir()
    value = 1

    with self.assertRaises(file_utils.FileAlreadyExistsError):
      file_utils.write_saved_model(value, temp_dir)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('list', []),
  )
  def test_raises_type_error_with_path(self, path):
    value = 1

    with self.assertRaises(TypeError):
      file_utils.write_saved_model(value, path)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_overwrite(self, overwrite):
    temp_dir = self.create_tempdir()
    value = 1

    with self.assertRaises(TypeError):
      file_utils.write_saved_model(value, temp_dir, overwrite)


if __name__ == '__main__':
  absltest.main()
