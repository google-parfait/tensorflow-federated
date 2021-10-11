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


class WriteSavedModelTest(parameterized.TestCase):

  def test_writes_to_new_file(self):
    temp_dir = self.create_tempdir()
    shutil.rmtree(temp_dir)
    obj = tf.Module()
    self.assertFalse(os.path.exists(temp_dir))

    file_utils.write_saved_model(obj, temp_dir)

    self.assertTrue(os.path.exists(temp_dir))

  def test_writes_to_existing_file(self):
    temp_dir = self.create_tempdir()
    obj = tf.Module()
    overwrite = True
    self.assertTrue(os.path.exists(temp_dir))

    file_utils.write_saved_model(obj, temp_dir, overwrite)

    self.assertTrue(os.path.exists(temp_dir))

  def test_raises_file_already_exists_error_with_existing_file(self):
    temp_dir = self.create_tempdir()
    obj = tf.Module()

    with self.assertRaises(file_utils.FileAlreadyExistsError):
      file_utils.write_saved_model(obj, temp_dir)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('list', []),
  )
  def test_raises_type_error_with_path(self, path):
    obj = tf.Module()

    with self.assertRaises(TypeError):
      file_utils.write_saved_model(obj, path)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_obj(self, obj):
    temp_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_utils.write_saved_model(obj, temp_dir)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_overwrite(self, overwrite):
    temp_dir = self.create_tempdir()
    obj = tf.Module()

    with self.assertRaises(TypeError):
      file_utils.write_saved_model(obj, temp_dir, overwrite)


if __name__ == '__main__':
  absltest.main()
