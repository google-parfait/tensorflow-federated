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
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from google.protobuf import message
from tensorflow_federated.python.program import file_utils


class ReadSavedModelTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_returns_value_with_path_str(self):
    module = file_utils._ValueModule(1)
    path = self.create_tempdir()
    path = path.full_path
    tf.saved_model.save(module, path, signatures={})

    actual_value = await file_utils.read_saved_model(path)

    self.assertEqual(actual_value, 1)

  async def test_returns_value_with_path_path_like(self):
    module = file_utils._ValueModule(1)
    path = self.create_tempdir()
    tf.saved_model.save(module, path.full_path, signatures={})

    actual_value = await file_utils.read_saved_model(path)

    self.assertEqual(actual_value, 1)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('list', []),
  )
  async def test_raises_type_error_with_path(self, path):
    with self.assertRaises(TypeError):
      await file_utils.read_saved_model(path)


class WriteSavedModelTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_writes_to_new_file_with_path_str(self):
    value = 1
    path = self.create_tempdir()
    path = path.full_path
    shutil.rmtree(path)
    self.assertFalse(os.path.exists(path))

    await file_utils.write_saved_model(value, path)

    self.assertTrue(os.path.exists(path))
    module = tf.saved_model.load(path)
    actual_value = module()
    self.assertEqual(actual_value, value)

  async def test_writes_to_new_file_with_path_path_like(self):
    value = 1
    path = self.create_tempdir()
    shutil.rmtree(path)
    self.assertFalse(os.path.exists(path))

    await file_utils.write_saved_model(value, path)

    self.assertTrue(os.path.exists(path))
    module = tf.saved_model.load(path.full_path)
    actual_value = module()
    self.assertEqual(actual_value, value)

  async def test_writes_to_existing_file(self):
    value = 1
    path = self.create_tempdir()
    self.assertTrue(os.path.exists(path))
    overwrite = True

    await file_utils.write_saved_model(value, path, overwrite)

    self.assertTrue(os.path.exists(path))
    module = tf.saved_model.load(path.full_path)
    actual_value = module()
    self.assertEqual(actual_value, value)

  async def test_raises_file_exists_error_with_existing_file(self):
    value = 1
    path = self.create_tempdir()

    with self.assertRaises(FileExistsError):
      await file_utils.write_saved_model(value, path)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('bool', True),
      ('list', []),
  )
  async def test_raises_type_error_with_path(self, path):
    value = 1

    with self.assertRaises(TypeError):
      await file_utils.write_saved_model(value, path)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_overwrite(self, overwrite):
    value = 1
    path = self.create_tempdir()

    with self.assertRaises(TypeError):
      await file_utils.write_saved_model(value, path, overwrite)

  async def test_does_not_raise_decode_error_with_large_value(self):
    path = self.create_tempdir()
    path = path.full_path
    shutil.rmtree(path)
    # This value is too large to serialize as a single proto.
    value = [np.zeros(shape=[20_000, 10_000], dtype=np.float32)] * 3

    try:
      await file_utils.write_saved_model(value, path)
    except message.DecodeError:
      self.fail('Raised `DecodeError` unexpectedly.')


if __name__ == '__main__':
  absltest.main()
