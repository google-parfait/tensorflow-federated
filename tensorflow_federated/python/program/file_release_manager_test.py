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

import os.path
import shutil

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import file_release_manager
from tensorflow_federated.python.program import test_utils


class SavedModelFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_dir = self.create_tempdir()
    shutil.rmtree(temp_dir)
    self.assertFalse(os.path.exists(temp_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir=temp_dir)

    self.assertTrue(os.path.exists(temp_dir))

  def test_does_not_raise_type_error_with_root_dir_str(self):
    try:
      file_release_manager.SavedModelFileReleaseManager(
          root_dir='/tmp', prefix='a_')
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_does_not_raise_type_error_with_root_dir_path_like(self):
    temp_dir = self.create_tempdir()

    try:
      file_release_manager.SavedModelFileReleaseManager(
          root_dir=temp_dir, prefix='a_')
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
  def test_returns_path_with_root_dir_and_prefix(self, root_dir, prefix, key,
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
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      release_mngr._get_path_for_key(key)


class SavedModelFileReleaseManagerReleaseTest(parameterized.TestCase,
                                              tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, [None]),
      ('bool', True, [tf.constant(True)]),
      ('int', 1, [tf.constant(1)]),
      ('str', 'a', [tf.constant('a')]),
      ('list',
       [True, 1, 'a'],
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('list_empty', [], []),
      ('list_nested',
       [[True, 1], ['a']],
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('dict',
       {'a': True, 'b': 1, 'c': 'a'},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('dict_empty', {}, []),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('attr',
       test_utils.TestAttrObject1(True, 1),
       [tf.constant(True), tf.constant(1)]),
      ('attr_nested',
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('tensor_int', tf.constant(1), [tf.constant(1)]),
      ('tensor_str', tf.constant('a'), [tf.constant('a')]),
      ('tensor_2d', tf.ones((2, 3)), [tf.ones((2, 3))]),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('numpy_int', np.int32(1), [tf.constant(1)]),
      ('numpy_2d', np.ones((2, 3)), [tf.ones((2, 3))]),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('server_array_reference', test_utils.TestServerArrayReference(1), [1]),
      ('server_array_reference_nested',
       {'a': [test_utils.TestServerArrayReference(True),
              test_utils.TestServerArrayReference(1)],
        'b': test_utils.TestServerArrayReference('a')},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('materialized_values_and_value_references',
       [1, test_utils.TestServerArrayReference(2)],
       [tf.constant(1), tf.constant(2)]),
  )
  # pyformat: enable
  def test_writes_value(self, value, expected_value):
    temp_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=temp_dir, prefix='a_')

    release_mngr.release(value, 1)

    path = release_mngr._get_path_for_key(1)
    module = tf.saved_model.load(path)
    actual_value = module()
    self.assertEqual(type(actual_value), type(expected_value))
    self.assertAllEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(TypeError):
      release_mngr.release(1, key)


if __name__ == '__main__':
  absltest.main()
