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
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import file_program_state_manager
from tensorflow_federated.python.program import program_state_manager
from tensorflow_federated.python.program import test_utils


class FileProgramStateManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_dir = self.create_tempdir()
    shutil.rmtree(temp_dir)
    self.assertFalse(os.path.exists(temp_dir))

    file_program_state_manager.FileProgramStateManager(root_dir=temp_dir)

    self.assertTrue(os.path.exists(temp_dir))

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

  def test_returns_versions_with_saved_program_state_and_other_files(self):
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
    shutil.rmtree(temp_dir)
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
      ('standard', '/tmp/a_123'),
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
      ('standard', '/tmp', 'a_', 123, '/tmp/a_123'),
      ('trailing_slash', '/tmp/', 'a_', 123, '/tmp/a_123'),
      ('no_prefix', '/tmp', '', 123, '/tmp/123'),
  )
  def test_returns_path_with_root_dir_and_prefix(self, root_dir, prefix,
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


class FileProgramStateManagerLoadTest(parameterized.TestCase, tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, None),
      ('bool', True, tf.constant(True)),
      ('int', 1, tf.constant(1)),
      ('str', 'a', tf.constant('a')),
      ('list',
       [True, 1, 'a'],
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('list_empty', [], []),
      ('list_nested',
       [[True, 1], ['a']],
       [[tf.constant(True), tf.constant(1)], [tf.constant('a')]]),
      ('dict',
       {'a': True, 'b': 1, 'c': 'a'},
       {'a': tf.constant(True), 'b': tf.constant(1), 'c': tf.constant('a')}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       {'x': {'a': tf.constant(True), 'b': tf.constant(1)},
        'y': {'c': tf.constant('a')}}),
      ('attr',
       test_utils.TestAttrObject1(True, 1),
       test_utils.TestAttrObject1(tf.constant(True), tf.constant(1))),
      ('attr_nested',
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')},
       {'a': [test_utils.TestAttrObject1(tf.constant(True), tf.constant(1))],
        'b': test_utils.TestAttrObject2(tf.constant('a'))}),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_2d', tf.ones((2, 3)), tf.ones((2, 3))),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]}),
      ('numpy_int', np.int32(1), tf.constant(1)),
      ('numpy_2d', np.ones((2, 3)), tf.ones((2, 3))),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]}),
      ('server_array_reference',
       test_utils.TestServerArrayReference(1),
       tf.constant(1)),
      ('server_array_reference_nested',
       {'a': [test_utils.TestServerArrayReference(True),
              test_utils.TestServerArrayReference(1)],
        'b': [test_utils.TestServerArrayReference('a')]},
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]}),
      ('materialized_values_and_value_references',
       [1, test_utils.TestServerArrayReference(2)],
       [tf.constant(1), tf.constant(2)]),
  )
  # pyformat: enable
  def test_returns_saved_program_state(self, program_state,
                                       expected_program_state):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.save(program_state, 1)
    structure = program_state

    actual_program_state = program_state_mngr.load(1, structure)

    self.assertEqual(type(actual_program_state), type(expected_program_state))
    self.assertAllEqual(actual_program_state, expected_program_state)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  def test_returns_saved_program_state_with_version(self, version):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    for i in range(3):
      program_state_mngr.save(f'state_{i}', i)
    structure = 'state'

    actual_program_state = program_state_mngr.load(version, structure)

    expected_program_state = f'state_{version}'
    self.assertEqual(actual_program_state, expected_program_state)

  def test_raises_version_not_found_error_with_no_saved_program_state(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStateNotFoundError):
      _ = program_state_mngr.load(0, None)

  def test_raises_version_not_found_error_with_unknown_version(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.save('state_1', 1)
    structure = 'state'

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStateNotFoundError):
      program_state_mngr.load(10, structure)

  def test_raises_structure_error(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')
    program_state_mngr.save('state_1', 1)
    structure = []

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStructureError):
      program_state_mngr.load(1, structure)

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
      program_state_mngr.load(version, None)


class FileProgramStateManagerRemoveTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  def test_removes_saved_program_state_with_version(self, version):
    temp_dir = self.create_tempdir()
    for version in range(3):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr._remove(version)

    expected_dirs = ['a_0', 'a_1', 'a_2']
    expected_dirs.remove(f'a_{version}')
    self.assertCountEqual(os.listdir(temp_dir), expected_dirs)

  def test_removes_saved_program_state_last(self):
    temp_dir = self.create_tempdir()
    os.mkdir(os.path.join(temp_dir, 'a_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr._remove(1)

    self.assertCountEqual(os.listdir(temp_dir), [])

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

    self.assertCountEqual(os.listdir(temp_dir), [f'a_{i}' for i in range(10)])

  def test_removes_saved_program_state_with_keep_first_true(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=3, keep_first=True)

    program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(temp_dir), ['a_0', 'a_8', 'a_9'])

  def test_removes_saved_program_state_with_keep_first_false(self):
    temp_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(temp_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=3, keep_first=False)

    program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(temp_dir), ['a_7', 'a_8', 'a_9'])


class FileProgramStateManagerSaveTest(parameterized.TestCase, tf.test.TestCase):

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
        'b': [test_utils.TestServerArrayReference('a')]},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('materialized_values_and_value_references',
       [1, test_utils.TestServerArrayReference(2)],
       [tf.constant(1), tf.constant(2)]),
  )
  # pyformat: enable
  def test_writes_program_state(self, program_state, expected_value):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_', keep_total=0)

    program_state_mngr.save(program_state, 1)

    path = program_state_mngr._get_path_for_version(1)
    module = tf.saved_model.load(path)
    actual_value = module()
    self.assertEqual(type(actual_value), type(expected_value))
    self.assertAllEqual(actual_value, expected_value)

  def test_removes_saved_program_state(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    with mock.patch.object(
        program_state_mngr,
        '_remove_old_program_state') as mock_remove_old_program_state:
      program_state_mngr.save('state_1', 1)

      mock_remove_old_program_state.assert_called_once()

  def test_raises_version_already_exists_error_with_existing_version(self):
    temp_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=temp_dir, prefix='a_')

    program_state_mngr.save('state_1', 1)

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStateAlreadyExistsError):
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


if __name__ == '__main__':
  absltest.main()
