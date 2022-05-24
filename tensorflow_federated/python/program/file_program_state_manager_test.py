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

import asyncio
import os
import os.path
import shutil
import tempfile
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.program import file_program_state_manager
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import program_state_manager
from tensorflow_federated.python.program import program_test_utils


class FileProgramStateManagerInitTest(parameterized.TestCase):

  def test_creates_new_dir_with_root_dir_str(self):
    root_dir = self.create_tempdir()
    root_dir = root_dir.full_path
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_program_state_manager.FileProgramStateManager(root_dir=root_dir)

    self.assertTrue(os.path.exists(root_dir))

  def test_creates_new_dir_with_root_dir_path_like(self):
    root_dir = self.create_tempdir()
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_program_state_manager.FileProgramStateManager(root_dir=root_dir)

    self.assertTrue(os.path.exists(root_dir))

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
    root_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir=root_dir, prefix=prefix)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_keep_total(self, keep_total):
    root_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir=root_dir, keep_total=keep_total)

  @parameterized.named_parameters(
      ('none', None),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_keep_first(self, keep_first):
    root_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir=root_dir, keep_first=keep_first)


class FileProgramStateManagerGetVersionsTest(parameterized.TestCase,
                                             unittest.IsolatedAsyncioTestCase):

  @parameterized.named_parameters(
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  async def test_returns_versions_with_saved_program_state(self, count):
    root_dir = self.create_tempdir()
    for version in range(count):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=0)

    actual_versions = await program_state_mngr.get_versions()

    expected_versions = list(range(count))
    self.assertEqual(actual_versions, expected_versions)

  async def test_returns_versions_with_saved_program_state_and_other_files(
      self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
      tempfile.mkstemp(prefix=os.path.join(root_dir, 'b_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=0)

    actual_versions = await program_state_mngr.get_versions()

    expected_versions = list(range(10))
    self.assertEqual(actual_versions, expected_versions)

  async def test_returns_versions_with_saved_program_state_and_prefixed_files(
      self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
      tempfile.mkstemp(prefix=os.path.join(root_dir, 'a_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=0)

    actual_versions = await program_state_mngr.get_versions()

    expected_versions = list(range(10))
    self.assertEqual(actual_versions, expected_versions)

  async def test_returns_none_if_root_dir_does_not_exist(self):
    root_dir = self.create_tempdir()
    shutil.rmtree(root_dir)
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    versions = await program_state_mngr.get_versions()

    self.assertIsNone(versions)

  async def test_returns_none_with_no_files(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    versions = await program_state_mngr.get_versions()

    self.assertIsNone(versions)

  async def test_returns_none_with_no_saved_program_state(self):
    root_dir = self.create_tempdir()
    for _ in range(10):
      tempfile.mkstemp(prefix=os.path.join(root_dir, 'a_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    versions = await program_state_mngr.get_versions()

    self.assertIsNone(versions)


class FileProgramStateManagerGetVersionForPathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', '/tmp/a_123', 123),
      ('no_root_dir', 'a_123', 123),
      ('top_level', '/a_123', 123),
  )
  def test_returns_version_with_path(self, path, expected_version):
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir='/tmp', prefix='a_')

    actual_version = program_state_mngr._get_version_for_path(path)

    self.assertEqual(actual_version, expected_version)

  @parameterized.named_parameters(
      ('wrong_prefix', '/tmp/b_123'),
      ('no_version', '/tmp/a_'),
      ('not_version', '/tmp/a_abc'),
  )
  def test_returns_none_with_path(self, path):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    version = program_state_mngr._get_version_for_path(path)

    self.assertIsNone(version)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_path(self, path):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

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
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    with self.assertRaises(TypeError):
      program_state_mngr._get_path_for_version(version)


class FileProgramStateManagerLoadTest(parameterized.TestCase,
                                      unittest.IsolatedAsyncioTestCase,
                                      tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, tf.constant(True)),
      ('int', 1, tf.constant(1)),
      ('str', 'a', tf.constant('a')),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_2d', tf.ones((2, 3)), tf.ones((2, 3))),
      ('numpy_int', np.int32(1), tf.constant(1)),
      ('numpy_2d', np.ones((2, 3)), tf.ones((2, 3))),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       tf.constant(1)),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('list_empty', [], []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       [[tf.constant(True), tf.constant(1)], [tf.constant('a')]]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       {'a': tf.constant(True), 'b': tf.constant(1), 'c': tf.constant('a')}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       {'x': {'a': tf.constant(True), 'b': tf.constant(1)},
        'y': {'c': tf.constant('a')}}),
      ('attr',
       program_test_utils.TestAttrObject2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       program_test_utils.TestAttrObject2(tf.constant(True), tf.constant(1))),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               tf.constant(True), tf.constant(1)),
           program_test_utils.TestAttrObject1(tf.constant('a')))),
  )
  # pyformat: enable
  async def test_returns_saved_program_state(self, program_state,
                                             expected_program_state):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')
    await program_state_mngr.save(program_state, 1)
    structure = program_state

    actual_program_state = await program_state_mngr.load(1, structure)

    if isinstance(actual_program_state, tf.data.Dataset):
      actual_program_state = list(actual_program_state)
    if isinstance(expected_program_state, tf.data.Dataset):
      expected_program_state = list(expected_program_state)
    self.assertAllEqual(actual_program_state, expected_program_state)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  async def test_returns_saved_program_state_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')
    for i in range(3):
      await program_state_mngr.save(f'state_{i}', i)
    structure = 'state'

    actual_program_state = await program_state_mngr.load(version, structure)

    expected_program_state = f'state_{version}'
    self.assertEqual(actual_program_state, expected_program_state)

  async def test_raises_version_not_found_error_with_no_saved_program_state(
      self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStateNotFoundError):
      await program_state_mngr.load(0, None)

  async def test_raises_version_not_found_error_with_unknown_version(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')
    await program_state_mngr.save('state_1', 1)
    structure = 'state'

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStateNotFoundError):
      await program_state_mngr.load(10, structure)

  async def test_raises_structure_error(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')
    await program_state_mngr.save('state_1', 1)
    structure = []

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStructureError):
      await program_state_mngr.load(1, structure)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    with self.assertRaises(TypeError):
      await program_state_mngr.load(version, None)


class FileProgramStateManagerRemoveTest(parameterized.TestCase,
                                        unittest.IsolatedAsyncioTestCase):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  async def test_removes_saved_program_state_with_version(self, version):
    root_dir = self.create_tempdir()
    for version in range(3):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    await program_state_mngr._remove(version)

    expected_dirs = ['a_0', 'a_1', 'a_2']
    expected_dirs.remove(f'a_{version}')
    self.assertCountEqual(os.listdir(root_dir), expected_dirs)

  async def test_removes_saved_program_state_last(self):
    root_dir = self.create_tempdir()
    os.mkdir(os.path.join(root_dir, 'a_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    await program_state_mngr._remove(1)

    self.assertCountEqual(os.listdir(root_dir), [])

  async def test_noops_with_unknown_version(self):
    root_dir = self.create_tempdir()
    os.mkdir(os.path.join(root_dir, 'a_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    await program_state_mngr._remove(10)

    self.assertCountEqual(os.listdir(root_dir), ['a_1'])

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    with self.assertRaises(TypeError):
      await program_state_mngr._remove(version)


class FileProgramStateManagerRemoveOldProgramStateTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def test_does_not_remove_saved_program_state_with_keep_total_0(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=0)

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(root_dir), [f'a_{i}' for i in range(10)])

  async def test_removes_saved_program_state_with_keep_first_true(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=3, keep_first=True)

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(root_dir), ['a_0', 'a_8', 'a_9'])

  async def test_removes_saved_program_state_with_keep_first_false(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'a_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=3, keep_first=False)

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(root_dir), ['a_7', 'a_8', 'a_9'])


class FileProgramStateManagerSaveTest(parameterized.TestCase,
                                      unittest.IsolatedAsyncioTestCase,
                                      tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, [None]),
      ('bool', True, [True]),
      ('int', 1, [1]),
      ('str', 'a', ['a']),
      ('list', [True, 1, 'a'], [True, 1, 'a']),
      ('list_empty', [], []),
      ('list_nested', [[True, 1], ['a']], [True, 1, 'a']),
      ('dict', {'a': True, 'b': 1, 'c': 'a'}, [True, 1, 'a']),
      ('dict_empty', {}, []),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       [True, 1, 'a']),
      ('attr', program_test_utils.TestAttrObject2(True, 1), [True, 1]),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       [True, 1, 'a']),
      ('tensor_int', tf.constant(1), [tf.constant(1)]),
      ('tensor_str', tf.constant('a'), [tf.constant('a')]),
      ('tensor_2d', tf.ones((2, 3)), [tf.ones((2, 3))]),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       [tf.constant(True), tf.constant(1), tf.constant('a')]),
      ('numpy_int', np.int32(1), [np.int32(1)]),
      ('numpy_2d', np.ones((2, 3)), [np.ones((2, 3))]),
      ('numpy_nested',
       {'a': [np.bool_(True), np.int32(1)], 'b': [np.str_('a')]},
       [np.bool_(True), np.int32(1), np.str_('a')]),
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1), [1]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [tf.data.Dataset.from_tensor_slices([1, 2, 3])]),
      ('materializable_value_reference_nested',
       {'a': [program_test_utils.TestMaterializableValueReference(True),
              program_test_utils.TestMaterializableValueReference(1)],
        'b': [program_test_utils.TestMaterializableValueReference('a')]},
       [True, 1, 'a']),
      ('materializable_value_reference_and_materialized_value',
       [1, program_test_utils.TestMaterializableValueReference(2)],
       [1, 2]),
  )
  # pyformat: enable
  async def test_writes_program_state(self, program_state, expected_value):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_', keep_total=0)

    future = asyncio.Future()
    future.set_result(None)
    with mock.patch.object(
        file_utils, 'write_saved_model',
        return_value=future) as mock_write_saved_model:
      await program_state_mngr.save(program_state, 1)

      mock_write_saved_model.assert_called_once()
      call = mock_write_saved_model.mock_calls[0]
      _, args, _ = call
      actual_value, _ = args

      def _normalize(value: Any) -> Any:
        if isinstance(value, tf.data.Dataset):
          return list(value)
        return value

      actual_value = tree.map_structure(_normalize, actual_value)
      expected_value = tree.map_structure(_normalize, expected_value)
      self.assertAllEqual(actual_value, expected_value)

  async def test_removes_saved_program_state(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    future = asyncio.Future()
    future.set_result(None)
    with mock.patch.object(
        program_state_mngr, '_remove_old_program_state',
        return_value=future) as mock_remove_old_program_state:
      await program_state_mngr.save('state_1', 1)

      mock_remove_old_program_state.assert_called_once()

  async def test_raises_version_already_exists_error_with_existing_version(
      self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    await program_state_mngr.save('state_1', 1)

    with self.assertRaises(
        program_state_manager.ProgramStateManagerStateAlreadyExistsError):
      await program_state_mngr.save('state_1', 1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir=root_dir, prefix='a_')

    with self.assertRaises(TypeError):
      await program_state_mngr.save('state', version)


if __name__ == '__main__':
  absltest.main()
