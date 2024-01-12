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

    file_program_state_manager.FileProgramStateManager(root_dir)

    self.assertTrue(os.path.exists(root_dir))

  def test_creates_new_dir_with_root_dir_path_like(self):
    root_dir = self.create_tempdir()
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_program_state_manager.FileProgramStateManager(root_dir)

    self.assertTrue(os.path.exists(root_dir))

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_root_dir(self, root_dir):
    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(root_dir)

  def test_raises_value_error_with_root_dir_empty(self):
    root_dir = ''

    with self.assertRaises(ValueError):
      file_program_state_manager.FileProgramStateManager(root_dir)

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
          root_dir, prefix=prefix
      )

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_keep_total(self, keep_total):
    root_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_program_state_manager.FileProgramStateManager(
          root_dir, keep_total=keep_total
      )

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
          root_dir, keep_first=keep_first
      )


class FileProgramStateManagerGetVersionsTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('1', 1),
      ('2', 2),
      ('10', 10),
  )
  async def test_returns_versions_with_program_state_only(self, count):
    root_dir = self.create_tempdir()
    for version in range(count):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=0
    )

    actual_versions = await program_state_mngr.get_versions()

    expected_versions = list(range(count))
    self.assertEqual(actual_versions, expected_versions)

  async def test_returns_versions_with_program_state_and_other_files(
      self,
  ):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
      tempfile.mkstemp(prefix=os.path.join(root_dir, 'file_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=0
    )

    actual_versions = await program_state_mngr.get_versions()

    expected_versions = list(range(10))
    self.assertEqual(actual_versions, expected_versions)

  async def test_returns_versions_with_program_state_and_prefixed_files(
      self,
  ):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
      tempfile.mkstemp(prefix=os.path.join(root_dir, 'program_state_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=0
    )

    actual_versions = await program_state_mngr.get_versions()

    expected_versions = list(range(10))
    self.assertEqual(actual_versions, expected_versions)

  async def test_returns_none_if_root_does_not_exist(self):
    root_dir = self.create_tempdir()
    shutil.rmtree(root_dir)
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    versions = await program_state_mngr.get_versions()

    self.assertIsNone(versions)

  async def test_returns_none_with_empty_root(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    versions = await program_state_mngr.get_versions()

    self.assertIsNone(versions)

  async def test_returns_none_with_other_files_only(self):
    root_dir = self.create_tempdir()
    for _ in range(10):
      tempfile.mkstemp(prefix=os.path.join(root_dir, 'file_'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    versions = await program_state_mngr.get_versions()

    self.assertIsNone(versions)


class FileProgramStateManagerGetVersionForPathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', '/tmp/program_state_123', 123),
      ('no_root_dir', 'program_state_123', 123),
      ('top_level', '/program_state_123', 123),
  )
  def test_returns_version_with_path(self, path, expected_version):
    root_dir = '/tmp'
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    actual_version = program_state_mngr._get_version_for_path(path)

    self.assertEqual(actual_version, expected_version)

  @parameterized.named_parameters(
      ('wrong_prefix', '/tmp/wrong_123'),
      ('no_version', '/tmp/program_state_'),
      ('not_version', '/tmp/program_state_abc'),
  )
  def test_returns_none_with_path(self, path):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

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
        root_dir
    )

    with self.assertRaises(TypeError):
      program_state_mngr._get_version_for_path(path)


class FileProgramStateManagerGetPathForVersionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', '/tmp', 'prefix_', 123, '/tmp/prefix_123'),
      ('trailing_slash', '/tmp/', 'prefix_', 123, '/tmp/prefix_123'),
      ('no_prefix', '/tmp', '', 123, '/tmp/123'),
  )
  def test_returns_path_with_root_dir_and_prefix(
      self, root_dir, prefix, version, expected_path
  ):
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, prefix=prefix
    )

    actual_path = program_state_mngr._get_path_for_version(version)

    self.assertEqual(actual_path, expected_path)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
  )
  async def test_does_not_raise_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    try:
      program_state_mngr._get_path_for_version(version)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    with self.assertRaises(TypeError):
      program_state_mngr._get_path_for_version(version)


class FileProgramStateManagerLoadTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, np.bool_(True)),
      ('int', 1, np.int32(1)),
      ('str', 'a', b'a'),
      ('tensor_int', tf.constant(1), np.int32(1)),
      ('tensor_str', tf.constant('a'), b'a'),
      ('tensor_array', tf.constant([1] * 3), np.array([1] * 3, np.int32)),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_array', np.array([1] * 3, np.int32), np.array([1] * 3, np.int32)),

      # materializable value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       np.int32(1)),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       program_test_utils.TestSerializable(1, 2)),

      # structures
      ('list',
       [
           True,
           1,
           'a',
           program_test_utils.TestMaterializableValueReference(2),
           program_test_utils.TestSerializable(3, 4),
       ],
       [
           np.bool_(True),
           np.int32(1),
           b'a',
           np.int32(2),
           program_test_utils.TestSerializable(3, 4),
       ]),
      ('list_empty', [], []),
      ('list_nested',
       [
           [
               True,
               1,
               'a',
               program_test_utils.TestMaterializableValueReference(2),
               program_test_utils.TestSerializable(3, 4),
           ],
           [5],
       ],
       [
           [
               np.bool_(True),
               np.int32(1),
               b'a',
               np.int32(2),
               program_test_utils.TestSerializable(3, 4),
           ],
           [np.int32(5)],
       ]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
           'e': program_test_utils.TestSerializable(3, 4),
       },
       {
           'a': np.bool_(True),
           'b': np.int32(1),
           'c': b'a',
           'd': np.int32(2),
           'e': program_test_utils.TestSerializable(3, 4),
       }),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': 5},
       },
       {
           'x': {
               'a': np.bool_(True),
               'b': np.int32(1),
               'c': b'a',
               'd': np.int32(2),
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': np.int32(5)}
       }),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2),
           e=program_test_utils.TestSerializable(3, 4),
       ),
       program_test_utils.TestNamedTuple1(
           a=np.bool_(True),
           b=np.int32(1),
           c=b'a',
           d=np.int32(2),
           e=program_test_utils.TestSerializable(3, 4),
       )),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2),
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       ),
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=np.bool_(True),
               b=np.int32(1),
               c=b'a',
               d=np.int32(2),
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=np.int32(5)),
       )),
  )
  # pyformat: enable
  async def test_returns_program_state(self, program_state, expected_state):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    version = 1
    await program_state_mngr.save(program_state, version)
    structure = program_state

    actual_state = await program_state_mngr.load(version, structure)

    tree.assert_same_structure(actual_state, expected_state)
    actual_state = program_test_utils.to_python(actual_state)
    expected_state = program_test_utils.to_python(expected_state)
    self.assertEqual(actual_state, expected_state)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  async def test_returns_program_state_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    for i in range(3):
      await program_state_mngr.save(program_state=f'state_{i}', version=i)
    structure = 'state'

    actual_state = await program_state_mngr.load(version, structure)

    expected_state = f'state_{version}'.encode()
    self.assertEqual(actual_state, expected_state)

  async def test_raises_program_state_not_found_error_with_no_program_state(
      self,
  ):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    version = 0
    structure = 'state'

    with self.assertRaises(program_state_manager.ProgramStateNotFoundError):
      await program_state_mngr.load(version, structure)

  async def test_raises_program_state_not_found_error_with_unknown_version(
      self,
  ):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    program_state = 'state_1'
    version = 1
    await program_state_mngr.save(program_state, version)
    unknown_version = 0
    structure = 'state'

    with self.assertRaises(program_state_manager.ProgramStateNotFoundError):
      await program_state_mngr.load(unknown_version, structure)

  async def test_raises_value_error_with_incorrect_structure(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    program_state = 'state_1'
    version = 1
    await program_state_mngr.save(program_state, version)
    structure = []

    with self.assertRaises(ValueError):
      await program_state_mngr.load(version, structure)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    structure = 'state'

    with self.assertRaises(TypeError):
      await program_state_mngr.load(version, structure)


class FileProgramStateManagerRemoveTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  async def test_removes_program_state_with_version(self, version):
    root_dir = self.create_tempdir()
    for version in range(3):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    await program_state_mngr._remove(version)

    expected_dirs = ['program_state_0', 'program_state_1', 'program_state_2']
    expected_dirs.remove(f'program_state_{version}')
    self.assertCountEqual(os.listdir(root_dir), expected_dirs)

  async def test_removes_program_state_last(self):
    root_dir = self.create_tempdir()
    os.mkdir(os.path.join(root_dir, 'program_state_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    version = 1

    await program_state_mngr._remove(version)

    self.assertCountEqual(os.listdir(root_dir), [])

  async def test_noops_with_unknown_version(self):
    root_dir = self.create_tempdir()
    os.mkdir(os.path.join(root_dir, 'program_state_1'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    version = 10

    await program_state_mngr._remove(version)

    self.assertCountEqual(os.listdir(root_dir), ['program_state_1'])

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
  )
  async def test_does_not_raise_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    try:
      await program_state_mngr._remove(version)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    with self.assertRaises(TypeError):
      await program_state_mngr._remove(version)


class FileProgramStateManagerRemoveOldProgramStateTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_does_not_remove_program_state_with_keep_total_0(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=0
    )

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(
        os.listdir(root_dir), [f'program_state_{i}' for i in range(10)]
    )

  async def test_removes_program_state_with_keep_first_true(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=3, keep_first=True
    )

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(
        os.listdir(root_dir),
        ['program_state_0', 'program_state_8', 'program_state_9'],
    )

  async def test_removes_program_state_with_keep_first_false(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=3, keep_first=False
    )

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(
        os.listdir(root_dir),
        ['program_state_7', 'program_state_8', 'program_state_9'],
    )

  async def test_removes_all_program_state_except_for_the_first(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=1, keep_first=True
    )

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(root_dir), ['program_state_0'])

  async def test_removes_all_program_state_except_for_the_last(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir, keep_total=1, keep_first=False
    )

    await program_state_mngr._remove_old_program_state()

    self.assertCountEqual(os.listdir(root_dir), ['program_state_9'])


class FileProgramStateManagerRemoveAllTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_noops_with_no_program_state(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    await program_state_mngr.remove_all()

    self.assertEqual(os.listdir(root_dir), [])

  async def test_removes_all_program_state(self):
    root_dir = self.create_tempdir()
    for version in range(10):
      os.mkdir(os.path.join(root_dir, f'program_state_{version}'))
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )

    await program_state_mngr.remove_all()

    self.assertEqual(os.listdir(root_dir), [])


class FileProgramStateManagerSaveTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, [None]),
      ('bool', True, [True]),
      ('int', 1, [1]),
      ('str', 'a', ['a']),
      ('tensor_int', tf.constant(1), [tf.constant(1)]),
      ('tensor_str', tf.constant('a'), [tf.constant('a')]),
      ('tensor_array', tf.constant([1] * 3), [tf.constant([1] * 3)]),
      ('numpy_int', np.int32(1), [np.int32(1)]),
      ('numpy_array',
       np.array([1] * 3, np.int32),
       [np.array([1] * 3, np.int32)]),

      # materializable value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       [1]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [tf.data.Dataset.from_tensor_slices([1, 2, 3])]),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       [program_test_utils.TestSerializable(1, 2).to_bytes()]),

      # structures
      ('list',
       [
           True,
           1,
           'a',
           program_test_utils.TestMaterializableValueReference(2),
           program_test_utils.TestSerializable(3, 4),
       ],
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4).to_bytes(),
       ]),
      ('list_empty', [], []),
      ('list_nested',
       [
           [
               True,
               1,
               'a',
               program_test_utils.TestMaterializableValueReference(2),
               program_test_utils.TestSerializable(3, 4),
           ],
           [5],
       ],
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4).to_bytes(),
           5,
       ]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
           'e': program_test_utils.TestSerializable(3, 4),
       },
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4).to_bytes(),
       ]),
      ('dict_empty', {}, []),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': 5},
       },
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4).to_bytes(),
           5,
       ]),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2),
           e=program_test_utils.TestSerializable(3, 4),
       ),
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4).to_bytes(),
       ]),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2),
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       ),
       [
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4).to_bytes(),
           5,
       ]),
  )
  # pyformat: enable
  async def test_writes_program_state(self, program_state, expected_value):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    version = 1

    with mock.patch.object(
        file_utils, 'write_saved_model'
    ) as mock_write_saved_model:
      await program_state_mngr.save(program_state, version)

      mock_write_saved_model.assert_called_once()
      call = mock_write_saved_model.mock_calls[0]
      _, args, kwargs = call
      actual_value, actual_path = args
      tree.assert_same_structure(actual_value, expected_value)
      actual_value = program_test_utils.to_python(actual_value)
      expected_value = program_test_utils.to_python(expected_value)
      self.assertEqual(actual_value, expected_value)
      expected_path = os.path.join(root_dir, f'program_state_{version}')
      self.assertEqual(actual_path, expected_path)
      self.assertEqual(kwargs, {})

  async def test_raises_not_encodable_error_program_state_attrs(self):
    program_state = program_test_utils.TestAttrs(1, 2)
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    version = 1

    with self.assertRaises(Exception):
      await program_state_mngr.save(program_state, version)

  async def test_removes_old_program_state(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    program_state = 'state_1'
    version = 1

    with mock.patch.object(
        program_state_mngr, '_remove_old_program_state'
    ) as mock_remove_old_program_state:
      await program_state_mngr.save(program_state, version)

      mock_remove_old_program_state.assert_called_once()

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
  )
  async def test_does_not_raise_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    program_state = 'state'

    try:
      await program_state_mngr.save(program_state, version)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_version(self, version):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    program_state = 'state'

    with self.assertRaises(TypeError):
      await program_state_mngr.save(program_state, version)

  async def test_raises_program_state_exists_error_with_existing_version(self):
    root_dir = self.create_tempdir()
    program_state_mngr = file_program_state_manager.FileProgramStateManager(
        root_dir
    )
    program_state = 'state'
    version = 1

    await program_state_mngr.save(program_state, version)

    with self.assertRaises(program_state_manager.ProgramStateExistsError):
      await program_state_mngr.save(program_state, version)


if __name__ == '__main__':
  absltest.main()
