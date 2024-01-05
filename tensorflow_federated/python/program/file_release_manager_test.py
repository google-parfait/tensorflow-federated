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

from collections.abc import Iterable, Mapping, Sequence
import csv
import os
import os.path
import shutil
from typing import Union
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.program import file_release_manager
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils


def _read_values_from_csv(
    file_path: Union[str, os.PathLike[str]]
) -> tuple[list[str], list[dict[str, release_manager.ReleasableStructure]]]:
  with tf.io.gfile.GFile(file_path, 'r') as file:
    reader = csv.DictReader(file)
    fieldnames = list(reader.fieldnames)
    values = list(reader)
  return fieldnames, values


def _write_values_to_csv(
    file_path: Union[str, os.PathLike[str]],
    fieldnames: Sequence[str],
    values: Iterable[Mapping[str, release_manager.ReleasableStructure]],
) -> None:
  with tf.io.gfile.GFile(file_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames)
    writer.writeheader()
    for value in values:
      writer.writerow(value)


class CSVFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_new_file_with_file_path_str(self):
    file_path = self.create_tempfile()
    file_path = file_path.full_path
    os.remove(file_path)
    self.assertFalse(os.path.exists(file_path))

    file_release_manager.CSVFileReleaseManager(file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_creates_new_file_with_file_path_path_like(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    self.assertFalse(os.path.exists(file_path))

    file_release_manager.CSVFileReleaseManager(file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_creates_new_dir_with_file_path_str(self):
    file_path = self.create_tempfile()
    file_path = file_path.full_path
    root_dir = os.path.dirname(file_path)
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.CSVFileReleaseManager(file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_creates_new_dir_with_file_path_path_like(self):
    file_path = self.create_tempfile()
    root_dir = os.path.dirname(file_path)
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.CSVFileReleaseManager(file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_initializes_with_empty_file(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(file_path, fieldnames=['key'], values=[])
    self.assertTrue(os.path.exists(file_path))

    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    self.assertIsNone(release_mngr._latest_key)

  def test_initializes_with_existing_file(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{'key': 1, 'a': 11, 'b': 12}],
    )
    self.assertTrue(os.path.exists(file_path))

    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    self.assertEqual(release_mngr._latest_key, 1)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_file_path(self, file_path):
    with self.assertRaises(TypeError):
      file_release_manager.CSVFileReleaseManager(file_path)

  def test_raises_value_error_with_file_path_empty(self):
    file_path = ''

    with self.assertRaises(ValueError):
      file_release_manager.CSVFileReleaseManager(file_path)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_save_mode(self, save_mode):
    file_path = self.create_tempfile()
    os.remove(file_path)

    with self.assertRaises(TypeError):
      file_release_manager.CSVFileReleaseManager(file_path, save_mode=save_mode)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_key_fieldname(self, key_fieldname):
    file_path = self.create_tempfile()
    os.remove(file_path)

    with self.assertRaises(TypeError):
      file_release_manager.CSVFileReleaseManager(
          file_path, key_fieldname=key_fieldname
      )

  def test_raises_value_error_with_key_fieldname_empty(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    key_fieldname = ''

    with self.assertRaises(ValueError):
      file_release_manager.CSVFileReleaseManager(
          file_path, key_fieldname=key_fieldname
      )

  def test_raises_key_fieldname_not_found_error_with_unknown_key_fieldname(
      self,
  ):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{'key': 1, 'a': 11, 'b': 12}],
    )
    key_fieldname = 'unknown'

    with self.assertRaises(file_release_manager.CSVKeyFieldnameNotFoundError):
      file_release_manager.CSVFileReleaseManager(
          file_path, key_fieldname=key_fieldname
      )

  def test_raises_key_fieldname_not_found_error_with_unknown_file(self):
    file_path = self.create_tempfile()

    with self.assertRaises(file_release_manager.CSVKeyFieldnameNotFoundError):
      file_release_manager.CSVFileReleaseManager(file_path)


class CSVFileReleaseManagerReadValuesTest(parameterized.TestCase):

  def test_returns_values_from_empty_file(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    fieldnames, values = release_mngr._read_values()

    self.assertEqual(fieldnames, ['key'])
    self.assertEqual(values, [])

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values', ['key', 'a', 'b'], []),
      ('one_value', ['key', 'a', 'b'], [{'key': 1, 'a': 11, 'b': 12}]),
      ('two_values', ['key', 'a', 'b'],
       [{'key': 1, 'a': 11, 'b': 12},
        {'key': 2, 'a': 21, 'b': 22}]),
  )
  # pyformat: enable
  def test_returns_values_from_existing_file(self, fieldnames, values):
    file_path = self.create_tempfile()
    _write_values_to_csv(file_path, fieldnames, values)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    actual_fieldnames, actual_values = release_mngr._read_values()

    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = structure_utils.map_structure(str, values)
    self.assertEqual(actual_values, expected_values)


class CSVFileReleaseManagerWriteValuesTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values', ['key', 'a', 'b'], []),
      ('one_value', ['key', 'a', 'b'], [{'key': 1, 'a': 11, 'b': 12}]),
      ('two_values', ['key', 'a', 'b'],
       [{'key': 1, 'a': 11, 'b': 12},
        {'key': 2, 'a': 21, 'b': 22}]),
  )
  # pyformat: enable
  def test_writes_values_to_empty_file(self, fieldnames, values):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    release_mngr._write_values(fieldnames, values)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = structure_utils.map_structure(str, values)
    self.assertEqual(actual_values, expected_values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values', ['key', 'a', 'b'], []),
      ('one_value', ['key', 'a', 'b'], [{'key': 1, 'a': 11, 'b': 12}]),
      ('two_values', ['key', 'a', 'b'],
       [{'key': 1, 'a': 11, 'b': 12},
        {'key': 2, 'a': 21, 'b': 22}]),
  )
  # pyformat: enable
  def test_writes_values_to_existing_file(self, fieldnames, values):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{'key': 1, 'a': 11, 'b': 12}],
    )
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    release_mngr._write_values(fieldnames, values)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = structure_utils.map_structure(str, values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list_none', [None]),
      ('list_bool', [True]),
      ('list_int', [1]),
  )
  def test_raises_type_error_with_fieldnames(self, fieldnames):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    values = [{'key': 1, 'a': 11, 'b': 12}]

    with self.assertRaises(TypeError):
      release_mngr._write_values(fieldnames, values)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list_none', [None]),
      ('list_bool', [True]),
      ('list_int', [1]),
      ('list_str', ['a']),
  )
  def test_raises_type_error_with_values(self, values):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    fieldnames = ['key', 'a', 'b']

    with self.assertRaises(TypeError):
      release_mngr._write_values(fieldnames, values)


class CSVFileReleaseManagerWriteValueTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  async def test_writes_value_to_empty_file(self, value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path, save_mode=file_release_manager.CSVSaveMode.WRITE
    )

    await release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = ['key']
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames]
    )
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [expected_value]
    expected_values = structure_utils.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('empty', {}),
      ('same_fields', {'a': 11, 'b': 12}),
      ('less_fields', {'a': 11}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  async def test_writes_value_to_existing_file(self, value):
    file_path = self.create_tempfile()
    existing_fieldnames = ['key', 'a', 'b']
    existing_value = {'key': 1, 'a': 11, 'b': 12}
    _write_values_to_csv(
        file_path,
        fieldnames=existing_fieldnames,
        values=[existing_value],
    )
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path, save_mode=file_release_manager.CSVSaveMode.WRITE
    )

    await release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = existing_fieldnames.copy()
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames]
    )
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value1 = {name: '' for name in expected_fieldnames}
    expected_value1.update(existing_value)
    expected_value2 = {name: '' for name in expected_fieldnames}
    expected_value2.update(value)
    expected_values = [expected_value1, expected_value2]
    expected_values = structure_utils.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_value(self, value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    with self.assertRaises(TypeError):
      await release_mngr._write_value(value)


class CSVFileReleaseManagerAppendValueTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  async def test_appends_value_to_empty_file(self, value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path, save_mode=file_release_manager.CSVSaveMode.APPEND
    )

    await release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = ['key']
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames]
    )
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [expected_value]
    expected_values = structure_utils.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('empty', {}),
      ('same_fields', {'a': 11, 'b': 12}),
      ('less_fields', {'a': 11}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  async def test_appends_value_to_existing_file(self, value):
    file_path = self.create_tempfile()
    existing_fieldnames = ['key', 'a', 'b']
    existing_value = {'key': 1, 'a': 11, 'b': 12}
    _write_values_to_csv(
        file_path,
        fieldnames=existing_fieldnames,
        values=[existing_value],
    )
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path, save_mode=file_release_manager.CSVSaveMode.APPEND
    )

    await release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = existing_fieldnames.copy()
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames]
    )
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value1 = {name: '' for name in expected_fieldnames}
    expected_value1.update(existing_value)
    expected_value2 = {name: '' for name in expected_fieldnames}
    expected_value2.update(value)
    expected_values = [expected_value1, expected_value2]
    expected_values = structure_utils.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_value(self, value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    with self.assertRaises(TypeError):
      await release_mngr._append_value(value)

  async def test_raises_permission_error(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path, save_mode=file_release_manager.CSVSaveMode.APPEND
    )

    with mock.patch.object(csv.DictWriter, 'writerow') as mock_writerow:
      mock_writerow.side_effect = csv.Error()

      with self.assertRaises(PermissionError):
        await release_mngr._append_value({})


class CSVFileReleaseManagerRemoveValuesGreaterThanKeyTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
  )
  async def test_removes_values_from_empty_file(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    await release_mngr._remove_values_greater_than_key(key)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    self.assertEqual(actual_fieldnames, ['key'])
    self.assertEqual(actual_values, [])

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  async def test_removes_values_from_existing_file(self, key):
    file_path = self.create_tempfile()
    existing_fieldnames = ['key', 'a', 'b']
    existing_values = [
        {'key': 1, 'a': 11, 'b': 12},
        {'key': 2, 'a': 21, 'b': 22},
    ]
    _write_values_to_csv(
        file_path,
        fieldnames=existing_fieldnames,
        values=existing_values,
    )
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    await release_mngr._remove_values_greater_than_key(key)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    if key == 0:
      expected_fieldnames = ['key']
    else:
      expected_fieldnames = existing_fieldnames
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_values = existing_values[0:key]
    expected_values = structure_utils.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
  )
  async def test_does_not_raise_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    try:
      await release_mngr._remove_values_greater_than_key(key)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)

    with self.assertRaises(TypeError):
      await release_mngr._remove_values_greater_than_key(key)


class CSVFileReleaseManagerReleaseTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, [{'key': '1', '': ''}]),
      ('bool', True, [{'key': '1', '': 'True'}]),
      ('int', 1, [{'key': '1', '': '1'}]),
      ('str', 'a', [{'key': '1', '': 'a'}]),
      ('tensor_int', tf.constant(1), [{'key': '1', '': '1'}]),
      ('tensor_str', tf.constant('a'), [{'key': '1', '': 'b\'a\''}]),
      ('tensor_array', tf.constant([1] * 3), [{'key': '1', '': '[1, 1, 1]'}]),
      ('numpy_int', np.int32(1), [{'key': '1', '': '1'}]),
      ('numpy_array',
       np.array([1] * 3, np.int32),
       [{'key': '1', '': '[1, 1, 1]'}]),

      # materializable value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       [{'key': '1', '': '1'}]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [{'key': '1', '': '[1, 2, 3]'}]),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       [{'key': '1', '': 'TestSerializable(a=1, b=2)'}]),

      # other values
      ('attrs',
       program_test_utils.TestAttrs(1, 2),
       [{'key': '1', '': 'TestAttrs(a=1, b=2)'}]),

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
           {
               'key': '1',
               '0': 'True',
               '1': '1',
               '2': 'a',
               '3': '2',
               '4': 'TestSerializable(a=3, b=4)',
           },
       ]),
      ('list_empty', [], [{'key': '1'}]),
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
           {
               'key': '1',
               '0/0': 'True',
               '0/1': '1',
               '0/2': 'a',
               '0/3': '2',
               '0/4': 'TestSerializable(a=3, b=4)',
               '1/0': '5',
           },
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
           {
               'key': '1',
               'a': 'True',
               'b': '1',
               'c': 'a',
               'd': '2',
               'e': 'TestSerializable(a=3, b=4)',
           },
       ]),
      ('dict_empty', {}, [{'key': '1'}]),
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
           {
               'key': '1',
               'x/a': 'True',
               'x/b': '1',
               'x/c': 'a',
               'x/d': '2',
               'x/e': 'TestSerializable(a=3, b=4)',
               'y/a': '5',
           },
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
           {
               'key': '1',
               'a': 'True',
               'b': '1',
               'c': 'a',
               'd': '2',
               'e': 'TestSerializable(a=3, b=4)',
           },
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
           {
               'key': '1',
               'x/a': 'True',
               'x/b': '1',
               'x/c': 'a',
               'x/d': '2',
               'x/e': 'TestSerializable(a=3, b=4)',
               'y/a': '5',
           },
       ]),
  )
  # pyformat: enable
  async def test_writes_value(self, value, expected_value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    key = 1

    await release_mngr.release(value, key=key)

    _, actual_value = _read_values_from_csv(file_path)
    tree.assert_same_structure(actual_value, expected_value)
    self.assertEqual(actual_value, expected_value)

  async def test_remove_values_greater_than_key_with_empty_file(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    value = {'a': 11, 'b': 21}
    key = 1

    with mock.patch.object(
        release_mngr, '_remove_values_greater_than_key'
    ) as mock_remove_values_greater_than_key:
      await release_mngr.release(value, key=key)

      mock_remove_values_greater_than_key.assert_called_once_with(0)

    self.assertEqual(release_mngr._latest_key, key)

  async def test_remove_values_greater_than_key_with_existing_file(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{'key': 1, 'a': 11, 'b': 12}],
    )
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    value = {'a': 11, 'b': 12}
    key = 1

    with mock.patch.object(
        release_mngr, '_remove_values_greater_than_key'
    ) as mock_remove_values_greater_than_key:
      await release_mngr.release(value, key=key)

      mock_remove_values_greater_than_key.assert_called_once_with(0)

    self.assertEqual(release_mngr._latest_key, key)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
  )
  async def test_does_not_raise_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    value = 1

    try:
      await release_mngr.release(value, key=key)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(file_path)
    value = 1

    with self.assertRaises(TypeError):
      await release_mngr.release(value, key=key)


class SavedModelFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_new_dir_with_root_dir_str(self):
    root_dir = self.create_tempdir()
    root_dir = root_dir.full_path
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir)

    self.assertTrue(os.path.exists(root_dir))

  def test_creates_new_dir_with_root_dir_path_like(self):
    root_dir = self.create_tempdir()
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir)

    self.assertTrue(os.path.exists(root_dir))

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_root_dir(self, root_dir):
    with self.assertRaises(TypeError):
      file_release_manager.SavedModelFileReleaseManager(root_dir)

  def test_raises_value_error_with_root_dir_empty(self):
    root_dir = ''

    with self.assertRaises(ValueError):
      file_release_manager.SavedModelFileReleaseManager(root_dir)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_prefix(self, prefix):
    root_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_release_manager.SavedModelFileReleaseManager(root_dir, prefix=prefix)


class SavedModelFileReleaseManagerGetPathForKeyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('standard', '/tmp', 'a_', 123, '/tmp/a_123'),
      ('trailing_slash', '/tmp/', 'a_', 123, '/tmp/a_123'),
      ('no_prefix', '/tmp', '', 123, '/tmp/123'),
  )
  def test_returns_path_with_root_dir_and_prefix(
      self, root_dir, prefix, key, expected_path
  ):
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir, prefix=prefix
    )

    actual_path = release_mngr._get_path_for_key(key)

    self.assertEqual(actual_path, expected_path)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
      ('str', 'a'),
  )
  async def test_does_not_raise_type_error_with_key(self, key):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)

    try:
      release_mngr._get_path_for_key(key)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')


class SavedModelFileReleaseManagerReleaseTest(
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
           True,
           1,
           'a',
           2,
           program_test_utils.TestSerializable(3, 4),
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
           program_test_utils.TestSerializable(3, 4),
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
           program_test_utils.TestSerializable(3, 4),
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
           program_test_utils.TestSerializable(3, 4),
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
           program_test_utils.TestSerializable(3, 4),
           5,
       ]),
  )
  # pyformat: enable
  async def test_writes_value(self, value, expected_value):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    key = 1

    with mock.patch.object(
        file_utils, 'write_saved_model'
    ) as mock_write_saved_model:
      await release_mngr.release(value, key=key)

      mock_write_saved_model.assert_called_once()
      call = mock_write_saved_model.mock_calls[0]
      _, args, kwargs = call
      actual_value, actual_path = args
      tree.assert_same_structure(actual_value, expected_value)
      actual_value = program_test_utils.to_python(actual_value)
      expected_value = program_test_utils.to_python(expected_value)
      self.assertEqual(actual_value, expected_value)
      expected_path = os.path.join(root_dir, f'release_{key}')
      self.assertEqual(actual_path, expected_path)
      self.assertEqual(kwargs, {'overwrite': True})

  # pyformat: disable
  @parameterized.named_parameters(
      # serializable values
      ('serializable_value', program_test_utils.TestSerializable(1, 2)),

      # other values
      ('attrs', program_test_utils.TestAttrs(1, 2)),
  )
  # pyformat: enable
  async def test_raises_not_encodable_error_with_value(self, value):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    key = 1

    with self.assertRaises(Exception):
      await release_mngr.release(value, key=key)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('negative', -1),
      ('numpy', np.int32(1)),
      ('str', 'a'),
  )
  async def test_does_not_raise_type_error_with_key(self, key):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    value = 1

    try:
      await release_mngr.release(value, key=key)
    except TypeError:
      self.fail('Raised `TypeError` unexpectedly.')


class SavedModelFileReleaseManagerGetValueTest(
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

      # structures
      ('list',
       [
           True,
           1,
           'a',
           program_test_utils.TestMaterializableValueReference(2),
           None,
       ],
       [
           np.bool_(True),
           np.int32(1),
           b'a',
           np.int32(2),
           None,
       ]),
      ('list_empty', [], []),
      ('list_nested',
       [
           [
               True,
               1,
               'a',
               program_test_utils.TestMaterializableValueReference(2),
               None,
           ],
           [5],
       ],
       [
           [
               np.bool_(True),
               np.int32(1),
               b'a',
               np.int32(2),
               None,
           ],
           [np.int32(5)],
       ]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
           'e': None,
       },
       {
           'a': np.bool_(True),
           'b': np.int32(1),
           'c': b'a',
           'd': np.int32(2),
           'e': None,
       }),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
               'e': None,
           },
           'y': {'a': 5},
       },
       {
           'x': {
               'a': np.bool_(True),
               'b': np.int32(1),
               'c': b'a',
               'd': np.int32(2),
               'e': None,
           },
           'y': {'a': np.int32(5)}
       }),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2),
           e=None,
       ),
       program_test_utils.TestNamedTuple1(
           a=np.bool_(True),
           b=np.int32(1),
           c=b'a',
           d=np.int32(2),
           e=None,
       )),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2),
               e=None,
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       ),
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=np.bool_(True),
               b=np.int32(1),
               c=b'a',
               d=np.int32(2),
               e=None,
           ),
           y=program_test_utils.TestNamedTuple2(a=np.int32(5)),
       )),
  )
  # pyformat: enable
  async def test_returns_saved_value(self, value, expected_value):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    key = 1
    await release_mngr.release(value, key=key)
    structure = value

    actual_value = await release_mngr.get_value(key, structure)

    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  async def test_returns_saved_value_with_key(self, key):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    for key in range(3):
      await release_mngr.release(f'value_{key}', key=key)
    structure = 'value'

    actual_value = await release_mngr.get_value(key, structure)

    expected_value = f'value_{key}'.encode()
    self.assertEqual(actual_value, expected_value)

  async def test_raises_released_value_not_found_error_with_no_saved_value(
      self,
  ):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    key = 1
    structure = 'value'

    with self.assertRaises(release_manager.ReleasedValueNotFoundError):
      await release_mngr.get_value(key, structure)

  async def test_raises_released_value_not_found_error_with_unknown_key(self):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(root_dir)
    value = 'value_1'
    key = 1
    await release_mngr.release(value, key=key)
    unknown_key = 10
    structure = 'value'

    with self.assertRaises(release_manager.ReleasedValueNotFoundError):
      await release_mngr.get_value(unknown_key, structure)


if __name__ == '__main__':
  absltest.main()
