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

import csv
import os
import os.path
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import file_release_manager
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import program_test_utils


def _read_values_from_csv(
    file_path: Union[str, os.PathLike[str]]
) -> Tuple[List[str], List[Dict[str, Any]]]:
  with tf.io.gfile.GFile(file_path, 'r') as file:
    reader = csv.DictReader(file)
    fieldnames = list(reader.fieldnames)
    values = list(reader)
  return fieldnames, values


def _write_values_to_csv(file_path: Union[str, os.PathLike[str]],
                         fieldnames: Sequence[str],
                         values: Iterable[Mapping[str, Any]]) -> None:
  with tf.io.gfile.GFile(file_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for value in values:
      writer.writerow(value)


class CSVFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_new_file_with_file_path_str(self):
    file_path = self.create_tempfile()
    file_path = file_path.full_path
    os.remove(file_path)
    self.assertFalse(os.path.exists(file_path))

    file_release_manager.CSVFileReleaseManager(file_path=file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_creates_new_file_with_file_path_path_like(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    self.assertFalse(os.path.exists(file_path))

    file_release_manager.CSVFileReleaseManager(file_path=file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_creates_new_dir_with_file_path_str(self):
    file_path = self.create_tempfile()
    file_path = file_path.full_path
    root_dir = os.path.dirname(file_path)
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.CSVFileReleaseManager(file_path=file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_creates_new_dir_with_file_path_path_like(self):
    file_path = self.create_tempfile()
    root_dir = os.path.dirname(file_path)
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.CSVFileReleaseManager(file_path=file_path)

    self.assertTrue(os.path.exists(file_path))

  def test_initializes_with_empty_file(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(file_path=file_path, fieldnames=['key'], values=[])
    self.assertTrue(os.path.exists(file_path))

    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    self.assertIsNone(release_mngr._latest_key)

  def test_initializes_with_existing_file(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path=file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{
            'key': 1,
            'a': 11,
            'b': 12
        }])
    self.assertTrue(os.path.exists(file_path))

    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    self.assertEqual(release_mngr._latest_key, 1)

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
      ('list', []),
  )
  def test_raises_type_error_with_file_path(self, file_path):
    with self.assertRaises(TypeError):
      file_release_manager.CSVFileReleaseManager(file_path=file_path)

  def test_raises_value_error_with_file_path_empty(self):
    with self.assertRaises(ValueError):
      file_release_manager.CSVFileReleaseManager(file_path='')

  def test_does_not_raise_type_error_with_save_mode(self):
    file_path = self.create_tempfile()
    os.remove(file_path)

    try:
      file_release_manager.CSVFileReleaseManager(
          file_path=file_path,
          save_mode=file_release_manager.CSVSaveMode.APPEND)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

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
      file_release_manager.CSVFileReleaseManager(
          file_path=file_path, save_mode=save_mode)

  def test_does_not_raise_type_error_with_key_fieldname(self):
    file_path = self.create_tempfile()
    os.remove(file_path)

    try:
      file_release_manager.CSVFileReleaseManager(
          file_path=file_path, key_fieldname='z')
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

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
          file_path=file_path, key_fieldname=key_fieldname)

  def test_raises_value_error_with_key_fieldname_empty(self):
    file_path = self.create_tempfile()
    os.remove(file_path)

    with self.assertRaises(ValueError):
      file_release_manager.CSVFileReleaseManager(
          file_path=file_path, key_fieldname='')

  def test_raises_incompatible_file_error_with_unknown_key_fieldname(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path=file_path,
        fieldnames=['z', 'a', 'b'],
        values=[{
            'z': 1,
            'a': 11,
            'b': 12
        }])

    with self.assertRaises(
        file_release_manager.FileReleaseManagerIncompatibleFileError):
      file_release_manager.CSVFileReleaseManager(file_path=file_path)

  def test_raises_incompatible_file_error_with_unknown_file(self):
    file_path = self.create_tempfile()

    with self.assertRaises(
        file_release_manager.FileReleaseManagerIncompatibleFileError):
      file_release_manager.CSVFileReleaseManager(file_path=file_path)


class CSVFileReleaseManagerReadValuesTest(parameterized.TestCase):

  def test_returns_values_from_empty_file(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

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
    _write_values_to_csv(
        file_path=file_path, fieldnames=fieldnames, values=values)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    actual_fieldnames, actual_values = release_mngr._read_values()

    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = tree.map_structure(str, values)
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
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    release_mngr._write_values(fieldnames=fieldnames, values=values)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = tree.map_structure(str, values)
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
        file_path=file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{
            'key': 1,
            'a': 11,
            'b': 12
        }])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    release_mngr._write_values(fieldnames=fieldnames, values=values)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = tree.map_structure(str, values)
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
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)
    values = [{'key': 1, 'a': 11, 'b': 12}]

    with self.assertRaises(TypeError):
      release_mngr._write_values(fieldnames=fieldnames, values=values)

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
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)
    fieldnames = ['key', 'a', 'b']

    with self.assertRaises(TypeError):
      release_mngr._write_values(fieldnames=fieldnames, values=values)


class CSVFileReleaseManagerWriteValueTest(parameterized.TestCase,
                                          unittest.IsolatedAsyncioTestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  # pyformat: enable
  async def test_writes_value_to_empty_file(self, value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.WRITE)

    await release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = ['key']
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [expected_value]
    expected_values = tree.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('same_fields', {'a': 11, 'b': 12}),
      ('less_fields', {'a': 11}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  # pyformat: enable
  async def test_writes_value_to_existing_file(self, value):
    file_path = self.create_tempfile()
    existing_fieldnames = ['key', 'a', 'b']
    existing_value = {'key': 1, 'a': 11, 'b': 12}
    _write_values_to_csv(
        file_path=file_path,
        fieldnames=existing_fieldnames,
        values=[existing_value])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.WRITE)

    await release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = existing_fieldnames.copy()
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value1 = {name: '' for name in expected_fieldnames}
    expected_value1.update(existing_value)
    expected_value2 = {name: '' for name in expected_fieldnames}
    expected_value2.update(value)
    expected_values = [expected_value1, expected_value2]
    expected_values = tree.map_structure(str, expected_values)
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
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    with self.assertRaises(TypeError):
      await release_mngr._write_value(value)


class CSVFileReleaseManagerAppendValueTest(parameterized.TestCase,
                                           unittest.IsolatedAsyncioTestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  # pyformat: enable
  async def test_appends_value_to_empty_file(self, value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.APPEND)

    await release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = ['key']
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [expected_value]
    expected_values = tree.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('same_fields', {'a': 11, 'b': 12}),
      ('less_fields', {'a': 11}),
      ('more_fields', {'a': 11, 'b': 12, 'c': 13}),
  )
  # pyformat: enable
  async def test_appends_value_to_existing_file(self, value):
    file_path = self.create_tempfile()
    existing_fieldnames = ['key', 'a', 'b']
    existing_value = {'key': 1, 'a': 11, 'b': 12}
    _write_values_to_csv(
        file_path=file_path,
        fieldnames=existing_fieldnames,
        values=[existing_value])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.APPEND)

    await release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = existing_fieldnames.copy()
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value1 = {name: '' for name in expected_fieldnames}
    expected_value1.update(existing_value)
    expected_value2 = {name: '' for name in expected_fieldnames}
    expected_value2.update(value)
    expected_values = [expected_value1, expected_value2]
    expected_values = tree.map_structure(str, expected_values)
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
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    with self.assertRaises(TypeError):
      await release_mngr._append_value(value)

  async def test_raises_permission_denied_error(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.APPEND)

    with mock.patch.object(csv.DictWriter, 'writerow') as mock_writerow:
      mock_writerow.side_effect = csv.Error()

      with self.assertRaises(
          file_release_manager.FileReleaseManagerPermissionDeniedError):
        await release_mngr._append_value({})


class CSVFileReleaseManagerRemoveValuesGreaterThanTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
  )
  async def test_removes_values_from_empty_file(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    await release_mngr._remove_values_greater_than(key)

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
        {
            'key': 1,
            'a': 11,
            'b': 12
        },
        {
            'key': 2,
            'a': 21,
            'b': 22
        },
    ]
    _write_values_to_csv(
        file_path=file_path,
        fieldnames=existing_fieldnames,
        values=existing_values)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    await release_mngr._remove_values_greater_than(key)

    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    if key == 0:
      expected_fieldnames = ['key']
    else:
      expected_fieldnames = existing_fieldnames
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_values = existing_values[0:key]
    expected_values = tree.map_structure(str, expected_values)
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    with self.assertRaises(TypeError):
      await release_mngr._remove_values_greater_than(key)


class CSVFileReleaseManagerReleaseTest(parameterized.TestCase,
                                       unittest.IsolatedAsyncioTestCase):

  async def test_calls_remove_values_greater_than_with_empty_file(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)
    value = {'a': 11, 'b': 21}
    type_signature = computation_types.StructType([
        ('a', tf.int32),
        ('b', tf.int32),
    ])

    with mock.patch.object(
        release_mngr,
        '_remove_values_greater_than') as mock_remove_values_greater_than:
      await release_mngr.release(value, type_signature, 1)

      mock_remove_values_greater_than.assert_called_once_with(0)

    self.assertEqual(release_mngr._latest_key, 1)

  async def test_calls_remove_values_greater_than_with_existing_file(self):
    file_path = self.create_tempfile()
    _write_values_to_csv(
        file_path=file_path,
        fieldnames=['key', 'a', 'b'],
        values=[{
            'key': 1,
            'a': 11,
            'b': 12
        }])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)
    value = {'a': 11, 'b': 12}
    type_signature = computation_types.StructType([
        ('a', tf.int32),
        ('b', tf.int32),
    ])

    with mock.patch.object(
        release_mngr,
        '_remove_values_greater_than') as mock_remove_values_greater_than:
      await release_mngr.release(value, type_signature, 1)

      mock_remove_values_greater_than.assert_called_once_with(0)

    self.assertEqual(release_mngr._latest_key, 1)

  async def test_calls_append_value(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.APPEND)
    value = {'a': 11, 'b': 12}
    type_signature = computation_types.StructType([
        ('a', tf.int32),
        ('b', tf.int32),
    ])

    with mock.patch.object(release_mngr, '_append_value') as mock_append_value:
      await release_mngr.release(value, type_signature, 1)

      mock_append_value.assert_called_once_with({'key': 1, 'a': 11, 'b': 12})

    self.assertEqual(release_mngr._latest_key, 1)

  async def test_calls_write_value(self):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path, save_mode=file_release_manager.CSVSaveMode.WRITE)
    value = {'a': 11, 'b': 12}
    type_signature = computation_types.StructType([
        ('a', tf.int32),
        ('b', tf.int32),
    ])

    with mock.patch.object(release_mngr, '_write_value') as mock_write_value:
      await release_mngr.release(value, type_signature, 1)

      mock_write_value.assert_called_once_with({'key': 1, 'a': 11, 'b': 12})

    self.assertEqual(release_mngr._latest_key, 1)

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none',
       None,
       computation_types.StructType([]),
       [{'key': '1', '': ''}]),
      ('bool',
       True,
       computation_types.TensorType(tf.bool),
       [{'key': '1', '': 'True'}]),
      ('int',
       1,
       computation_types.TensorType(tf.int32),
       [{'key': '1', '': '1'}]),
      ('str',
       'a',
       computation_types.TensorType(tf.string),
       [{'key': '1', '': 'a'}]),
      ('tensor_int',
       tf.constant(1),
       computation_types.TensorType(tf.int32),
       [{'key': '1', '': '1'}]),
      ('tensor_str',
       tf.constant('a'),
       computation_types.TensorType(tf.string),
       [{'key': '1', '': 'b\'a\''}]),
      ('tensor_array',
       tf.ones([3], tf.int32),
       computation_types.TensorType(tf.int32, [3]),
       [{'key': '1', '': '[1, 1, 1]'}]),
      ('numpy_int',
       np.int32(1),
       computation_types.TensorType(tf.int32),
       [{'key': '1', '': '1'}]),
      ('numpy_array',
       np.ones([3], np.int32),
       computation_types.TensorType(tf.int32, [3]),
       [{'key': '1', '': '[1, 1, 1]'}]),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32),
       [{'key': '1', '': '1'}]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(tf.int32),
       [{'key': '1', '': '[1, 2, 3]'}]),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       computation_types.SequenceType([tf.bool, tf.int32, tf.string]),
       [{'key': '1', '0': 'True', '1': '1', '2': 'a'}]),
      ('list_empty', [], computation_types.SequenceType([]), [{'key': '1'}]),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       computation_types.SequenceType([[tf.bool, tf.int32], [tf.string]]),
       [{'key': '1', '0/0': 'True', '0/1': '1', '1/0': 'a'}]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       computation_types.SequenceType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string)]),
       [{'key': '1', 'a': 'True', 'b': '1', 'c': 'a'}]),
      ('dict_empty', {}, computation_types.SequenceType([]), [{'key': '1'}]),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       computation_types.SequenceType([
           ('x', [('a', tf.bool), ('b', tf.int32)]),
           ('y', [('c', tf.string)])]),
       [{'key': '1', 'x/a': 'True', 'x/b': '1', 'y/c': 'a'}]),
      ('attr',
       program_test_utils.TestAttrObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.SequenceType([('a', tf.bool), ('b', tf.int32)]),
       [{'key': '1', 'a': 'True', 'b': '1'}]),
      ('attr_nested',
       program_test_utils.TestAttrObj2(
           program_test_utils.TestAttrObj2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObj1('a')),
       computation_types.SequenceType([
           ('a', [('a', tf.bool), ('b', tf.int32)]),
           ('b', [('c', tf.string)])]),
       [{'key': '1', 'a/a': 'True', 'a/b': '1', 'b/a': 'a'}]),
      ('namedtuple',
       program_test_utils.TestNamedtupleObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.SequenceType([('a', tf.bool), ('b', tf.int32)]),
       [{'key': '1', 'a': 'True', 'b': '1'}]),
      ('namedtuple_nested',
       program_test_utils.TestNamedtupleObj2(
           program_test_utils.TestNamedtupleObj2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestNamedtupleObj1('a')),
       computation_types.SequenceType([
           ('a', [('a', tf.bool), ('b', tf.int32)]),
           ('b', [('c', tf.string)])]),
       [{'key': '1', 'a/a': 'True', 'a/b': '1', 'b/a': 'a'}]),
  )
  # pyformat: enable
  async def test_writes_value(self, value, type_signature, expected_value):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)

    await release_mngr.release(value, type_signature, 1)

    _, actual_value = _read_values_from_csv(file_path)
    program_test_utils.assert_types_equal(actual_value, expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('negative_1', -1),
      ('0', 0),
      ('1', 1),
  )
  async def test_does_not_raise_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    try:
      await release_mngr.release(value, type_signature, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_key(self, key):
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=file_path)
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(TypeError):
      await release_mngr.release(value, type_signature, key)


class SavedModelFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_new_dir_with_root_dir_str(self):
    root_dir = self.create_tempdir()
    root_dir = root_dir.full_path
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir=root_dir)

    self.assertTrue(os.path.exists(root_dir))

  def test_creates_new_dir_with_root_dir_path_like(self):
    root_dir = self.create_tempdir()
    shutil.rmtree(root_dir)
    self.assertFalse(os.path.exists(root_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir=root_dir)

    self.assertTrue(os.path.exists(root_dir))

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
    root_dir = self.create_tempdir()

    with self.assertRaises(TypeError):
      file_release_manager.SavedModelFileReleaseManager(
          root_dir=root_dir, prefix=prefix)


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
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=root_dir, prefix='a_')

    with self.assertRaises(TypeError):
      release_mngr._get_path_for_key(key)


class SavedModelFileReleaseManagerReleaseTest(parameterized.TestCase,
                                              unittest.IsolatedAsyncioTestCase,
                                              tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, computation_types.StructType([]), [None]),
      ('bool', True, computation_types.TensorType(tf.bool), [True]),
      ('int', 1, computation_types.TensorType(tf.int32), [1]),
      ('str', 'a', computation_types.TensorType(tf.string), ['a']),
      ('tensor_int',
       tf.constant(1),
       computation_types.TensorType(tf.int32),
       [tf.constant(1)]),
      ('tensor_str',
       tf.constant('a'),
       computation_types.TensorType(tf.string),
       [tf.constant('a')]),
      ('tensor_array',
       tf.ones([3], tf.int32),
       computation_types.TensorType(tf.int32, [3]),
       [tf.ones([3])]),
      ('numpy_int',
       np.int32(1),
       computation_types.TensorType(tf.int32),
       [np.int32(1)]),
      ('numpy_array',
       np.ones([3], np.int32),
       computation_types.TensorType(tf.int32, [3]),
       [np.ones([3], np.int32)]),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       computation_types.TensorType(tf.int32),
       [1]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       computation_types.SequenceType(tf.int32),
       [tf.data.Dataset.from_tensor_slices([1, 2, 3])]),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       computation_types.SequenceType([tf.bool, tf.int32, tf.string]),
       [True, 1, 'a']),
      ('list_empty', [], computation_types.SequenceType([]), []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       computation_types.SequenceType([[tf.bool, tf.int32], [tf.string]]),
       [True, 1, 'a']),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       computation_types.SequenceType([
           ('a', tf.bool),
           ('b', tf.int32),
           ('c', tf.string)]),
       [True, 1, 'a']),
      ('dict_empty', {}, computation_types.SequenceType([]), []),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       computation_types.SequenceType([
           ('x', [('a', tf.bool), ('b', tf.int32)]),
           ('y', [('c', tf.string)])]),
       [True, 1, 'a']),
      ('attr',
       program_test_utils.TestAttrObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.SequenceType([('a', tf.bool), ('b', tf.int32)]),
       [True, 1]),
      ('attr_nested',
       program_test_utils.TestAttrObj2(
           program_test_utils.TestAttrObj2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObj1('a')),
       computation_types.SequenceType([
           ('a', [('a', tf.bool), ('b', tf.int32)]),
           ('b', [('c', tf.string)])]),
       [True, 1, 'a']),
      ('namedtuple',
       program_test_utils.TestNamedtupleObj2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       computation_types.SequenceType([('a', tf.bool), ('b', tf.int32)]),
       [True, 1]),
      ('namedtuple_nested',
       program_test_utils.TestNamedtupleObj2(
           program_test_utils.TestNamedtupleObj2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestNamedtupleObj1('a')),
       computation_types.SequenceType([
           ('a', [('a', tf.bool), ('b', tf.int32)]),
           ('b', [('c', tf.string)])]),
       [True, 1, 'a']),
  )
  # pyformat: enable
  async def test_writes_value(self, value, type_signature, expected_value):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=root_dir, prefix='a_')

    with mock.patch.object(file_utils,
                           'write_saved_model') as mock_write_saved_model:
      await release_mngr.release(value, type_signature, 1)

      mock_write_saved_model.assert_called_once()
      call = mock_write_saved_model.mock_calls[0]
      _, args, kwargs = call
      actual_value, actual_path = args
      program_test_utils.assert_types_equal(actual_value, expected_value)

      def _normalize(value: Any) -> Any:
        if isinstance(value, tf.data.Dataset):
          return list(value)
        return value

      actual_value = tree.map_structure(_normalize, actual_value)
      expected_value = tree.map_structure(_normalize, expected_value)
      self.assertAllEqual(actual_value, expected_value)
      expected_path = os.path.join(root_dir, 'a_1')
      self.assertEqual(actual_path, expected_path)
      self.assertEqual(kwargs, {'overwrite': True})

  @parameterized.named_parameters(
      ('negative_1', -1),
      ('0', 0),
      ('1', 1),
  )
  async def test_does_not_raise_type_error_with_key(self, key):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=root_dir, prefix='a_')
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    try:
      await release_mngr.release(value, type_signature, key)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  async def test_raises_type_error_with_key(self, key):
    root_dir = self.create_tempdir()
    release_mngr = file_release_manager.SavedModelFileReleaseManager(
        root_dir=root_dir, prefix='a_')
    value = 1
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(TypeError):
      await release_mngr.release(value, type_signature, key)


if __name__ == '__main__':
  absltest.main()
