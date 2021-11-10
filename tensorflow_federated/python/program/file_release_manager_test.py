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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.program import file_release_manager
from tensorflow_federated.python.program import test_utils


def _read_values_from_csv(
    file_path: os.PathLike) -> Tuple[List[str], List[Dict[str, Any]]]:
  with tf.io.gfile.GFile(file_path, 'r') as file:
    reader = csv.DictReader(file)
    fieldnames = list(reader.fieldnames)
    values = list(reader)
  return fieldnames, values


def _write_values_to_csv(file_path: os.PathLike, fieldnames: Sequence[str],
                         values: Iterable[Mapping[str, Any]]):
  with tf.io.gfile.GFile(file_path, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for value in values:
      writer.writerow(value)


class CSVFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    self.assertFalse(os.path.exists(temp_file))

    file_release_manager.CSVFileReleaseManager(file_path=temp_file)

    self.assertTrue(os.path.exists(temp_file))

  def test_initializes_with_empty_file(self):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=[file_release_manager._CSV_KEY_FIELDNAME],
        values=[])
    self.assertTrue(os.path.exists(temp_file))

    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    self.assertIsNone(release_mngr._latest_key)

  def test_initializes_with_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=[file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._CSV_KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    self.assertTrue(os.path.exists(temp_file))

    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    self.assertEqual(release_mngr._latest_key, 1)

  def test_does_not_raise_type_error_with_file_path_str(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)

    try:
      file_release_manager.CSVFileReleaseManager(file_path=temp_file.full_path)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_does_not_raise_type_error_with_file_path_path_like(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)

    try:
      file_release_manager.CSVFileReleaseManager(file_path=temp_file)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

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
    temp_file = self.create_tempfile()
    os.remove(temp_file)

    try:
      file_release_manager.CSVFileReleaseManager(
          file_path=temp_file,
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
    temp_file = self.create_tempfile()
    os.remove(temp_file)

    with self.assertRaises(TypeError):
      file_release_manager.CSVFileReleaseManager(
          file_path=temp_file, save_mode=save_mode)

  def test_raises_incompatible_file_error_with_unknown_file(self):
    temp_file = self.create_tempfile()

    with self.assertRaises(
        file_release_manager.FileReleaseManagerIncompatibleFileError):
      file_release_manager.CSVFileReleaseManager(file_path=temp_file)


class CSVFileReleaseManagerReadValuesTest(parameterized.TestCase):

  def test_returns_values_from_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    fieldnames, values = release_mngr._read_values()

    self.assertEqual(fieldnames, [file_release_manager._CSV_KEY_FIELDNAME])
    self.assertEqual(values, [])

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       []),
      ('one_value',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20}]),
      ('two_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20},
        {file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 11, 'b': 21}]),
  )
  # pyformat: enable
  def test_returns_values_from_existing_file(self, fieldnames, values):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file, fieldnames=fieldnames, values=values)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    actual_fieldnames, actual_values = release_mngr._read_values()

    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = tree.map_structure(str, values)
    self.assertEqual(actual_values, expected_values)


class CSVFileReleaseManagerWriteValuesTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       []),
      ('one_value',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20}]),
      ('two_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20},
        {file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 11, 'b': 21}]),
  )
  # pyformat: enable
  def test_writes_values_to_empty_file(self, fieldnames, values):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr._write_values(fieldnames=fieldnames, values=values)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = tree.map_structure(str, values)
    self.assertEqual(actual_values, expected_values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       []),
      ('one_value',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20}]),
      ('two_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20},
        {file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 11, 'b': 21}]),
  )
  # pyformat: enable
  def test_writes_values_to_existing_file(self, fieldnames, values):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=[file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._CSV_KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr._write_values(fieldnames=fieldnames, values=values)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    self.assertEqual(actual_fieldnames, fieldnames)
    expected_values = tree.map_structure(str, values)
    self.assertEqual(actual_values, expected_values)


class CSVFileReleaseManagerWriteValueTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 11, 'b': 21, 'c': 31}),
  )
  # pyformat: enable
  def test_writes_value_to_empty_file(self, value):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.WRITE)

    release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    expected_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME]
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
      ('same_fields', {'a': 11, 'b': 21}),
      ('less_fields', {'a': 11}),
      ('more_fields', {'a': 11, 'b': 21, 'c': 31}),
  )
  # pyformat: enable
  def test_writes_value_to_existing_file(self, value):
    temp_file = self.create_tempfile()
    existing_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b']
    existing_value = {
        file_release_manager._CSV_KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    }
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=existing_fieldnames,
        values=[existing_value])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.WRITE)

    release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
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


class CSVFileReleaseManagerAppendValueTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 11, 'b': 21, 'c': 31}),
  )
  # pyformat: enable
  def test_appends_value_to_empty_file(self, value):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.APPEND)

    release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    expected_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME]
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
      ('same_fields', {'a': 11, 'b': 21}),
      ('less_fields', {'a': 11}),
      ('more_fields', {'a': 11, 'b': 21, 'c': 31}),
  )
  # pyformat: enable
  def test_appends_value_to_existing_file(self, value):
    temp_file = self.create_tempfile()
    existing_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b']
    existing_value = {
        file_release_manager._CSV_KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    }
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=existing_fieldnames,
        values=[existing_value])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.APPEND)

    release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
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

  def test_raises_permission_denied_error(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.APPEND)

    with mock.patch('csv.DictWriter.writerow') as mock_writerow:
      mock_writerow.side_effect = csv.Error()

      with self.assertRaises(
          file_release_manager.FileReleaseManagerPermissionDeniedError):
        release_mngr._append_value({})


class CSVFileReleaseManagerRemoveAllValuesTest(parameterized.TestCase):

  def test_removes_values_from_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr._remove_all_values()

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    self.assertEqual(actual_fieldnames,
                     [file_release_manager._CSV_KEY_FIELDNAME])
    self.assertEqual(actual_values, [])

  # pyformat: disable
  @parameterized.named_parameters(
      ('no_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       []),
      ('one_value',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20}]),
      ('two_values',
       [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
       [{file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 10, 'b': 20},
        {file_release_manager._CSV_KEY_FIELDNAME: 1, 'a': 11, 'b': 21}]),
  )
  # pyformat: enable
  def test_removes_values_from_existing_file(self, fieldnames, values):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file, fieldnames=fieldnames, values=values)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr._remove_all_values()

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    self.assertEqual(actual_fieldnames,
                     [file_release_manager._CSV_KEY_FIELDNAME])
    self.assertEqual(actual_values, [])


class CSVFileReleaseManagerRemoveValuesAfterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
  )
  def test_removes_values_from_empty_file(self, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr._remove_values_after(key)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    self.assertEqual(actual_fieldnames,
                     [file_release_manager._CSV_KEY_FIELDNAME])
    self.assertEqual(actual_values, [])

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
      ('2', 2),
  )
  def test_removes_values_from_existing_file(self, key):
    temp_file = self.create_tempfile()
    existing_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b']
    existing_values = [
        {
            file_release_manager._CSV_KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20
        },
        {
            file_release_manager._CSV_KEY_FIELDNAME: 2,
            'a': 11,
            'b': 21
        },
    ]
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=existing_fieldnames,
        values=existing_values)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr._remove_values_after(key)

    actual_fieldnames, actual_values = _read_values_from_csv(temp_file)
    if key == 0:
      expected_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME]
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
  def test_raises_type_error_with_key(self, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with self.assertRaises(TypeError):
      release_mngr._remove_values_after(key)


class CSVFileReleaseManagerReleaseTest(parameterized.TestCase,
                                       tf.test.TestCase):

  def test_calls_remove_all_values_with_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=[file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._CSV_KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(release_mngr,
                           '_remove_all_values') as mock_remove_all_values:
      release_mngr.release({'a': 11, 'b': 21}, 0)

      mock_remove_all_values.assert_called_once()

    self.assertEqual(release_mngr._latest_key, 0)

  def test_does_not_call_remove_all_values_with_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(release_mngr,
                           '_remove_all_values') as mock_remove_all_values:
      release_mngr.release({'a': 10, 'b': 20}, 0)

      mock_remove_all_values.assert_not_called()

    self.assertEqual(release_mngr._latest_key, 0)

  def test_calls_remove_values_after_with_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values_to_csv(
        file_path=temp_file,
        fieldnames=[file_release_manager._CSV_KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._CSV_KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(release_mngr,
                           '_remove_values_after') as mock_remove_values_after:
      release_mngr.release({'a': 11, 'b': 21}, 1)

      mock_remove_values_after.assert_called_with(0)

    self.assertEqual(release_mngr._latest_key, 1)

  def test_does_not_call_remove_values_after_with_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(release_mngr,
                           '_remove_values_after') as mock_remove_values_after:
      release_mngr.release({'a': 10, 'b': 20}, 1)

      mock_remove_values_after.assert_not_called()

    self.assertEqual(release_mngr._latest_key, 1)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}, 1),
      ('more_fields', {'a': 10, 'b': 20}, 1),
  )
  # pyformat: enable
  def test_calls_append_value(self, value, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.APPEND)

    with mock.patch.object(release_mngr, '_append_value') as mock_append_value:
      release_mngr.release(value, key)

      mock_append_value.assert_called_once()
      call = mock_append_value.mock_calls[0]
      _, args, _ = call
      actual_value, = args
      expected_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME]
      expected_fieldnames.extend(
          [x for x in value.keys() if x not in expected_fieldnames])
      expected_value = {name: '' for name in expected_fieldnames}
      expected_value.update({file_release_manager._CSV_KEY_FIELDNAME: key})
      expected_value.update(value)
      self.assertEqual(actual_value, expected_value)

    self.assertEqual(release_mngr._latest_key, key)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}, 1),
      ('more_fields', {'a': 10, 'b': 20}, 1),
  )
  # pyformat: enable
  def test_calls_write_value(self, value, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.CSVSaveMode.WRITE)

    with mock.patch.object(release_mngr, '_write_value') as mock_write_value:
      release_mngr.release(value, key)

      mock_write_value.assert_called_once()
      call = mock_write_value.mock_calls[0]
      _, args, _ = call
      actual_value, = args
      expected_fieldnames = [file_release_manager._CSV_KEY_FIELDNAME]
      expected_fieldnames.extend(
          [x for x in value.keys() if x not in expected_fieldnames])
      expected_value = {name: '' for name in expected_fieldnames}
      expected_value.update({file_release_manager._CSV_KEY_FIELDNAME: key})
      expected_value.update(value)
      self.assertEqual(actual_value, expected_value)

    self.assertEqual(release_mngr._latest_key, key)

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, [{'round_num': '1', '': ''}]),
      ('bool', True, [{'round_num': '1', '': 'True'}]),
      ('int', 1, [{'round_num': '1', '': '1'}]),
      ('str', 'a', [{'round_num': '1', '': 'a'}]),
      ('list',
       [True, 1, 'a'],
       [{'round_num': '1', '0': 'True', '1': '1', '2': 'a'}]),
      ('list_empty', [], [{'round_num': '1'}]),
      ('list_nested',
       [[True, 1], ['a']],
       [{'round_num': '1', '0/0': 'True', '0/1': '1', '1/0': 'a'}]),
      ('dict',
       {'a': True, 'b': 1, 'c': 'a'},
       [{'round_num': '1', 'a': 'True', 'b': '1', 'c': 'a'}]),
      ('dict_empty', {}, [{'round_num': '1'}]),
      ('dict_nested',
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}},
       [{'round_num': '1', 'x/a': 'True', 'x/b': '1', 'y/c': 'a'}]),
      ('attr',
       test_utils.TestAttrObject1(True, 1),
       [{'round_num': '1', 'a': 'True', 'b': '1'}]),
      ('attr_nested',
       {'a': [test_utils.TestAttrObject1(True, 1)],
        'b': test_utils.TestAttrObject2('a')},
       [{'round_num': '1', 'a/0/a': 'True', 'a/0/b': '1', 'b/a': 'a'}]),
      ('tensor_int', tf.constant(1), [{'round_num': '1', '': '1'}]),
      ('tensor_str', tf.constant('a'), [{'round_num': '1', '': 'b\'a\''}]),
      ('tensor_2d',
       tf.ones((2, 3)),
       [{'round_num': '1', '': '[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]'}]),
      ('tensor_nested',
       {'a': [tf.constant(True), tf.constant(1)], 'b': [tf.constant('a')]},
       [{'round_num': '1', 'a/0': 'True', 'a/1': '1', 'b/0': 'b\'a\''}]),
      ('numpy_int', np.int32(1), [{'round_num': '1', '': '1'}]),
      ('numpy_2d',
       np.ones((2, 3)),
       [{'round_num': '1', '': '[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]'}]),
      ('numpy_nested',
       {'a': [np.bool(True), np.int32(1)], 'b': [np.str_('a')]},
       [{'round_num': '1', 'a/0': 'True', 'a/1': '1', 'b/0': 'a'}]),
      ('server_array_reference',
       test_utils.TestServerArrayReference(1),
       [{'round_num': '1', '': '1'}]),
      ('server_array_reference_nested',
       {'a': [test_utils.TestServerArrayReference(True),
              test_utils.TestServerArrayReference(1)],
        'b': [test_utils.TestServerArrayReference('a')]},
       [{'round_num': '1', 'a/0': 'True', 'a/1': '1', 'b/0': 'a'}]),
      ('materialized_values_and_value_references',
       [1, test_utils.TestServerArrayReference(2)],
       [{'round_num': '1', '0': '1', '1': '2'}]),
  )
  # pyformat: enable
  def test_writes_value(self, value, expected_value):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    release_mngr.release(value, 1)

    _, actual_value = _read_values_from_csv(temp_file)
    self.assertAllEqual(type(actual_value), type(expected_value))
    self.assertAllEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with self.assertRaises(TypeError):
      release_mngr.release({}, key)


class SavedModelFileReleaseManagerInitTest(parameterized.TestCase):

  def test_creates_root_dir(self):
    temp_dir = self.create_tempdir()
    shutil.rmtree(temp_dir)
    self.assertFalse(os.path.exists(temp_dir))

    file_release_manager.SavedModelFileReleaseManager(root_dir=temp_dir)

    self.assertTrue(os.path.exists(temp_dir))

  def test_does_not_raise_type_error_with_root_dir_str(self):
    temp_dir = self.create_tempdir()

    try:
      file_release_manager.SavedModelFileReleaseManager(
          root_dir=temp_dir.full_path, prefix='a_')
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
        'b': [test_utils.TestServerArrayReference('a')]},
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
