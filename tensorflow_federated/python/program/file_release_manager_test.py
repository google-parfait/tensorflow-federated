# Copyright 2019, The TensorFlow Federated Authors.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.program import file_release_manager


def _read_values(
    file_path: os.PathLike) -> Tuple[List[str], List[Dict[str, Any]]]:
  with tf.io.gfile.GFile(file_path, 'r') as file:
    reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC)
    fieldnames = list(reader.fieldnames)
    values = list(reader)
  return fieldnames, values


def _write_values(file_path: os.PathLike, fieldnames: Sequence[str],
                  values: Iterable[Mapping[str, Any]]):
  with tf.io.gfile.GFile(file_path, 'w') as file:
    writer = csv.DictWriter(
        file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()
    for value in values:
      writer.writerow(value)


class CSVFileReleaseManagerInitTest(parameterized.TestCase):

  def test_writes_new_file_path(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    self.assertFalse(os.path.exists(temp_file))

    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    self.assertTrue(os.path.exists(temp_file))
    self.assertIsNone(file_release_mngr._latest_key)

  def test_loads_existing_file_path(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
        }])
    self.assertTrue(os.path.exists(temp_file))

    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    self.assertEqual(file_release_mngr._latest_key, 1)

  def test_loads_existing_file_path_empty(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME],
        values=[])
    self.assertTrue(os.path.exists(temp_file))

    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    self.assertIsNone(file_release_mngr._latest_key)

  def test_does_not_raise_type_error_with_file_path_str(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)

    try:
      file_release_manager.CSVFileReleaseManager(file_path=temp_file)
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
          file_path=temp_file, save_mode=file_release_manager.SaveMode.APPEND)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('none', None),
      ('bool', True),
      ('int', 1),
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


class CSVFileReleaseManagerReadValuesTest(absltest.TestCase):

  def test_returns_values_from_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    fieldnames, values = file_release_mngr._read_values()

    self.assertEqual(fieldnames, [file_release_manager._KEY_FIELDNAME])
    self.assertEqual(values, [])

  def test_returns_values_from_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    fieldnames, values = file_release_mngr._read_values()

    self.assertEqual(fieldnames, [
        file_release_manager._KEY_FIELDNAME,
        'a',
        'b',
    ])
    self.assertEqual(values, [{
        file_release_manager._KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    }])


class CSVFileReleaseManagerWriteValuesTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', [], []),
      ('no_values', ['a', 'b'], []),
      ('one_value', ['a', 'b'], [
          {'a': 10, 'b': 20},
      ]),
      ('two_values', ['a', 'b'], [
          {'a': 10, 'b': 20},
          {'a': 11, 'b': 21},
      ]),
  )
  # pyformat: enable
  def test_writes_values_to_empty_file(self, fieldnames, values):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._write_values(fieldnames=fieldnames, values=values)

    actual_fieldnames, actual_values = _read_values(temp_file)
    self.assertEqual(actual_fieldnames, fieldnames)
    self.assertEqual(actual_values, values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', [], []),
      ('no_values', ['a', 'b'], []),
      ('one_value', ['a', 'b'], [
          {'a': 11, 'b': 21},
      ]),
      ('two_values', ['a', 'b'], [
          {'a': 11, 'b': 21},
          {'a': 12, 'b': 22},
      ]),
  )
  # pyformat: enable
  def test_writes_values_to_existing_file(self, fieldnames, values):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._write_values(fieldnames=fieldnames, values=values)

    actual_fieldnames, actual_values = _read_values(temp_file)
    self.assertEqual(actual_fieldnames, fieldnames)
    self.assertEqual(actual_values, values)


class CSVFileReleaseManagerWriteValueTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 10, 'b': 20}),
  )
  # pyformat: enable
  def test_writes_value_to_empty_file(self, value):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.WRITE)

    file_release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values(temp_file)
    expected_fieldnames = [file_release_manager._KEY_FIELDNAME]
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [expected_value]
    self.assertEqual(actual_values, expected_values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('same_fields', {'a': 11, 'b': 21}),
      ('more_fields', {'a': 11, 'b': 21, 'c': 31}),
  )
  # pyformat: enable
  def test_write_value_to_existing_file(self, value):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.WRITE)

    file_release_mngr._write_value(value)

    actual_fieldnames, actual_values = _read_values(temp_file)
    expected_fieldnames = [file_release_manager._KEY_FIELDNAME, 'a', 'b']
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    existing_value = {name: '' for name in expected_fieldnames}
    existing_value.update({
        file_release_manager._KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    })
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [existing_value, expected_value]
    self.assertEqual(actual_values, expected_values)


class CSVFileReleaseManagerAppendValueTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('more_fields', {'a': 10, 'b': 20}),
  )
  # pyformat: enable
  def test_appends_value_to_empty_file(self, value):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.APPEND)

    file_release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values(temp_file)
    expected_fieldnames = [file_release_manager._KEY_FIELDNAME]
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [expected_value]
    self.assertEqual(actual_values, expected_values)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}),
      ('same_fields', {'a': 11, 'b': 21}),
      ('more_fields', {'a': 11, 'b': 21, 'c': 31}),
  )
  # pyformat: enable
  def test_appends_value_to_existing_file(self, value):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.APPEND)

    file_release_mngr._append_value(value)

    actual_fieldnames, actual_values = _read_values(temp_file)
    expected_fieldnames = [file_release_manager._KEY_FIELDNAME, 'a', 'b']
    expected_fieldnames.extend(
        [x for x in value.keys() if x not in expected_fieldnames])
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    existing_value = {name: '' for name in expected_fieldnames}
    existing_value.update({
        file_release_manager._KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    })
    expected_value = {name: '' for name in expected_fieldnames}
    expected_value.update(value)
    expected_values = [existing_value, expected_value]
    self.assertEqual(actual_values, expected_values)

  def test_raises_permission_denied_error(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.APPEND)

    with mock.patch('csv.DictWriter.writerow') as mock_writerow:
      mock_writerow.side_effect = csv.Error()

      with self.assertRaises(
          file_release_manager.FileReleaseManagerPermissionDeniedError):
        file_release_mngr._append_value({})


class CSVFileReleaseManagerRemoveAllValuesTest(absltest.TestCase):

  def test_removes_values_from_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._remove_all_values()

    actual_fieldnames, actual_values = _read_values(temp_file)
    self.assertEqual(actual_fieldnames, [file_release_manager._KEY_FIELDNAME])
    self.assertEqual(actual_values, [])

  def test_removes_values_from_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._remove_all_values()

    actual_fieldnames, actual_values = _read_values(temp_file)
    self.assertEqual(actual_fieldnames, [file_release_manager._KEY_FIELDNAME])
    self.assertEqual(actual_values, [])


class CSVFileReleaseManagerRemoveValuesAfterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('0', 0),
      ('1', 1),
  )
  def test_removes_values_from_empty_file(self, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._remove_values_after(key)

    actual_fieldnames, actual_values = _read_values(temp_file)
    self.assertEqual(actual_fieldnames, [file_release_manager._KEY_FIELDNAME])
    self.assertEqual(actual_values, [])

  def test_removes_values_from_existing_file_with_key_0(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20
        }, {
            file_release_manager._KEY_FIELDNAME: 2,
            'a': 11,
            'b': 21
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._remove_values_after(0)

    actual_fieldnames, actual_values = _read_values(temp_file)
    self.assertEqual(actual_fieldnames, [file_release_manager._KEY_FIELDNAME])
    self.assertEqual(actual_values, [])

  def test_removes_values_from_existing_file_with_key_1(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20
        }, {
            file_release_manager._KEY_FIELDNAME: 2,
            'a': 11,
            'b': 21
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._remove_values_after(1)

    actual_fieldnames, actual_values = _read_values(temp_file)
    expected_fieldnames = [file_release_manager._KEY_FIELDNAME, 'a', 'b']
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_values = [{
        file_release_manager._KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    }]
    self.assertEqual(actual_values, expected_values)

  def test_removes_values_from_existing_file_with_key_2(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }, {
            file_release_manager._KEY_FIELDNAME: 2,
            'a': 11,
            'b': 21,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    file_release_mngr._remove_values_after(2)

    actual_fieldnames, actual_values = _read_values(temp_file)
    expected_fieldnames = [file_release_manager._KEY_FIELDNAME, 'a', 'b']
    self.assertEqual(actual_fieldnames, expected_fieldnames)
    expected_values = [{
        file_release_manager._KEY_FIELDNAME: 1,
        'a': 10,
        'b': 20,
    }, {
        file_release_manager._KEY_FIELDNAME: 2,
        'a': 11,
        'b': 21,
    }]
    self.assertEqual(actual_values, expected_values)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with self.assertRaises(TypeError):
      file_release_mngr._remove_values_after(key)


class CSVFileReleaseManagerReleaseTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}, 1),
      ('more_fields', {'a': 10, 'b': 20}, 1),
  )
  # pyformat: enable
  def test_calls_append_value(self, value, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.APPEND)

    with mock.patch.object(file_release_mngr,
                           '_append_value') as mock_append_value:
      file_release_mngr.release(value, key)

      mock_append_value.assert_called_once()
      call = mock_append_value.mock_calls[0]
      _, args, _ = call
      actual_value, = args
      expected_fieldnames = [file_release_manager._KEY_FIELDNAME]
      expected_fieldnames.extend(
          [x for x in value.keys() if x not in expected_fieldnames])
      expected_value = {name: '' for name in expected_fieldnames}
      expected_value.update({file_release_manager._KEY_FIELDNAME: key})
      expected_value.update(value)
      self.assertEqual(actual_value, expected_value)

    self.assertEqual(file_release_mngr._latest_key, key)

  # pyformat: disable
  @parameterized.named_parameters(
      ('empty', {}, 1),
      ('more_fields', {'a': 10, 'b': 20}, 1),
  )
  # pyformat: enable
  def test_calls_write_value(self, value, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file, save_mode=file_release_manager.SaveMode.WRITE)

    with mock.patch.object(file_release_mngr,
                           '_write_value') as mock_write_value:
      file_release_mngr.release(value, key)

      mock_write_value.assert_called_once()
      call = mock_write_value.mock_calls[0]
      _, args, _ = call
      actual_value, = args
      expected_fieldnames = [file_release_manager._KEY_FIELDNAME]
      expected_fieldnames.extend(
          [x for x in value.keys() if x not in expected_fieldnames])
      expected_value = {name: '' for name in expected_fieldnames}
      expected_value.update({file_release_manager._KEY_FIELDNAME: key})
      expected_value.update(value)
      self.assertEqual(actual_value, expected_value)

    self.assertEqual(file_release_mngr._latest_key, key)

  def test_does_not_call_remote_all_values_with_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(file_release_mngr,
                           '_remove_all_values') as mock_remove_all_values:
      file_release_mngr.release({'a': 10, 'b': 20}, 0)

      mock_remove_all_values.assert_not_called()

    self.assertEqual(file_release_mngr._latest_key, 0)

  def test_does_not_call_remove_values_after_with_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(file_release_mngr,
                           '_remove_values_after') as mock_remove_values_after:
      file_release_mngr.release({'a': 10, 'b': 20}, 1)

      mock_remove_values_after.assert_not_called()

    self.assertEqual(file_release_mngr._latest_key, 1)

  def test_calls_remote_all_values_with_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(file_release_mngr,
                           '_remove_all_values') as mock_remove_all_values:
      file_release_mngr.release({'a': 11, 'b': 21}, 0)

      mock_remove_all_values.assert_called_once()

    self.assertEqual(file_release_mngr._latest_key, 0)

  def test_calls_remove_values_after_with_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with mock.patch.object(file_release_mngr,
                           '_remove_values_after') as mock_remove_values_after:
      file_release_mngr.release({'a': 11, 'b': 21}, 1)

      mock_remove_values_after.assert_called_with(0)

    self.assertEqual(file_release_mngr._latest_key, 1)

  @parameterized.named_parameters(
      ('none', None),
      ('str', 'a'),
      ('list', []),
  )
  def test_raises_type_error_with_key(self, key):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    with self.assertRaises(TypeError):
      file_release_mngr.release({}, key)


class CSVFileReleaseManagerValuesTest(parameterized.TestCase):

  def test_returns_values_with_empty_file(self):
    temp_file = self.create_tempfile()
    os.remove(temp_file)
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    values = file_release_mngr.values()

    self.assertEqual(values, {})

  def test_returns_values_with_existing_file(self):
    temp_file = self.create_tempfile()
    _write_values(
        file_path=temp_file,
        fieldnames=[file_release_manager._KEY_FIELDNAME, 'a', 'b'],
        values=[{
            file_release_manager._KEY_FIELDNAME: 1,
            'a': 10,
            'b': 20,
        }])
    file_release_mngr = file_release_manager.CSVFileReleaseManager(
        file_path=temp_file)

    values = file_release_mngr.values()

    self.assertEqual(values, {1: {
        'a': 10,
        'b': 20,
    }})


if __name__ == '__main__':
  absltest.main()
