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
"""Utilities for releasing values from a federated program to a file system."""

import collections
import csv
import enum
import os
import os.path
import random
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Sequence, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils

# DO_NOT_SUBMIT: This might need a class specific prefix.
_KEY_FIELDNAME = 'round_num'


class FileReleaseManagerIncompatibleFileError(Exception):
  pass


class FileReleaseManagerPermissionDeniedError(Exception):
  pass


# DO_NOT_SUBMIT: This might need a class specific prefix.
@enum.unique
class SaveMode(enum.Enum):
  APPEND = 'append'
  WRITE = 'write'


class CSVFileReleaseManager(release_manager.ReleaseManager):
  """A `tff.program.ReleaseManager` that releases values to a CSV file.

  A `tff.program.CSVFileReleaseManager` is a utility for releasing values
  from a federated program to a CSV file and is used to release values from
  platform storage to customer storage in a federated program.

  This manager can be configured to release values using a `save_mode` of either
  `SaveMode.APPEND` or `SaveMode.WRITE`. In append mode, when a value is
  released, the manager will try and append the value to the CSV file instead of
  overwriting the existing file. While potentially more efficient, append mode
  is incompatible with compressed files (e.g. `.bz2` formats) and encoded
  directories. This mode is equivalent to write mode when releasing a value with
  a different structure than the currently released values, so it may not be
  useful when values with different structures are being released frequently. In
  write mode (or in append mode when releasing new structures), each time a
  value is realeased, this manager reads the entire CSV file and overwrites the
  existing file with the additional values. This can be slower than append mode
  in some cases, but is compatible with compressed file formats and encoded
  directories.
  """

  def __init__(self,
               file_path: Union[str, os.PathLike],
               save_mode: SaveMode = SaveMode.APPEND):
    """Returns an initialized `tff.program.CSVFileReleaseManager`.

    Args:
      file_path: A path on the file system to save release values. If this file
        does not exist it will be created.
      save_mode: A `tff.program.SaveMode` specifying how to save release values.

    Raises:
      ValueError: If `file_path` is an empty string.
      FileReleaseManagerIncompatibleFileError: If the file exists but can not be
        read by the `tff.program.CSVFileReleaseManager`.
    """
    py_typecheck.check_type(file_path, (str, os.PathLike))
    py_typecheck.check_type(save_mode, SaveMode)
    if not file_path:
      raise ValueError('Expected `file_path` to not be an empty string.')
    self._file_path = file_path
    self._save_mode = save_mode

    if tf.io.gfile.exists(self._file_path):
      fieldnames, values = self._read_values()
      if _KEY_FIELDNAME not in fieldnames:
        raise FileReleaseManagerIncompatibleFileError(
            f'The file \'{self._file_path}\' exists but does not have a '
            f'`{_KEY_FIELDNAME}` column. It is possible that this file was not '
            'created by a `tff.program.CSVFileReleaseManager`.')
      if values:
        self._latest_key = values[-1][_KEY_FIELDNAME]
      else:
        self._latest_key = None
    else:
      self._write_values([_KEY_FIELDNAME], [])
      self._latest_key = None

  def _read_values(self) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Returns a tuple of fieldnames and values from the managed CSV."""
    with tf.io.gfile.GFile(self._file_path, 'r') as file:
      reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC)
      if reader.fieldnames is not None:
        fieldnames = list(reader.fieldnames)
      else:
        fieldnames = []
      values = list(reader)
    return fieldnames, values

  def _write_values(self, fieldnames: Sequence[str],
                    values: Iterable[Mapping[str, Any]]):
    """Writes `fieldnames` and `values` to the managed CSV."""
    path = os.fspath(self._file_path)

    # Create a temporary file.
    temp_path = f'{path}_temp{random.randint(1000, 9999)}'
    if tf.io.gfile.exists(temp_path):
      tf.io.gfile.remove(temp_path)

    # Write to the temporary file.
    with tf.io.gfile.GFile(temp_path, 'w') as file:
      writer = csv.DictWriter(
          file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
      writer.writeheader()
      writer.writerows(values)

    # Rename the temporary file to the final location atomically.
    tf.io.gfile.rename(temp_path, self._file_path, overwrite=True)

  def _write_value(self, value: Mapping[str, Any]):
    """Writes `value` to the managed CSV."""
    fieldnames, values = self._read_values()
    fieldnames.extend([x for x in value.keys() if x not in fieldnames])
    values.append(value)
    self._write_values(fieldnames, values)

  def _append_value(self, value: Mapping[str, Any]):
    """Appends `value` to the managed CSV."""
    with tf.io.gfile.GFile(self._file_path, 'a+') as file:
      reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC)
      if all(key in reader.fieldnames for key in value.keys()):
        writer = csv.DictWriter(
            file, fieldnames=reader.fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        try:
          writer.writerow(value)
        except (tf.errors.PermissionDeniedError, csv.Error) as e:
          raise FileReleaseManagerPermissionDeniedError(
              f'Could not append a value to the file \'{self._file_path}\'. It '
              'is possible that this file is compressed or encoded, please use '
              'write mode instead of append mode to release values to this '
              'file using a `tff.program.CSVFileReleaseManager`.') from e
      else:
        self._write_value(value)

  def _remove_all_values(self):
    """Removes all values from the managed CSV."""
    self._write_values([_KEY_FIELDNAME], [])
    self._latest_key = None

  def _remove_values_after(self, key: int):
    """Removes all values after `key` from the managed CSV."""
    py_typecheck.check_type(key, int)
    filtered_fieldnames = [_KEY_FIELDNAME]
    filtered_values = []
    _, values = self._read_values()
    for value in values:
      if value[_KEY_FIELDNAME] <= key:
        fieldnames = [x for x in value.keys() if x not in filtered_fieldnames]
        filtered_fieldnames.extend(fieldnames)
        filtered_values.append(value)
    self._write_values(filtered_fieldnames, filtered_values)
    self._latest_key = key

  # DO_NOT_SUBMIT: What does key represent? A strictly increasing integer value?
  def release(self, value: Any, key: int):
    """Releases `value` from a federated program.

    Args:
      value: The value to release.
      key: A integer to use to reference the released `value`, `key` represents
        a ???
    """
    py_typecheck.check_type(key, int)
    if self._latest_key is not None and key <= self._latest_key:
      if key == 0:
        self._remove_all_values()
      else:
        self._remove_values_after(key - 1)
    flattened_value = structure_utils.flatten(value)
    flattened_value[_KEY_FIELDNAME] = key
    if self._save_mode == SaveMode.APPEND:
      self._append_value(flattened_value)
    elif self._save_mode == SaveMode.WRITE:
      self._write_value(flattened_value)
    self._latest_key = key

  # DO_NOT_SUBMIT: This returns flattened released values not released values.
  def values(self) -> Dict[int, Dict[str, Any]]:
    """Returns a dict of all keys and released values."""
    result = collections.OrderedDict()
    _, values = self._read_values()
    for value in values:
      key = value.pop(_KEY_FIELDNAME)
      result[key] = value
    return result
