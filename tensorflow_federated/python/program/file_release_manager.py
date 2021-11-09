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
"""Utilities for releasing values from a federated program to a file system.

Note: This library uses `tf.io.gfile` to perform file system operations, this
means that this library:

  * supports all the file systems supported by `tf.io.gfile`
  * encodes files in the same way as `tf.io.gfile`
"""

import collections
import csv
import enum
import os
import os.path
import random
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Sequence, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


class FileReleaseManagerIncompatibleFileError(Exception):
  pass


class FileReleaseManagerPermissionDeniedError(Exception):
  pass


_CSV_KEY_FIELDNAME = 'round_num'


@enum.unique
class CSVSaveMode(enum.Enum):
  APPEND = 'append'
  WRITE = 'write'


class CSVFileReleaseManager(release_manager.ReleaseManager):
  """A `tff.program.ReleaseManager` that releases values to a CSV file.

  A `tff.program.CSVFileReleaseManager` is a utility for releasing values
  from a federated program to a CSV file and is used to release values from
  platform storage to customer storage in a federated program.

  Values are released to the file system as a CSV file and are quoted as
  strings. When the value is released, if the value is a value reference or a
  structure containing value references, each value reference is materialized.
  The value is then flattened, converted to a `numpy.ndarray`, and then
  converted to a nested list of Python scalars, and released as a CSV file.
  For example, `1` will be written as `'1'` and `tf.ones([2, 2])` will be
  written as `'[[1.0, 1.0], [1.0, 1.0]'`.

  This manager can be configured to release values using a `save_mode` of either
  `CSVSaveMode.APPEND` or `CSVSaveMode.WRITE`.

  * In append mode, when a value is released, this manager will try and append
    the value to the CSV file instead of overwriting the existing file. While
    potentially more efficient, append mode is incompatible with compressed
    files (e.g. `.bz2` formats) and encoded directories. This mode is equivalent
    to write mode when releasing a value with a different structure than the
    currently released values, so it may not be useful when values with
    different structures are being released frequently.

  * In write mode (or in append mode when releasing new structures), when a
    value is realeased, this manager reads the entire CSV file and overwrites
    the existing file with the additional values. This can be slower than append
    mode, but is compatible with compressed files (e.g. `.bz2` formats) and
    encoded directories.
  """

  def __init__(self,
               file_path: Union[str, os.PathLike],
               save_mode: CSVSaveMode = CSVSaveMode.APPEND):
    """Returns an initialized `tff.program.CSVFileReleaseManager`.

    Args:
      file_path: A path on the file system to save release values. If this file
        does not exist it will be created.
      save_mode: A `tff.program.CSVSaveMode` specifying how to save release
        values.

    Raises:
      ValueError: If `file_path` is an empty string.
      FileReleaseManagerIncompatibleFileError: If the file exists but is
        incompatible with the `tff.program.CSVFileReleaseManager`.
    """
    py_typecheck.check_type(file_path, (str, os.PathLike))
    py_typecheck.check_type(save_mode, CSVSaveMode)
    if not file_path:
      raise ValueError('Expected `file_path` to not be an empty string.')
    self._file_path = file_path
    self._save_mode = save_mode

    if tf.io.gfile.exists(self._file_path):
      fieldnames, values = self._read_values()
      if _CSV_KEY_FIELDNAME not in fieldnames:
        raise FileReleaseManagerIncompatibleFileError(
            f'The file \'{self._file_path}\' exists but is incompatible with '
            'the `tff.program.CSVFileReleaseManager`. It is possible that this '
            'file was not created by a `tff.program.CSVFileReleaseManager`.')
      if values:
        self._latest_key = int(values[-1][_CSV_KEY_FIELDNAME])
      else:
        self._latest_key = None
    else:
      self._write_values([_CSV_KEY_FIELDNAME], [])
      self._latest_key = None

  def _read_values(self) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Returns a tuple of fieldnames and values from the managed CSV."""
    with tf.io.gfile.GFile(self._file_path, 'r') as file:
      reader = csv.DictReader(file)
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
      writer = csv.DictWriter(file, fieldnames=fieldnames)
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
      reader = csv.DictReader(file)
      if all(key in reader.fieldnames for key in value.keys()):
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        try:
          writer.writerow(value)
        except (tf.errors.PermissionDeniedError, csv.Error) as e:
          raise FileReleaseManagerPermissionDeniedError(
              f'Could not append a value to the file \'{self._file_path}\'. It '
              'is possible that this file is compressed or encoded. Olease use '
              'write mode instead of append mode to release values to this '
              'file using a `tff.program.CSVFileReleaseManager`.') from e
      else:
        self._write_value(value)

  def _remove_all_values(self):
    """Removes all values from the managed CSV."""
    self._write_values([_CSV_KEY_FIELDNAME], [])
    self._latest_key = None

  def _remove_values_after(self, key: int):
    """Removes all values after `key` from the managed CSV."""
    py_typecheck.check_type(key, int)
    filtered_fieldnames = [_CSV_KEY_FIELDNAME]
    filtered_values = []
    _, values = self._read_values()
    for value in values:
      current_key = int(value[_CSV_KEY_FIELDNAME])
      if current_key <= key:
        fieldnames = [x for x in value.keys() if x not in filtered_fieldnames]
        filtered_fieldnames.extend(fieldnames)
        filtered_values.append(value)
    self._write_values(filtered_fieldnames, filtered_values)
    self._latest_key = key

  def release(self, value: Any, key: int):
    """Releases `value` from a federated program.

    This method will atomically update the managed CSV file by removing all
    values previously released with a key greater than or equal to `key` before
    writing `value`.

    Args:
      value: A materialized value, a value reference, or a structure of
        materialized values and value references representing the value to
        release.
      key: An integer used to reference the released `value`, `key` represents a
        step in a federated program.
    """
    py_typecheck.check_type(key, int)
    if self._latest_key is not None and key <= self._latest_key:
      if key == 0:
        self._remove_all_values()
      else:
        self._remove_values_after(key - 1)
    materialized_value = value_reference.materialize_value(value)
    flattened_value = structure_utils.flatten(materialized_value)

    normalized_value = collections.OrderedDict()
    for x, y in flattened_value.items():
      normalized_value[x] = np.array(y).tolist()

    normalized_value[_CSV_KEY_FIELDNAME] = key
    if self._save_mode == CSVSaveMode.APPEND:
      self._append_value(normalized_value)
    elif self._save_mode == CSVSaveMode.WRITE:
      self._write_value(normalized_value)
    self._latest_key = key


class SavedModelFileReleaseManager(release_manager.ReleaseManager):
  """A `tff.program.ReleaseManager` that releases values to a file system.

  A `tff.program.SavedModelFileReleaseManager` is a utility for releasing values
  from a federated program to a file system and is used to release values from
  platform storage to customer storage in a federated program.

  Values are released to the file system using the SavedModel (see
  `tf.saved_model`) format. When the value is released, if the value is a value
  reference or a structure containing value references, each value reference is
  materialized. The value is then flattened and released using the SavedModel
  format. The structure of the value is discarded.

  Note: The SavedModel format can only contain values that can be converted to a
  `tf.Tensor` (see `tf.convert_to_tensor`), releasing any other values will
  result in an error.

  See https://www.tensorflow.org/guide/saved_model for more infromation about
  the SavedModel format.
  """

  def __init__(self,
               root_dir: Union[str, os.PathLike],
               prefix: str = 'release_'):
    """Returns an initialized `tff.program.SavedModelFileReleaseManager`.

    Args:
      root_dir: A path on the file system to save program state. If this path
        does not exist it will be created.
      prefix: A string to use as the prefix for filenames.

    Raises:
      ValueError: If `root_dir` is an empty string.
    """
    py_typecheck.check_type(root_dir, (str, os.PathLike))
    py_typecheck.check_type(prefix, str)
    if not root_dir:
      raise ValueError('Expected `root_dir` to not be an empty string.')
    if not tf.io.gfile.exists(root_dir):
      tf.io.gfile.makedirs(root_dir)
    self._root_dir = root_dir
    self._prefix = prefix

  def _get_path_for_key(self, key: int) -> str:
    """Returns the path for the given `key`.

    This method does not assert that the given `key` or the returned path
    represent released values.

    Args:
      key: The key used to construct the path.
    """
    py_typecheck.check_type(key, int)
    basename = f'{self._prefix}{key}'
    return os.path.join(self._root_dir, basename)

  def release(self, value: Any, key: int):
    """Releases `value` from a federated program.

    Args:
      value: A materialized value, a value reference, or a structure of
        materialized values and value references representing the value to
        release.
      key: An integer used to reference the released `value`, `key` represents a
        step in a federated program.
    """
    py_typecheck.check_type(key, int)
    path = self._get_path_for_key(key)
    materialized_value = value_reference.materialize_value(value)
    flattened_value = tf.nest.flatten(materialized_value)
    module = file_utils.ValueModule(flattened_value)
    file_utils.write_saved_model(module, path, overwrite=True)
