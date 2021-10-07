# Copyright 2021, Google LLC.
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

import os
from typing import Any, Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import file_utils
from tensorflow_federated.python.program import release_manager


class SavedModelFileReleaseManager(release_manager.ReleaseManager):
  """A `tff.program.ReleaseManager` that releases values to a file system.

  A `tff.program.SavedModelFileReleaseManager` is a utility for releasing values
  from a federated program to a file system using the SavedModel format and is
  used to release values from platform storage to customer storage in a
  federated program.

  See https://www.tensorflow.org/guide/saved_model for more infromation about
  using the SavedModel format.
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
      key: The version to use to construct the path.
    """
    py_typecheck.check_type(key, int)
    basename = f'{self._prefix}{key}'
    return os.path.join(self._root_dir, basename)

  # TODO(b/202418342): Add support for `ValueReference`.
  def release(self, value: Any, key: int):
    """Releases `value` from a federated program.

    Args:
      value: The value to release.
      key: A integer to use to reference the released `value`, `key` represents
        a step in a federated program.
    """
    py_typecheck.check_type(key, int)
    path = self._get_path_for_key(key)
    flat_obj = tf.nest.flatten(value)
    model = tf.Module()
    model.obj = flat_obj
    model.build_obj_fn = tf.function(lambda: model.obj, input_signature=())
    file_utils.write_saved_model(model, path, overwrite=True)
