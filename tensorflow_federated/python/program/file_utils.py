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
"""Utilities for working with file systems."""

import os
import random
from typing import Union

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


class FileAlreadyExistsError(Exception):
  pass


def write_saved_model(obj: tf.Module,
                      path: Union[str, os.PathLike],
                      overwrite: bool = False):
  """Writes a `tf.Module` using the SavedModel format."""
  py_typecheck.check_type(path, (str, os.PathLike))
  py_typecheck.check_type(obj, tf.Module)
  py_typecheck.check_type(overwrite, bool)

  # Create a temporary directory.
  if isinstance(path, os.PathLike):
    path = os.fspath(path)
  temp_path = f'{path}_temp{random.randint(1000, 9999)}'
  if tf.io.gfile.exists(temp_path):
    tf.io.gfile.rmtree(temp_path)
  tf.io.gfile.makedirs(temp_path)

  # Write to the temporary directory.
  tf.saved_model.save(obj, temp_path, signatures={})

  # Rename the temporary directory to the final location atomically.
  if tf.io.gfile.exists(path):
    if not overwrite:
      raise FileAlreadyExistsError(f'File already exists for path: {path}')
    tf.io.gfile.rmtree(path)
  tf.io.gfile.rename(temp_path, path)
