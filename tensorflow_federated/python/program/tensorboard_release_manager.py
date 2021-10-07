# Copyright 2020, Google LLC.
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
"""Utilities for releasing values from a federated program to TensorBoard."""

import os
from typing import Any, Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils


class TensorboardReleaseManager(release_manager.ReleaseManager):
  """A `tff.program.ReleaseManager` that releases values to TensorBoard.

  A `tff.program.TensorboardReleaseManager` is a utility for releasing values
  from a federated program to TensorBoard and is used to release values from
  platform storage to customer storage in a federated program.

  Note: This manager releases values as summary data using `tf.summary` and
  summary data can only contain booleans, integers, unsigned integers, and
  floats, releasing any other values will be silently ignored.

  See https://www.tensorflow.org/api_docs/python/tf/summary for more information
  about summary data and how to visualize summary data using TensorBoard.
  """

  def __init__(self, summary_dir: Union[str, os.PathLike]):
    """Returns an initialized `tff.program.TensorboardReleaseManager`.

    Args:
      summary_dir: A path on the file system to save release values. If this
        path does not exist it will be created.

    Raises:
      ValueError: If `summary_dir` is an empty string.
    """
    py_typecheck.check_type(summary_dir, (str, os.PathLike))
    if not summary_dir:
      raise ValueError('Expected `summary_dir` to not be an empty string.')
    if not tf.io.gfile.exists(summary_dir):
      tf.io.gfile.makedirs(summary_dir)
    if isinstance(summary_dir, os.PathLike):
      summary_dir = os.fspath(summary_dir)
    self._summary_writer = tf.summary.create_file_writer(summary_dir)

  # TODO(b/202418342): Add support for `ValueReference`.
  def release(self, value: Any, key: int):
    """Releases `value` from a federated program.

    Args:
      value: The value to release.
      key: A integer to use to reference the released `value`, `key` represents
        a step in a federated program.
    """
    py_typecheck.check_type(key, int)
    flattened_value = structure_utils.flatten(value)

    with self._summary_writer.as_default():
      for name, value in flattened_value.items():
        value_array = np.array(value)
        # Summary data can only contain booleans, integers, unsigned integers,
        # and floats, releasing any other values will be silently ignored.
        if value_array.dtype.kind in ('b', 'i', 'u', 'f'):
          if value_array.shape:
            tf.summary.histogram(name, value, step=key)
          else:
            tf.summary.scalar(name, value, step=key)
