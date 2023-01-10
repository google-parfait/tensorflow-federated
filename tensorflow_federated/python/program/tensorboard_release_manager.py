# Copyright 2020, The TensorFlow Federated Authors.
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
from typing import Union

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference


class TensorBoardReleaseManager(
    release_manager.ReleaseManager[release_manager.ReleasableStructure, int]
):
  """A `tff.program.ReleaseManager` that releases values to TensorBoard.

  A `tff.program.TensorBoardReleaseManager` is a utility for releasing values
  from a federated program to TensorBoard and is used to release values from
  platform storage to customer storage in a federated program.

  Values are released as summary data using `tf.summary`. When the value is
  released, if the value is a value reference or a structure containing value
  references, each value reference is materialized. The value is then flattened
  and released as summary data. The structure of the value is used as the name
  of the summary data. Scalar values are released using `tf.summary.scalar` and
  non-scalar values are released using `tf.summary.histogram`.

  Warning: The summary data can only contain booleans, integers, unsigned
  integers, and floats, releasing any other values will be silently ignored.

  See https://www.tensorflow.org/api_docs/python/tf/summary for more information
  about summary data and how to visualize summary data using TensorBoard.
  """

  def __init__(self, summary_dir: Union[str, os.PathLike[str]]):
    """Returns an initialized `tff.program.TensorBoardReleaseManager`.

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

  async def release(
      self,
      value: value_reference.MaterializableStructure,
      type_signature: computation_types.Type,
      key: int,
  ) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.MaterializableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A integer used to reference the released `value`; `key` represents a
        step in a federated program.
    """
    del type_signature  # Unused.
    py_typecheck.check_type(key, (int, np.integer))

    materialized_value = await value_reference.materialize_value(value)
    flattened_value = structure_utils.flatten_with_name(materialized_value)

    def _normalize(
        value: value_reference.MaterializedValue,
    ) -> value_reference.MaterializedValue:
      if isinstance(value, tf.data.Dataset):
        value = list(value)
      return value

    with self._summary_writer.as_default():
      for name, value in flattened_value:
        normalized_value = _normalize(value)

        # Summary data can only contain booleans, integers, unsigned integers,
        # and floats, releasing any other values will be silently ignored.
        value_array = np.array(normalized_value)
        if value_array.dtype.kind in ('b', 'i', 'u', 'f'):
          if value_array.shape:
            tf.summary.histogram(name, normalized_value, step=key)
          else:
            tf.summary.scalar(name, normalized_value, step=key)
