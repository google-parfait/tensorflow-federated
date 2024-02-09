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
"""Utilities for releasing values from a federated program to logs."""

from typing import Optional

from absl import logging

from tensorflow_federated.python.program import release_manager
from tensorflow_federated.python.program import value_reference


class LoggingReleaseManager(
    release_manager.ReleaseManager[
        release_manager.ReleasableStructure, release_manager.Key
    ]
):
  """A `tff.program.ReleaseManager` that releases values to logs.

  A `tff.program.LoggingReleaseManager` is a utility for releasing values from a
  federated program to logs and is used to release values from platform storage
  to customer storage in a federated program.

  Values are released to logs as string representations of Python objects. When
  the value is released, if the value is a value reference or a structure
  containing value references, each value reference is materialized.
  """

  async def release(
      self,
      value: release_manager.ReleasableStructure,
      key: Optional[release_manager.Key],
  ) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      key: An optional value used to reference the released `value`.
    """
    materialized_value = await value_reference.materialize_value(value)
    logging.info('Releasing')
    logging.info('  value: %s', materialized_value)
    if key is not None:
      logging.info('  key: %s', key)
