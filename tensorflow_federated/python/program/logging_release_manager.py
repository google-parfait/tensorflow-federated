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
"""Utilities to log values from a federated program."""

from typing import Any, Optional

from absl import logging

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.program import release_manager


class LoggingReleaseManager(release_manager.ReleaseManager):
  """A `ReleaseManager` that logs values."""

  def release(self, value: Any, key: Optional[int] = None):
    """Releases a value from a federated program.

    Args:
      value: The value to release.
      key: An optional integer to use as the key for the released `value`, if
        specified, this value represents a round number in the federated
        program.
    """
    if key is not None:
      py_typecheck.check_type(key, int)
      logging.info("Releasing value at round %d: %s", key, value)
    else:
      logging.info("Releasing value: %s", value)
