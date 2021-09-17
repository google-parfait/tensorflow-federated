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
"""Utilities to release values from a federated program."""

import abc
from typing import Any


class ReleaseManager(metaclass=abc.ABCMeta):
  """An abstract interface for `ReleaseManager`s.

  A `ReleaseManager` is a utility to release values from a federated program
  from platform storage to customer storage.
  """

  @abc.abstractmethod
  def release(self, value: Any, key: Any = None):
    """Releases the `value` from a federated program.

    A concrete implementation of this interface should be specific about the
    types of `value` and `key` for this method and should document how the `key`
    will be used. This allows a federated program to understand how to create at
    `key` for the `value` before it is released. For example, a `ReleaseManager`
    that releases metrics keyed by a strictly increasing integer might specify a
    `value` type of `Mapping[str, Any]` and a `key` type of `int`.

    Args:
      value: The value to release, the specific structure of `key` is left up to
        the concrete implementations of `ReleaseManager`.
      key: An optional value to use as the key for the released `value`, the
         specific structure of `key` and how `key` is used is left up to the
         concrete implementations of`ReleaseManager`.
    """
    raise NotImplementedError
