# Copyright 2023, The TensorFlow Federated Authors.
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
"""Defines an abstract interface for objects that can be serialized."""

import abc


class Serializable(abc.ABC):

  @classmethod
  @abc.abstractmethod
  def from_bytes(cls, buffer: bytes) -> 'Serializable':
    """Deserializes the object from bytes."""
    raise NotImplementedError

  @abc.abstractmethod
  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    raise NotImplementedError
