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
"""Defines the abstract interface for classes that reference values."""

import abc
from typing import Any

from tensorflow_federated.python.core.impl.types import typed_object


class ValueReference(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """An abstract interface for classes that reference values.

  This interfaces provides the capability to maniplutate values without
  requiring them to be materialized as Python objects.
  """

  @abc.abstractmethod
  def get_value(self) -> Any:
    pass
