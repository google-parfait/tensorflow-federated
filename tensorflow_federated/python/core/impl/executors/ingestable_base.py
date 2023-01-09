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
"""Abstract interface for objects able to ingest themselves into an executor."""

import abc

from tensorflow_federated.python.core.impl.types import typed_object


class Ingestable(typed_object.TypedObject, metaclass=abc.ABCMeta):
  """Abstract interface for objects able to ingest themselves in an executor."""

  @abc.abstractmethod
  async def ingest(self, executor):
    """Causes this object to ingest itself into the given executor.

    Args:
      executor: An instance of `executor_base.Executor` to ingest into.

    Returns:
      An instance of `executor_value_base.ExecutorValue` returned by the
      executor.
    """
    raise NotImplementedError
