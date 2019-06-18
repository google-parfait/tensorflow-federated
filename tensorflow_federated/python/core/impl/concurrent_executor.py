# Lint as: python3
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
"""A concurrent executor that does work asynchronously in multiple threads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base


class ConcurrentExecutor(executor_base.Executor):
  """The concurrent executor partitions work across multiple threads.

  This executor only handles multithreading. It delegates all execution to an
  underlying pool of target executors.
  """

  def __init__(self, target_executors):
    """Creates a concurrent executor backed by a pool of target executors.

    Args:
      target_executors: An enumerable of target executors to use.
    """
    # TODO(b/134543154): Actually implement this.

    self._target_executors = []
    for executor in target_executors:
      py_typecheck.check_type(executor, executor_base.Executor)
      self._target_executors.append(executor)

  async def ingest(self, value, type_spec):
    raise NotImplementedError

  async def invoke(self, comp, arg):
    raise NotImplementedError
