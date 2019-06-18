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
"""An executor that transforms computations prior to executing them."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl import executor_base


class TransformingExecutor(executor_base.Executor):
  """This executor transforms computations prior to executing them.

  This executor only performs transformations. All other aspects of execution
  are delegated to the underlying target executor.
  """

  # TODO(b/134543154): Add transformations options, including an option to
  # consolidate all unplaced logic into a single block of TensorFlow code.

  # TODO(b/134543154): Actually implement this.

  def __init__(self, target_executor):
    """Creates a transforming executor backed by a given target executor.

    Args:
      target_executor: The target executor to use.
    """
    py_typecheck.check_type(target_executor, executor_base.Executor)
    self._target_executor = target_executor

  async def ingest(self, value, type_spec):
    raise NotImplementedError

  async def invoke(self, comp, arg):
    raise NotImplementedError
