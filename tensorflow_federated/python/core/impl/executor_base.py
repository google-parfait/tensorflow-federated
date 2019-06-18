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
"""A base Python interface for all types of executors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Executor(object):
  """Represents the abstract interface that all executors must implement."""

  # TODO(b/134543154): Migrate the reference executor over this new interface.

  # TODO(b/134543154): Standardize and document the kinds of values that can be
  # embedded and must be understood by all executor implementations, possibly
  # factoring out parts of reference executor's `to_representation_for_type()`.

  @abc.abstractmethod
  async def ingest(self, value, type_spec):
    """A coroutine that ingests value `value` of type `type_spec`.

    This function is used to embed a value within the executor. Once embedded,
    the value can be further passed around. The value being embedded can be a
    computation. Embedding a computation prior to invoking it potentally allows
    the executor to amortize overhead across multiple calls.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: The `tff.Type` of the value represented by this object, or
        something convertible to it.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents the
      ingested value.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def invoke(self, comp, arg):
    """A coruotine that invokes computation `comp` with argument `arg`.

    Args:
      comp: The computation to invoke. If `comp` has not been ingested by this
        executor yet, it is ingested first.
      arg: The optional argument of the call, or `None` if no argument was
        supplied. If it has not been ingested by the executor yet, it is first
        ingested prior to doing anything else.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents the
      result of invocation.
    """
    raise NotImplementedError
