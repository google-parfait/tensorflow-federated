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

import abc


class Executor(object, metaclass=abc.ABCMeta):
  """Represents the abstract interface that all executors must implement.

  NOTE: This component is only available in Python 3.
  """

  # TODO(b/134543154): Migrate the reference executor over this new interface.

  # TODO(b/134543154): Standardize and document the kinds of values that can be
  # embedded and must be understood by all executor implementations, possibly
  # factoring out parts of reference executor's `to_representation_for_type()`.

  @abc.abstractmethod
  async def create_value(self, value, type_spec=None):
    """A coroutine that creates embedded value from `value` of type `type_spec`.

    This function is used to embed a value within the executor. The argument
    can be one of the plain Python types, a nested structure, a representation
    of a TFF computation, etc. Once embedded, the value can be further passed
    around within the executor. For functional values, embedding them prior to
    invocation potentally allows the executor to amortize overhead across
    multiple calls.

    Args:
      value: An object that represents the value to embed within the executor.
      type_spec: An optional `tff.Type` of the value represented by this object,
        or something convertible to it. The type can only be omitted if the
        value is a instance of `tff.TypedObject`.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents the
      embedded value.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_call(self, comp, arg=None):
    """A coroutine that creates a call to `comp` with optional argument `arg`.

    Args:
      comp: The computation to invoke. It must have been first embedded in the
        executor by calling `create_value()` on it first.
      arg: An optional argument of the call, or `None` if no argument was
        supplied. If it is present, it must have been embedded in the executor
        by calling `create_value()` on it first.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents the
      constructed vall.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_tuple(self, elements):
    """A coroutine that creates a tuple of `elements`.

    Args:
      elements: An enumerable or dict with the elements to create a tuple from.
        The elements must all have been embedded in this executor by invoking
        `create_value()` on them first.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents the
      constructed tuple.
    """
    raise NotImplementedError

  @abc.abstractmethod
  async def create_selection(self, source, index=None, name=None):
    """A coroutine that creates a selection from `source`.

    Args:
      source: The source to select from. The source must have been embedded in
        this executor by invoking `create_value()` on it first.
      index: An optional integer index. Either this, or `name` must be present.
      name: An optional string name. Either this, or `index` must be present.

    Returns:
      An instance of `executor_value_base.ExecutorValue` that represents the
      constructed selection.
    """
    raise NotImplementedError
