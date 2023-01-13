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
"""Utilities for releasing values from a federated program."""

import abc
import asyncio
import collections
from collections.abc import Callable, Sequence
from typing import Any, Generic, Optional, TypeVar, Union

import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference

# ReleaseManager's may release any value (including materialized values) in
# addition to materializable values.
ReleasableValue = Union[
    Any,
    value_reference.MaterializableValueReference,
]
ReleasableStructure = TypeVar(
    'ReleasableStructure',
    bound=structure_utils.Structure[ReleasableValue],
)
Key = TypeVar('Key')


class ReleaseManager(abc.ABC, Generic[ReleasableStructure, Key]):
  """An interface for releasing values from a federated program.

  A `tff.program.ReleaseManager` is used to release values from platform storage
  to customer storage in a federated program.
  """

  @abc.abstractmethod
  async def release(
      self,
      value: ReleasableStructure,
      type_signature: computation_types.Type,
      key: Key,
  ) -> None:
    """Releases `value` from a federated program.

    An implementation of this interface should be specific about the types of
    `value` and `key` for this method and should document how the `key` will be
    used. This allows a federated program to understand how to create a `key`
    for the `value` before it is released. For example, a
    `tff.program.ReleaseManager` that releases metrics keyed by a strictly
    increasing integer might specify a `value` type of
    `Mapping[str, ReleasableValue]` and a `key` type of `int`.

    Args:
      value: A `tff.program.MaterializableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A value used to reference the released `value`.
    """
    raise NotImplementedError


_FILTERED_SUBTREE = object()


class FilteringReleaseManager(ReleaseManager[ReleasableStructure, Key]):
  """A `tff.program.ReleaseManager` that filters values before releasing them.

  A `tff.program.FilteringReleaseManager` is a utility for filtering values
  before releasing the values and is used to release values from platform
  storage to customer storage in a federated program.

  Values are filtered and released using the given `release_manager`.
  """

  def __init__(
      self,
      release_manager: ReleaseManager[ReleasableStructure, Key],
      filter_fn: Callable[[tuple[Union[str, int], ...]], bool],
  ):
    """Returns an initialized `tff.program.FilteringReleaseManager`.

    The `filter_fn` is a `Callable` that has a single parameter `path` and
    returns a `bool`, and is used to filter values before they are released. A
    `path` is a tuple of indices and/or keys which uniquely identifies the
    position of the corresponding item in the `value`; `path` matches the
    expectations of the `dm-tree` library. For example, this function could be
    used to release a single metric with the name 'loss':

    ```
    def filter_fn(path) -> bool:
      if path == ('loss', ):
        return True
      else:
        return False
    ```

    Args:
      release_manager: A `tff.program.ReleaseManager` used to release values to.
      filter_fn: A `Callable` used to filter values before they are released,
        this function has a single parameter `path` and returns a `bool`.
    """
    py_typecheck.check_type(release_manager, ReleaseManager)
    py_typecheck.check_callable(filter_fn)

    self._release_manager = release_manager
    self._filter_fn = filter_fn

  async def release(
      self,
      value: ReleasableStructure,
      type_signature: computation_types.Type,
      key: Key,
  ) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.MaterializableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A value used to reference the released `value`.
    """

    def _fn(
        path: tuple[Union[str, int], ...], subtree: tree.Structure
    ) -> Optional[tree.Structure]:
      if not tree.is_nested(subtree):
        if self._filter_fn(path):
          return None
        else:
          return _FILTERED_SUBTREE
      else:
        if isinstance(subtree, collections.OrderedDict):
          return collections.OrderedDict(
              [(k, v) for k, v in subtree.items() if v is not _FILTERED_SUBTREE]
          )
        elif isinstance(subtree, dict):
          items = sorted(subtree.items())
          return {k: v for k, v in items if v is not _FILTERED_SUBTREE}
        elif isinstance(
            subtree, (list, tuple)
        ) and not py_typecheck.is_named_tuple(subtree):
          return [x for x in subtree if x is not _FILTERED_SUBTREE]
        else:
          raise NotImplementedError(f'Unexpected type found: {type(subtree)}.')

    filtered_value = tree.traverse_with_path(_fn, value, top_down=False)

    if type_signature.is_struct():
      type_signature = structure.to_odict_or_tuple(type_signature)
    filtered_type = tree.traverse_with_path(_fn, type_signature, top_down=False)
    if tree.is_nested(filtered_type):
      filtered_type = computation_types.to_type(filtered_type)

    await self._release_manager.release(filtered_value, filtered_type, key)


class GroupingReleaseManager(ReleaseManager[ReleasableStructure, Key]):
  """A `tff.program.ReleaseManager` that releases values to other release managers.

  A `tff.program.GroupingReleaseManager` is a utility for release values from a
  federated program to a collection of other release managers and is used to
  release values from platform storage to customer storage in a federated
  program.

  Values are released using each of the `tff.program.ReleaseManager`s in the
  given `release_managers`.
  """

  def __init__(
      self, release_managers: Sequence[ReleaseManager[ReleasableStructure, Key]]
  ):
    """Returns an initialized `tff.program.GroupingReleaseManager`.

    Args:
      release_managers: A sequence of `tff.program.ReleaseManager` used to
        release values to.

    Raises:
      ValueError: If `release_managers` is empty.
    """
    py_typecheck.check_type(release_managers, Sequence)
    if not release_managers:
      raise ValueError('Expected `release_managers` to not be empty.')
    for release_manager in release_managers:
      py_typecheck.check_type(release_manager, ReleaseManager)

    self._release_managers = release_managers

  async def release(
      self,
      value: ReleasableStructure,
      type_signature: computation_types.Type,
      key: Key,
  ) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.MaterializableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A value used to reference the released `value`.
    """
    await asyncio.gather(
        *[m.release(value, type_signature, key) for m in self._release_managers]
    )
