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
from collections.abc import Callable, Mapping, Sequence
import functools
import operator
import typing
from typing import Generic, Optional, Protocol, TypeVar, Union

import attrs
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.program import structure_utils
from tensorflow_federated.python.program import value_reference

# ReleaseManager's may release any value in addition to materializable values.
ReleasableValue = Union[
    object,
    value_reference.MaterializableValue,
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
      value: A `tff.program.ReleasableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A value used to reference the released `value`.
    """
    raise NotImplementedError


class NotFilterableError(Exception):
  """Raised when the structure can not be filtered."""


class FilterMismatchError(Exception):
  """Raised when there is a mismatch filtering the value and type signature."""


@typing.runtime_checkable
class _NamedTuple(Protocol):

  @property
  def _fields(self) -> tuple[str, ...]:
    ...

  def _asdict(self) -> dict[str, object]:
    ...


# Sentinel object used by the `tff.program.FilteringReleaseManager` to indicate
# that a subtree can be filtered when traversing structures of values and type
# signatures.
_FILTERED_SUBTREE = object()


class FilteringReleaseManager(ReleaseManager[ReleasableStructure, Key]):
  """A `tff.program.ReleaseManager` that filters values before releasing them.

  A `tff.program.FilteringReleaseManager` is a utility for filtering values
  before releasing the values and is used to release values from platform
  storage to customer storage in a federated program.

  Values are filtered using a `filter_fn` and released to the `release_manager`.

  The `filter_fn` is a `Callable` that has a single parameter `path` and returns
  a `bool`, and is used to filter values before they are released. A `path` is a
  tuple of indices and/or keys which uniquely identifies the position of the
  corresponding item in the `value`; `path` matches the expectations of the
  `tree` library.

  The `filter_fn` is applied to the items in the structure but not the structure
  itself. If all the items in a structure are filtered out, then the structure
  will be filtered out as well.

  For example:

  ```
  filtering_manager = tff.program.FilteringReleaseManager(
      release_manager=...,
      filter_fn=...,
  )

  value = {
    'loss': 1.0,
    'accuracy': 0.5,
  }
  await filtering_manager.release(value, ...)
  ```

  If `filter_fn` is:

  * `lambda _: True` then the entire structure is released.
  * `lambda _: False` then nothing is released.
  * `lambda path: path == ('loss',)` then `{'loss': 1.0}` is released.

  Note: The path `()` corresponds to the root of the structure; because the
  `filter_fn` is applied to the items in the structure but not the structure
  itself, this path can be used to filter individual values from structures of
  values.

  Important: Most `tff.program.ReleasableStructure` can be filtered, including
  individual values, structures, and structures nested in `NamedTuple`s.
  However, the fields of a `NamedTuple` can not be filtered.
  """

  def __init__(
      self,
      release_manager: ReleaseManager[ReleasableStructure, Key],
      filter_fn: Callable[[tuple[Union[str, int], ...]], bool],
  ):
    """Returns an initialized `tff.program.FilteringReleaseManager`.

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
      value: A `tff.program.ReleasableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A value used to reference the released `value`.

    Raises:
      NotFilterableError: If the `value` can not be filtered.
      FilterMismatchError: If there is a mismatch filtering the `value` and
        `type_signature`.
    """

    def _filter_value(
        path: tuple[Union[str, int], ...],
        subtree: ReleasableStructure,
    ) -> Optional[Union[ReleasableStructure, type(_FILTERED_SUBTREE)]]:
      """The function to apply when filtering the `value`.

      This function is meant to be used with `tree.traverse_with_path` to filter
      the `value`. The function `tree.traverse_with_path` is used because the
      traversal functions from `tree` apply a function to the structure and
      the leaves, whereas the map functions only apply a function to the leaves.
      Additionally, `path` is used to determine which parts of the structure to
      filter.

      See https://tree.readthedocs.io/en/latest/api.html#tree.traverse for more
      information.

      Args:
        path: A tuple of indices and/or keys which uniquely identifies the
          position of `subtree` in the `value`.
        subtree: A substructure in `value`.

      Returns:
        A filtered value or `_FILTERED_SUBTREE` if the entire structure was
        filtered.

      Raises:
        NotFilterableError: If `subtree` can not be filtered.
      """
      if tree.is_nested(subtree) and not attrs.has(type(subtree)):
        # TODO(b/224484886): Downcasting to all handled types.
        subtree = typing.cast(
            Union[Sequence[object], Mapping[str, object]], subtree
        )
        if isinstance(subtree, Sequence):
          elements = [x for x in subtree if x is not _FILTERED_SUBTREE]
          if not elements:
            return _FILTERED_SUBTREE
          elif isinstance(subtree, _NamedTuple):
            if len(subtree) != len(elements):
              fields = list(type(subtree)._fields)
              missing_fields = [
                  k
                  for k, v in subtree._asdict().items()
                  if v is _FILTERED_SUBTREE
              ]
              raise NotFilterableError(
                  'The fields of a `NamedTuple` can not be filtered. Expected '
                  f'`{type(subtree)}` to have fields `{fields}`, found it was '
                  f'missing fields `{missing_fields}`.'
              )

            return type(subtree)(*elements)
          else:
            # Assumes the `Sequence` has a constructor that accepts `elements`,
            #  this is safe because `tree` makes the same assumption.
            return type(subtree)(elements)  # pytype: disable=wrong-arg-count
        elif isinstance(subtree, Mapping):
          items = [
              (k, v) for k, v in subtree.items() if v is not _FILTERED_SUBTREE
          ]
          if not items:
            return _FILTERED_SUBTREE
          else:
            # Assumes the `Mapping` has a constructor that accepts `items`,
            # this is safe because `tree` makes the same assumption.
            return type(subtree)(items)  # pytype: disable=wrong-arg-count
        else:
          raise NotImplementedError(f'Unexpected type found: {type(subtree)}.')
      else:
        if self._filter_fn(path):
          return None
        else:
          return _FILTERED_SUBTREE

    filtered_value = tree.traverse_with_path(
        _filter_value, value, top_down=False
    )

    def _create_filtered_type(
        type_spec: computation_types.Type,
        path: tuple[Union[str, int], ...] = (),
    ) -> Union[computation_types.Type, type(_FILTERED_SUBTREE)]:
      """Creates a `tff.Type` from the `type_signature`.

      This function mirrors `_filter_value` in the way the way `path` and
      `_FILTERED_SUBTREE` are used ; however, it does not use `tree` to do the
      traversal because `tree` does not know how to traverse `tff.Type`s.
      Instead the traversal is performed manually.

      Args:
        type_spec: A `tff.Type` in `type_signature`.
        path: A tuple of indices and/or keys which uniquely identifies the
          position of `subtree` in the `value`.

      Returns:
        A filtered value or `_FILTERED_SUBTREE` if the entire structure was
        filtered.
      """
      if isinstance(type_spec, computation_types.StructType) and not attrs.has(
          type(type_spec.python_container)
      ):
        # An empty tuple may represent the type signature for the value `None`.
        # If that is the case, do not filter the empty structure, instead treat
        # `type_spec` as a leaf and apply the filter function.
        if type_spec == computation_types.StructType([]):
          value_at_path = functools.reduce(operator.getitem, path, value)
          if value_at_path is None:
            if self._filter_fn(path):
              return type_spec
            else:
              return _FILTERED_SUBTREE

        elements = []
        element_types = structure.iter_elements(type_spec)
        for index, (name, element_type) in enumerate(element_types):
          element_path = path + (name or index,)
          filtered_element_type = _create_filtered_type(
              element_type, element_path
          )
          if filtered_element_type is not _FILTERED_SUBTREE:
            elements.append((name, filtered_element_type))

        if not elements:
          return _FILTERED_SUBTREE
        elif isinstance(type_spec, computation_types.StructWithPythonType):
          # Note: The fields of a `NamedTuple` can not be filtered. However,
          # raising an error here can be skipped, because the appropriate error
          # is raised when filtering the `value`.
          return computation_types.StructWithPythonType(
              elements, type_spec.python_container
          )
        else:
          return computation_types.StructType(elements)
      else:
        if self._filter_fn(path):
          return type_spec
        else:
          return _FILTERED_SUBTREE

    filtered_type = _create_filtered_type(type_signature)

    if (
        filtered_value is not _FILTERED_SUBTREE
        and filtered_type is not _FILTERED_SUBTREE
    ):
      await self._release_manager.release(filtered_value, filtered_type, key)
    elif filtered_value is not filtered_type:
      if filtered_value is _FILTERED_SUBTREE:
        value_label = 'empty'
      else:
        value_label = 'not empty'
      if filtered_type is _FILTERED_SUBTREE:
        type_signature_label = 'empty'
      else:
        type_signature_label = 'not empty'
      raise FilterMismatchError(
          'Expected `value` and `type_signature` to be filtered consistently, '
          f'found the filtered `value` was {value_label} and the filtered '
          f'`type_signature` was {type_signature_label}.'
      )


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
      value: A `tff.program.ReleasableStructure` to release.
      type_signature: The `tff.Type` of `value`.
      key: A value used to reference the released `value`.
    """
    await asyncio.gather(
        *[m.release(value, type_signature, key) for m in self._release_managers]
    )
