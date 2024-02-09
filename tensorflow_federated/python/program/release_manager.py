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
import datetime
import typing
from typing import Generic, Optional, TypeVar, Union

import attrs
import tree

from tensorflow_federated.python.common_libs import py_typecheck
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


class ReleasedValueNotFoundError(Exception):
  """Raised when a released value cannot be found."""

  def __init__(self, key: object):
    super().__init__(f'No released value found for key: {key}.')


class ReleaseManager(abc.ABC, Generic[ReleasableStructure, Key]):
  """An interface for releasing values from a federated program.

  A `tff.program.ReleaseManager` is used to release values from platform storage
  to customer storage in a federated program.
  """

  @abc.abstractmethod
  async def release(self, value: ReleasableStructure, key: Key) -> None:
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
      key: A value used to reference the released `value`.
    """
    raise NotImplementedError


class NotFilterableError(Exception):
  """Raised when the structure cannot be filtered."""


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
  However, the fields of a `NamedTuple` cannot be filtered.
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

    self._release_manager = release_manager
    self._filter_fn = filter_fn

  async def release(self, value: ReleasableStructure, key: Key) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      key: A value used to reference the released `value`.

    Raises:
      NotFilterableError: If the `value` cannot be filtered.
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
        NotFilterableError: If `subtree` cannot be filtered.
      """
      if tree.is_nested(subtree) and not attrs.has(type(subtree)):
        # TODO: b/224484886 - Downcasting to all handled types.
        subtree = typing.cast(
            Union[Sequence[object], Mapping[str, object]], subtree
        )
        if isinstance(subtree, Sequence):
          elements = [x for x in subtree if x is not _FILTERED_SUBTREE]
          if not elements:
            return _FILTERED_SUBTREE
          elif isinstance(subtree, py_typecheck.SupportsNamedTuple):
            if len(subtree) != len(elements):
              fields = list(type(subtree)._fields)
              missing_fields = [
                  k
                  for k, v in subtree._asdict().items()
                  if v is _FILTERED_SUBTREE
              ]
              raise NotFilterableError(
                  'The fields of a `NamedTuple` cannot be filtered. Expected '
                  f'{type(subtree)} to have fields {fields}, found it was '
                  f'missing fields {missing_fields}.'
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

    if filtered_value is not _FILTERED_SUBTREE:
      await self._release_manager.release(filtered_value, key=key)


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

  async def release(self, value: ReleasableStructure, key: Key) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      key: A value used to reference the released `value`.
    """
    await asyncio.gather(
        *[m.release(value, key=key) for m in self._release_managers]
    )


class PeriodicReleaseManager(ReleaseManager[ReleasableStructure, Key]):
  """A `tff.program.ReleaseManager` that releases values at regular intervals.

  A `tff.program.PeriodicReleaseManager` is a utility for releasing values at
  regular intervals and is used to release values from platform storage to
  customer storage in a federated program.

  The interval can be controlled at construction time by setting the
  `periodicity`. The `periodicity` can be a positive integer or
  `datetime.timedelta`. A `periodicity` of `3` means that every third value is
  released to the `release_manager`, and invoking `release` ten times will
  release the third, sixth, and ninth values. A `periodicity` of
  `datetime.timedelta(hours=3)` means that three hours after the previously
  released value the next value is released to the `release_manager`.

   Note: that a `periodicity` of one or a very small `datetime.timedelta` will
   release every value, making the `tff.program.PeriodicReleaseManager` a noop
   wrapper around the `release_manager`.
  """

  def __init__(
      self,
      release_manager: ReleaseManager[ReleasableStructure, Key],
      periodicity: Union[int, datetime.timedelta],
  ):
    """Returns an initialized `tff.program.PeriodicReleaseManager`.

    Args:
      release_manager: A `tff.program.ReleaseManager` used to release values to.
      periodicity: The interval to release values. Must be a positive integer or
        `datetime.timedelta`.

    Raises:
      ValueError: If `periodicity` is not a positive integer or
      `datetime.timedelta`.
    """
    if (isinstance(periodicity, int) and periodicity < 1) or (
        isinstance(periodicity, datetime.timedelta)
        and periodicity.total_seconds() < 1.0
    ):
      raise ValueError(
          'Expected `periodicity` to be a positive integer or'
          f' `datetime.timedelta`, found {periodicity}.'
      )

    self._release_manager = release_manager
    self._periodicity = periodicity
    if isinstance(periodicity, int):
      self._count = 0
    elif isinstance(periodicity, datetime.timedelta):
      self._timestamp = datetime.datetime.now()
    else:
      raise NotImplementedError(
          f'Unexpected `periodicity` found: {type(periodicity)}.'
      )

  async def release(self, value: ReleasableStructure, key: Key) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      key: A value used to reference the released `value`.
    """
    if isinstance(self._periodicity, int):
      self._count += 1
      if self._count % self._periodicity == 0:
        await self._release_manager.release(value, key=key)
    elif isinstance(self._periodicity, datetime.timedelta):
      now = datetime.datetime.now()
      if now >= self._timestamp + self._periodicity:
        self._timestamp = now
        await self._release_manager.release(value, key=key)
    else:
      raise NotImplementedError(
          f'Unexpected `periodicity` found: {type(self._periodicity)}.'
      )


class DelayedReleaseManager(ReleaseManager[ReleasableStructure, Key]):
  """A `tff.program.ReleaseManager` that releases values after specified delay.

  A `tff.program.DelayedReleaseManager` is a utility for releasing values in a
  federated program, where releases only take place after a specified delay
  count. I.e., releases from platform storage to customer storage will take
  place only after a certain number of instances that the `.release()` method is
  called. After this delay, further calls to `.release()` will release values
  (in accordance with the `release_manager` that was provided).

  For example, in a federated program that runs for a long time, one may want to
  skip releasing values until the program has run for a sufficiently long-enough
  period.

  The delay count is specified at construction time by setting the `delay`
  argument (an integer). A `delay` of `3` means that all values will start to be
  released once `release` has been invoked at least three times.

   Note: that a `delay` of one will release every value, making the
   `tff.program.DelayedReleaseManager` a noop wrapper around the
   `release_manager`.
  """

  def __init__(
      self,
      release_manager: ReleaseManager[ReleasableStructure, Key],
      delay: int,
  ):
    """Returns an initialized `tff.program.DelayedReleaseManager`.

    Args:
      release_manager: A `tff.program.ReleaseManager` used to release values to.
      delay: The delay duration before releasing values. Must be a positive
        integer.

    Raises:
      ValueError: If `delay` is not positive.
    """
    if delay < 1:
      raise ValueError(f'The `delay` must be positive but found {delay}.')

    self._release_manager = release_manager
    self._count = 0
    self._delay = delay

  async def release(self, value: ReleasableStructure, key: Key) -> None:
    """Releases `value` from a federated program.

    Args:
      value: A `tff.program.ReleasableStructure` to release.
      key: A value used to reference the released `value`.
    """
    self._count += 1
    if self._count >= self._delay:
      await self._release_manager.release(value, key=key)
