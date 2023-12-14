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
"""Utilities for working with structured data."""

from collections.abc import Callable, Iterable, Mapping, Sequence
import typing
from typing import Optional, TypeVar, Union

import attrs
import tree


T = TypeVar('T')
# This type defines the structures supported by the `tff.program` API, meaning
# values of type `T` nested in structures defined by this type. For an example
# of how to use this type see `tff.program.MaterializedStructure`.
Structure = Union[
    T,
    Sequence['Structure[T]'],
    Mapping[str, 'Structure[T]'],
]


def _filter_structure(structure: Structure[object]) -> Structure[object]:
  """Returns a filtered `tff.program.Structure`.

  Containers that are not explicity supported by `tff.program.Structure` are
  filtered out (by converting them to `None`) and objects are converted to
  `None`.

  Args:
    structure: A `tff.program.Structure`.
  """

  structure_types = []
  for arg in typing.get_args(Structure):
    origin_type = typing.get_origin(arg)
    if origin_type is not None:
      structure_types.append(origin_type)

  def _fn(structure: Structure[object]) -> Optional[object]:
    if tree.is_nested(structure) and not attrs.has(type(structure)):
      if isinstance(structure, tuple(structure_types)):
        return None
      else:
        return tree.MAP_TO_NONE
    else:
      return tree.MAP_TO_NONE

  return tree.traverse(_fn, structure)


def flatten_with_name(structure: Structure[T]) -> list[tuple[str, T]]:
  """Creates a flattened representation of the `structure` with names.

  Args:
    structure: A `tff.program.Structure`.

  Returns:
    A `list` of `(name, value)` `tuples` representing the flattened `structure`,
    where `name` uniquely identifies the position of the `value` in the
    `structure`.
  """
  filtered_structure = _filter_structure(structure)
  flattened = tree.flatten_with_path_up_to(filtered_structure, structure)

  def _name(path: Iterable[Union[int, str]]) -> str:
    return '/'.join(map(str, path))

  return [(_name(path), value) for path, value in flattened]


def flatten(structure: Structure[T]) -> list[T]:
  """Flattens a `tff.program.Structure` into a `list`."""
  filtered_structure = _filter_structure(structure)
  return tree.flatten_up_to(filtered_structure, structure)


def unflatten_as(
    structure: Structure[T], flat_sequence: Sequence[T]
) -> Structure[T]:
  """Unflattens a sequence into a `tff.program.Structure`."""
  filtered_structure = _filter_structure(structure)
  return tree.unflatten_as(filtered_structure, flat_sequence)


def map_structure(
    fn: Callable[..., T], *structures: Structure[T], **kwargs: object
) -> Structure[T]:
  """Maps `fn` through the `tff.program.Structure`s."""
  if not structures:
    raise ValueError('Expected at least one structure.')
  first_structure = structures[0]
  if len(structures) > 1:
    check_types = kwargs.get('check_types', True)
    for structure in structures[1:]:
      tree.assert_same_structure(first_structure, structure, check_types)
  filtered_structure = _filter_structure(first_structure)
  return tree.map_structure_up_to(filtered_structure, fn, *structures, **kwargs)
