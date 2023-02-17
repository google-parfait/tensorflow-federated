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

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar, Union

import pytype_extensions
import tree


T = TypeVar('T')
# This type defines the structures supported by the `tff.program` API, meaning
# values of type `T` nested in structures defined by this type. For an example
# of how to use this type see `tff.program.MaterializedStructure`.
Structure = Union[
    T,
    Sequence['Structure[T]'],
    Mapping[str, 'Structure[T]'],
    pytype_extensions.Attrs['Structure[T]'],
]


def flatten_with_name(structure: Structure[T]) -> list[tuple[str, T]]:
  """Creates a flattened representation of the `structure` with names.

  Args:
    structure: A potentially nested structure.

  Returns:
    A `list` of `(name, value)` `tuples` representing the flattened `structure`,
    where `name` uniquely identifies the position of the `value` in the
    `structure`.
  """
  flattened = tree.flatten_with_path(structure)

  def _name(path: Iterable[Union[int, str]]) -> str:
    return '/'.join(map(str, path))

  return [(_name(path), value) for path, value in flattened]
