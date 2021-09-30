# Copyright 2021, Google LLC.
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

from typing import Any, List, Tuple

import tree


def flatten(structure: Any) -> List[Tuple[str, Any]]:
  """Creates a list representing a flattened version of the given `structure`.

  Args:
    structure: A possibly nested structure.

  Returns:
    A list of `(path, value)` tuples representing the flattened version of the
    given `structure`, where `path` is a string uniquely identifying the
    position of `value` in the structure of the given `structure`. The returned
    list is sorted by `path`.
  """
  flat = tree.flatten_with_path(structure)

  def name(path):
    return '/'.join(map(str, path))

  return sorted([(name(path), item) for path, item in flat])
