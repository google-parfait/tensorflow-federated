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
"""Helpers for converting Python data representations for CPP bindings."""

import collections
from typing import Mapping

from tensorflow_federated.python.core.impl.types import placements


def convert_cardinalities_dict_to_string_keyed(
    cardinalities: Mapping[placements.PlacementLiteral,
                           int]) -> Mapping[str, int]:
  """Ensures incoming cardinalities dict is formatted correctly."""
  if not isinstance(cardinalities, collections.abc.Mapping):
    raise TypeError('`cardinalities` must be a `Mapping`. Received a type: '
                    f'{type(cardinalities)}.')
  uri_cardinalities = {}
  for placement, cardinality in cardinalities.items():
    if not isinstance(placement, placements.PlacementLiteral):
      raise TypeError('`cardinalities` must be a `Mapping` with '
                      '`PlacementLiteral` (e.g. `tff.CLIENTS`) keys. '
                      f'Received a key of type: {type(placement)}.')
    if not isinstance(cardinality, int):
      raise TypeError('`cardinalities` must be a `Mapping` with `int` values. '
                      f'Received a value of type: {type(cardinality)} for '
                      f'placement {placement}.')
    uri_cardinalities[placement.uri] = cardinality
  return uri_cardinalities
