# Copyright 2018, The TensorFlow Federated Authors.
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
"""Placement literals for use in computation and type definitions."""


class PlacementLiteral:
  """A representation of one of the globally recognized placement literals."""

  def __init__(
      self, name: str, uri: str, default_all_equal: bool, description: str
  ):
    self._name = name
    self._uri = uri
    self._default_all_equal = default_all_equal
    self._description = description

  @property
  def name(self) -> str:
    return self._name

  @property
  def uri(self) -> str:
    return self._uri

  @property
  def default_all_equal(self) -> bool:
    return self._default_all_equal

  def is_server(self) -> bool:
    return self is SERVER

  def is_clients(self) -> bool:
    return self is CLIENTS

  def __doc__(self) -> str:
    return self._description

  def __str__(self) -> str:
    return self._name

  def __repr__(self) -> str:
    return "PlacementLiteral('{}')".format(self._uri)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, PlacementLiteral):
      return NotImplemented
    return self._uri == other.uri

  def __hash__(self) -> int:
    return hash(self._uri)


# TODO: b/113112108 - Define the remaining placement literals (e.g.,
# intermediate coordinators). Possibly rename SERVER to COORDINATOR or some such
# if desired.

CLIENTS = PlacementLiteral(
    'CLIENTS',
    'clients',
    default_all_equal=False,
    description='The collective of all client devices.',
)

SERVER = PlacementLiteral(
    'SERVER',
    'server',
    default_all_equal=True,
    description='The single top-level central coordinator.',
)


def uri_to_placement_literal(uri: str) -> PlacementLiteral:
  """Returns the placement literal that corresponds to the given URI.

  Args:
    uri: The URI of the placement.

  Returns:
    The placement literal.

  Raises:
    ValueError: if there is no known placement literal with such URI.
  """
  for literal in [CLIENTS, SERVER]:
    if uri == literal.uri:
      return literal
  raise ValueError(f'There is no known literal with uri `{uri}`.')
