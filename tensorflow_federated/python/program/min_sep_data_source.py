# Copyright 2025, The TensorFlow Federated Authors.
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
"""Utilities for representing data sources with min-sep round participation."""

import random
from typing import Optional
import federated_language


class MinSepDataSourceIterator(
    federated_language.program.FederatedDataSourceIterator
):
  """A `FederatedDataSourceIterator` providing min-sep round participation.

  Clients, which are represented by client ids, are eligible for participation
  in rounds that are exactly `min_sep` rounds apart. The round indices for which
  a client is eligible are computed randomly at initialization time, and when
  `select` is called, the requested number of clients are randomly selected from
  those eligible for the current round.
  """

  def __init__(
      self,
      client_ids: list[str],
      min_sep: int,
      federated_type: federated_language.FederatedType,
  ):
    """Returns an initialized `tff.program.MinSepDataSourceIterator`.

    Args:
      client_ids: A list of strings representing the clients from this data
        source. Must not be empty.
      min_sep: The number of rounds that must elapse between successive
        participations for the same client. Must be a positive integer.
      federated_type: The type of data represented by this data source iterator.

    Raises:
      ValueError: If `client_ids` is empty or if `min_sep` is not a positive
        integer.
    """
    if not client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    if min_sep <= 0:
      raise ValueError(
          'Expected `min_sep` to be a positive integer, found `min_sep` of '
          f'{min_sep}.'
      )

    # The client ids are randomly assigned to `min_sep` rounds. A client id
    # wil be eligible for participation in round `i` if it is assigned to
    # the `i % min_sep`th entry in `_client_id_round_assignments`.
    self._client_id_round_assignments = [[] for _ in range(min_sep)]
    for client_id in client_ids:
      self._client_id_round_assignments[random.randint(0, min_sep - 1)].append(
          client_id
      )

    self._min_sep = min_sep
    self._federated_type = federated_type
    self._round_index = 0

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'MinSepDataSourceIterator':
    """Deserializes the object from bytes."""
    raise NotImplementedError()

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    raise NotImplementedError()

  @property
  def federated_type(self) -> federated_language.FederatedType:
    """The type of the data returned by calling `select`."""
    return self._federated_type

  def select(self, k: Optional[int] = None) -> object:
    """Returns a new selection of client ids for the present round.

    Args:
      k: A number of elements to select. Must be a positive integer. If greater
        than the number of eligible clients for this round, fewer than k client
        ids will be returned.

    Raises:
      ValueError: If `k` is not a positive integer.
    """
    # Obtain the eligible client ids for the current round.
    eligible_ids = self._client_id_round_assignments[
        self._round_index % self._min_sep
    ]

    if k is None or k < 0:
      raise ValueError(
          'Expected `k` to be a positive integer, found `k` of {k}.'
      )

    selected_ids = random.sample(eligible_ids, min(len(eligible_ids), k))
    self._round_index += 1
    return selected_ids


class MinSepDataSource(federated_language.program.FederatedDataSource):
  """A `FederatedDataSource` providing min-sep round participation behavior."""

  def __init__(
      self,
      client_ids: list[str],
      min_sep: int,
      federated_type: federated_language.FederatedType,
  ):
    """Returns an initialized `tff.program.MinSepDataSource`.

    Args:
      client_ids: A list of strings representing the clients from this data
        source. Must not be empty.
      min_sep: The number of rounds that must elapse between successive
        participations for the same client. Must be a positive integer.
      federated_type: The type of data represented by this data source.

    Raises:
      ValueError: If `client_ids` is empty or if `min_sep` is not a positive
        integer.
    """
    if not client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    if min_sep <= 0:
      raise ValueError(
          'Expected `min_sep` to be a positive integer, found `min_sep` of '
          f'{min_sep}.'
      )

    self._client_ids = client_ids
    self._min_sep = min_sep
    self._federated_type = federated_type

  @property
  def federated_type(self) -> federated_language.FederatedType:
    return self._federated_type

  def iterator(self) -> MinSepDataSourceIterator:
    return MinSepDataSourceIterator(
        self._client_ids, self._min_sep, self._federated_type
    )
