# Copyright 2022, The TensorFlow Federated Authors.
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
"""Utilities for representing data sources backed by client ids."""

from collections.abc import Sequence
import random
from typing import Optional

import numpy as np

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import serialization_utils


class ClientIdDataSourceIterator(data_source.FederatedDataSourceIterator):
  """A `tff.program.FederatedDataSourceIterator` backed by client ids.

  A `tff.program.FederatedDataSourceIterator` backed by sequence of client ids,
  one client id per client. It selects client ids uniformly at random, with
  replacement over successive calls of `select()` but without replacement within
  a single call of `select()`.
  """

  def __init__(self, client_ids: Sequence[str]):
    """Returns an initialized `tff.program.ClientIdDataSourceIterator`.

    Args:
      client_ids: A sequence of client ids to use to yield the ids from this
        data source.

    Raises:
      ValueError: If `client_ids` is empty.
    """
    py_typecheck.check_type(client_ids, Sequence)
    for client_id in client_ids:
      py_typecheck.check_type(client_id, str)
    if not client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    self._client_ids = client_ids
    self._federated_type = computation_types.FederatedType(
        np.str_, placements.CLIENTS
    )

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'ClientIdDataSourceIterator':
    """Deserializes the object from bytes."""
    client_ids, _ = serialization_utils.unpack_sequence_from(
        serialization_utils.unpack_str_from, buffer
    )
    return ClientIdDataSourceIterator(client_ids)

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    client_ids_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_str, self._client_ids
    )
    return client_ids_bytes

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select`."""
    return self._federated_type

  def select(self, k: Optional[int] = None) -> object:
    """Returns a new selection of client ids from this iterator.

    Args:
      k: A number of elements to select. Must be a positive integer and less
        than the number of `client_ids`.

    Raises:
      ValueError: If `k` is not a positive integer or if `k` is not less than
        the number of `client_ids`.
    """
    if k is not None:
      py_typecheck.check_type(k, int)
    if k is None or k < 0 or k > len(self._client_ids):
      raise ValueError(
          'Expected `k` to be a positive integer and less than the number of '
          f'`client_ids`, found `k` of {k} and number of `client_ids` of '
          f'{len(self._client_ids)}.'
      )

    return random.sample(self._client_ids, k)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, ClientIdDataSourceIterator):
      return NotImplemented
    return self._client_ids == other._client_ids


class ClientIdDataSource(data_source.FederatedDataSource):
  """A `tff.program.FederatedDataSource` backed by client ids."""

  def __init__(self, client_ids: Sequence[str]):
    """Returns an initialized `tff.program.ClientIdDataSource`.

    Args:
      client_ids: A sequence of strings used to yield the client ids from this
        data source. Must not be empty.

    Raises:
      ValueError: If `client_ids` is empty.
    """
    py_typecheck.check_type(client_ids, Sequence)
    for client_id in client_ids:
      py_typecheck.check_type(client_id, str)
    if not client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    self._client_ids = client_ids
    self._federated_type = computation_types.FederatedType(
        np.str_, placements.CLIENTS
    )

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._federated_type

  def iterator(self) -> data_source.FederatedDataSourceIterator:
    """Returns a new iterator for retrieving client ids from this data source."""
    return ClientIdDataSourceIterator(self._client_ids)
