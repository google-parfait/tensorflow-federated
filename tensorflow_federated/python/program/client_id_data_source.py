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

import collections
import random
from typing import Any, List, Optional, Sequence

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source


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
    py_typecheck.check_type(client_ids, collections.abc.Sequence)
    for client_id in client_ids:
      py_typecheck.check_type(client_id, str)
    if not client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    self._client_ids = client_ids
    self._federated_type = computation_types.FederatedType(
        tf.string, placements.CLIENTS)

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._federated_type

  def select(self, number_of_clients: Optional[int] = None) -> Any:
    """Returns a new selection of client ids from this iterator.

    Args:
      number_of_clients: A number of clients to use when selecting data, must be
        a positive integer and less than the total number of `client_ids`.

    Raises:
      ValueError: If `number_of_clients` is not a positive integer or if
        `number_of_clients` is not less than the total number of `client_ids`.
    """
    if number_of_clients is not None:
      py_typecheck.check_type(number_of_clients, int)
    if (number_of_clients is None or number_of_clients < 0 or
        number_of_clients > len(self._client_ids)):
      raise ValueError('Expected `number_of_clients` to be a positive integer '
                       'and less than the number of `client_ids`, found '
                       f'number_of_clients: {number_of_clients}, '
                       f'number of client_ids: {len(self._client_ids)}')
    return random.sample(self._client_ids, number_of_clients)


class ClientIdDataSource(data_source.FederatedDataSource):
  """A tff.program.FederatedDataSource managing data in form of client ids."""

  def __init__(self, client_ids: Sequence[str]):
    """Returns an initialized `tff.program.ClientIdDataSource`.

    Args:
      client_ids: A sequence of client ids to use to yield the ids from this
        data source.

    Raises:
      ValueError: If `client_ids` is empty or not a sequence, or if one of the
        ids is not a string.
    """
    py_typecheck.check_type(client_ids, collections.abc.Sequence)
    for client_id in client_ids:
      py_typecheck.check_type(client_id, str)
    if not client_ids:
      raise ValueError('Expected `client_ids` to not be empty.')

    self._client_ids = client_ids
    self._federated_type = computation_types.FederatedType(
        tf.string, placements.CLIENTS)
    self._capabilities = [data_source.Capability.RANDOM_UNIFORM]

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._federated_type

  @property
  def capabilities(self) -> List[data_source.Capability]:
    """The list of capabilities supported by this data source."""
    return self._capabilities

  def iterator(self) -> data_source.FederatedDataSourceIterator:
    """Returns a new iterator for retrieving client ids from this data source."""
    return ClientIdDataSourceIterator(self._client_ids)
