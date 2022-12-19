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
"""Defines abstract interfaces for representing data sources."""

import abc
import enum
from typing import Any, Optional

from tensorflow_federated.python.core.impl.types import computation_types


class Capability(enum.Enum):
  """Capability that can be supported by a data source."""

  # Subsequent iterations over the data in this data source yield the exact
  # same sequence of results (provided that one issues the exact same sequence
  # of requests, e.g., samples of the same size). One example of a data source
  # with this property would be a source that processes data in a sequential
  # order in which it's physically stored, based on a numeric or lexicographic
  # order, or that uses a fixed random seed to determine the order in which the
  # data is retrieved.
  DETERMINISTIC = 1

  # During each individual iteration over the data in this data source, none
  # of the data items are repeated. One example of a data source with this
  # property would be a source that performs sampling without replacement in
  # order to prevent repetitions, or that simply traverses a corpus of data in
  # sequential order (e.g., based on how it's stored).
  NO_REPETITIONS = 2

  # During each individual iteration over the data, and for each individual
  # request within this iteration, the selected subset of users whose data is
  # returned is chosen uniformly at random.
  # NOTE: This property is mutually exclusive with either `DETERMINISTIC` or
  # `NO_REPETITIONS`.
  RANDOM_UNIFORM = 3

  # Samples of data from this data source (i.e., obtained by invoking `select`
  # on the iterators created from this data source) can be supplied as an
  # argument to multiple computation invocations (concurrenly or sequentially).
  # Some data sources that place limits on the amount of time during which the
  # samples of data are available for processing may not support this.
  SUPPORTS_REUSE = 4


class FederatedDataSourceIterator(abc.ABC):
  """An abstract interface for representing federated data source iterators.

  This interface abstracts away the specifics of iterating over data in a data
  source.

  Things one can do with a data source iterator:

  * Determine the type of the data supplied by this iterator by inspecting
    the `federated_type` property. The type returned must match that of the data
    source that returned this iterator.

  * Return a new selection of federated data from the iterator by invoking
    `select`.

  Please see `tff.program.FederatedDataSource` for additional context and the
  high-level description of how to use data sources.
  """

  @property
  @abc.abstractmethod
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select`."""
    raise NotImplementedError

  @abc.abstractmethod
  def select(self, num_clients: Optional[int] = None) -> Any:
    """Returns a new selection of federated data from this iterator.

    The selection contains data distributed across a cohort of logical clients.
    The manner in which this cohort is constructed is a function of the
    arguments supplied here by the caller (such as `num_clients`) as well as by
    the capabilities of the data source itself (e.g., whether it offers data
    selected uniformly at random, performs sampling without replacement,
    guarantees determinism, etc.).

    Args:
      num_clients: Optional, the number of clients to select. Must be a positive
        integer, or `None` if unspecified.

    Returns:
      An object of type `federated_type` representing the selected data, and
      that can be supplied as an argument to a `tff.Computation`. See
      `tff.program.FederatedContext` for more information about these types.
    """
    raise NotImplementedError


class FederatedDataSource(abc.ABC):
  """An abstract interface for representing federated data sources.

  This interface abstracts away the specifics of working with various types of
  data sources.

  Things one can do with a data source:

  * Determine the type of the data supplied by this data source by inspecting
    the `federated_type` property. The type returned should be a federated type.
    Note that depending on whether this data source contains one or a number of
    federated datasets, the type may or may not be a struct (with individual
    datasets appearing as elements of this struct).

  * Determine the guarantees/features offered by this data source by inspecting
    the `capabilities` property.

  * Construct a new iterator for this data source by invoking `iterator` on it.
    Each iterator represents an independent pass over the data from this data
    source.

  See `tff.program.Capability` for the descriptions of the formal guarantees.
  """

  @property
  @abc.abstractmethod
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def capabilities(self) -> list[Capability]:
    """The list of capabilities supported by this data source."""
    raise NotImplementedError

  @abc.abstractmethod
  def iterator(self) -> FederatedDataSourceIterator:
    """Returns a new iterator for retrieving data from this data source."""
    raise NotImplementedError
