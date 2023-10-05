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
from typing import Optional

from tensorflow_federated.python.common_libs import serializable
from tensorflow_federated.python.core.impl.types import computation_types


class FederatedDataSourceIterator(serializable.Serializable, abc.ABC):
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
  def select(self, k: Optional[int] = None) -> object:
    """Returns a new selection of federated data from this iterator.

    Args:
      k: An optional number of elements to select. Must be a positive integer,
        or `None` if unspecified.

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

  * Construct a new iterator for this data source by invoking `iterator` on it.
    Each iterator represents an independent pass over the data from this data
    source.
  """

  @property
  @abc.abstractmethod
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    raise NotImplementedError

  @abc.abstractmethod
  def iterator(self) -> FederatedDataSourceIterator:
    """Returns a new iterator for retrieving data from this data source."""
    raise NotImplementedError
