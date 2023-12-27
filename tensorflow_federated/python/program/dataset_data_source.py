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
"""Utilities for representing data sources backed by `tf.data.Dataset`s."""

from collections.abc import Sequence
import random
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.program import data_source
from tensorflow_federated.python.program import serialization_utils


class DatasetDataSourceIterator(data_source.FederatedDataSourceIterator):
  """A `tff.program.FederatedDataSourceIterator` backed by `tf.data.Dataset`s.

  A `tff.program.FederatedDataSourceIterator` backed by a sequence of
  `tf.data.Dataset's, one `tf.data.Dataset' per client. It selects datasources
  uniformly at random, with replacement over successive calls of `select()` but
  without replacement within a single call of `select()`.
  """

  def __init__(
      self,
      datasets: Sequence[tf.data.Dataset],
      federated_type: computation_types.FederatedType,
  ):
    """Returns an initialized `tff.program.DatasetDataSourceIterator`.

    Args:
      datasets: A sequence of `tf.data.Dataset's to use to yield the data from
        this data source.
      federated_type: The type of the data returned by calling `select` on an
        iterator.

    Raises:
      ValueError: If `datasets` is empty or if each `tf.data.Dataset` in
        `datasets` does not have the same type specification.
    """
    py_typecheck.check_type(datasets, Sequence)
    if not datasets:
      raise ValueError('Expected `datasets` to not be empty.')
    for dataset in datasets:
      py_typecheck.check_type(dataset, tf.data.Dataset)
      element_spec = datasets[0].element_spec
      if dataset.element_spec != element_spec:
        raise ValueError(
            'Expected each `tf.data.Dataset` in `datasets` to have the same '
            f'type specification, found {element_spec} and '
            f'{dataset.element_spec}.'
        )
    py_typecheck.check_type(federated_type, computation_types.FederatedType)

    self._datasets = datasets
    self._federated_type = federated_type

  @classmethod
  def from_bytes(cls, buffer: bytes) -> 'DatasetDataSourceIterator':
    """Deserializes the object from bytes."""
    offset = 0
    datasets, datasets_size = serialization_utils.unpack_sequence_from(
        serialization_utils.unpack_dataset_from, buffer, offset=offset
    )
    offset += datasets_size
    federated_type, _ = serialization_utils.unpack_type_spec_from(
        buffer, offset=offset
    )
    if not isinstance(federated_type, computation_types.FederatedType):
      raise TypeError(
          'Expected `federated_type` to be a `tff.FederatedType`, found '
          f'{type(federated_type)}.'
      )
    return DatasetDataSourceIterator(
        datasets=datasets, federated_type=federated_type
    )

  def to_bytes(self) -> bytes:
    """Serializes the object to bytes."""
    datasets_bytes = serialization_utils.pack_sequence(
        serialization_utils.pack_dataset, self._datasets
    )
    federated_type_bytes = serialization_utils.pack_type_spec(
        self._federated_type
    )
    return datasets_bytes + federated_type_bytes

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select`."""
    return self._federated_type

  def select(self, k: Optional[int] = None) -> object:
    """Returns a new selection of data from this iterator.

    Args:
      k: A number of elements to select. Must be a positive integer and less
        than the number of `datasets`.

    Raises:
      ValueError: If `k` is not a positive integer or if `k` is not less than
        the number of `datasets`.
    """
    if k is not None:
      py_typecheck.check_type(k, int)
    if k is None or k < 0 or k > len(self._datasets):
      raise ValueError(
          'Expected `k` to be a positive integer and less than the number of '
          f'`datasets`, found `k` of {k} and number of `datasets` of '
          f'{len(self._datasets)}.'
      )

    return random.sample(self._datasets, k)

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    elif not isinstance(other, DatasetDataSourceIterator):
      return NotImplemented
    if self._federated_type != other._federated_type:
      return False
    for self_dataset, other_dataset in zip(self._datasets, other._datasets):
      if list(self_dataset) != list(other_dataset):
        return False
    return True


class DatasetDataSource(data_source.FederatedDataSource):
  """A `tff.program.FederatedDataSource` backed by `tf.data.Dataset`s.

  A `tff.program.FederatedDataSource` backed by a sequence of
  `tf.data.Dataset's, one `tf.data.Dataset' per client, and selects data
  uniformly random with replacement.
  """

  def __init__(self, datasets: Sequence[tf.data.Dataset]):
    """Returns an initialized `tff.program.DatasetDataSource`.

    Args:
      datasets: A sequence of `tf.data.Dataset's to use to yield the data from
        this data source. Must not be empty and each `tf.data.Dataset' must have
        the same type specification.

    Raises:
      ValueError: If `datasets` is empty or if each `tf.data.Dataset` in
        `datasets` does not have the same type specification.
    """
    py_typecheck.check_type(datasets, Sequence)
    if not datasets:
      raise ValueError('Expected `datasets` to not be empty.')
    for dataset in datasets:
      py_typecheck.check_type(dataset, tf.data.Dataset)
      element_spec = datasets[0].element_spec
      if dataset.element_spec != element_spec:
        raise ValueError(
            'Expected each `tf.data.Dataset` in `datasets` to have the same '
            f'type specification, found {element_spec} and '
            f'{dataset.element_spec}.'
        )

    self._datasets = datasets
    element_type = computation_types.tensorflow_to_type(element_spec)
    self._federated_type = computation_types.FederatedType(
        computation_types.SequenceType(element_type), placements.CLIENTS
    )

  @property
  def federated_type(self) -> computation_types.FederatedType:
    """The type of the data returned by calling `select` on an iterator."""
    return self._federated_type

  def iterator(self) -> DatasetDataSourceIterator:
    """Returns a new iterator for retrieving datasets from this data source."""
    return DatasetDataSourceIterator(self._datasets, self._federated_type)
