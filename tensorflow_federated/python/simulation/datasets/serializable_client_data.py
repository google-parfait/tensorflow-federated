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
"""Library methods for working with centralized data used in simulation."""

import abc
import collections
from typing import Callable, Iterable, List, Optional

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation.datasets import client_data

# TODO(b/186139255): Merge with client_data.py once ConcreteClientData is
# removed.


class SerializableClientData(client_data.ClientData, metaclass=abc.ABCMeta):
  """Object to hold a federated dataset with serializable dataset construction.

  In contrast to `tff.simulation.datasets.ClientData`, this implementation
  uses a serializable dataset constructor for each client. This enables all
  sub-classes to access `SerializableClientData.dataset_computation`.
  """

  @abc.abstractproperty
  def serializable_dataset_fn(self):
    """A callable accepting a client ID and returning a `tf.data.Dataset`.

    Note that this callable must be traceable by TF, as it will be used in the
    context of a `tf.function`.
    """
    pass

  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    """Creates a new `tf.data.Dataset` containing the client training examples.

    This function will create a dataset for a given client, given that
    `client_id` is contained in the `client_ids` property of the `ClientData`.
    Unlike `create_dataset`, this method need not be serializable.

    Args:
      client_id: The string client_id for the desired client.

    Returns:
      A `tf.data.Dataset` object.
    """
    if client_id not in self.client_ids:
      raise ValueError(
          'ID [{i}] is not a client in this ClientData. See '
          'property `client_ids` for the list of valid ids.'.format(
              i=client_id))
    return self.serializable_dataset_fn(client_id)

  @property
  def dataset_computation(self):
    """A `tff.Computation` accepting a client ID, returning a dataset.

    Note: the `dataset_computation` property is intended as a TFF-specific
    performance optimization for distributed execution.
    """
    if (not hasattr(self, '_cached_dataset_computation')) or (
        self._cached_dataset_computation is None):

      @computations.tf_computation(tf.string)
      def dataset_computation(client_id):
        return self.serializable_dataset_fn(client_id)

      self._cached_dataset_computation = dataset_computation
    return self._cached_dataset_computation

  def create_tf_dataset_from_all_clients(self,
                                         seed: Optional[int] = None
                                        ) -> tf.data.Dataset:
    """Creates a new `tf.data.Dataset` containing _all_ client examples.

    This function is intended for use training centralized, non-distributed
    models (num_clients=1). This can be useful as a point of comparison
    against federated models.

    Currently, the implementation produces a dataset that contains
    all examples from a single client in order, and so generally additional
    shuffling should be performed.

    Args:
      seed: Optional, a seed to determine the order in which clients are
        processed in the joined dataset. The seed can be any 32-bit unsigned
        integer or an array of such integers.

    Returns:
      A `tf.data.Dataset` object.
    """
    client_ids = self.client_ids.copy()
    np.random.RandomState(seed=seed).shuffle(client_ids)
    nested_dataset = tf.data.Dataset.from_tensor_slices(client_ids)
    # We apply serializable_dataset_fn here to avoid loading all client datasets
    # in memory, which is slow. Note that tf.data.Dataset.map implicitly wraps
    # the input mapping in a tf.function.
    example_dataset = nested_dataset.flat_map(self.serializable_dataset_fn)
    return example_dataset

  def preprocess(
      self, preprocess_fn: Callable[[tf.data.Dataset], tf.data.Dataset]
  ) -> 'PreprocessSerializableClientData':
    """Applies `preprocess_fn` to each client's data."""
    py_typecheck.check_callable(preprocess_fn)
    return PreprocessSerializableClientData(self, preprocess_fn)

  @classmethod
  def from_clients_and_tf_fn(
      cls,
      client_ids: Iterable[str],
      serializable_dataset_fn: Callable[[str], tf.data.Dataset],
  ) -> 'ConcreteClientData':
    """Constructs a `ClientData` based on the given function.

    Args:
      client_ids: A non-empty list of strings to use as input to
        `create_dataset_fn`.
      serializable_dataset_fn: A function that takes a client_id from the above
        list, and returns a `tf.data.Dataset`. This function must be
        serializable and usable within the context of a `tf.function` and
        `tff.Computation`.

    Returns:
      A `ConcreteSerializableClientData` object.
    """
    return ConcreteSerializableClientData(client_ids, serializable_dataset_fn)


class PreprocessSerializableClientData(SerializableClientData):
  """Applies a preprocessing function to every dataset it returns.

  This class delegates all other aspects of implementation to its underlying
  `SerializableClientData` object, simply wiring in its `preprocess_fn`
  where necessary. Note that this `preprocess_fn` must be serializable by
  TensorFlow.
  """

  def __init__(  # pylint: disable=super-init-not-called
      self, underlying_client_data: SerializableClientData,
      preprocess_fn: Callable[[tf.data.Dataset], tf.data.Dataset]):
    py_typecheck.check_type(underlying_client_data, SerializableClientData)
    py_typecheck.check_callable(preprocess_fn)
    self._underlying_client_data = underlying_client_data
    self._preprocess_fn = preprocess_fn
    example_dataset = self._preprocess_fn(
        self._underlying_client_data.create_tf_dataset_for_client(
            next(iter(underlying_client_data.client_ids))))
    self._element_type_structure = example_dataset.element_spec
    self._dataset_computation = None

    def serializable_dataset_fn(client_id: str) -> tf.data.Dataset:
      return self._preprocess_fn(
          self._underlying_client_data.serializable_dataset_fn(client_id))  # pylint:disable=protected-access

    self._serializable_dataset_fn = serializable_dataset_fn

  @property
  def serializable_dataset_fn(self):
    return self._serializable_dataset_fn

  @property
  def client_ids(self):
    return self._underlying_client_data.client_ids

  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    return self._preprocess_fn(
        self._underlying_client_data.create_tf_dataset_for_client(client_id))

  @property
  def element_type_structure(self):
    return self._element_type_structure


class ConcreteSerializableClientData(SerializableClientData):
  """A generic `SerializableClientData` object.

  This is a simple implementation of `SerializableClientData`, where datasets
  are specified as a function from `client_id` to a `tf.data.Dataset`, where
  this function must be serializable by a `tf.function`.
  """

  def __init__(  # pylint: disable=super-init-not-called
      self,
      client_ids: Iterable[str],
      serializable_dataset_fn: Callable[[str], tf.data.Dataset],
  ):
    """Creates a `SerializableClientData` from clients and a mapping function.

    Args:
      client_ids: A non-empty iterable of `string` objects, representing ids for
        each client.
      serializable_dataset_fn: A function that takes as input a `string`, and
        returns a `tf.data.Dataset`. This must be traceable by TF and TFF. That
        is, it must be compatible with both `tf.function` and `tff.Computation`
        wrappers.
    """
    py_typecheck.check_type(client_ids, collections.abc.Iterable)
    py_typecheck.check_callable(serializable_dataset_fn)

    if not client_ids:
      raise ValueError('At least one client_id is required.')

    self._client_ids = list(client_ids)
    self._serializable_dataset_fn = serializable_dataset_fn

    example_dataset = serializable_dataset_fn(next(iter(client_ids)))
    self._element_type_structure = example_dataset.element_spec

  @property
  def client_ids(self) -> List[str]:
    return self._client_ids

  @property
  def serializable_dataset_fn(self):
    return self._serializable_dataset_fn

  @property
  def element_type_structure(self):
    return self._element_type_structure
