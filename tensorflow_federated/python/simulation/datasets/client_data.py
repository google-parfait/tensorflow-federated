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
from typing import Callable, Iterable, List, Optional, Tuple

from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations


class ClientData(object, metaclass=abc.ABCMeta):
  """Object to hold a federated dataset.

  The federated dataset is represented as a list of client ids, and
  a function to look up the local dataset for each client id.

  Note: Cross-device federated learning does not use client IDs or perform any
  tracking of clients. However in simulation experiments using centralized test
  data the experimenter may select specific clients to be processed per round.
  The concept of a client ID is only available at the preprocessing stage when
  preparing input data for the simulation and is not part of the TensorFlow
  Federated core APIs.

  Each client's local dataset is represented as a `tf.data.Dataset`, but
  generally this class (and the corresponding datasets hosted by TFF) can
  easily be consumed by any Python-based ML framework as `numpy` arrays:

  ```python
  import tensorflow as tf
  import tensorflow_federated as tff
  import tensorflow_datasets as tfds

  for client_id in sampled_client_ids[:5]:
    client_local_dataset = tfds.as_numpy(
        emnist_train.create_tf_dataset_for_client(client_id))
    # client_local_dataset is an iterable of structures of numpy arrays
    for example in client_local_dataset:
      print(example)
  ```

  If desiring a manner for constructing ClientData objects for testing purposes,
  please see the `tff.simulation.datasets.TestClientData` class, as it provides
  an easy way to construct toy federated datasets.
  """

  @abc.abstractproperty
  def client_ids(self) -> List[str]:
    """A list of string identifiers for clients in this dataset."""
    pass

  @abc.abstractmethod
  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    """Creates a new `tf.data.Dataset` containing the client training examples.

    Args:
      client_id: The string client_id for the desired client.

    Returns:
      A `tf.data.Dataset` object.
    """
    pass

  @abc.abstractproperty
  def dataset_computation(self) -> computation_base.Computation:
    """A `tff.Computation` accepting a client ID, returning a dataset.

    Note: the `dataset_computation` property is intended as a TFF-specific
    performance optimization for distributed execution, and subclasses of
    `ClientData` may or may not support it.

    `ClientData` implementations that don't support `dataset_computation`
    should raise `NotImplementedError` if this attribute is accessed.
    """
    pass

  @abc.abstractproperty
  def element_type_structure(self):
    """The element type information of the client datasets.

    Returns:
      A nested structure of `tf.TensorSpec` objects defining the type of the
    elements returned by datasets in this `ClientData` object.
    """
    pass

  def datasets(self,
               limit_count: Optional[int] = None,
               seed: Optional[int] = None) -> Iterable[tf.data.Dataset]:
    """Yields the `tf.data.Dataset` for each client in random order.

    This function is intended for use building a static array of client data
    to be provided to the top-level federated computation.

    Args:
      limit_count: Optional, a maximum number of datasets to return.
      seed: Optional, a seed to determine the order in which clients are
        processed in the joined dataset. The seed can be any 32-bit unsigned
        integer or an array of such integers.
    """
    # Create a copy to prevent the original list being reordered
    client_ids = self.client_ids.copy()
    np.random.RandomState(seed=seed).shuffle(client_ids)
    count = 0
    for client_id in client_ids:
      if limit_count is not None and count >= limit_count:
        return
      count += 1
      dataset = self.create_tf_dataset_for_client(client_id)
      py_typecheck.check_type(dataset, tf.data.Dataset)
      yield dataset

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
    # Note: simply calling Dataset.concatenate() will result in too deep
    # recursion depth.
    # Note: Tests are via the simple concrete from_tensor_slices_client_data.
    client_datasets = list(self.datasets(seed=seed))
    nested_dataset = tf.data.Dataset.from_tensor_slices(client_datasets)
    example_dataset = nested_dataset.flat_map(lambda x: x)
    return example_dataset

  def preprocess(
      self, preprocess_fn: Callable[[tf.data.Dataset], tf.data.Dataset]
  ) -> 'PreprocessClientData':
    """Applies `preprocess_fn` to each client's data."""
    py_typecheck.check_callable(preprocess_fn)
    return PreprocessClientData(self, preprocess_fn)

  @classmethod
  def from_clients_and_fn(
      cls,
      client_ids: Iterable[str],
      create_tf_dataset_for_client_fn: Callable[[str], tf.data.Dataset],
  ) -> 'ConcreteClientData':
    """Constructs a `ClientData` based on the given function.

    Args:
      client_ids: A non-empty list of client_ids which are valid inputs to the
        create_tf_dataset_for_client_fn.
      create_tf_dataset_for_client_fn: A function that takes a client_id from
        the above list, and returns a `tf.data.Dataset`. If this function is
        additionally a `tff.Computation`, the constructed `ClientData`
        will expose a `dataset_computation` attribute which can be used for
        high-performance distributed simulations.

    Returns:
      A `ClientData`.
    """
    return ConcreteClientData(client_ids, create_tf_dataset_for_client_fn)

  @classmethod
  def train_test_client_split(
      cls,
      client_data: 'ClientData',
      num_test_clients: int,
      seed: Optional[int] = None) -> Tuple['ClientData', 'ClientData']:
    """Returns a pair of (train, test) `ClientData`.

    This method partitions the clients of `client_data` into two `ClientData`
    objects with disjoint sets of `ClientData.client_ids`. All clients in the
    test `ClientData` are guaranteed to have non-empty datasets, but the
    training `ClientData` may have clients with no data.

    Note: This method may be expensive, and so it may be useful to avoid calling
    multiple times and holding on to the results.

    Args:
      client_data: The base `ClientData` to split.
      num_test_clients: How many clients to hold out for testing. This can be at
        most len(client_data.client_ids) - 1, since we don't want to produce
        empty `ClientData`.
      seed: Optional seed to fix shuffling of clients before splitting.

    Returns:
      A pair (train_client_data, test_client_data), where test_client_data
      has `num_test_clients` selected at random, subject to the constraint they
      each have at least 1 batch in their dataset.

    Raises:
      ValueError: If `num_test_clients` cannot be satistifed by `client_data`,
        or too many clients have empty datasets.
    """
    if num_test_clients <= 0:
      raise ValueError('Please specify num_test_clients > 0.')

    if len(client_data.client_ids) <= num_test_clients:
      raise ValueError('The client_data supplied has only {} clients, but '
                       '{} test clients were requested.'.format(
                           len(client_data.client_ids), num_test_clients))

    train_client_ids = list(client_data.client_ids)
    np.random.RandomState(seed).shuffle(train_client_ids)
    # These clients will be added back into the training set at the end.
    clients_with_insufficient_batches = []
    test_client_ids = []
    while len(test_client_ids) < num_test_clients:
      if not train_client_ids or (
          # Arbitrarily threshold where "many" (relative to num_test_clients)
          # clients have no data. Note: If needed, we could make this limit
          # configurable.
          len(clients_with_insufficient_batches) > 5 * num_test_clients + 10):

        raise ValueError('Encountered too many clients with no data.')

      client_id = train_client_ids.pop()
      dataset = client_data.create_tf_dataset_for_client(client_id)
      try:
        _ = next(dataset.__iter__())
      except StopIteration:
        logging.warning('Client %s had no data, skipping.', client_id)
        clients_with_insufficient_batches.append(client_id)
        continue

      test_client_ids.append(client_id)

    # Invariant for successful exit of the above loop:
    assert len(test_client_ids) == num_test_clients

    def from_ids(client_ids: Iterable[str]) -> 'ConcreteClientData':
      return cls.from_clients_and_fn(client_ids,
                                     client_data.create_tf_dataset_for_client)

    return (from_ids(train_client_ids + clients_with_insufficient_batches),
            from_ids(test_client_ids))


class PreprocessClientData(ClientData):
  """Applies a preprocessing function to every dataset it returns.

  This `ClientData` subclass delegates all other aspects of implementation to
  its underlying `ClientData` object, simply wiring in its `preprocess_fn`
  where necessary.
  """

  def __init__(self, underlying_client_data: ClientData,
               preprocess_fn: Callable[[tf.data.Dataset], tf.data.Dataset]):
    py_typecheck.check_type(underlying_client_data, ClientData)
    py_typecheck.check_callable(preprocess_fn)
    self._underlying_client_data = underlying_client_data
    self._preprocess_fn = preprocess_fn
    example_dataset = self._preprocess_fn(
        self._underlying_client_data.create_tf_dataset_for_client(
            next(iter(underlying_client_data.client_ids))))
    self._element_type_structure = example_dataset.element_spec
    self._dataset_computation = None

  @property
  def client_ids(self):
    return self._underlying_client_data.client_ids

  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    return self._preprocess_fn(
        self._underlying_client_data.create_tf_dataset_for_client(client_id))

  @property
  def dataset_computation(self):
    if self._dataset_computation is None:

      @computations.tf_computation(tf.string)
      def dataset_comp(client_id):
        return self._preprocess_fn(
            self._underlying_client_data.dataset_computation(client_id))

      self._dataset_computation = dataset_comp

    return self._dataset_computation

  @property
  def element_type_structure(self):
    return self._element_type_structure


class ConcreteClientData(ClientData):
  """A generic `ClientData` object.

  This is a simple implementation of client_data, where Datasets are specified
  as a function from client_id to Dataset.

  The `ConcreteClientData.preprocess` classmethod is provided as a utility
  used to wrap another `ClientData` with an additional preprocessing function.
  """

  def __init__(
      self,
      client_ids: Iterable[str],
      create_tf_dataset_for_client_fn: Callable[[str], tf.data.Dataset],
  ):
    """Arguments correspond to the corresponding members of `ClientData`.

    Args:
      client_ids: A non-empty list of string client_ids.
      create_tf_dataset_for_client_fn: A function that takes a client_id from
        the above list, and returns a `tf.data.Dataset`. If this function is
        additionally a `tff.Computation`, the constructed `ConcreteClientData`
        will expose a `dataset_computation` attribute which can be used for
        high-performance distributed simulations.
    """
    py_typecheck.check_type(client_ids, collections.abc.Iterable)
    py_typecheck.check_callable(create_tf_dataset_for_client_fn)

    if not client_ids:
      raise ValueError('At least one client_id is required.')

    self._client_ids = list(client_ids)
    self._create_tf_dataset_for_client_fn = create_tf_dataset_for_client_fn

    if isinstance(self._create_tf_dataset_for_client_fn,
                  computation_base.Computation):
      self._dataset_computation = self._create_tf_dataset_for_client_fn
    else:
      self._dataset_computation = None

    example_dataset = create_tf_dataset_for_client_fn(next(iter(client_ids)))
    self._element_type_structure = example_dataset.element_spec

  @property
  def client_ids(self) -> List[str]:
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    return self._create_tf_dataset_for_client_fn(client_id)

  @property
  def element_type_structure(self):
    return self._element_type_structure

  @property
  def dataset_computation(self):
    if self._dataset_computation is not None:
      return self._dataset_computation
    raise NotImplementedError
