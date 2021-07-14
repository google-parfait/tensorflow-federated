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
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations


class IncompatiblePreprocessFnError(TypeError):

  def __init__(self):
    message = (
        'The preprocess_fn must not be a tff.Computation. Please use a python'
        ' callable or tf.function instead. This restriction is because '
        '`tf.data.Dataset.map` wraps preprocessing functions with a '
        '`tf.function` decorator, which cannot call to a `tff.Computation`.')
    super().__init__(message)


def is_nonnegative_32_bit_int(x: Any) -> bool:
  if isinstance(x, int) and 0 <= x and x < 2**32:
    return True
  return False


def check_numpy_random_seed(seed: Any) -> None:
  """Determines if an input is a valid random seed for `np.random.RandomState`.

  Specifically, this method returns `True` if the input is a nonnegative 32-bit
  integer, a sequence of such integers, or `None`.

  Args:
    seed: The argument that we wish to determine is a valid random seed.

  Raises:
    InvalidRandomSeedError: If the input argument does not meet any of the
      types above.
  """
  if seed is None:
    return
  elif is_nonnegative_32_bit_int(seed):
    return
  elif isinstance(seed, Sequence) and all(
      [is_nonnegative_32_bit_int(x) for x in seed]):
    return
  raise InvalidRandomSeedError(type(seed))


class InvalidRandomSeedError(TypeError):

  def __init__(self, seed_type):
    message = (
        'The seed must be a nonnegative 32-bit integer, a sequence of such '
        'integers, or None. Found {} instead.'.format(seed_type))
    super().__init__(message)


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

  @abc.abstractproperty
  def element_type_structure(self):
    """The element type information of the client datasets.

    Returns:
      A nested structure of `tf.TensorSpec` objects defining the type of the
    elements returned by datasets in this `ClientData` object.
    """
    pass

  def datasets(
      self,
      limit_count: Optional[int] = None,
      seed: Optional[Union[int, Sequence[int]]] = None
  ) -> Iterable[tf.data.Dataset]:
    """Yields the `tf.data.Dataset` for each client in random order.

    This function is intended for use building a static array of client data
    to be provided to the top-level federated computation.

    Args:
      limit_count: Optional, a maximum number of datasets to return.
      seed: Optional, a seed to determine the order in which clients are
        processed in the joined dataset. The seed can be any nonnegative 32-bit
        integer, an array of such integers, or `None`.
    """
    check_numpy_random_seed(seed)
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

  def create_tf_dataset_from_all_clients(
      self,
      seed: Optional[Union[int, Sequence[int]]] = None) -> tf.data.Dataset:
    """Creates a new `tf.data.Dataset` containing _all_ client examples.

    This function is intended for use training centralized, non-distributed
    models (num_clients=1). This can be useful as a point of comparison
    against federated models.

    Currently, the implementation produces a dataset that contains
    all examples from a single client in order, and so generally additional
    shuffling should be performed.

    Args:
      seed: Optional, a seed to determine the order in which clients are
        processed in the joined dataset. The seed can be any nonnegative 32-bit
        integer, an array of such integers, or `None`.

    Returns:
      A `tf.data.Dataset` object.
    """
    check_numpy_random_seed(seed)
    client_ids = self.client_ids.copy()
    np.random.RandomState(seed=seed).shuffle(client_ids)
    nested_dataset = tf.data.Dataset.from_tensor_slices(client_ids)
    # We apply serializable_dataset_fn here to avoid loading all client datasets
    # in memory, which is slow. Note that tf.data.Dataset.map implicitly wraps
    # the input mapping in a tf.function.
    example_dataset = nested_dataset.flat_map(self.serializable_dataset_fn)
    return example_dataset

  def preprocess(
      self, preprocess_fn: Callable[[tf.data.Dataset],
                                    tf.data.Dataset]) -> 'ClientData':
    """Applies `preprocess_fn` to each client's data.

    Args:
      preprocess_fn: A callable accepting a `tf.data.Dataset` and returning a
        preprocessed `tf.data.Dataset`. This function must be traceable by TF.

    Returns:
      A `tff.simulation.datasets.ClientData`.

    Raises:
      IncompatiblePreprocessFnError: If `preprocess_fn` is a `tff.Computation`.
    """
    py_typecheck.check_callable(preprocess_fn)
    if isinstance(preprocess_fn, computation_base.Computation):
      raise IncompatiblePreprocessFnError()
    return PreprocessClientData(self, preprocess_fn)

  @classmethod
  def from_clients_and_tf_fn(
      cls,
      client_ids: Iterable[str],
      serializable_dataset_fn: Callable[[str], tf.data.Dataset],
  ) -> 'ClientData':
    """Constructs a `ClientData` based on the given function.

    Args:
      client_ids: A non-empty list of strings to use as input to
        `create_dataset_fn`.
      serializable_dataset_fn: A function that takes a client_id from the above
        list, and returns a `tf.data.Dataset`. This function must be
        serializable and usable within the context of a `tf.function` and
        `tff.Computation`.

    Returns:
      A `ClientData` object.
    """
    return ConcreteClientData(client_ids, serializable_dataset_fn)

  @classmethod
  def train_test_client_split(
      cls,
      client_data: 'ClientData',
      num_test_clients: int,
      seed: Optional[Union[int, Sequence[int]]] = None
  ) -> Tuple['ClientData', 'ClientData']:
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
      seed: Optional seed to fix shuffling of clients before splitting. The seed
        can be any nonnegative 32-bit integer, an array of such integers, or
        `None`.

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

    check_numpy_random_seed(seed)
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

    def from_ids(client_ids: Iterable[str]) -> 'ClientData':
      return cls.from_clients_and_tf_fn(client_ids,
                                        client_data.serializable_dataset_fn)

    return (from_ids(train_client_ids + clients_with_insufficient_batches),
            from_ids(test_client_ids))


class PreprocessClientData(ClientData):
  """Applies a preprocessing function to every dataset it returns.

  This `ClientData` subclass delegates all other aspects of implementation to
  its underlying `ClientData` object, simply wiring in its `preprocess_fn`
  where necessary.
  """

  def __init__(  # pylint: disable=super-init-not-called
      self, underlying_client_data: ClientData,
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


class ConcreteClientData(ClientData):
  """A generic `ClientData` object.

  This is a simple implementation of client_data, where Datasets are specified
  as a function from client_id to Dataset.

  The `ConcreteClientData.preprocess` classmethod is provided as a utility
  used to wrap another `ClientData` with an additional preprocessing function.
  """

  def __init__(  # pylint: disable=super-init-not-called
      self,
      client_ids: Iterable[str],
      serializable_dataset_fn: Callable[[str], tf.data.Dataset],
  ):
    """Creates a `ClientData` from clients and a mapping function.

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
