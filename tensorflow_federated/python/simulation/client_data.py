# Lint as: python3
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
"""Library methods for working with centralized data used in simulation.

N.B. Federated Learning does not use client IDs or perform any tracking of
clients. However in simulation experiments using centralized test data the
experimenter may select specific clients to be processed per round. The concept
of a client ID is only available at the preprocessing stage when preparing input
data for the simulation and is not part of the TensorFlow Federated core APIs.
"""

import abc

from absl import logging
import numpy as np
import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck


@six.add_metaclass(abc.ABCMeta)
class ClientData(object):
  """Object to hold a dataset and a mapping of clients to examples."""

  @abc.abstractproperty
  def client_ids(self):
    """The list of string identifiers for clients in this dataset."""
    pass

  @abc.abstractmethod
  def create_tf_dataset_for_client(self, client_id):
    """Creates a new `tf.data.Dataset` containing the client training examples.

    Args:
      client_id: The string client_id for the desired client.

    Returns:
      A `tf.data.Dataset` object.
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

  def create_tf_dataset_from_all_clients(self, seed=None):
    """Creates a new `tf.data.Dataset` containing _all_ client examples.

    NOTE: the returned `tf.data.Dataset` is not serializable and runnable on
    other devices, as it uses `tf.py_func` internally.

    Currently, the implementation produces a dataset that contains
    all examples from a single client in order, and so generally additional
    shuffling should be performed.

    Args:
      seed: Optional, a seed to determine the order in which clients are
        processed in the joined dataset.

    Returns:
      A `tf.data.Dataset` object.
    """

    # NOTE: simply calling Dataset.concatenate() will result in too deep
    # recursion depth.
    # NOTE: Tests are via the simple concrete from_tensor_slices_client_data.
    def _generator():
      client_ids = list(self.client_ids)
      np.random.RandomState(seed=seed).shuffle(client_ids)
      for client_id in client_ids:
        for example in self.create_tf_dataset_for_client(client_id):
          yield example

    types = tf.nest.map_structure(lambda t: t.dtype,
                                  self.element_type_structure)
    shapes = tf.nest.map_structure(lambda t: t.shape,
                                   self.element_type_structure)

    return tf.data.Dataset.from_generator(_generator, types, shapes)

  def preprocess(self, preprocess_fn):
    """Applies `preprocess_fn` to each client's data."""
    py_typecheck.check_callable(preprocess_fn)

    def get_dataset(client_id):
      return preprocess_fn(self.create_tf_dataset_for_client(client_id))

    return ConcreteClientData(self.client_ids, get_dataset)

  @classmethod
  def from_clients_and_fn(cls, client_ids, create_tf_dataset_for_client_fn):
    """Constructs a `ClientData` based on the given function.

    Args:
      client_ids: A non-empty list of client_ids which are valid inputs to the
        create_tf_dataset_for_client_fn.
      create_tf_dataset_for_client_fn: A function that takes a client_id from
        the above list, and returns a `tf.data.Dataset`.

    Returns:
      A `ClientData`.
    """
    return ConcreteClientData(client_ids, create_tf_dataset_for_client_fn)

  @classmethod
  def train_test_client_split(cls, client_data, num_test_clients):
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
    np.random.shuffle(train_client_ids)
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
        _ = next(iter(dataset))
      except StopIteration:
        logging.warning('Client %s had no data, skipping.', client_id)
        clients_with_insufficient_batches.append(client_id)
        continue

      test_client_ids.append(client_id)

    # Invariant for successful exit of the above loop:
    assert len(test_client_ids) == num_test_clients

    def from_ids(client_ids):
      return cls.from_clients_and_fn(client_ids,
                                     client_data.create_tf_dataset_for_client)

    return (from_ids(train_client_ids + clients_with_insufficient_batches),
            from_ids(test_client_ids))


class ConcreteClientData(ClientData):
  """A generic `ClientData` object.

  This is a simple implementation of client_data, where Datasets are specified
  as a function from client_id to Dataset.

  The `ConcreteClientData.preprocess` classmethod is provided as a utility
  used to wrap another `ClientData` with an additional preprocessing function.
  """

  def __init__(self, client_ids, create_tf_dataset_for_client_fn):
    """Arguments correspond to the corresponding members of `ClientData`.

    Args:
      client_ids: A non-empty list of client_ids.
      create_tf_dataset_for_client_fn: A function that takes a client_id from
        the above list, and returns a `tf.data.Dataset`.
    """
    py_typecheck.check_type(client_ids, list)
    py_typecheck.check_callable(create_tf_dataset_for_client_fn)
    if not client_ids:
      raise ValueError('At least one client_id is required.')

    self._client_ids = client_ids
    self._create_tf_dataset_for_client_fn = create_tf_dataset_for_client_fn

    example_dataset = create_tf_dataset_for_client_fn(client_ids[0])
    self._element_type_structure = tf.data.experimental.get_structure(
        example_dataset)

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    return self._create_tf_dataset_for_client_fn(client_id)

  @property
  def element_type_structure(self):
    return self._element_type_structure
