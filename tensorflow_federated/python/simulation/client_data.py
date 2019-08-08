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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
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
  def output_types(self):
    """Returns the type of each component of an element of the client datasets.

    Any `tf.data.Dataset` constructed by this class is expected have matching
    `output_types` properties when accessed via
    `tf.compat.v1.data.get_output_types(dataset)`.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of the client datasets.
    """
    pass

  @abc.abstractproperty
  def output_shapes(self):
    """Returns the shape of each component of an element of the client datasets.

    Any `tf.data.Dataset` constructed by this class is expected to have matching
    `output_shapes` properties when accessed via
    `tf.compat.v1.data.get_output_shapes(dataset)`.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of the client datasets.
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

    return tf.data.Dataset.from_generator(_generator, self.output_types,
                                          self.output_shapes)

  def preprocess(self, preprocess_fn):
    """Applies `preprocess_fn` to each client's data."""
    py_typecheck.check_callable(preprocess_fn)

    def get_dataset(client_id):
      return preprocess_fn(self.create_tf_dataset_for_client(client_id))

    return ConcreteClientData(self.client_ids, get_dataset)


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
    self._output_types = example_dataset.output_types
    self._output_shapes = example_dataset.output_shapes

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    return self._create_tf_dataset_for_client_fn(client_id)

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes
