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

  def create_tf_dataset_from_all_clients(self, seed=None):
    """Creates a new `tf.data.Dataset` containing _all_ client examples.

    NOTE: the returned `tf.data.Dataset` is not serializable and runnable on
    other devices, as it uses `tf.py_func` internally.

    Currently, the implementation produces a dataset that contains
    all examples from a single client in order, and so generally additional
    shuffling should be performed.

    Args:
      seed: Optional, a seed to determine the order in which clients
        are processed in the joined dataset.

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

  @abc.abstractproperty
  def output_types(self):
    """Returns the type of each component of an element of the client datasets.

    Any `tf.data.Dataset` constructed by this class is expected have matching
    `tf.data.Dataset.output_types` properties.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of the client datasets.
    """
    pass

  @abc.abstractproperty
  def output_shapes(self):
    """Returns the shape of each component of an element of the client datasets.

    Any `tf.data.Dataset` constructed by this class is expected to have matching
    `tf.data.Dataset.output_shapes` properties.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of the client datasets.
    """
    pass
