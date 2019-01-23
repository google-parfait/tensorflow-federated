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

import six


@six.add_metaclass(abc.ABCMeta)
class ClientData(object):
  """Object to hold a dataset and a mapping of clients to examples."""

  @abc.abstractproperty
  def client_ids(self):
    """The list of identifiers for clients in this dataset.

    A client identifier can be any type understood by the
    `tff.simulation.ClientData.create_tf_dataset_for_client` method, determined
    by the implementation.
    """
    pass

  @abc.abstractmethod
  def create_tf_dataset_for_client(self, client_id):
    """Creates a new `tf.data.Dataset` containing the client training examples.

    Args:
      client_id: The identifier for the desired client.

    Returns:
      A `tf.data.Dataset` object.
    """
    pass

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
