# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""A simple ClientData based on in-memory tensor slices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation import client_data


class FromTensorSlicesClientData(client_data.ClientData):
  """ClientData based on `tf.data.Dataset.from_tensor_slices`."""

  def __init__(self, tensor_slices_dict):
    """Constructs the object from a dictionary of client data.

    NOTE: All clients are required to have non-empty data.

    Args:
      tensor_slices_dict: A dictionary keyed by client_id, where values are
        structures suitable for passing to `tf.data.Dataset.from_tensor_slices`.

    Raises:
      ValueError: If a client with no data is found.
    """
    py_typecheck.check_type(tensor_slices_dict, dict)
    self._tensor_slices_dict = tensor_slices_dict
    example_dataset = self.create_tf_dataset_for_client(self.client_ids[0])
    self._output_types = tf.compat.v1.data.get_output_types(example_dataset)
    self._output_shapes = tf.compat.v1.data.get_output_shapes(example_dataset)

  @property
  def client_ids(self):
    return list(self._tensor_slices_dict.keys())

  def create_tf_dataset_for_client(self, client_id):
    tensor_slices = self._tensor_slices_dict[client_id]
    if tensor_slices:
      return tf.data.Dataset.from_tensor_slices(tensor_slices)
    else:
      raise ValueError('No data found for client {}'.format(client_id))

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes
