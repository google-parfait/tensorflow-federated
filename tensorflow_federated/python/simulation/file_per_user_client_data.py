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
"""Implementations of the ClientData abstract base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class FilePerUserClientData(client_data.ClientData):
  """A `tf.simulation.ClientData` that maps a set of files to a dataset.

  This mapping is restricted to one file per user.
  """

  def __init__(self, client_ids, create_tf_dataset_fn):
    """Constructs a `tf.simulation.ClientData` object.

    Args:
      client_ids: A list of `client_id`s.
      create_tf_dataset_fn: A callable that takes a `client_id` and returns a
        `tf.data.Dataset` object.
    """
    py_typecheck.check_type(client_ids, list)
    if not client_ids:
      raise ValueError('`client_ids` must have at least one client ID')
    py_typecheck.check_callable(create_tf_dataset_fn)
    self._client_ids = sorted(client_ids)
    self._create_tf_dataset_fn = create_tf_dataset_fn

    g = tf.Graph()
    with g.as_default():
      tf_dataset = self._create_tf_dataset_fn(self._client_ids[0])
      self._output_types = tf_dataset.output_types
      self._output_shapes = tf_dataset.output_shapes

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    tf_dataset = self._create_tf_dataset_fn(client_id)
    tensor_utils.check_nested_equal(tf_dataset.output_types, self._output_types)
    tensor_utils.check_nested_equal(tf_dataset.output_shapes,
                                    self._output_shapes)
    return tf_dataset

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes

  @classmethod
  def create_from_dir(cls, path, create_tf_dataset_fn=tf.data.TFRecordDataset):
    """Builds a `tff.simulation.FilePerUserClientData`.

    Iterates over all files in `path`, using the filename as the client ID. Does
    not recursively search `path`.

    Args:
      path: A directory path to search for per-client files.
      create_tf_dataset_fn: A callable that creates a `tf.data.Datasaet` object
        for a given file in the directory specified in `path`.

    Returns:
      A `tff.simulation.FilePerUserClientData` object.
    """
    client_ids_to_paths_dict = {
        filename: os.path.join(path, filename) for filename in os.listdir(path)
    }

    def create_dataset_for_filename_fn(client_id):
      return create_tf_dataset_fn(client_ids_to_paths_dict[client_id])

    return FilePerUserClientData(
        list(client_ids_to_paths_dict.keys()), create_dataset_for_filename_fn)
