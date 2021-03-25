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

import collections
import os.path
from typing import Callable, Mapping

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class FilePerUserClientData(client_data.ClientData):
  """A `tff.simulation.datasets.ClientData` that maps a set of files to a dataset.

  This mapping is restricted to one file per user.
  """

  def __init__(self, client_ids_to_files: Mapping[str, str],
               dataset_fn: Callable[[str], tf.data.Dataset]):
    """Constructs a `tff.simulation.datasets.ClientData` object.

    Args:
      client_ids_to_files: A mapping from string client IDs to filepaths
        containing the user's data.
      dataset_fn: A factory function that takes a filepath (must accept both
        strings and tensors) and returns a `tf.data.Dataset` corresponding to
        this path.
    """
    py_typecheck.check_type(client_ids_to_files, collections.abc.Mapping)
    if not client_ids_to_files:
      raise ValueError('`client_ids` must have at least one client ID')
    py_typecheck.check_callable(dataset_fn)
    self._client_ids = sorted(client_ids_to_files.keys())

    def create_dataset_for_filename_fn(client_id):
      return dataset_fn(client_ids_to_files[client_id])

    @computations.tf_computation(tf.string)
    def dataset_computation(client_id):
      client_ids_to_path = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(
              list(client_ids_to_files.keys()),
              list(client_ids_to_files.values())), '')
      client_path = client_ids_to_path.lookup(client_id)
      return dataset_fn(client_path)

    self._create_tf_dataset_fn = create_dataset_for_filename_fn
    self._dataset_computation = dataset_computation

    g = tf.Graph()
    with g.as_default():
      tf_dataset = self._create_tf_dataset_fn(self._client_ids[0])
      self._element_type_structure = tf_dataset.element_spec

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    tf_dataset = self._create_tf_dataset_fn(client_id)
    tensor_utils.check_nested_equal(tf_dataset.element_spec,
                                    self._element_type_structure)
    return tf_dataset

  @property
  def element_type_structure(self):
    return self._element_type_structure

  @classmethod
  def create_from_dir(cls, path, create_tf_dataset_fn=tf.data.TFRecordDataset):
    """Builds a `tff.simulation.datasets.FilePerUserClientData`.

    Iterates over all files in `path`, using the filename as the client ID. Does
    not recursively search `path`.

    Args:
      path: A directory path to search for per-client files.
      create_tf_dataset_fn: A callable that creates a `tf.data.Datasaet` object
        for a given file in the directory specified in `path`.

    Returns:
      A `tff.simulation.datasets.FilePerUserClientData` object.
    """
    client_ids_to_paths_dict = {
        filename: os.path.join(path, filename)
        for filename in tf.io.gfile.listdir(path)
    }

    return FilePerUserClientData(client_ids_to_paths_dict, create_tf_dataset_fn)

  @property
  def dataset_computation(self):
    return self._dataset_computation
