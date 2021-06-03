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
"""Implementations of `ClientData` backed by a file system."""

import collections
import os.path
from typing import Callable, Mapping

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class FilePerUserClientData(client_data.SerializableClientData):
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

    # Creates a dataset in a manner that can be serialized by TF.
    def serializable_dataset_fn(client_id: str) -> tf.data.Dataset:
      client_ids_to_path = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(
              list(client_ids_to_files.keys()),
              list(client_ids_to_files.values())), '')
      client_path = client_ids_to_path.lookup(client_id)
      return dataset_fn(client_path)

    self._serializable_dataset_fn = serializable_dataset_fn

    tf_dataset = serializable_dataset_fn(tf.constant(self._client_ids[0]))
    self._element_type_structure = tf_dataset.element_spec

  @property
  def serializable_dataset_fn(self):
    """Creates a `tf.data.Dataset` for a client in a TF-serializable manner."""
    return self._serializable_dataset_fn

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
    """Creates a new `tf.data.Dataset` containing the client training examples.

    This function will create a dataset for a given client if `client_id` is
    contained in the `client_ids` property of the `FilePerUserClientData`.
    Unlike `self.serializable_dataset_fn`, this method is not serializable.

    Args:
      client_id: The string identifier for the desired client.

    Returns:
      A `tf.data.Dataset` object.
    """
    if client_id not in self.client_ids:
      raise ValueError(
          'ID [{i}] is not a client in this ClientData. See '
          'property `client_ids` for the list of valid ids.'.format(
              i=client_id))

    client_dataset = self.serializable_dataset_fn(tf.constant(client_id))
    tensor_utils.check_nested_equal(client_dataset.element_spec,
                                    self._element_type_structure)
    return client_dataset

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
