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
"""Implementation of HDF5 backed ClientData."""

import collections

import h5py
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.tensorflow_libs import tensor_utils


class HDF5ClientData(client_data.ClientData):
  """A `tff.simulation.ClientData` backed by an HDF5 file.

  This class expects that the HDF5 file has a top-level group `examples` which
  contains further subgroups, one per user, named by the user ID. Further, the
  users must have identical keys.

  The `tf.data.Dataset` returned by
  `HDF5ClientData.create_tf_dataset_for_client(client_id)` yields ordered dicts
  from zipping all datasets that were found at `/data/client_id` group, in a
  similar fashion to `tf.data.Dataset.from_tensor_slices()`.
  """

  _EXAMPLES_GROUP = "examples"

  def __init__(self, hdf5_filepath):
    """Constructs a `tff.simulation.ClientData` object.

    Args:
      hdf5_filepath: String path to the hdf5 file.
    """
    py_typecheck.check_type(hdf5_filepath, str)
    self._filepath = hdf5_filepath

    self._h5_file = h5py.File(self._filepath, "r")
    self._client_ids = sorted(
        list(self._h5_file[HDF5ClientData._EXAMPLES_GROUP].keys()))
    self._client_keys = self._h5_file[HDF5ClientData._EXAMPLES_GROUP][self._client_ids[0]].keys()

    # Get the types and shapes from the first client. We do it once during
    # initialization so we can get both properties in one go.
    example_tf_dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict((name, ds[()]) for name, ds in sorted(
          self._h5_file[HDF5ClientData._EXAMPLES_GROUP][self._client_ids[0]].items())))
    self._element_type_structure = example_tf_dataset.element_spec

    @computations.tf_computation(tf.string)
    def dataset_computation(client_id):
      client_datasets = collections.OrderedDict()
      for key in self._client_keys:
        client_datasets[key] = tfio.IODataset.from_hdf5(
            filename=self._filepath,
            dataset=tf.strings.join(separator="/", inputs=("", HDF5ClientData._EXAMPLES_GROUP,  client_id,  key)),
            spec=self._element_type_structure[key])
      return tf.data.Dataset.zip(client_datasets)

    self._dataset_computation = dataset_computation

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id):
    if client_id not in self.client_ids:
      raise ValueError(
          "ID [{i}] is not a client in this ClientData. See "
          "property `client_ids` for the list of valid ids.".format(
              i=client_id))
    tf_dataset = self._dataset_computation(client_id)
    return tf_dataset

  @property
  def element_type_structure(self):
    return self._element_type_structure

  @property
  def dataset_computation(self):
    return self._dataset_computation
