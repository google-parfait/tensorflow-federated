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

from typing import Callable, Mapping
import warnings

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import file_per_user_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.


class FilePerUserClientData(file_per_user_client_data.FilePerUserClientData):
  """A `tff.simulation.datasets.ClientData` that maps files to a dataset.

  This mapping is restricted to one file per user.

  WARNING: this class is deprecated and is slated for removal in April 2021.
  Please use `tff.simulation.datasets.FilePerUserClientData` instead.
  """

  def __init__(self, client_ids_to_files: Mapping[str, str],
               dataset_fn: Callable[[str], tf.data.Dataset]):
    """Constructs a `tff.simulation.datasets.ClientData` object.

    WARNING: this class is deprecated and is slated for removal in April 2021.
    Please use `tff.simulation.datasets.FilePerUserClientData` instead.

    Args:
      client_ids_to_files: A mapping from string client IDs to filepaths
        containing the user's data.
      dataset_fn: A factory function that takes a filepath (must accept both
        strings and tensors) and returns a `tf.data.Dataset` corresponding to
        this path.
    """
    warnings.warn(
        'tff.simulation.FilePerUserClientData is deprecated and slated for '
        'removal in April 2021. Please use '
        'tff.simulation.datasets.FilePerUserClientData instead.',
        DeprecationWarning)
    file_per_user_client_data.FilePerUserClientData.__init__(
        self, client_ids_to_files, dataset_fn)
