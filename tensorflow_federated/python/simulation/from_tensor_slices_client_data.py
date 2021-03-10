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

import warnings

from tensorflow_federated.python.simulation.datasets import from_tensor_slices_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.


class FromTensorSlicesClientData(
    from_tensor_slices_client_data.FromTensorSlicesClientData):
  """ClientData based on `tf.data.Dataset.from_tensor_slices`.

  Useful for constructing toy federated datasets for testing purposes.

  Using this ClientData for large datasets is *not* recommended, as all the data
  gets directly baked into the TensorFlow graph (which is memory intensive).

  WARNING: this class is deprecated and is slated for removal in April 2021.
  Please use `tff.simulation.datasets.FromTensorSlicesClientData` instead.
  """

  def __init__(self, tensor_slices_dict):
    """Constructs the object from a dictionary of client data.

    Note: All clients are required to have non-empty data.

    WARNING: this class is deprecated and is slated for removal in April 2021.
    Please use `tff.simulation.datasets.FromTensorSlicesClientData` instead.

    Args:
      tensor_slices_dict: A dictionary keyed by client_id, where values are
        lists, tuples, or dicts for passing to
        `tf.data.Dataset.from_tensor_slices`. Note that namedtuples and attrs
        classes are not explicitly supported, but a user can convert their data
        from those formats to a dict, and then use this class.

    Raises:
      ValueError: If a client with no data is found.
      TypeError: If `tensor_slices_dict` is not a dictionary, or its value
        structures are namedtuples, or its value structures are not either
        strictly lists, strictly (standard, non-named) tuples, or strictly
        dictionaries.
      TypeError: If flattened values in tensor_slices_dict convert to different
        TensorFlow data types.
    """
    warnings.warn(
        'tff.simulation.FromTensorSlicesClientData is deprecated and slated for '
        'removal in April 2021. Please use '
        'tff.simulation.datasets.FromTensorSlicesClientData instead.',
        DeprecationWarning)
    from_tensor_slices_client_data.FromTensorSlicesClientData.__init__(
        self, tensor_slices_dict)
