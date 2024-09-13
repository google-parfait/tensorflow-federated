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

import copy

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation.datasets import client_data


def isnamedtuple(example_structure):
  return isinstance(example_structure, tuple) and hasattr(
      example_structure, '_fields'
  )


class TestClientData(client_data.ClientData):
  """A `tff.simulation.datasets.ClientData` intended for test purposes.

  The implementation is based on `tf.data.Dataset.from_tensor_slices.` This
  class is intended only for constructing toy federated datasets, especially
  to support simulation tests. Using this for large datasets is *not*
  recommended, as it requires putting all client data into the underlying
  TensorFlow graph (which is memory intensive).
  """

  def __init__(self, tensor_slices_dict):
    """Constructs the object from a dictionary of client data.

    Note: All clients are required to have non-empty data.

    Args:
      tensor_slices_dict: A dictionary keyed by client_id, where values are
        lists, tuples, or dicts for passing to
        `tf.data.Dataset.from_tensor_slices`. Note that namedtuples and attrs
        classes are not explicitly supported, but a user can convert their data
        from those formats to a dict, and then use this class. The leaves of
        this dictionary must not be `tf.Tensor`s, in order to avoid putting
        eager tensors into graphs.

    Raises:
      ValueError: If a client with no data is found.
      TypeError: If `tensor_slices_dict` is not a dictionary, or its value
        structures are namedtuples, or its value structures are not either
        strictly lists, strictly (standard, non-named) tuples, or strictly
        dictionaries.
      TypeError: If any leaf of `tensor_slices_dict` is a `tf.Tensor`.
    """
    py_typecheck.check_type(tensor_slices_dict, dict)
    tensor_slices_dict = copy.deepcopy(tensor_slices_dict)
    structures = list(tensor_slices_dict.values())
    example_structure = structures[0]
    # The structure should not be a namedtuple.
    if isnamedtuple(example_structure):
      raise TypeError(
          'The tensor slices in the client data dictionary must be strictly '
          'lists, tuples (not namedtuples), or dictionaries, but the provided '
          'data was a namedtuple. Suggest using ._asdict() to convert the '
          'namedtuples to dictionaries to work with this class.'
      )
    # The structures must be lists, tuples, or dictionaries.
    py_typecheck.check_type(example_structure, (list, tuple, dict))
    # The structures must all be the same.
    for structure in structures:
      py_typecheck.check_type(structure, type(example_structure))

    for leaf in tf.nest.flatten(tensor_slices_dict):
      if tf.is_tensor(leaf):
        raise TypeError(
            'Tensor slices must not be TF tensors. Consider converting them to'
            ' numpy values instead.'
        )

    self._tensor_slices_dict = tensor_slices_dict
    example_dataset = self.create_tf_dataset_for_client(self.client_ids[0])
    self._element_type_structure = example_dataset.element_spec

  @property
  def client_ids(self):
    return list(self._tensor_slices_dict.keys())

  @tf.function
  def _create_dataset(self, client_id):
    """A tf.function taking id of a client and returning that client's data.

    This method serializes the data in `tensor_slices_dict` into the
    TensorFlow graph; since we don't know the client_id that will need
    looking up until graph execution time, we need to bake the entire
    dataset in. Consequently, this is not really a recommended pattern for
    heavy use, but it works for very simple, toy datasets, such as we use in
    testing.

    The order of operations is roughly:
    0 - Check that `client_id` is in the dataset.
    1 - Serialize the contents of `_tensor_slices_dict`.
    2 - Store the serialized data into hash tables, keyed by the client ids.
    3 - Do the hash tables lookups, to recover the data for `client_id`.
    4 - Profit (i.e, convert into tf.data.Dataset and return).

    All these steps are handled inside the tf.function; to reiterate, we
    need all the data in the graph, so that we can handle looking up any
    client's id at graph execution time.

    Previous experiences with using lookup tables inside tf.function have
    been mixed. This code may be more fragile (than other code in TFF), but
    as it's simply a test utility not intended for production, it doesn't
    have as strong an SLA.

    Args:
      client_id: The string identifier for particular client in the dataset.

    Returns:
      A tf.data.Dataset of `client_id`'s data.

    Raises:
      tf.errors.InvalidArgumentError: If no data can be found for the
        `client_id` provided (i.e., it's not in the set of clients).
    """
    keys = [x for x in self._tensor_slices_dict.keys()]
    client_id_valid = tf.math.reduce_any(tf.math.equal(client_id, keys))
    tf.Assert(client_id_valid, ['No data found for client ', client_id])
    datasets = [
        tf.data.Dataset.from_tensor_slices(x)
        for x in self._tensor_slices_dict.values()
    ]
    out_dataset = datasets[0]
    for k, d in zip(keys, datasets):
      if tf.math.equal(k, client_id):
        out_dataset = d
    return out_dataset

  @property
  def serializable_dataset_fn(self):
    return self._create_dataset

  def create_tf_dataset_for_client(self, client_id):
    tensor_slices = self._tensor_slices_dict[client_id]
    if tensor_slices:
      return tf.data.Dataset.from_tensor_slices(tensor_slices)
    else:
      raise ValueError('No data found for client {}'.format(client_id))

  @property
  def element_type_structure(self):
    return self._element_type_structure
