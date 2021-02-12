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

import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.simulation import client_data


def isnamedtuple(example_structure):
  return isinstance(example_structure, tuple) and hasattr(
      example_structure, '_fields')


class FromTensorSlicesClientData(client_data.ClientData):
  """ClientData based on `tf.data.Dataset.from_tensor_slices`.

  Useful for constructing toy federated datasets for testing purposes.

  Using this ClientData for large datasets is *not* recommended, as all the data
  gets directly baked into the TensorFlow graph (which is memory intensive).
  """

  def __init__(self, tensor_slices_dict):
    """Constructs the object from a dictionary of client data.

    Note: All clients are required to have non-empty data.

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
    py_typecheck.check_type(tensor_slices_dict, dict)
    structures = list(tensor_slices_dict.values())
    example_structure = structures[0]
    # The structure should not be a namedtuple.
    if isnamedtuple(example_structure):
      raise TypeError(
          'The tensor slices in the client data dictionary must be strictly '
          'lists, tuples (not namedtuples), or dictionaries, but the provided '
          'data was a namedtuple. Suggest using ._asdict() to convert the '
          'namedtuples to dictionaries to work with this class.')
    # The structures must be lists, tuples, or dictionaries.
    py_typecheck.check_type(example_structure, (list, tuple, dict))
    # The structures must all be the same.
    for structure in structures:
      py_typecheck.check_type(structure, type(example_structure))

    def check_types_match(tensors, expected_dtypes):
      for tensor, expected_dtype in zip(tensors, expected_dtypes):
        if tensor.dtype is not expected_dtype:
          raise TypeError(
              'The input tensor_slices_dict must have entries that convert '
              'to identical TensorFlow data types, but found two different '
              'entries with values of %s and %s' %
              (expected_dtype, tensor.dtype))

    if isinstance(example_structure, dict):

      # This is needed to keep data that was loosely specified in a list or
      # tuple together in a common object (a tf.Tensor or tf.RaggedTensor), for
      # correct flattening.
      def convert_any_lists_of_strings_or_bytes_to_ragged_tensors(structure):
        for key, entries in structure.items():
          if isinstance(entries, (list, tuple)):
            if isinstance(entries[0], (bytes, str)):
              structure[key] = tf.ragged.constant(entries)
            else:
              structure[key] = tf.constant(entries)

      convert_any_lists_of_strings_or_bytes_to_ragged_tensors(example_structure)
      self._example_structure = example_structure
      self._dtypes = [
          tf.constant(x).dtype for x in tf.nest.flatten(example_structure)
      ]

      for s in structures:
        convert_any_lists_of_strings_or_bytes_to_ragged_tensors(s)
        check_types_match([tf.constant(x) for x in tf.nest.flatten(s)],
                          self._dtypes)
    else:
      self._example_structure = None
      self._dtypes = [tf.constant(example_structure).dtype]
      for s in structures:
        check_types_match([tf.constant(s)], self._dtypes)

    self._tensor_slices_dict = tensor_slices_dict
    example_dataset = self.create_tf_dataset_for_client(self.client_ids[0])
    self._element_type_structure = example_dataset.element_spec

    self._dataset_computation = None

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
  def element_type_structure(self):
    return self._element_type_structure

  @property
  def dataset_computation(self):
    if self._dataset_computation is None:

      @computations.tf_computation(tf.string)
      @tf.function
      def construct_dataset(client_id):
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
        keys = tf.constant(list(self._tensor_slices_dict.keys()))

        client_id_valid = tf.math.reduce_any(tf.math.equal(client_id, keys))
        tf.Assert(client_id_valid, ['No data found for client ', client_id])

        # An alternative strategy to the one below might be to have a
        # tf.constant ordered in the same order as the keys (client ids),
        # containing the data entries for each client. This proved complicated
        # to get working in practice, and so the `tf.lookup.StaticHashTable` was
        # opted for instead. If at some point problems are encountered with the
        # hashtables (such as memory leaks), this might be revisited.

        # Serialize and flatten (if necessary) the contents of the input dict.
        serialized_flat_structures = [[] for _ in range(len(self._dtypes))]
        for s in self._tensor_slices_dict.values():
          flat_structure = tf.nest.flatten(s) if isinstance(s, dict) else [s]
          for i, x in enumerate(flat_structure):
            serialized_flat_structures[i].append(
                tf.io.serialize_tensor(tf.constant(x)))

        # Put the data into TF hash tables. There is one hash table for each
        # field in the client data.
        hash_tables = []
        for i in range(len(self._dtypes)):
          hash_tables.append(
              tf.lookup.StaticHashTable(
                  initializer=tf.lookup.KeyValueTensorInitializer(
                      keys=keys, values=serialized_flat_structures[i]),
                  # Note: This default_value should never be encountered, as
                  # we do a check above that the client_id is in the set of
                  # keys.
                  default_value='unknown_value'))

        # Recover data relating to the given client_id from the hash table.
        tensor_slices_list = [
            tf.io.parse_tensor(table.lookup(client_id), out_type=dtype)
            for table, dtype in zip(hash_tables, self._dtypes)
        ]

        # If necessary, unflatten the structures back into desired structure.
        if self._example_structure is not None:
          tensor_slices = tf.nest.pack_sequence_as(self._example_structure,
                                                   tensor_slices_list)
          for k, v in self._example_structure.items():
            tensor_slices[k] = tf.stack(tensor_slices[k])
            tensor_slices[k].set_shape([None] + list(v.shape)[1:])
        else:
          tensor_slices = tensor_slices_list[0]

        return tf.data.Dataset.from_tensor_slices(tensor_slices)

      self._dataset_computation = construct_dataset

    return self._dataset_computation
