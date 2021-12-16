# Copyright 2021, The TensorFlow Federated Authors.
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
"""Utilities for constructing, serializing and parsing SQL-backed ClientData."""

import collections
import os
import tempfile
from typing import Callable, List, Mapping

from absl import logging

import sqlite3
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import sql_client_data


class ElementSpecCompatibilityError(TypeError):
  """Exception if the element_spec does not follow `Mapping[str, tf.TensorSpec]`."""
  pass


def _bytes_feature(tensor) -> tf.train.Feature:
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor.numpy()]))


def _float_feature(tensor) -> tf.train.Feature:
  """Returns a float_list from a float / double."""
  return tf.train.Feature(
      float_list=tf.train.FloatList(value=tensor.numpy().flatten()))


def _int64_feature(tensor) -> tf.train.Feature:
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=tensor.numpy().flatten()))


def _validate_element_spec(element_spec: Mapping[str, tf.TensorSpec]):
  """Validate the type of element_spec."""

  if not isinstance(element_spec, collections.abc.Mapping):
    raise ElementSpecCompatibilityError(
        f'{element_spec} has type {type(element_spec)}, but expected '
        f'`Mapping[str, tf.TensorSpec]`.')
  for key, tensor_spec in element_spec.items():
    if not isinstance(key, str):
      raise ElementSpecCompatibilityError(
          f'{key} has type {type(key)}, but expected str.')
    if not isinstance(tensor_spec, tf.TensorSpec):
      raise ElementSpecCompatibilityError(
          f'{tensor_spec} has type {type(tensor_spec)}, but expected '
          '`tf.TensorSpec`.')


def _build_serializer(
    element_spec: Mapping[str, tf.TensorSpec]
) -> Callable[[Mapping[str, tf.Tensor]], bytes]:
  """Build a serializer based on the element_spec of a dataset."""

  _validate_element_spec(element_spec)
  feature_funcs = collections.OrderedDict()
  for key, tensor_spec in element_spec.items():

    if tensor_spec.dtype is tf.string:
      feature_funcs[key] = _bytes_feature
    elif tensor_spec.dtype.is_floating:
      feature_funcs[key] = _float_feature
    elif tensor_spec.dtype.is_integer:
      feature_funcs[key] = _int64_feature
    else:
      raise ElementSpecCompatibilityError(
          f'Unsupported dtype {tensor_spec.dtype}.')

  def serializer(element: Mapping[str, tf.Tensor]) -> bytes:

    feature = {
        key: feature_funcs[key](tensor) for key, tensor in element.items()
    }

    return tf.train.Example(features=tf.train.Features(
        feature=feature)).SerializeToString()

  return serializer


def _build_parser(
    element_spec: Mapping[str, tf.TensorSpec]
) -> Callable[[bytes], Mapping[str, tf.Tensor]]:
  """Build a parser based on the element_spec of a dataset."""

  _validate_element_spec(element_spec)

  parse_specs = collections.OrderedDict()
  for key, tensor_spec in element_spec.items():
    if tensor_spec.dtype is tf.string:
      parser_dtype = tf.string
    elif tensor_spec.dtype.is_floating:
      parser_dtype = tf.float32
    elif tensor_spec.dtype.is_integer:
      parser_dtype = tf.int64
    else:
      raise ValueError(f'unsupported dtype {tensor_spec.dtype}')

    parse_specs[key] = tf.io.FixedLenFeature(
        shape=tensor_spec.shape, dtype=parser_dtype)

  def parser(tensor_proto: bytes) -> Mapping[str, tf.Tensor]:
    parsed_features = tf.io.parse_example(tensor_proto, parse_specs)

    result = collections.OrderedDict()

    for key, tensor_spec in element_spec.items():
      result[key] = tf.cast(parsed_features[key], tensor_spec.dtype)

    return result

  return parser


def save_to_sql_client_data(
    client_ids: List[str],
    dataset_fn: Callable[[str], tf.data.Dataset],
    database_filepath: str,
    allow_overwrite: bool = False,
) -> None:
  """Serialize a federated dataset into a SQL database compatible with `SqlClientData`.

  Note: All the clients must share the same dataset.element_spec of type
  `Mapping[str, TensorSpec]`.

  Args:
    client_ids: A list of string identifiers for clients in this dataset.
    dataset_fn: A callable that accepts a `str` as an argument and returns a
      `tf.data.Dataset` instance. Unlike `from_client_and_tf_dataset_fn`, it
      does not require a TF serializable dataset function.
    database_filepath: A `str` filepath to the SQL database.
    allow_overwrite: A boolean indicating whether to allow overwriting if file
      already exists at dataset_filepath.

  Raises:
    FileExistsError: if file exists at `dataset_filepath` and `allow_overwrite`
      is False. If overwriting is intended, please use allow_overwrite = True.
    ElementSpecCompatibilityError: if the element_spec of local datasets are not
      identical across clients, or if the element_spec of datasets are not of
      type `Mapping[str, TensorSpec]`.
  """

  if tf.io.gfile.exists(database_filepath) and not allow_overwrite:
    raise FileExistsError(f'File already exists at {database_filepath}')

  tmp_database_filepath = tempfile.mkstemp()[1]
  logging.info('Building local SQL database at %s.', tmp_database_filepath)
  example_client_id = client_ids[0]
  example_dataset = dataset_fn(example_client_id)
  example_element_spec = example_dataset.element_spec

  if not isinstance(example_element_spec, Mapping):
    raise ElementSpecCompatibilityError(
        'The element_spec of the local dataset must be a Mapping, '
        f'found {example_element_spec} instead')
  for key, val in example_element_spec.items():
    if not isinstance(val, tf.TensorSpec):
      raise ElementSpecCompatibilityError(
          'The element_spec of the local dataset must be a Mapping[str, TensorSpec], '
          f'and must not be nested, found {key}:{val} instead.')

  serializer = _build_serializer(example_element_spec)
  parser = _build_parser(example_element_spec)

  with sqlite3.connect(tmp_database_filepath) as con:
    test_setup_queries = [
        """CREATE TABLE examples (
           split_name TEXT NOT NULL,
           client_id TEXT NOT NULL,
           serialized_example_proto BLOB NOT NULL);""",
        # The `client_metadata` table is required, though not documented.
        """CREATE TABLE client_metadata (
           client_id TEXT NOT NULL,
           split_name TEXT NOT NULL,
           num_examples INTEGER NOT NULL);""",
    ]
    for q in test_setup_queries:
      con.execute(q)

    logging.info('Starting writing to SQL database at scratch path {%s}.',
                 tmp_database_filepath)

    for client_id in client_ids:
      local_ds = dataset_fn(client_id)

      if local_ds.element_spec != example_element_spec:
        raise ElementSpecCompatibilityError(f"""
            All the clients must share the same dataset element type.
            The local dataset of client '{client_id}' has element type
            {local_ds.element_spec}, which is different from client
            '{example_client_id}' which has element type {example_element_spec}.
            """)

      num_elem = 0
      for elem in local_ds:
        num_elem += 1
        con.execute(
            'INSERT INTO examples '
            '(split_name, client_id, serialized_example_proto) '
            'VALUES (?, ?, ?);', ('N/A', client_id, serializer(elem)))

      con.execute(
          'INSERT INTO client_metadata (client_id, split_name, num_examples) '
          'VALUES (?, ?, ?);', (client_id, 'N/A', num_elem))

  if tf.io.gfile.exists(database_filepath):
    tf.io.gfile.remove(database_filepath)
  tf.io.gfile.makedirs(os.path.dirname(database_filepath))
  tf.io.gfile.copy(tmp_database_filepath, database_filepath)
  tf.io.gfile.remove(tmp_database_filepath)
  logging.info('SQL database saved at %s', database_filepath)


def load_and_parse_sql_client_data(
    database_filepath: str,
    element_spec: Mapping[str, tf.TensorSpec]) -> client_data.ClientData:
  """Load a `ClientData` arises by parsing a serialized `SqlClientData`.

  Args:
    database_filepath: A `str` filepath of the SQL database. This function will
      first fetch the SQL database to a local temporary directory if
      `database_filepath` is a remote directory.
    element_spec: The `element_spec` of the local dataset. This is used to parse
      the serialized `tff.simulation.datasets.SqlClientData`. The `element_spec`
      must be of type `Mapping[str, TensorSpec]`.

  Returns:
    A `tff.simulation.datasets.ClientData` instance arised from parsing a
    `tff.simulation.datasets.SqlClientData`.

  Raises:
    FileNotFoundError: if database_filepath does not exist.
    ElementSpecCompatibilityError: if the `element_spec` of datasets are not of
      type `Mapping[str, TensorSpec]`.
  """
  parser = _build_parser(element_spec)

  def dataset_parser(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)

  if not tf.io.gfile.exists(database_filepath):
    raise FileNotFoundError(f'No such file or directory: {database_filepath}')
  elif not os.path.exists(database_filepath):
    logging.info('Starting fetching SQL database to local.')
    tmp_dir = tempfile.mkdtemp()
    tmp_database_filepath = tf.io.gfile.join(
        tmp_dir, os.path.basename(database_filepath))
    tf.io.gfile.copy(database_filepath, tmp_database_filepath, overwrite=True)
    database_filepath = tmp_database_filepath
    logging.info('Finished fetching SQL database to local.')

  return sql_client_data.SqlClientData(database_filepath).preprocess(
      dataset_parser)
