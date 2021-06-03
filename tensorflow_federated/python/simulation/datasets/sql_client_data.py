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
"""Implementation of `ClientData` backed by an SQL database."""

from typing import Iterator, Optional

from absl import logging
import sqlite3
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.simulation.datasets import client_data


class DatabaseFormatError(Exception):
  pass


REQUIRED_TABLES = frozenset(["examples", "client_metadata"])
REQUIRED_EXAMPLES_COLUMNS = frozenset(
    ["split_name", "client_id", "serialized_example_proto"])


def _check_database_format(database_filepath: str):
  """Validates the format of a SQLite database.

  Args:
    database_filepath: A string filepath to a SQLite database.

  Raises:
    DatabaseFormatError: If the required tables or columns are missing from the
      database at `database_filepath`.
  """
  connection = sqlite3.connect(database_filepath)
  # Make sure `examples` and `client_metadata` tables exists.
  result = connection.execute("SELECT name FROM sqlite_master;")
  table_names = {r[0] for r in result}
  missing_tables = REQUIRED_TABLES - table_names
  if missing_tables:
    raise DatabaseFormatError(
        f"Database at [{database_filepath}] does not have the required "
        f"{missing_tables} tables.")
  column_names = set()
  for r in connection.execute("PRAGMA table_info(examples);"):
    column_names.add(r[1])
  missing_required_columns = REQUIRED_EXAMPLES_COLUMNS - column_names
  if missing_required_columns:
    raise DatabaseFormatError(
        "Database table `examples` must contain columns "
        f"{REQUIRED_EXAMPLES_COLUMNS}, "
        f"but is missing columns {missing_required_columns}.")


def _fetch_client_ids(database_filepath: str,
                      split_name: Optional[str] = None) -> Iterator[str]:
  """Fetches the list of client_ids.

  Args:
    database_filepath: A path to a SQL database.
    split_name: An optional split name to filter on. If `None`, all client ids
      are returned.

  Returns:
    An iterator of string client ids.
  """
  connection = sqlite3.connect(database_filepath)
  query = "SELECT DISTINCT client_id FROM client_metadata"
  if split_name is not None:
    query += f" WHERE split_name = '{split_name}'"
  query += ";"
  result = connection.execute(query)
  return map(lambda x: x[0], result)


class SqlClientData(client_data.SerializableClientData):
  """A `tff.simulation.datasets.ClientData` backed by an SQL file.

  This class expects that the SQL file has an `examples` table where each
  row is an example in the dataset. The table must contain at least the
  following columns:

     -   `split_name`: `TEXT` column used to split test, holdout, and
         training examples.
     -   `client_id`: `TEXT` column identifying which user the example belongs
         to.
     -   `serialized_example_proto`: A serialized `tf.train.Example` protocol
         buffer containing containing the example data.
  """

  def __init__(self, database_filepath: str, split_name: Optional[str] = None):
    """Constructs a `tff.simulation.datasets.SqlClientData` object.

    Args:
      database_filepath: A `str` filepath to a SQL database.
      split_name: An optional `str` identifier for the split of the database to
        use. This filters clients and examples based on the `split_name` column.
        A value of `None` means no filtering, selecting all examples.
    """
    py_typecheck.check_type(database_filepath, str)
    _check_database_format(database_filepath)
    self._filepath = database_filepath
    self._split_name = split_name
    self._client_ids = sorted(
        list(_fetch_client_ids(database_filepath, split_name)))
    logging.info("Loaded %d client ids from SQL database.",
                 len(self._client_ids))
    # SQLite returns a single column of bytes which are serialized protocol
    # buffer messages.
    self._element_type_structure = tf.TensorSpec(dtype=tf.string, shape=())

  def _create_dataset(self, client_id):
    """Creates a `tf.data.Dataset` for a client in a TF-serializable manner."""
    query_parts = [
        "SELECT serialized_example_proto FROM examples WHERE client_id = '",
        client_id, "'"
    ]
    if self._split_name is not None:
      query_parts.extend([" and split_name ='", self._split_name, "'"])
    return tf.data.experimental.SqlDataset(
        driver_name="sqlite",
        data_source_name=self._filepath,
        query=tf.strings.join(query_parts),
        output_types=(tf.string))

  @property
  def serializable_dataset_fn(self):
    return self._create_dataset

  @property
  def client_ids(self):
    return self._client_ids

  def create_tf_dataset_for_client(self, client_id: str):
    """Creates a new `tf.data.Dataset` containing the client training examples.

    This function will create a dataset for a given client if `client_id` is
    contained in the `client_ids` property of the `SQLClientData`. Unlike
    `self.serializable_dataset_fn`, this method is not serializable.

    Args:
      client_id: The string identifier for the desired client.

    Returns:
      A `tf.data.Dataset` object.
    """
    if client_id not in self.client_ids:
      raise ValueError(
          "ID [{i}] is not a client in this ClientData. See "
          "property `client_ids` for the list of valid ids.".format(
              i=client_id))
    return self._create_dataset(client_id)

  @property
  def element_type_structure(self):
    return self._element_type_structure
