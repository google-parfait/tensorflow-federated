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

from typing import Optional
import warnings

from tensorflow_federated.python.simulation.datasets import sql_client_data

# TODO(b/182305417): Delete this once the full deprecation period has passed.


class SqlClientData(sql_client_data.SqlClientData):
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

  WARNING: this class is deprecated and is slated for removal in April 2021.
  Please use `tff.simulation.datasets.SqlClientData` instead.
  """

  def __init__(self, database_filepath: str, split_name: Optional[str] = None):
    """Constructs a `tff.simulation.datasets.SqlClientData` object.

    WARNING: this class is deprecated and is slated for removal in April 2021.
    Please use `tff.simulation.datasets.SqlClientData` instead.

    Args:
      database_filepath: A `str` filepath to a SQL database.
      split_name: An optional `str` identifier for the split of the database to
        use. This filters clients and examples based on the `split_name` column.
        A value of `None` means no filtering, selecting all examples.
    """
    warnings.warn(
        'tff.simulation.SqlClientData is deprecated and slated for '
        'removal in April 2021. Please use '
        'tff.simulation.datasets.SqlClientData instead.', DeprecationWarning)
    sql_client_data.SqlClientData.__init__(self, database_filepath, split_name)
