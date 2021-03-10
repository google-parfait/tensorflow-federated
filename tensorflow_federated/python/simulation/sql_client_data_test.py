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

import os
import warnings

from absl import flags
import sqlite3
import tensorflow as tf

from tensorflow_federated.python.simulation import sql_client_data

FLAGS = flags.FLAGS

# TODO(b/182305417): Delete this once the full deprecation period has passed.


def test_dataset_filepath():
  return os.path.join(FLAGS.test_tmpdir, 'test.sqlite')


def make_test_example(client_id: str, e: int) -> bytes:
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'client_id':
                  tf.train.Feature(
                      bytes_list=tf.train.BytesList(
                          value=[client_id.encode('utf-8')])),
              'example_num':
                  tf.train.Feature(int64_list=tf.train.Int64List(value=[e])),
          })).SerializeToString()


def setUpModule():
  with sqlite3.connect(test_dataset_filepath()) as connection:
    test_setup_queries = [
        """CREATE TABLE examples (
           split_name TEXT NOT NULL,
           client_id TEXT NOT NULL,
           serialized_example_proto BLOB NOT NULL);""",
        """CREATE INDEX idx_client_id_split
           ON examples (split_name, client_id);""",
        """CREATE TABLE client_metadata (
           client_id TEXT NOT NULL,
           split_name TEXT NOT NULL,
           num_examples INTEGER NOT NULL);""",
        """CREATE INDEX idx_metadata_client_id
           ON client_metadata (client_id);""",
    ]
    for q in test_setup_queries:
      connection.execute(q)
    for i, client_id in enumerate(['test_a', 'test_b', 'test_c']):
      num_examples = i + 1
      split_counts = {'train': 0, 'test': 0}
      for e in range(num_examples):
        split_name = 'train' if e % 2 == 0 else 'test'
        split_counts[split_name] += 1
        connection.execute(
            'INSERT INTO examples '
            '(split_name, client_id, serialized_example_proto) '
            'VALUES (?, ?, ?);',
            (split_name, client_id, make_test_example(client_id, e)))
      for split_name, count in split_counts.items():
        if count == 0:
          continue
        connection.execute(
            'INSERT INTO client_metadata (client_id, split_name, num_examples) '
            'VALUES (?, ?, ?);', (client_id, split_name, count))


class SqlClientDataTest(tf.test.TestCase):

  def test_deprecation_warning_raised_on_init(self):

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      sql_client_data.SqlClientData(test_dataset_filepath())
      self.assertNotEmpty(w)
      self.assertEqual(w[0].category, DeprecationWarning)
      self.assertRegex(
          str(w[0].message), 'tff.simulation.SqlClientData is deprecated')


if __name__ == '__main__':
  tf.test.main()
