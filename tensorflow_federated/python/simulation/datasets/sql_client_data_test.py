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
"""Tests for tensorflow_federated.python.simulation.sql_client_data."""

import os

from absl import flags
import sqlite3
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.datasets import sql_client_data

FLAGS = flags.FLAGS


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

  def test_client_missing(self):
    client_data = sql_client_data.SqlClientData(test_dataset_filepath())
    with self.assertRaisesRegex(ValueError, 'not a client in this ClientData'):
      client_data.create_tf_dataset_for_client('missing_client_id')

  def test_create_dataset_for_client(self):

    def test_split(split_name, example_counts):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      self.assertEqual(client_data.client_ids, list(example_counts.keys()))
      self.assertEqual(client_data.element_type_structure,
                       tf.TensorSpec(shape=(), dtype=tf.string))
      for client_id, expected_examples in example_counts.items():
        dataset = client_data.create_tf_dataset_for_client(client_id)
        actual_examples = dataset.reduce(0, lambda s, x: s + 1)
        self.assertEqual(actual_examples, expected_examples, msg=client_id)

    with self.subTest('no_split'):
      test_split(None, {'test_a': 1, 'test_b': 2, 'test_c': 3})
    with self.subTest('train_split'):
      test_split('train', {'test_a': 1, 'test_b': 1, 'test_c': 2})
    with self.subTest('test_split'):
      # The `test` split has no examples for client `test_a`.
      test_split('test', {'test_b': 1, 'test_c': 1})

  def test_create_dataset_from_all_clients(self):

    def test_split(split_name, example_counts):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      self.assertEqual(client_data.client_ids, list(example_counts.keys()))
      self.assertEqual(client_data.element_type_structure,
                       tf.TensorSpec(shape=(), dtype=tf.string))
      expected_examples = sum(example_counts.values())
      dataset = client_data.create_tf_dataset_from_all_clients()
      actual_examples = dataset.reduce(0, lambda s, x: s + 1)
      self.assertEqual(actual_examples, expected_examples)

    with self.subTest('no_split'):
      test_split(None, {'test_a': 1, 'test_b': 2, 'test_c': 3})
    with self.subTest('train_split'):
      test_split('train', {'test_a': 1, 'test_b': 1, 'test_c': 2})
    with self.subTest('test_split'):
      # The `test` split has no examples for client `test_a`.
      test_split('test', {'test_b': 1, 'test_c': 1})

  def test_dataset_computation(self):

    def test_split(split_name, expected_examples):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      self.assertEqual(
          str(client_data.dataset_computation.type_signature),
          '(string -> string*)')
      dataset = client_data.dataset_computation('test_c')
      actual_examples = dataset.reduce(0, lambda s, x: s + 1)
      self.assertEqual(actual_examples, expected_examples)

    with self.subTest('no_split'):
      test_split(None, 3)
    with self.subTest('train'):
      test_split('train', 2)
    with self.subTest('test'):
      test_split('test', 1)


class PreprocessSqlClientDataTest(tf.test.TestCase):

  def test_preprocess_with_identity_gives_same_structure(self):

    def test_split(split_name):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      preprocessed_client_data = client_data.preprocess(lambda x: x)
      self.assertEqual(preprocessed_client_data.element_type_structure,
                       client_data.element_type_structure)

    with self.subTest('no_split'):
      test_split(None)
    with self.subTest('train_split'):
      test_split('train')
    with self.subTest('test_split'):
      test_split('test')

  def test_create_dataset_for_client_with_identity_preprocess(self):

    def test_split(split_name, example_counts):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      client_data = client_data.preprocess(lambda x: x)
      self.assertEqual(client_data.client_ids, list(example_counts.keys()))
      self.assertEqual(client_data.element_type_structure,
                       tf.TensorSpec(shape=(), dtype=tf.string))
      for client_id, expected_examples in example_counts.items():
        dataset = client_data.create_tf_dataset_for_client(client_id)
        actual_examples = dataset.reduce(0, lambda s, x: s + 1)
        self.assertEqual(actual_examples, expected_examples, msg=client_id)

    with self.subTest('no_split'):
      test_split(None, {'test_a': 1, 'test_b': 2, 'test_c': 3})
    with self.subTest('train_split'):
      test_split('train', {'test_a': 1, 'test_b': 1, 'test_c': 2})
    with self.subTest('test_split'):
      # The `test` split has no examples for client `test_a`.
      test_split('test', {'test_b': 1, 'test_c': 1})

  def test_create_dataset_from_all_clients_with_identity_preprocess(self):

    def test_split(split_name, example_counts):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      client_data = client_data.preprocess(lambda x: x)
      self.assertEqual(client_data.client_ids, list(example_counts.keys()))
      self.assertEqual(client_data.element_type_structure,
                       tf.TensorSpec(shape=(), dtype=tf.string))
      expected_examples = sum(example_counts.values())
      dataset = client_data.create_tf_dataset_from_all_clients()
      actual_examples = dataset.reduce(0, lambda s, x: s + 1)
      self.assertEqual(actual_examples, expected_examples)

    with self.subTest('no_split'):
      test_split(None, {'test_a': 1, 'test_b': 2, 'test_c': 3})
    with self.subTest('train_split'):
      test_split('train', {'test_a': 1, 'test_b': 1, 'test_c': 2})
    with self.subTest('test_split'):
      # The `test` split has no examples for client `test_a`.
      test_split('test', {'test_b': 1, 'test_c': 1})

  def test_dataset_computation_with_identity_preprocess(self):

    def test_split(split_name, expected_examples):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      client_data = client_data.preprocess(lambda x: x)
      self.assertEqual(
          str(client_data.dataset_computation.type_signature),
          '(string -> string*)')
      dataset = client_data.dataset_computation('test_c')
      actual_examples = dataset.reduce(0, lambda s, x: s + 1)
      self.assertEqual(actual_examples, expected_examples)

    with self.subTest('no_split'):
      test_split(None, 3)
    with self.subTest('train'):
      test_split('train', 2)
    with self.subTest('test'):
      test_split('test', 1)

  def test_create_dataset_for_client_with_take_preprocess(self):

    def test_split(split_name, example_counts):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      client_data = client_data.preprocess(lambda x: x.take(1))
      self.assertEqual(client_data.client_ids, list(example_counts.keys()))
      self.assertEqual(client_data.element_type_structure,
                       tf.TensorSpec(shape=(), dtype=tf.string))
      for client_id, expected_examples in example_counts.items():
        dataset = client_data.create_tf_dataset_for_client(client_id)
        actual_examples = dataset.reduce(0, lambda s, x: s + 1)
        self.assertEqual(actual_examples, expected_examples, msg=client_id)

    with self.subTest('no_split'):
      test_split(None, {'test_a': 1, 'test_b': 1, 'test_c': 1})
    with self.subTest('train_split'):
      test_split('train', {'test_a': 1, 'test_b': 1, 'test_c': 1})
    with self.subTest('test_split'):
      # The `test` split has no examples for client `test_a`.
      test_split('test', {'test_b': 1, 'test_c': 1})

  def test_create_dataset_from_all_clients_with_take_preprocess(self):

    def test_split(split_name):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      client_data = client_data.preprocess(lambda x: x.take(1))
      expected_examples = len(client_data.client_ids)
      dataset = client_data.create_tf_dataset_from_all_clients()
      actual_examples = dataset.reduce(0, lambda s, x: s + 1)
      self.assertEqual(actual_examples, expected_examples)

    with self.subTest('no_split'):
      test_split(None)
    with self.subTest('train_split'):
      test_split('train')
    with self.subTest('test_split'):
      test_split('test')

  def test_dataset_computation_with_take_preprocess(self):

    def test_split(split_name, expected_examples):
      client_data = sql_client_data.SqlClientData(
          test_dataset_filepath(), split_name=split_name)
      client_data = client_data.preprocess(lambda x: x.take(1))
      self.assertEqual(
          str(client_data.dataset_computation.type_signature),
          '(string -> string*)')
      dataset = client_data.dataset_computation('test_c')
      actual_examples = dataset.reduce(0, lambda s, x: s + 1)
      self.assertEqual(actual_examples, expected_examples)

    with self.subTest('no_split'):
      test_split(None, 1)
    with self.subTest('train'):
      test_split('train', 1)
    with self.subTest('test'):
      test_split('test', 1)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
