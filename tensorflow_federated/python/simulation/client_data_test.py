# Lint as: python3
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

from absl.testing import absltest
import tensorflow.compat.v2 as tf

from tensorflow_federated.python.simulation import client_data as cd


class ConcreteClientDataTest(tf.test.TestCase, absltest.TestCase):

  def test_concrete_client_data(self):
    client_ids = [1, 2, 3]

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    self.assertEqual(client_data.element_type_structure,
                     tf.TensorSpec(shape=(), dtype=tf.int64))

    def length(ds):
      return tf.data.experimental.cardinality(ds).numpy()

    for i in client_ids:
      self.assertEqual(length(client_data.create_tf_dataset_for_client(i)), i)

    # Preprocess to only take the first example from each client
    client_data = client_data.preprocess(lambda d: d.take(1))
    for i in client_ids:
      self.assertEqual(length(client_data.create_tf_dataset_for_client(i)), 1)

  def get_test_client_data(self):

    def create_dataset_fn(client_id):
      num_examples = 1 if client_id % 2 == 0 else 0
      return tf.data.Dataset.range(num_examples)

    client_ids = list(range(10))
    return cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

  def test_split_train_test_selects_nonempty_test_clients(self):
    # Only even client_ids have data:
    client_data = self.get_test_client_data()

    train, test = cd.ClientData.train_test_client_split(
        client_data, num_test_clients=3)
    # Test that all clients end up in one of the two ClientData:
    self.assertCountEqual(client_data.client_ids,
                          train.client_ids + test.client_ids)
    self.assertLen(test.client_ids, 3)
    for client_id in test.client_ids:
      self.assertEqual(client_id % 2, 0)

    train, test = cd.ClientData.train_test_client_split(
        client_data, num_test_clients=5)
    self.assertLen(test.client_ids, 5)
    self.assertLen(train.client_ids, 5)

  def test_split_train_test_not_enough_nonempty_clients(self):
    client_data = self.get_test_client_data()
    with self.assertRaisesRegex(ValueError, 'too many clients with no data.'):
      cd.ClientData.train_test_client_split(client_data, num_test_clients=6)

  def test_split_train_test_too_few_clients(self):
    client_data = self.get_test_client_data()
    with self.assertRaisesRegex(ValueError, 'has only 10 clients.*11'):
      cd.ClientData.train_test_client_split(client_data, num_test_clients=11)

  def test_split_train_test_no_test_clients_requested(self):
    client_data = self.get_test_client_data()
    with self.assertRaisesRegex(ValueError, 'Please specify'):
      cd.ClientData.train_test_client_split(client_data, num_test_clients=0)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
