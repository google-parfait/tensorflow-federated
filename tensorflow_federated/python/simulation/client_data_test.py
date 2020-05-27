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
import tensorflow as tf

from tensorflow_federated.python.simulation import client_data as cd

tf.compat.v1.enable_v2_behavior()


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

  def test_datasets_lists_all_elements(self):
    client_ids = [1, 2, 3]

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    def ds_iterable_to_list_set(datasets):
      return set(tuple(ds.as_numpy_iterator()) for ds in datasets)

    datasets = ds_iterable_to_list_set(client_data.datasets())
    expected = ds_iterable_to_list_set(
        (create_dataset_fn(cid) for cid in client_ids))
    self.assertEqual(datasets, expected)

  def test_datasets_is_lazy(self):
    client_ids = [1, 2, 3]

    # Note: this is called once on initialization of ClientData
    # with client_ids[0] in order to get the element type.
    # After that, it should be called lazily when `next` is called
    # on a `.datasets()` iterator.
    called_count = 0

    def only_call_me_thrice(client_id):
      nonlocal called_count
      called_count += 1
      if called_count == 1:
        self.assertEqual(client_id, client_ids[0])
      if called_count > 3:
        raise Exception('called too many times')
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=only_call_me_thrice)

    datasets_iter = client_data.datasets()
    next(datasets_iter)
    next(datasets_iter)
    with self.assertRaisesRegex(Exception, 'called too many times'):
      next(datasets_iter)

  def test_datasets_limit_count(self):
    client_ids = [1, 2, 3]

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    ds = list(client_data.datasets(limit_count=1))
    self.assertLen(ds, 1)

  def test_datasets_doesnt_shuffle_client_ids_list(self):
    client_ids = [1, 2, 3]
    client_ids_copy = client_ids.copy()

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    client_data.datasets()
    self.assertEqual(client_ids, client_ids_copy)
    client_data.datasets()
    self.assertEqual(client_ids, client_ids_copy)
    client_data.datasets()
    self.assertEqual(client_ids, client_ids_copy)

  def test_create_tf_dataset_from_all_clients(self):
    client_ids = [1, 2, 3]

    def create_dataset_fn(client_id):
      return tf.data.Dataset.from_tensor_slices([client_id])

    client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    dataset = client_data.create_tf_dataset_from_all_clients()
    dataset_list = list(dataset.as_numpy_iterator())
    self.assertCountEqual(client_ids, dataset_list)

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
  tf.test.main()
