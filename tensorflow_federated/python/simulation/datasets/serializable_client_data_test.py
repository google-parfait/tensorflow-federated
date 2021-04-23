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

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.datasets import serializable_client_data


class ConcreteClientDataTest(tf.test.TestCase, absltest.TestCase):

  def test_concrete_client_data(self):
    client_ids = ['1', '2', '3']

    def create_dataset_fn(client_id):
      num_examples = tf.strings.to_number(client_id, out_type=tf.int64)
      return tf.data.Dataset.range(num_examples)

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

    self.assertEqual(client_data.element_type_structure,
                     tf.TensorSpec(shape=(), dtype=tf.int64))

    def length(ds):
      return tf.data.experimental.cardinality(ds).numpy()

    for i in client_ids:
      self.assertEqual(
          length(client_data.create_tf_dataset_for_client(i)), int(i))

    # Preprocess to only take the first example from each client
    client_data = client_data.preprocess(lambda d: d.take(1))
    for i in client_ids:
      self.assertEqual(length(client_data.create_tf_dataset_for_client(i)), 1)

    # One example per client, so the whole dataset should be `num_clients` long.
    num_clients = len(
        list(iter(client_data.create_tf_dataset_from_all_clients())))
    self.assertLen(client_ids, num_clients)

  def get_test_client_data(self):

    def create_dataset_fn(client_id):
      num_examples = 1 if client_id % 2 == 0 else 0
      return tf.data.Dataset.range(num_examples)

    client_ids = list(range(10))
    return serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

  def test_datasets_lists_all_elements(self):
    client_ids = [1, 2, 3]

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

    def ds_iterable_to_list_set(datasets):
      return set(tuple(ds.as_numpy_iterator()) for ds in datasets)

    datasets = ds_iterable_to_list_set(client_data.datasets())
    expected = ds_iterable_to_list_set(
        (create_dataset_fn(cid) for cid in client_ids))
    self.assertEqual(datasets, expected)

  def test_dataset_computation_lists_all_elements(self):
    client_ids = ['1', '2', '3']

    def create_dataset_fn(client_id):
      num_examples = tf.strings.to_number(client_id, out_type=tf.int64)
      return tf.data.Dataset.range(num_examples)

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

    for client_id in client_ids:
      expected_dataset = create_dataset_fn(client_id)
      actual_dataset = client_data.dataset_computation(client_id)
      expected_values = tuple(expected_dataset.as_numpy_iterator())
      actual_values = tuple(actual_dataset.as_numpy_iterator())
      self.assertEqual(expected_values, actual_values)

  def test_dataset_from_large_client_list(self):
    client_ids = [str(x) for x in range(1_000_000)]

    def create_dataset(_):
      return tf.data.Dataset.range(100)

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset)
    # Ensure this completes within the test timeout without raising error.
    # Previous implementations caused this to take an very long time via Python
    # list -> generator -> list transformations.
    try:
      client_data.create_tf_dataset_from_all_clients(seed=42)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

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

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=only_call_me_thrice)

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

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

    ds = list(client_data.datasets(limit_count=1))
    self.assertLen(ds, 1)

  def test_datasets_doesnt_shuffle_client_ids_list(self):
    client_ids = [1, 2, 3]
    client_ids_copy = client_ids.copy()

    def create_dataset_fn(client_id):
      num_examples = client_id
      return tf.data.Dataset.range(num_examples)

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

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

    client_data = serializable_client_data.SerializableClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

    dataset = client_data.create_tf_dataset_from_all_clients()
    dataset_list = list(dataset.as_numpy_iterator())
    self.assertCountEqual(client_ids, dataset_list)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
