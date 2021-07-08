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

from typing import Union
import warnings

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.datasets import client_data as cd


class CheckRandomSeedTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('integer1', 0),
      ('integer2', 1234),
      ('integer3', 2**32 - 1),
      ('sequence1', [0, 2, 323]),
      ('sequence2', [1024]),
      ('sequence3', [2**31, 2**32 - 1]),
      ('none', None),
  )
  def test_validate_does_not_raise_on_expected_inputs(self, seed):
    cd.check_numpy_random_seed(seed)

  @parameterized.named_parameters(
      ('integer1', -1),
      ('integer2', 2**40),
      ('integer3', 2**32),
      ('sequence1', [0, 2, -1]),
      ('sequence2', [2**33]),
      ('sequence3', [2**31, 2**32 - 1, -500]),
      ('string', 'bad_seed'),
  )
  def test_validate_raises_on_unexpected_inputs(self, seed):
    with self.assertRaises(cd.InvalidRandomSeedError):
      cd.check_numpy_random_seed(seed)


def create_concrete_client_data(
    serializable: bool = False,
) -> Union[cd.ConcreteSerializableClientData, cd.ConcreteClientData]:
  """Creates a simple `ConcreteSerializableClientData` instance.

  The resulting `ClientData` has the following clients and datasets (written as
  lists):
  *   client `1`: [0]
  *   client `2`: [0, 1]
  *   client `3`: [0, 1, 2]

  Args:
    serializable: A boolean indicating whether to create a `ConcreteClientData`
      (`serializable = False`) or a `ConcreteSerializableClientData`
      (`serializable = True`).

  Returns:
    A `ConcreteSerializableClientData` instance.
  """
  client_ids = ['1', '2', '3']

  def create_dataset_fn(client_id):
    num_examples = tf.strings.to_number(client_id, out_type=tf.int64)
    return tf.data.Dataset.range(num_examples)

  if serializable:
    concrete_client_data = cd.ClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)
  else:
    concrete_client_data = cd.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)
  return concrete_client_data


def dataset_length(dataset):
  return dataset.reduce(0, lambda x, _: x + 1)


class TrainTestClientSplitTest(tf.test.TestCase, parameterized.TestCase):

  def get_even_odd_client_data(self):
    """Creates a `ClientData` where only clients with even IDs have data."""

    def create_dataset_fn(client_id):
      client_id_as_int = tf.strings.to_number(client_id, out_type=tf.int64)
      num_examples = 1 if client_id_as_int % 2 == 0 else 0
      return tf.data.Dataset.range(num_examples)

    client_ids = [str(x) for x in range(10)]
    return cd.ClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)

  def test_split_train_test_selects_nonempty_test_clients(self):
    # Only even client_ids have data:
    client_data = self.get_even_odd_client_data()

    train, test = cd.ClientData.train_test_client_split(
        client_data, num_test_clients=3)
    # Test that all clients end up in one of the two ClientData:
    self.assertCountEqual(client_data.client_ids,
                          train.client_ids + test.client_ids)
    self.assertLen(test.client_ids, 3)
    for client_id in test.client_ids:
      self.assertEqual(int(client_id) % 2, 0)

    train, test = cd.ClientData.train_test_client_split(
        client_data, num_test_clients=5)
    self.assertLen(test.client_ids, 5)
    self.assertLen(train.client_ids, 5)

  def test_split_train_test_not_enough_nonempty_clients(self):
    client_data = self.get_even_odd_client_data()
    with self.assertRaisesRegex(ValueError, 'too many clients with no data.'):
      cd.ClientData.train_test_client_split(client_data, num_test_clients=6)

  def test_split_train_test_too_few_clients(self):
    client_data = self.get_even_odd_client_data()
    with self.assertRaisesRegex(ValueError, 'has only 10 clients.*11'):
      cd.ClientData.train_test_client_split(client_data, num_test_clients=11)

  def test_split_train_test_no_test_clients_requested(self):
    client_data = self.get_even_odd_client_data()
    with self.assertRaisesRegex(ValueError, 'Please specify'):
      cd.ClientData.train_test_client_split(client_data, num_test_clients=0)

  def test_split_train_test_fixed_seed(self):
    client_data = self.get_even_odd_client_data()

    train_0, test_0 = cd.ClientData.train_test_client_split(
        client_data, num_test_clients=3, seed=0)
    train_1, test_1 = cd.ClientData.train_test_client_split(
        client_data, num_test_clients=3, seed=0)

    self.assertEqual(train_0.client_ids, train_1.client_ids)
    self.assertEqual(test_0.client_ids, test_1.client_ids)

  @parameterized.named_parameters(
      ('integer1', 0),
      ('integer2', 1234),
      ('integer3', 2**32 - 1),
      ('sequence1', [0, 2, 323]),
      ('sequence2', [1024]),
      ('sequence3', [2**31, 2**32 - 1]),
      ('none', None),
  )
  def test_split_does_not_raise_on_expected_random_seed(self, seed):
    client_data = client_data = self.get_even_odd_client_data()
    cd.ClientData.train_test_client_split(
        client_data, num_test_clients=3, seed=seed)

  @parameterized.named_parameters(
      ('integer1', -1),
      ('integer2', 2**40),
      ('integer3', 2**32),
      ('sequence1', [0, 2, -1]),
      ('sequence2', [2**33]),
      ('sequence3', [2**31, 2**32 - 1, -500]),
      ('string', 'bad_seed'),
  )
  def test_split_raises_on_unexpected_random_seed(self, seed):
    client_data = client_data = self.get_even_odd_client_data()
    with self.assertRaises(cd.InvalidRandomSeedError):
      cd.ClientData.train_test_client_split(
          client_data, num_test_clients=3, seed=seed)


class ConcreteClientDataTest(tf.test.TestCase, parameterized.TestCase):

  def test_deprecation_warning_raised_on_from_clients_and_fn(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      create_concrete_client_data(serializable=False)
      self.assertNotEmpty(w)
      self.assertEqual(w[0].category, DeprecationWarning)
      self.assertRegex(
          str(w[0].message),
          'tff.simulation.datasets.ClientData.from_clients_and_fn is deprecated'
      )

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_concrete_client_data_create_expected_datasets(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)
    self.assertEqual(client_data.element_type_structure,
                     tf.TensorSpec(shape=(), dtype=tf.int64))
    for i in client_data.client_ids:
      client_dataset = client_data.create_tf_dataset_for_client(i)
      self.assertEqual(dataset_length(client_dataset), int(i))

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_datasets_lists_all_elements(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)

    def ds_iterable_to_list_set(datasets):
      return set(tuple(ds.as_numpy_iterator()) for ds in datasets)

    datasets = ds_iterable_to_list_set(client_data.datasets())
    expected = ds_iterable_to_list_set(
        (client_data.create_tf_dataset_for_client(cid)
         for cid in client_data.client_ids))
    self.assertEqual(datasets, expected)

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_datasets_is_lazy(self, serializable):
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

    if serializable:
      client_data = cd.ClientData.from_clients_and_tf_fn(
          client_ids=client_ids, serializable_dataset_fn=only_call_me_thrice)
    else:
      client_data = cd.ClientData.from_clients_and_fn(
          client_ids=client_ids,
          create_tf_dataset_for_client_fn=only_call_me_thrice)

    datasets_iter = client_data.datasets()
    next(datasets_iter)
    next(datasets_iter)
    with self.assertRaisesRegex(Exception, 'called too many times'):
      next(datasets_iter)

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_datasets_limit_count(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)
    dataset_list = list(client_data.datasets(limit_count=1))
    self.assertLen(dataset_list, 1)

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_datasets_doesnt_shuffle_client_ids_list(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)
    client_ids_copy = client_data.client_ids.copy()

    client_data.datasets()
    self.assertEqual(client_data.client_ids, client_ids_copy)
    client_data.datasets()
    self.assertEqual(client_data.client_ids, client_ids_copy)
    client_data.datasets()
    self.assertEqual(client_data.client_ids, client_ids_copy)

  @parameterized.named_parameters(
      ('integer1', 0),
      ('integer2', 1234),
      ('integer3', 2**32 - 1),
      ('sequence1', [0, 2, 323]),
      ('sequence2', [1024]),
      ('sequence3', [2**31, 2**32 - 1]),
      ('none', None),
  )
  def test_datasets_does_not_raise_on_expected_random_seed(self, seed):
    client_data = create_concrete_client_data(serializable=False)
    next(client_data.datasets(seed=seed))

  @parameterized.named_parameters(
      ('integer1', -1),
      ('integer2', 2**40),
      ('integer3', 2**32),
      ('sequence1', [0, 2, -1]),
      ('sequence2', [2**33]),
      ('sequence3', [2**31, 2**32 - 1, -500]),
      ('string', 'bad_seed'),
  )
  def test_datasets_raises_on_unexpected_random_seed(self, seed):
    client_data = create_concrete_client_data(serializable=False)
    with self.assertRaises(cd.InvalidRandomSeedError):
      next(client_data.datasets(seed=seed))

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_create_tf_dataset_from_all_clients(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)
    dataset = client_data.create_tf_dataset_from_all_clients()
    dataset_list = list(dataset.as_numpy_iterator())
    self.assertCountEqual(dataset_list, [0, 0, 0, 1, 1, 2])

  @parameterized.named_parameters(
      ('integer1', 0),
      ('integer2', 1234),
      ('integer3', 2**32 - 1),
      ('sequence1', [0, 2, 323]),
      ('sequence2', [1024]),
      ('sequence3', [2**31, 2**32 - 1]),
      ('none', None),
  )
  def test_create_from_all_does_not_raise_on_expected_random_seed(self, seed):
    client_data = create_concrete_client_data(serializable=False)
    client_data.create_tf_dataset_from_all_clients(seed=seed)

  @parameterized.named_parameters(
      ('integer1', -1),
      ('integer2', 2**40),
      ('integer3', 2**32),
      ('sequence1', [0, 2, -1]),
      ('sequence2', [2**33]),
      ('sequence3', [2**31, 2**32 - 1, -500]),
      ('string', 'bad_seed'),
  )
  def test_create_from_all_raises_on_unexpected_random_seed(self, seed):
    client_data = create_concrete_client_data(serializable=False)
    with self.assertRaises(cd.InvalidRandomSeedError):
      client_data.create_tf_dataset_from_all_clients(seed=seed)


class ConcreteSerializableClientDataTest(tf.test.TestCase,
                                         parameterized.TestCase):

  def test_dataset_computation_lists_all_elements(self):
    client_data = create_concrete_client_data(serializable=True)
    for client_id in client_data.client_ids:
      expected_values = list(range(int(client_id)))
      client_dataset = client_data.dataset_computation(client_id)
      actual_values = list(client_dataset.as_numpy_iterator())
      self.assertEqual(expected_values, actual_values)

  def test_dataset_from_large_client_list(self):
    client_ids = [str(x) for x in range(1_000_000)]

    def create_dataset(_):
      return tf.data.Dataset.range(100)

    client_data = cd.ClientData.from_clients_and_tf_fn(
        client_ids=client_ids, serializable_dataset_fn=create_dataset)
    # Ensure this completes within the test timeout without raising error.
    # Previous implementations caused this to take an very long time via Python
    # list -> generator -> list transformations.
    try:
      client_data.create_tf_dataset_from_all_clients(seed=42)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

  @parameterized.named_parameters(
      ('integer1', 0),
      ('integer2', 1234),
      ('integer3', 2**32 - 1),
      ('sequence1', [0, 2, 323]),
      ('sequence2', [1024]),
      ('sequence3', [2**31, 2**32 - 1]),
      ('none', None),
  )
  def test_create_from_all_does_not_raise_on_expected_random_seed(self, seed):
    client_data = create_concrete_client_data(serializable=True)
    client_data.create_tf_dataset_from_all_clients(seed=seed)

  @parameterized.named_parameters(
      ('integer1', -1),
      ('integer2', 2**40),
      ('integer3', 2**32),
      ('sequence1', [0, 2, -1]),
      ('sequence2', [2**33]),
      ('sequence3', [2**31, 2**32 - 1, -500]),
      ('string', 'bad_seed'),
  )
  def test_create_from_all_raises_on_unexpected_random_seed(self, seed):
    client_data = create_concrete_client_data(serializable=True)
    with self.assertRaises(cd.InvalidRandomSeedError):
      client_data.create_tf_dataset_from_all_clients(seed=seed)


class PreprocessClientDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_preprocess_creates_expected_client_datasets(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)

    def preprocess_fn(dataset):
      return dataset.map(lambda x: 2 * x)

    preprocess_client_data = client_data.preprocess(preprocess_fn)
    for client_id in client_data.client_ids:
      expected_dataset = [2 * a for a in range(int(client_id))]
      actual_dataset = preprocess_client_data.create_tf_dataset_for_client(
          client_id)
      self.assertEqual(expected_dataset,
                       list(actual_dataset.as_numpy_iterator()))

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_preprocess_with_take_one(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)
    preprocess_fn = lambda x: x.take(1)

    preprocess_client_data = client_data.preprocess(preprocess_fn)
    for client_id in client_data.client_ids:
      dataset = preprocess_client_data.create_tf_dataset_for_client(client_id)
      self.assertEqual(dataset_length(dataset), 1)

    self.assertLen(
        client_data.client_ids,
        dataset_length(
            preprocess_client_data.create_tf_dataset_from_all_clients()))

  def test_preprocess_creates_expected_client_datasets_with_dataset_comp(self):
    # We only use `serializable=True`, since it has a `dataset_computation`
    # attribute.
    client_data = create_concrete_client_data(serializable=True)

    def preprocess_fn(dataset):
      return dataset.map(lambda x: 2 * x)

    preprocess_client_data = client_data.preprocess(preprocess_fn)
    for client_id in client_data.client_ids:
      expected_dataset = [2 * a for a in range(int(client_id))]
      actual_dataset = preprocess_client_data.dataset_computation(client_id)
      self.assertEqual(expected_dataset,
                       list(actual_dataset.as_numpy_iterator()))

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_preprocess_creates_expected_amalgamated_dataset(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)

    def preprocess_fn(dataset):
      return dataset.map(lambda x: 2 * x)

    preprocess_client_data = client_data.preprocess(preprocess_fn)
    expected_amalgamated_dataset = [0, 0, 2, 0, 2, 4]
    actual_amalgamated_dataset = (
        preprocess_client_data.create_tf_dataset_from_all_clients())
    self.assertCountEqual(expected_amalgamated_dataset,
                          list(actual_amalgamated_dataset.as_numpy_iterator()))

  @parameterized.named_parameters(('nonserializable', False),
                                  ('serializable', True))
  def test_preprocess_raises_on_tff_computation(self, serializable):
    client_data = create_concrete_client_data(serializable=serializable)

    @computations.tf_computation
    def foo():
      return 1

    with self.assertRaises(cd.IncompatiblePreprocessFnError):
      client_data.preprocess(foo)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
