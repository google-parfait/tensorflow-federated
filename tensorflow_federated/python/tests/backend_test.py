# Copyright 2020, The TensorFlow Federated Authors.
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

import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.tests import temperature_sensor_example
from tensorflow_federated.python.tests import test_contexts


class NoClientAggregationsTest(parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_executes_null_aggregate(self):
    unit_type = tff.StructWithPythonType([], tuple)

    @tff.tensorflow.computation(unit_type, unit_type)
    def accumulate(a, b):
      del b  # Unused
      return a

    @tff.tensorflow.computation(unit_type)
    def report(a):
      return a

    @tff.federated_computation()
    def empty_agg():
      val_at_clients = tff.federated_value([], tff.CLIENTS)
      return tff.federated_aggregate(
          val_at_clients, [], accumulate, accumulate, report
      )

    result = empty_agg()
    self.assertEqual(result, ())

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_executes_empty_sum(self):

    @tff.federated_computation(tff.FederatedType(np.int32, tff.CLIENTS))
    def fed_sum(x):
      return tff.federated_sum(x)

    result = fed_sum([])
    self.assertEqual(result, 0)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_empty_mean_returns_nan(self):
    self.skipTest('b/200970992')
    # TODO: b/200970992 - Standardize handling of this case. We currently have a
    # ZeroDivisionError, a RuntimeError, and a context that returns nan.

    @tff.federated_computation(tff.FederatedType(np.float32, tff.CLIENTS))
    def fed_mean(x):
      return tff.federated_mean(x)

    with self.assertRaises(RuntimeError):
      fed_mean([])


class DatasetManipulationTest(parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_executes_passthru_dataset(self):

    @tff.tensorflow.computation(tff.SequenceType(np.int64))
    def passthru_dataset(ds):
      return ds

    input_data = tf.data.Dataset.range(10)
    ds = passthru_dataset(input_data)
    self.assertIsInstance(ds, list)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_executes_dataset_concat_aggregation(self):
    element_type = tff.TensorType(np.float32, [2])

    @tff.tensorflow.computation
    def create_dataset():
      return tf.data.Dataset.from_tensor_slices([[0.0, 0.0]])

    @tff.tensorflow.computation
    def concat_datasets(ds1, ds2):
      return ds1.concatenate(ds2)

    @tff.tensorflow.computation
    def identity(ds):
      return ds

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(element_type), tff.CLIENTS)
    )
    def do_a_federated_aggregate(client_ds):
      return tff.federated_aggregate(
          value=client_ds,
          zero=create_dataset(),
          accumulate=concat_datasets,
          merge=concat_datasets,
          report=identity,
      )

    input_data = tf.data.Dataset.from_tensor_slices([[0.1, 0.2]])
    ds = do_a_federated_aggregate([input_data])
    self.assertIsInstance(ds, list)


class TemperatureSensorExampleTest(parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_temperature_sensor_example_with_clients_datasets(self):
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float),
    ]
    threshold = 10.0

    result = temperature_sensor_example.mean_over_threshold(
        temperatures, threshold
    )
    self.assertEqual(result, 12.5)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_temperature_sensor_example_with_clients_lists(self):
    temperatures = [
        [float(x) for x in range(10)],
        [float(x) for x in range(20)],
        [float(x) for x in range(30)],
    ]
    threshold = 10.0

    result = temperature_sensor_example.mean_over_threshold(
        temperatures, threshold
    )
    self.assertEqual(result, 12.5)


class FederatedComputationTest(parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_constant(self):
    @tff.federated_computation
    def foo():
      return 10

    result = foo()
    self.assertEqual(result, 10)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_empty_tuple(self):
    @tff.federated_computation
    def foo():
      return ()

    result = foo()
    self.assertEqual(result, ())

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_federated_value(self):
    @tff.federated_computation
    def foo(x):
      return tff.federated_value(x, tff.SERVER)

    result = foo(10)
    self.assertIsNotNone(result)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_federated_zip(self):

    @tff.federated_computation([tff.FederatedType(np.int32, tff.CLIENTS)] * 2)
    def foo(x):
      return tff.federated_zip(x)

    result = foo([[1, 2], [3, 4]])
    self.assertIsNotNone(result)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_federated_zip_with_twenty_elements(self):
    # This test will fail if execution scales factorially with number of
    # elements zipped.
    num_element = 20
    num_clients = 2

    @tff.federated_computation(
        [tff.FederatedType(np.int32, tff.CLIENTS)] * num_element
    )
    def foo(x):
      return tff.federated_zip(x)

    value = [list(range(num_clients))] * num_element
    result = foo(value)
    self.assertIsNotNone(result)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_repeated_invocations_of_map(self):

    @tff.tensorflow.computation(np.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.FederatedType(np.int32, tff.CLIENTS))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    result1 = map_add_one([0, 1])
    result2 = map_add_one([0, 1])

    self.assertIsNotNone(result1)
    self.assertEqual(result1, result2)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_polymorphism(self):

    @tff.tensorflow.computation(np.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.FederatedType(np.int32, tff.CLIENTS))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    result1 = map_add_one([0, 1])
    result2 = map_add_one([0, 1, 2])

    self.assertIsNotNone(result1)
    self.assertIsNotNone(result2)

    self.assertLen(result1, 2)
    self.assertLen(result2, 3)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_runs_unplaced_lambda(self):

    @tff.federated_computation(np.int32, np.int32)
    def bar(x, y):
      del y  # Unused
      return x

    result = bar(1, 2)
    self.assertEqual(result, 1)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_runs_server_placed_lambda(self):

    @tff.federated_computation(np.int32, np.int32)
    def foo(x, y):
      del y  # Unused
      return x

    @tff.federated_computation(
        tff.FederatedType(
            collections.OrderedDict(x=np.int32, y=np.int32), tff.SERVER
        )
    )
    def bar(server_tuple):
      return tff.federated_map(foo, server_tuple)

    result = bar(collections.OrderedDict(x=1, y=2))
    self.assertEqual(result, 1)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_runs_clients_placed_lambda(self):

    @tff.federated_computation(np.int32, np.int32)
    def foo(x, y):
      del y  # Unused
      return x

    @tff.federated_computation(
        tff.FederatedType(
            collections.OrderedDict(x=np.int32, y=np.int32), tff.CLIENTS
        )
    )
    def bar(clients_tuple):
      return tff.federated_map(foo, clients_tuple)

    result = bar([collections.OrderedDict(x=1, y=2)])
    self.assertEqual(result, [1])

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_bad_type_coercion_raises(self):
    tensor_type = tff.TensorType(np.float32, [None])

    @tff.tensorflow.computation(tensor_type)
    def foo(x):
      # We will pass in a tensor which passes the TFF type check, but fails the
      # reshape.
      return tf.reshape(x, [])

    @tff.federated_computation(tff.FederatedType(tensor_type, tff.CLIENTS))
    def map_foo_at_clients(x):
      return tff.federated_map(foo, x)

    @tff.federated_computation(tff.FederatedType(tensor_type, tff.SERVER))
    def map_foo_at_server(x):
      return tff.federated_map(foo, x)

    bad_tensor = np.array([1.0] * 10, dtype=np.float32)
    good_tensor = np.array([1.0], dtype=np.float32)
    # Ensure running this computation at both placements, or unplaced, still
    # raises.
    with self.assertRaises(Exception):
      foo(bad_tensor)
    with self.assertRaises(Exception):
      map_foo_at_server(bad_tensor)
    with self.assertRaises(Exception):
      map_foo_at_clients([bad_tensor] * 10)
    # We give the distributed runtime a chance to clean itself up, otherwise
    # workers may be getting SIGABRT while they are handling another exception,
    # causing the test infra to crash. Making a successful call ensures that
    # cleanup happens after failures have been handled.
    map_foo_at_clients([good_tensor] * 10)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_runs_federated_select(self):
    keys_per_client = 3
    max_key = 5
    selectee_type = tff.TensorType(np.str_, [None])

    @tff.tensorflow.computation(selectee_type, np.int32)
    def gather(selectee, key):
      return tf.gather(selectee, key)

    @tff.federated_computation(
        tff.FederatedType(selectee_type, tff.SERVER),
        tff.FederatedType(
            tff.TensorType(np.int32, [keys_per_client]), tff.CLIENTS
        ),
    )
    def select(server_val, client_keys):
      max_key_at_server = tff.federated_value(max_key, tff.SERVER)
      return tff.federated_select(
          client_keys, max_key_at_server, server_val, gather
      )

    result = select(
        ['zero', 'one', 'two', 'three', 'four'],
        [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
    )
    self.assertEqual(
        result,
        [
            [b'zero', b'one', b'two'],
            [b'one', b'two', b'three'],
            [b'two', b'three', b'four'],
        ],
    )


class TensorFlowComputationTest(tf.test.TestCase, parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_create_call_take_two_from_stateful_dataset(self):
    vocab = ['a', 'b', 'c', 'd', 'e', 'f']

    @tff.tensorflow.computation(tff.SequenceType(np.str_))
    def take_two(ds):
      table = tf.lookup.StaticVocabularyTable(
          tf.lookup.KeyValueTensorInitializer(
              vocab, tf.range(len(vocab), dtype=tf.int64)
          ),
          num_oov_buckets=1,
      )
      ds = ds.map(table.lookup)
      return ds.take(2)

    ds = tf.data.Dataset.from_tensor_slices(vocab)
    result = take_two(ds)
    self.assertCountEqual(result, [0, 1])

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_twice_used_variable_keeps_separate_state(self):
    def count_one_body():
      variable = tf.Variable(initial_value=0, name='var_of_interest')
      with tf.control_dependencies([variable.assign_add(1)]):
        return variable.read_value()

    count_one_1 = tff.tensorflow.computation(count_one_body)
    count_one_2 = tff.tensorflow.computation(count_one_body)

    @tff.tensorflow.computation
    def count_one_twice():
      return count_one_1(), count_one_1(), count_one_2()

    self.assertEqual((1, 1, 1), count_one_twice())

  @tff.test.with_contexts(
      (
          'native_sync_local',
          tff.backends.native.create_sync_local_cpp_execution_context,
      ),
  )
  def test_dynamic_lookup_table(self):

    @tff.tensorflow.computation(
        tff.TensorType(np.str_, [None]),
        tff.TensorType(np.str_, [None]),
    )
    def comp(table_args, to_lookup):
      values = tf.range(tf.shape(table_args)[0])
      initializer = tf.lookup.KeyValueTensorInitializer(table_args, values)
      table = tf.lookup.StaticHashTable(initializer, default_value=101)
      return table.lookup(to_lookup)

    result = comp(tf.constant(['a', 'b', 'c']), tf.constant(['a', 'z', 'c']))
    self.assertAllEqual(result, [0, 101, 2])

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_reinitialize_dynamic_lookup_table(self):

    @tff.tensorflow.computation(
        tff.TensorType(np.str_, [None]),
        tff.TensorType(np.str_, []),
    )
    def comp(table_args, to_lookup):
      values = tf.range(tf.shape(table_args)[0])
      initializer = tf.lookup.KeyValueTensorInitializer(table_args, values)
      table = tf.lookup.StaticHashTable(initializer, default_value=101)
      return table.lookup(to_lookup)

    expected_zero = comp(tf.constant(['a', 'b', 'c']), tf.constant('a'))
    expected_three = comp(tf.constant(['a', 'b', 'c', 'd']), tf.constant('d'))

    self.assertEqual(expected_zero, 0)
    self.assertEqual(expected_three, 3)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_returns_constant(self):

    @tff.tensorflow.computation
    def foo():
      return 10

    result = foo()
    self.assertEqual(result, 10)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_returns_empty_tuple(self):

    @tff.tensorflow.computation
    def foo():
      return ()

    result = foo()
    self.assertEqual(result, ())

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_returns_variable(self):

    @tff.tensorflow.computation
    def foo():
      return tf.Variable(10, name='var')

    result = foo()
    self.assertEqual(result, 10)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_returns_result_with_typed_fn(self):

    @tff.tensorflow.computation(np.int32, np.int32)
    def foo(x, y):
      return x + y

    result = foo(1, 2)
    self.assertEqual(result, 3)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_raises_type_error_with_typed_fn(self):

    @tff.tensorflow.computation(np.int32, np.int32)
    def foo(x, y):
      return x + y

    with self.assertRaises(TypeError):
      foo(1.0, 2.0)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_returns_result_with_polymorphic_fn(self):

    @tff.tensorflow.computation
    def foo(x, y):
      return x + y

    result = foo(1, 2)
    self.assertEqual(result, 3)
    result = foo(1.0, 2.0)
    self.assertEqual(result, 3.0)


class NonDeterministicTest(parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_computation_called_once_is_invoked_once(self):

    @tff.tensorflow.computation
    def get_random():
      return tf.random.normal([])

    @tff.federated_computation
    def get_one_random_twice():
      value = get_random()
      return value, value

    first_random, second_random = get_one_random_twice()
    self.assertEqual(first_random, second_random)

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_computation_called_twice_is_invoked_twice(self):

    @tff.tensorflow.computation
    def get_random():
      return tf.random.normal([])

    @tff.federated_computation
    def get_two_random():
      return get_random(), get_random()

    first_random, second_random = get_two_random()
    self.assertNotEqual(first_random, second_random)


class KerasIntegrationTest(parameterized.TestCase):

  @tff.test.with_contexts(*test_contexts.get_all_contexts())
  def test_keras_model_with_activity_regularization_runs_fedavg(self):
    self.skipTest('b/171358068')

    num_clients = 5

    x_train = np.zeros([100, 100]).astype(np.float16)
    x_train_clients = np.array_split(x_train, num_clients)

    def map_fn(example):
      return collections.OrderedDict(x=example, y=example)

    def client_data(n):
      ds = tf.data.Dataset.from_tensor_slices(x_train_clients[n])
      return ds.batch(128).map(map_fn)

    train_data = [client_data(n) for n in range(num_clients)]
    input_spec = train_data[0].element_spec
    input_dim = x_train.shape[1]

    def model_fn():
      model = tf.keras.models.Sequential([
          tf.keras.layers.InputLayer(input_shape=(input_dim,)),
          tf.keras.layers.Dense(
              50, activity_regularizer=tf.keras.regularizers.l1(0.1)
          ),
          tf.keras.layers.Dense(input_dim, activation='sigmoid'),
      ])
      tff_model = tff.learning.models.from_keras_model(
          keras_model=model,
          input_spec=input_spec,
          loss=tf.keras.losses.MeanSquaredError(),
      )
      return tff_model

    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn, client_optimizer_fn=tff.learning.optimizers.build_sgdm(0.1)
    )

    state = trainer.initialize()
    for _ in range(2):
      state = trainer.next(state, train_data).state


if __name__ == '__main__':
  tff.test.set_no_default_context()
  absltest.main()
