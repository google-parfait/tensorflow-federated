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

from tensorflow_federated.python.tests import remote_runtime_test_utils
from tensorflow_federated.python.tests import temperature_sensor_example
from tensorflow_federated.python.tests import test_contexts


class ExampleTest(parameterized.TestCase):

  @test_contexts.with_contexts
  def test_temperature_sensor_example(self):
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float),
    ]
    threshold = 10.0

    result = temperature_sensor_example.mean_over_threshold(
        temperatures, threshold)
    self.assertEqual(result, 12.5)


class FederatedComputationTest(parameterized.TestCase):

  @test_contexts.with_contexts
  def test_constant(self):

    @tff.federated_computation
    def foo():
      return 10

    result = foo()
    self.assertEqual(result, 10)

  @test_contexts.with_contexts
  def test_empty_tuple(self):

    @tff.federated_computation
    def foo():
      return ()

    result = foo()
    self.assertEqual(result, ())

  @test_contexts.with_contexts
  def test_federated_value(self):

    @tff.federated_computation
    def foo(x):
      return tff.federated_value(x, tff.SERVER)

    result = foo(10)
    self.assertIsNotNone(result)

  @test_contexts.with_contexts
  def test_federated_zip(self):

    @tff.federated_computation([tff.type_at_clients(tf.int32)] * 2)
    def foo(x):
      return tff.federated_zip(x)

    result = foo([[1, 2], [3, 4]])
    self.assertIsNotNone(result)

  @test_contexts.with_contexts
  def test_federated_zip_with_twenty_elements(self):
    # This test will fail if execution scales factorially with number of
    # elements zipped.
    num_element = 20
    num_clients = 2

    @tff.federated_computation([tff.type_at_clients(tf.int32)] * num_element)
    def foo(x):
      return tff.federated_zip(x)

    value = [list(range(num_clients))] * num_element
    result = foo(value)
    self.assertIsNotNone(result)

  @test_contexts.with_contexts
  def test_repeated_invocations_of_map(self):

    @tff.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    result1 = map_add_one([0, 1])
    result2 = map_add_one([0, 1])

    self.assertIsNotNone(result1)
    self.assertEqual(result1, result2)

  @test_contexts.with_contexts
  def test_polymorphism(self):

    @tff.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    result1 = map_add_one([0, 1])
    result2 = map_add_one([0, 1, 2])

    self.assertIsNotNone(result1)
    self.assertIsNotNone(result2)

    self.assertLen(result1, 2)
    self.assertLen(result2, 3)

  @test_contexts.with_contexts
  def test_runs_unplaced_lambda(self):

    @tff.federated_computation(tf.int32, tf.int32)
    def bar(x, y):
      del y  # Unused
      return x

    result = bar(1, 2)
    self.assertEqual(result, 1)

  @test_contexts.with_contexts
  def test_runs_server_placed_lambda(self):

    @tff.federated_computation(tf.int32, tf.int32)
    def foo(x, y):
      del y  # Unused
      return x

    @tff.federated_computation(
        tff.FederatedType(
            collections.OrderedDict(x=tf.int32, y=tf.int32), tff.SERVER))
    def bar(server_tuple):
      return tff.federated_map(foo, server_tuple)

    result = bar(collections.OrderedDict(x=1, y=2))
    self.assertEqual(result, 1)

  @test_contexts.with_contexts
  def test_runs_clients_placed_lambda(self):

    @tff.federated_computation(tf.int32, tf.int32)
    def foo(x, y):
      del y  # Unused
      return x

    @tff.federated_computation(
        tff.FederatedType(
            collections.OrderedDict(x=tf.int32, y=tf.int32), tff.CLIENTS))
    def bar(clients_tuple):
      return tff.federated_map(foo, clients_tuple)

    result = bar([collections.OrderedDict(x=1, y=2)])
    self.assertEqual(result, [1])


class TensorFlowComputationTest(parameterized.TestCase):

  @test_contexts.with_contexts
  def test_returns_constant(self):

    @tff.tf_computation
    def foo():
      return 10

    result = foo()
    self.assertEqual(result, 10)

  @test_contexts.with_contexts
  def test_returns_empty_tuple(self):

    @tff.tf_computation
    def foo():
      return ()

    result = foo()
    self.assertEqual(result, ())

  @test_contexts.with_contexts
  def test_returns_variable(self):

    @tff.tf_computation
    def foo():
      return tf.Variable(10, name='var')

    result = foo()
    self.assertEqual(result, 10)

  # pyformat: disable
  @test_contexts.with_contexts(
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_local_caching', test_contexts.create_native_local_caching_context()),
      ('native_remote',
       remote_runtime_test_utils.create_localhost_remote_context(test_contexts.WORKER_PORTS),
       remote_runtime_test_utils.create_inprocess_worker_contexts(test_contexts.WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(test_contexts.AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_inprocess_aggregator_contexts(test_contexts.WORKER_PORTS, test_contexts.AGGREGATOR_PORTS)),
      ('native_sizing', tff.backends.native.create_sizing_execution_context()),
      ('native_thread_debug',
       tff.backends.native.create_thread_debugging_execution_context()),
  )
  # pyformat: enable
  def test_takes_infinite_dataset(self):

    @tff.tf_computation
    def foo(ds):
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    ds = tf.data.Dataset.range(10).repeat()
    actual_result = foo(ds)

    expected_result = ds.take(10).reduce(np.int64(0), lambda x, y: x + y)
    self.assertEqual(actual_result, expected_result)

  # pyformat: disable
  @test_contexts.with_contexts(
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_local_caching', test_contexts.create_native_local_caching_context()),
      ('native_remote',
       remote_runtime_test_utils.create_localhost_remote_context(test_contexts.WORKER_PORTS),
       remote_runtime_test_utils.create_inprocess_worker_contexts(test_contexts.WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(test_contexts.AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_inprocess_aggregator_contexts(test_contexts.WORKER_PORTS, test_contexts.AGGREGATOR_PORTS)),
      ('native_sizing', tff.backends.native.create_sizing_execution_context()),
      ('native_thread_debug',
       tff.backends.native.create_thread_debugging_execution_context()),
  )
  # pyformat: enable
  def test_returns_infinite_dataset(self):

    @tff.tf_computation
    def foo():
      return tf.data.Dataset.range(10).repeat()

    actual_result = foo()

    expected_result = tf.data.Dataset.range(10).repeat()
    self.assertEqual(
        actual_result.take(100).reduce(np.int64(0), lambda x, y: x + y),
        expected_result.take(100).reduce(np.int64(0), lambda x, y: x + y))

  @test_contexts.with_contexts
  def test_returns_result_with_typed_fn(self):

    @tff.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    result = foo(1, 2)
    self.assertEqual(result, 3)

  @test_contexts.with_contexts
  def test_raises_type_error_with_typed_fn(self):

    @tff.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    with self.assertRaises(TypeError):
      foo(1.0, 2.0)

  @test_contexts.with_contexts
  def test_returns_result_with_polymorphic_fn(self):

    @tff.tf_computation
    def foo(x, y):
      return x + y

    result = foo(1, 2)
    self.assertEqual(result, 3)
    result = foo(1.0, 2.0)
    self.assertEqual(result, 3.0)


class NonDeterministicTest(parameterized.TestCase):

  @test_contexts.with_contexts
  def test_computation_called_once_is_invoked_once(self):

    @tff.tf_computation
    def get_random():
      return tf.random.normal([])

    @tff.federated_computation
    def get_one_random_twice():
      value = get_random()
      return value, value

    first_random, second_random = get_one_random_twice()
    self.assertEqual(first_random, second_random)

  @test_contexts.with_contexts
  def test_computation_called_twice_is_invoked_twice(self):
    self.skipTest(
        'b/139135080: Recognize distinct instantiations of the same TF code as '
        '(potentially) distinct at construction time.')

    @tff.tf_computation
    def get_random():
      return tf.random.normal([])

    @tff.federated_computation
    def get_two_random():
      return get_random(), get_random()

    first_random, second_random = get_two_random()
    self.assertNotEqual(first_random, second_random)


class SizingExecutionContextTest(parameterized.TestCase):

  @test_contexts.with_context(
      tff.backends.native.create_sizing_execution_context())
  def test_get_size_info(self):
    num_clients = 10
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [tf.data.Dataset.range(10).map(to_float)] * num_clients
    threshold = 15.0

    temperature_sensor_example.mean_over_threshold(temperatures, threshold)
    context = tff.framework.get_context_stack().current
    size_info = context.executor_factory.get_size_info()

    # Each client receives a tf.float32 and uploads two tf.float32 values.
    expected_broadcast_bits = [num_clients * 32]
    expected_aggregate_bits = [num_clients * 32 * 2]
    expected_broadcast_history = {
        (('CLIENTS', num_clients),): [[1, tf.float32]] * num_clients
    }
    expected_aggregate_history = {
        (('CLIENTS', num_clients),): [[1, tf.float32]] * num_clients * 2
    }
    self.assertEqual(size_info.broadcast_history, expected_broadcast_history)
    self.assertEqual(size_info.aggregate_history, expected_aggregate_history)
    self.assertEqual(size_info.broadcast_bits, expected_broadcast_bits)
    self.assertEqual(size_info.aggregate_bits, expected_aggregate_bits)


class KerasIntegrationTest(parameterized.TestCase):

  @test_contexts.with_contexts
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
          tf.keras.layers.Input(shape=(input_dim,)),
          tf.keras.layers.Dense(
              50, activity_regularizer=tf.keras.regularizers.l1(.1)),
          tf.keras.layers.Dense(input_dim, activation='sigmoid'),
      ])
      tff_model = tff.learning.from_keras_model(
          keras_model=model,
          input_spec=input_spec,
          loss=tf.keras.losses.MeanSquaredError(),
      )
      return tff_model

    trainer = tff.learning.build_federated_averaging_process(
        model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))

    state = trainer.initialize()
    for _ in range(2):
      state, _ = trainer.next(state, train_data)


if __name__ == '__main__':
  tff.test.set_no_default_context()
  absltest.main()
