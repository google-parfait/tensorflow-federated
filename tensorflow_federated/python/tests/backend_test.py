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
import contextlib
import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import portpicker
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.tests import remote_runtime_test_utils
from tensorflow_federated.python.tests import temperature_sensor_example


_WORKER_PORTS = [portpicker.pick_unused_port() for _ in range(2)]
_AGGREGATOR_PORTS = [portpicker.pick_unused_port() for _ in range(2)]


def _create_native_local_caching_context():
  local_ex_factory = tff.framework.local_executor_factory()

  def _wrap_local_executor_with_caching(cardinalities):
    local_ex = local_ex_factory.create_executor(cardinalities)
    return tff.framework.CachingExecutor(local_ex)

  return tff.framework.ExecutionContext(
      tff.framework.ResourceManagingExecutorFactory(
          _wrap_local_executor_with_caching))


def _get_all_contexts():
  # pyformat: disable
  return [
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_local_caching', _create_native_local_caching_context()),
      ('native_remote_request_reply',
       remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS, rpc_mode='REQUEST_REPLY'),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)),
      ('native_remote_streaming',
       remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS, rpc_mode='STREAMING'),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(_AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_localhost_aggregator_contexts(_WORKER_PORTS, _AGGREGATOR_PORTS)),
      ('native_sizing', tff.backends.native.create_sizing_execution_context()),
      ('native_thread_debug',
       tff.backends.native.create_thread_debugging_execution_context()),
      ('reference', tff.backends.reference.create_reference_context()),
  ]
  # pyformat: enable


def with_context(context):
  """A decorator for running tests in the given `context`."""

  def decorator_context(fn):

    @functools.wraps(fn)
    def wrapper_context(self):
      context_stack = tff.framework.get_context_stack()
      with context_stack.install(context):
        fn(self)

    return wrapper_context

  return decorator_context


def with_environment(server_contexts):
  """A decorator for running tests in an environment."""

  def decorator_environment(fn):

    @functools.wraps(fn)
    def wrapper_environment(self):
      with contextlib.ExitStack() as stack:
        for server_context in server_contexts:
          stack.enter_context(server_context)
        fn(self)

    return wrapper_environment

  return decorator_environment


def with_contexts(*args):
  """A decorator for creating tests parameterized by context."""

  def decorator_contexts(fn, *named_contexts):
    if not named_contexts:
      named_contexts = _get_all_contexts()

    @parameterized.named_parameters(*named_contexts)
    def wrapper_contexts(self, context, server_contexts=None):
      with_context_decorator = with_context(context)
      decorated_fn = with_context_decorator(fn)
      if server_contexts is not None:
        with_environment_decorator = with_environment(server_contexts)
        decorated_fn = with_environment_decorator(decorated_fn)
      decorated_fn(self)

    return wrapper_contexts

  if len(args) == 1 and callable(args[0]):
    return decorator_contexts(args[0])
  else:
    return lambda fn: decorator_contexts(fn, *args)


class ExampleTest(parameterized.TestCase):

  @with_contexts
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

  @with_contexts
  def test_constant(self):

    @tff.federated_computation
    def foo():
      return 10

    result = foo()
    self.assertEqual(result, 10)

  @with_contexts
  def test_empty_tuple(self):

    @tff.federated_computation
    def foo():
      return ()

    result = foo()
    self.assertEqual(result, ())

  @with_contexts
  def test_federated_value(self):

    @tff.federated_computation
    def foo(x):
      return tff.federated_value(x, tff.SERVER)

    result = foo(10)
    self.assertIsNotNone(result)

  @with_contexts
  def test_federated_zip(self):

    @tff.federated_computation([tff.type_at_clients(tf.int32)] * 2)
    def foo(x):
      return tff.federated_zip(x)

    result = foo([[1, 2], [3, 4]])
    self.assertIsNotNone(result)

  @with_contexts
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

  @with_contexts
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

  @with_contexts
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


class TensorFlowComputationTest(parameterized.TestCase):

  @with_contexts
  def test_returns_constant(self):

    @tff.tf_computation
    def foo():
      return 10

    result = foo()
    self.assertEqual(result, 10)

  @with_contexts
  def test_returns_empty_tuple(self):

    @tff.tf_computation
    def foo():
      return ()

    result = foo()
    self.assertEqual(result, ())

  @with_contexts
  def test_returns_variable(self):

    @tff.tf_computation
    def foo():
      return tf.Variable(10, name='var')

    result = foo()
    self.assertEqual(result, 10)

  # pyformat: disable
  @with_contexts(
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_local_caching', _create_native_local_caching_context()),
      ('native_remote',
       remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(_AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_localhost_aggregator_contexts(_WORKER_PORTS, _AGGREGATOR_PORTS)),
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
  @with_contexts(
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_local_caching', _create_native_local_caching_context()),
      ('native_remote',
       remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(_AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_localhost_aggregator_contexts(_WORKER_PORTS, _AGGREGATOR_PORTS)),
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

  @with_contexts
  def test_returns_result_with_typed_fn(self):

    @tff.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    result = foo(1, 2)
    self.assertEqual(result, 3)

  @with_contexts
  def test_raises_type_error_with_typed_fn(self):

    @tff.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    with self.assertRaises(TypeError):
      foo(1.0, 2.0)

  @with_contexts
  def test_returns_result_with_polymorphic_fn(self):

    @tff.tf_computation
    def foo(x, y):
      return x + y

    result = foo(1, 2)
    self.assertEqual(result, 3)
    result = foo(1.0, 2.0)
    self.assertEqual(result, 3.0)


class NonDeterministicTest(parameterized.TestCase):

  @with_contexts
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

  @with_contexts
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

  @with_context(tff.backends.native.create_sizing_execution_context())
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

  @with_contexts
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
