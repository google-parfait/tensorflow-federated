# Copyright 2019, The TensorFlow Federated Authors.
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
import asyncio
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.types import placement_literals


def _temperature_sensor_example_next_fn():

  @computations.tf_computation(
      computation_types.SequenceType(tf.float32), tf.float32)
  def count_over(ds, t):
    return ds.reduce(
        np.float32(0), lambda n, x: n + tf.cast(tf.greater(x, t), tf.float32))

  @computations.tf_computation(computation_types.SequenceType(tf.float32))
  def count_total(ds):
    return ds.reduce(np.float32(0.0), lambda n, _: n + 1.0)

  @computations.federated_computation(
      computation_types.at_clients(computation_types.SequenceType(tf.float32)),
      computation_types.at_server(tf.float32))
  def comp(temperatures, threshold):
    return intrinsics.federated_mean(
        intrinsics.federated_map(
            count_over,
            intrinsics.federated_zip(
                [temperatures,
                 intrinsics.federated_broadcast(threshold)])),
        intrinsics.federated_map(count_total, temperatures))

  return comp


def _create_concurrent_maxthread_tuples():
  tuples = []
  for concurrency in range(1, 5):
    local_ex_string = 'local_executor_{}_clients_per_thread'.format(concurrency)
    ex_factory = executor_stacks.local_executor_factory(
        clients_per_thread=concurrency)
    tuples.append((local_ex_string, ex_factory, concurrency))
    sizing_ex_string = 'sizing_executor_{}_client_thread'.format(concurrency)
    ex_factory = executor_stacks.sizing_executor_factory(
        clients_per_thread=concurrency)
    tuples.append((sizing_ex_string, ex_factory, concurrency))
    debug_ex_string = 'debug_executor_{}_client_thread'.format(concurrency)
    ex_factory = executor_stacks.thread_debugging_executor_factory(
        clients_per_thread=concurrency)
    tuples.append((debug_ex_string, ex_factory, concurrency))
  return tuples


class ExecutorMock(mock.MagicMock, executor_base.Executor):

  async def create_value(self, *args):
    pass

  async def create_call(self, *args):
    pass

  async def create_selection(self, *args):
    pass

  async def create_struct(self, *args):
    pass

  async def close(self, *args):
    pass


class ResourceManagingExecutorFactoryTest(absltest.TestCase):

  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.eager_tf_executor.EagerTFExecutor',
      return_value=ExecutorMock())
  def test_ensure_closed_closes_executor_passed_at_initialization(
      self, mock_ex):

    def _stack_fn(x):
      del x  # Unused
      return ExecutorMock()

    resource_manager = executor_stacks.ResourceManagingExecutorFactory(
        _stack_fn, ensure_closed=[mock_ex])
    resource_manager.clean_up_executors()
    mock_ex.close.assert_called_once()


class ConcreteExecutorFactoryTest(parameterized.TestCase):

  def _maybe_wrap_stack_fn(self, stack_fn, ex_factory):
    """The stack_fn for SizingExecutorFactory requires two outputs.

    If required, we will wrap the stack_fn and provide a dummy value as the
    second return value.

    Args:
      stack_fn: The original stack_fn
      ex_factory: A class which inherits from ExecutorFactory.

    Returns:
      A stack_fn that might additionally return a list as the second value.
    """
    if ex_factory == executor_stacks.SizingExecutorFactory:
      return lambda x: (stack_fn(x), [])
    else:
      return stack_fn

  def test_subclass_base_fails_no_create_method(self):

    class NotCallable(executor_factory.ExecutorFactory):

      def clean_up_executors(self):
        pass

    with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
      NotCallable()

  def test_subclass_base_fails_no_cleanup(self):

    class NoCleanup(executor_factory.ExecutorFactory):

      def create_executor(self, x):
        pass

    with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
      NoCleanup()

  def test_instantiation_succeeds_both_methods_specified(self):

    class Fine(executor_factory.ExecutorFactory):

      def create_executor(self, x):
        pass

      def clean_up_executors(self):
        pass

    Fine()

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_stacks.SizingExecutorFactory),
      ('ResourceManagingExecutorFactory',
       executor_stacks.ResourceManagingExecutorFactory))
  def test_concrete_class_instantiates_stack_fn(self, ex_factory):

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    self.assertIsInstance(factory, ex_factory)

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_stacks.SizingExecutorFactory),
      ('ResourceManagingExecutorFactory',
       executor_stacks.ResourceManagingExecutorFactory))
  def test_call_constructs_executor(self, ex_factory):

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    ex = factory.create_executor({})
    self.assertIsInstance(ex, executor_base.Executor)

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_stacks.SizingExecutorFactory),
      ('ResourceManagingExecutorFactory',
       executor_stacks.ResourceManagingExecutorFactory))
  def test_cleanup_succeeds_without_init(self, ex_factory):

    def _stack_fn(x):
      del x  # Unused
      return eager_tf_executor.EagerTFExecutor()

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    factory.clean_up_executors()

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_stacks.SizingExecutorFactory),
      ('ResourceManagingExecutorFactory',
       executor_stacks.ResourceManagingExecutorFactory))
  def test_cleanup_calls_close(self, ex_factory):
    ex = eager_tf_executor.EagerTFExecutor()
    ex.close = mock.MagicMock()

    def _stack_fn(x):
      del x  # Unused
      return ex

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    factory.create_executor({})
    factory.clean_up_executors()
    ex.close.assert_called_once()

  @parameterized.named_parameters(
      ('SizingExecutorFactory', executor_stacks.SizingExecutorFactory),
      ('ResourceManagingExecutorFactory',
       executor_stacks.ResourceManagingExecutorFactory))
  def test_construction_with_multiple_cardinalities_reuses_existing_stacks(
      self, ex_factory):
    ex = eager_tf_executor.EagerTFExecutor()
    ex.close = mock.MagicMock()
    num_times_invoked = 0

    def _stack_fn(x):
      del x  # Unused
      nonlocal num_times_invoked
      num_times_invoked += 1
      return ex

    maybe_wrapped_stack_fn = self._maybe_wrap_stack_fn(_stack_fn, ex_factory)
    factory = ex_factory(maybe_wrapped_stack_fn)
    for _ in range(2):
      factory.create_executor({})
      factory.create_executor({placement_literals.SERVER: 1})
    self.assertEqual(num_times_invoked, 2)


class ExecutorStacksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
      ('debug_executor', executor_stacks.thread_debugging_executor_factory),
  )
  def test_construction_with_no_args(self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn()
    self.assertIsInstance(executor_factory_impl,
                          executor_stacks.ResourceManagingExecutorFactory)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_construction_raises_with_max_fanout_one(self, executor_factory_fn):
    with self.assertRaises(ValueError):
      executor_factory_fn(max_fanout=1)

  @parameterized.named_parameters(
      ('local_executor_none_clients', executor_stacks.local_executor_factory()),
      ('sizing_executor_none_clients',
       executor_stacks.sizing_executor_factory()),
      ('local_executor_three_clients',
       executor_stacks.local_executor_factory(num_clients=3)),
      ('sizing_executor_three_clients',
       executor_stacks.sizing_executor_factory(num_clients=3)),
  )
  @test_utils.skip_test_for_multi_gpu
  def test_execution_of_temperature_sensor_example(self, executor):
    comp = _temperature_sensor_example_next_fn()
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float),
    ]
    threshold = 15.0

    with executor_test_utils.install_executor(executor):
      result = comp(temperatures, threshold)

    self.assertAlmostEqual(result, 8.333, places=3)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_execution_with_inferred_clients_larger_than_fanout(
      self, executor_factory_fn):

    @computations.federated_computation(computation_types.at_clients(tf.int32))
    def foo(x):
      return intrinsics.federated_sum(x)

    executor = executor_factory_fn(max_fanout=3)
    with executor_test_utils.install_executor(executor):
      result = foo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    self.assertEqual(result, 55)

  @parameterized.named_parameters(
      ('local_executor_none_clients', executor_stacks.local_executor_factory()),
      ('sizing_executor_none_clients',
       executor_stacks.sizing_executor_factory()),
      ('debug_executor_none_clients',
       executor_stacks.thread_debugging_executor_factory()),
      ('local_executor_one_client',
       executor_stacks.local_executor_factory(num_clients=1)),
      ('sizing_executor_one_client',
       executor_stacks.sizing_executor_factory(num_clients=1)),
      ('debug_executor_one_client',
       executor_stacks.thread_debugging_executor_factory(num_clients=1)),
  )
  def test_execution_of_tensorflow(self, executor):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    with executor_test_utils.install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)

  @parameterized.named_parameters(*_create_concurrent_maxthread_tuples())
  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.eager_tf_executor.EagerTFExecutor',
      return_value=ExecutorMock())
  def test_limiting_concurrency_constructs_one_eager_executor(
      self, ex_factory, clients_per_thread, tf_executor_mock):
    num_clients = 10
    ex_factory.create_executor({placement_literals.CLIENTS: num_clients})
    concurrency_level = math.ceil(num_clients / clients_per_thread)
    args_list = tf_executor_mock.call_args_list
    # One for server executor, one for unplaced executor, concurrency_level for
    # clients.
    self.assertLen(args_list, concurrency_level + 2)

  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.reference_resolving_executor.ReferenceResolvingExecutor',
      return_value=ExecutorMock())
  def test_thread_debugging_executor_constructs_exactly_one_reference_resolving_executor(
      self, executor_mock):
    executor_stacks.thread_debugging_executor_factory().create_executor(
        {placement_literals.CLIENTS: 10})
    executor_mock.assert_called_once()

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
      ('debug_executor', executor_stacks.thread_debugging_executor_factory),
  )
  def test_create_executor_raises_with_wrong_cardinalities(
      self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn(num_clients=5)
    cardinalities = {
        placement_literals.SERVER: 1,
        None: 1,
        placement_literals.CLIENTS: 1,
    }
    with self.assertRaises(ValueError,):
      executor_factory_impl.create_executor(cardinalities)


class UnplacedExecutorFactoryTest(parameterized.TestCase):

  def test_constructs_executor_factory(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    self.assertIsInstance(unplaced_factory, executor_factory.ExecutorFactory)

  def test_constructs_executor_factory_without_caching(self):
    unplaced_factory_no_caching = executor_stacks.UnplacedExecutorFactory(
        use_caching=False)
    self.assertIsInstance(unplaced_factory_no_caching,
                          executor_factory.ExecutorFactory)

  def test_create_executor_returns_executor(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    unplaced_executor = unplaced_factory.create_executor(cardinalities={})
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  def test_create_executor_raises_with_nonempty_cardinalitites(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    with self.assertRaises(ValueError):
      unplaced_factory.create_executor(
          cardinalities={placement_literals.SERVER: 1})

  @parameterized.named_parameters(('server_on_cpu', 'CPU'),
                                  ('server_on_gpu', 'GPU'),
                                  ('server_on_tpu', 'TPU'))
  def test_create_executor_with_server_device(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=False, server_device=server_tf_device)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  @parameterized.named_parameters(('clients_on_cpu', 'CPU'),
                                  ('clients_on_gpu', 'GPU'),
                                  ('clients_on_tpu', 'TPU'))
  def test_create_executor_with_client_devices(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=False, client_devices=tf_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  @parameterized.named_parameters(('server_clients_on_cpu', 'CPU'),
                                  ('server_clients_on_gpu', 'GPU'),
                                  ('server_clients_on_tpu', 'TPU'))
  def test_create_executor_with_server_client_devices(self, tf_device):
    tf_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not tf_devices else tf_devices[0]
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=False,
        server_device=server_tf_device,
        client_devices=tf_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)

  @parameterized.named_parameters(('clients_on_cpu', 'CPU'),
                                  ('clients_on_gpu', 'GPU'),
                                  ('clients_on_tpu', 'TPU'))
  def test_create_executor_with_server_cpu_client_devices(self, tf_device):
    cpu_devices = tf.config.list_logical_devices('CPU')
    client_devices = tf.config.list_logical_devices(tf_device)
    server_tf_device = None if not cpu_devices else cpu_devices[0]
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(
        use_caching=False,
        server_device=server_tf_device,
        client_devices=client_devices)
    unplaced_executor = unplaced_factory.create_executor()
    self.assertIsInstance(unplaced_executor, executor_base.Executor)


class FederatingExecutorFactoryTest(absltest.TestCase):

  def test_constructs_executor_factory(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    self.assertIsInstance(federating_factory, executor_factory.ExecutorFactory)

  def test_raises_on_access_of_nonexistent_sizing_executors(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    with self.assertRaisesRegex(ValueError, 'not configured'):
      _ = federating_factory.sizing_executors

  def test_returns_empty_list_of_sizing_executors_if_configured(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1,
        unplaced_ex_factory=unplaced_factory,
        use_sizing=True)
    sizing_ex_list = federating_factory.sizing_executors
    self.assertIsInstance(sizing_ex_list, list)
    self.assertEmpty(sizing_ex_list)

  def test_constructs_as_many_sizing_executors_as_client_executors(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=2,
        unplaced_ex_factory=unplaced_factory,
        use_sizing=True)
    federating_factory.create_executor(
        cardinalities={placement_literals.CLIENTS: 10})
    sizing_ex_list = federating_factory.sizing_executors
    self.assertIsInstance(sizing_ex_list, list)
    self.assertLen(sizing_ex_list, 5)

  def test_reinvocation_of_create_executor_extends_sizing_executors(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=2,
        unplaced_ex_factory=unplaced_factory,
        use_sizing=True)
    federating_factory.create_executor(
        cardinalities={placement_literals.CLIENTS: 10})
    federating_factory.create_executor(
        cardinalities={placement_literals.CLIENTS: 12})
    sizing_ex_list = federating_factory.sizing_executors
    self.assertIsInstance(sizing_ex_list, list)
    self.assertLen(sizing_ex_list, 5 + 6)

  def test_create_executor_raises_mismatched_num_clients(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        num_clients=3,
        clients_per_thread=1,
        unplaced_ex_factory=unplaced_factory,
        use_sizing=True)
    with self.assertRaisesRegex(ValueError, 'configured to return 3 clients'):
      federating_factory.create_executor(
          cardinalities={placement_literals.CLIENTS: 5})


class MinimalLengthFlatStackFnTest(parameterized.TestCase):

  def test_callable_raises_negative_clients(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        2, federating_factory)
    with self.assertRaises(ValueError):
      flat_stack_fn({placement_literals.CLIENTS: -1})

  def test_returns_singleton_list_for_zero_clients(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        3, federating_factory)
    executor_list = flat_stack_fn({placement_literals.CLIENTS: 0})
    self.assertLen(executor_list, 1)

  @parameterized.named_parameters(
      ('max_3_clients_3_clients', 3, 3),
      ('max_3_clients_10_clients', 3, 10),
  )
  def test_constructs_correct_length_list(self, max_clients_per_stack,
                                          num_clients):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        max_clients_per_stack, federating_factory)
    executor_list = flat_stack_fn({placement_literals.CLIENTS: num_clients})
    self.assertLen(executor_list,
                   math.ceil(num_clients / max_clients_per_stack))


class ComposingExecutorFactoryTest(absltest.TestCase):

  def test_constructs_executor_factory_with_federated_factory(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        2, federating_factory)
    composing_ex_factory = executor_stacks.ComposingExecutorFactory(
        max_fanout=2,
        unplaced_ex_factory=unplaced_factory,
        flat_stack_fn=flat_stack_fn)
    self.assertIsInstance(composing_ex_factory,
                          executor_factory.ExecutorFactory)

  def test_constructs_executor_factory_with_child_executors(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    child_executors = [unplaced_factory.create_executor() for _ in range(5)]
    flat_stack_fn = lambda _: child_executors
    composing_ex_factory = executor_stacks.ComposingExecutorFactory(
        max_fanout=2,
        unplaced_ex_factory=unplaced_factory,
        flat_stack_fn=flat_stack_fn,
    )
    self.assertIsInstance(composing_ex_factory,
                          executor_factory.ExecutorFactory)

  def test_construction_raises_with_max_fanout_one(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        2, federating_factory)
    with self.assertRaises(ValueError):
      executor_stacks.ComposingExecutorFactory(
          max_fanout=1,
          unplaced_ex_factory=unplaced_factory,
          flat_stack_fn=flat_stack_fn)

  def test_creates_executor_with_large_fanout(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        2, federating_factory)
    composing_ex_factory = executor_stacks.ComposingExecutorFactory(
        max_fanout=200,
        unplaced_ex_factory=unplaced_factory,
        flat_stack_fn=flat_stack_fn)
    ex = composing_ex_factory.create_executor({placement_literals.CLIENTS: 10})
    self.assertIsInstance(ex, executor_base.Executor)

  def test_creates_executor_with_small_fanout(self):
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=1, unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        2, federating_factory)
    composing_ex_factory = executor_stacks.ComposingExecutorFactory(
        max_fanout=2,
        unplaced_ex_factory=unplaced_factory,
        flat_stack_fn=flat_stack_fn)
    ex = composing_ex_factory.create_executor({placement_literals.CLIENTS: 10})
    self.assertIsInstance(ex, executor_base.Executor)

  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.federated_composing_strategy.FederatedComposingStrategy.factory',
      return_value=ExecutorMock())
  def test_executor_with_small_fanout_calls_correct_number_of_composing_strategies(
      self, composing_strategy_mock):
    num_clients = 10
    max_fanout = 2
    clients_per_thread = 1
    unplaced_factory = executor_stacks.UnplacedExecutorFactory(use_caching=True)
    federating_factory = executor_stacks.FederatingExecutorFactory(
        clients_per_thread=clients_per_thread,
        unplaced_ex_factory=unplaced_factory)
    flat_stack_fn = executor_stacks.create_minimal_length_flat_stack_fn(
        2, federating_factory)
    composing_ex_factory = executor_stacks.ComposingExecutorFactory(
        max_fanout=max_fanout,
        unplaced_ex_factory=unplaced_factory,
        flat_stack_fn=flat_stack_fn)
    composing_ex_factory.create_executor(
        {placement_literals.CLIENTS: num_clients})
    args_list = composing_strategy_mock.call_args_list
    # 5 at the first layer, 1 at the second
    self.assertLen(args_list, 6)


class RemoteExecutorFactoryTest(absltest.TestCase):

  def _make_set_cardinalities_patch(self, mock_obj):

    async def set_cardinalities_patch(self, *args, **kwargs):
      del self  # Unused
      return mock_obj(*args, **kwargs)

    return set_cardinalities_patch

  def setUp(self):
    super().setUp()
    self.coro_mock = mock.Mock()
    self.cardinalities_patcher = mock.patch(
        'tensorflow_federated.python.core.impl.executors.remote_executor.RemoteExecutor.set_cardinalities',
        new=self._make_set_cardinalities_patch(self.coro_mock))
    self.ready_patcher = mock.patch(
        'tensorflow_federated.python.core.impl.executors.remote_executor.RemoteExecutor.is_ready',
        new=lambda _: True)
    self.cardinalities_patcher.start()
    self.ready_patcher.start()

  def tearDown(self):
    self.cardinalities_patcher.stop()
    self.ready_patcher.stop()
    super().tearDown()

  def test_fewer_clients_than_workers_only_passes_one_client(self):
    channels = [
        grpc.insecure_channel('localhost:1'),
        grpc.insecure_channel('localhost:2')
    ]
    remote_ex_factory = executor_stacks.remote_executor_factory(channels)
    remote_ex_factory.create_executor({placement_literals.CLIENTS: 1})
    self.assertLen(self.coro_mock.call_args_list, 1)
    self.coro_mock.assert_called_once_with({placement_literals.CLIENTS: 1})

  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.executor_stacks.ComposingExecutorFactory._aggregate_stacks',
      return_value=ExecutorMock())
  def test_fewer_clients_than_workers_returns_only_one_live_worker(
      self, mock_obj):
    channels = [
        grpc.insecure_channel('localhost:1'),
        grpc.insecure_channel('localhost:2')
    ]
    remote_ex_factory = executor_stacks.remote_executor_factory(channels)
    remote_ex_factory.create_executor({placement_literals.CLIENTS: 1})
    self.assertLen(mock_obj.call_args_list, 1)
    # Assert that aggregate stacks was passed only one executor.
    mock_obj.assert_called_once_with([mock.ANY])

  @mock.patch(
      'tensorflow_federated.python.core.impl.executors.federating_executor.FederatingExecutor',
      return_value=ExecutorMock())
  def test_single_worker_construction_invokes_federating_executor(
      self, mock_obj):
    channels = [
        grpc.insecure_channel('localhost:1'),
    ]
    remote_ex_factory = executor_stacks.remote_executor_factory(channels)
    remote_ex_factory.create_executor({placement_literals.CLIENTS: 10})
    mock_obj.assert_called_once()

  def test_configuration_succeeds_while_event_loop_is_running(self):
    loop = asyncio.get_event_loop()
    channels = [
        grpc.insecure_channel('localhost:1'),
        grpc.insecure_channel('localhost:2')
    ]

    async def coro_func():
      remote_ex_factory = executor_stacks.remote_executor_factory(channels)
      remote_ex_factory.create_executor({placement_literals.CLIENTS: 1})

    loop.run_until_complete(coro_func())
    loop.stop()
    loop.close()


if __name__ == '__main__':
  absltest.main()
