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
import collections
from unittest import mock

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.executors import caching_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_test_utils
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor


def create_test_executor_factory():
  executor = eager_tf_executor.EagerTFExecutor()
  executor = caching_executor.CachingExecutor(executor)
  executor = reference_resolving_executor.ReferenceResolvingExecutor(executor)
  return executor_factory.ExecutorFactoryImpl(lambda _: executor)


def _make_executor_and_tracer_for_test():
  tracer = executor_test_utils.TracingExecutor(
      eager_tf_executor.EagerTFExecutor())
  ex = caching_executor.CachingExecutor(tracer)
  return ex, tracer


def _tensor_to_id(iterable):
  # Tensor is not hashable in TF 2.0 so we hash it using id().
  return [
      item if not isinstance(item, tf.Tensor) else id(item) for item in iterable
  ]


class TestError(Exception):
  """An error for unittests."""


async def raise_error(*args, **kwargs):
  """A function for mock executors that always raises an error."""
  del args  # Unused.
  del kwargs  # Unused.
  await asyncio.sleep(1)
  raise TestError()


# An arbitrary value for testing.
TEST_VALUE = True


async def create_test_value(*args, **kwargs):
  """A function for mock executors that returns an arbitrary value."""
  del args  # Unused.
  del kwargs  # Unused.
  await asyncio.sleep(1)
  return TEST_VALUE


@computations.tf_computation
def foo():
  return tf.constant(10)


class CachingExecutorTest(absltest.TestCase):

  def test_create_value_does_not_cache_error(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_value(1.0, tf.float32))
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_value(1.0, tf.float32))
    # Ensure create_value was called twice on the mock (not cached and only
    # called once).
    mock_executor.create_value.assert_has_calls([
        mock.call(1.0, computation_types.TensorType(tf.float32)),
        mock.call(1.0, computation_types.TensorType(tf.float32))
    ])

  def test_create_value_does_not_cache_error_avoids_double_cache_delete(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    future1 = cached_executor.create_value(1.0, tf.float32)
    future2 = cached_executor.create_value(1.0, tf.float32)
    results = loop.run_until_complete(
        asyncio.gather(future1, future2, return_exceptions=True))
    # Ensure create_call is only called once, since the first call inserts the
    # inner executor future into the cache. However we expect two errors to be
    # returned.
    mock_executor.create_value.assert_called_once_with(
        1.0, computation_types.TensorType(tf.float32))
    self.assertLen(results, 2)
    self.assertIsInstance(results[0], TestError)
    self.assertIsInstance(results[1], TestError)

  def test_create_call_does_not_cache_error(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = create_test_value
    mock_executor.create_call.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    v = loop.run_until_complete(cached_executor.create_value(foo))
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_call(v))
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_call(v))
    # Ensure create_call was called twice on the mock (not cached and only
    # called once).
    mock_executor.create_call.assert_has_calls(
        [mock.call(TEST_VALUE), mock.call(TEST_VALUE)])

  def test_create_call_does_not_cache_error_avoids_double_cache_delete(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = create_test_value
    mock_executor.create_call.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    v = loop.run_until_complete(cached_executor.create_value(foo))
    future_call1 = cached_executor.create_call(v)
    future_call2 = cached_executor.create_call(v)
    results = loop.run_until_complete(
        asyncio.gather(future_call1, future_call2, return_exceptions=True))
    # Ensure create_call is only called once, since the first call inserts the
    # inner executor future into the cache. However we expect two errors to be
    # returned.
    mock_executor.create_call.assert_called_once_with(TEST_VALUE)
    self.assertLen(results, 2)
    self.assertIsInstance(results[0], TestError)
    self.assertIsInstance(results[1], TestError)

  def test_create_tuple_does_not_cache_error(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = create_test_value
    mock_executor.create_tuple.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    value = loop.run_until_complete(cached_executor.create_value(foo))
    value_tuple = (value, value)
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_tuple(value_tuple))
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_tuple(value_tuple))
    # Ensure create_tuple was called twice on the mock (not cached and only
    # called once).
    anon_tuple_value = anonymous_tuple.AnonymousTuple([(None, TEST_VALUE),
                                                       (None, TEST_VALUE)])
    mock_executor.create_tuple.assert_has_calls(
        [mock.call(anon_tuple_value),
         mock.call(anon_tuple_value)])

  def test_create_tuple_does_not_cache_error_avoids_double_delete(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = create_test_value
    mock_executor.create_tuple.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    value = loop.run_until_complete(cached_executor.create_value(foo))
    value_tuple = (value, value)
    future1 = cached_executor.create_tuple(value_tuple)
    future2 = cached_executor.create_tuple(value_tuple)
    results = loop.run_until_complete(
        asyncio.gather(future1, future2, return_exceptions=True))
    # Ensure create_call is only called once, since the first call inserts the
    # inner executor future into the cache. However we expect two errors to be
    # returned.
    mock_executor.create_tuple.assert_called_once_with(
        anonymous_tuple.AnonymousTuple([(None, TEST_VALUE),
                                        (None, TEST_VALUE)]))
    self.assertLen(results, 2)
    self.assertIsInstance(results[0], TestError)
    self.assertIsInstance(results[1], TestError)

  def test_create_selection_does_not_cache_error(self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = create_test_value
    mock_executor.create_selection.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    value = loop.run_until_complete(
        cached_executor.create_value((1, 2),
                                     computation_types.NamedTupleType(
                                         (tf.int32, tf.int32))))
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_selection(value, 1))
    with self.assertRaises(TestError):
      _ = loop.run_until_complete(cached_executor.create_selection(value, 1))
    # Ensure create_tuple was called twice on the mock (not cached and only
    # called once).
    mock_executor.create_selection.assert_has_calls([])

  def test_create_selection_does_not_cache_error_avoids_double_cache_delete(
      self):
    loop = asyncio.get_event_loop()
    mock_executor = mock.create_autospec(executor_base.Executor)
    mock_executor.create_value.side_effect = create_test_value
    mock_executor.create_selection.side_effect = raise_error
    cached_executor = caching_executor.CachingExecutor(mock_executor)
    value = loop.run_until_complete(
        cached_executor.create_value((1, 2),
                                     computation_types.NamedTupleType(
                                         (tf.int32, tf.int32))))
    future1 = cached_executor.create_selection(value, 1)
    future2 = cached_executor.create_selection(value, 1)
    results = loop.run_until_complete(
        asyncio.gather(future1, future2, return_exceptions=True))
    # Ensure create_tuple was called twice on the mock (not cached and only
    # called once).
    mock_executor.create_selection.assert_has_calls([])
    self.assertLen(results, 2)
    self.assertIsInstance(results[0], TestError)
    self.assertIsInstance(results[1], TestError)

  def test_close_clears_cache(self):
    ex, _ = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()
    v1 = loop.run_until_complete(ex.create_value(10, tf.int32))
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIs(v2, v1)
    ex.close()
    v3 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIsNot(v3, v1)

  def test_with_integer_constant(self):
    ex, tracer = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()
    v1 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIsInstance(v1, caching_executor.CachedValue)
    self.assertEqual(str(v1.identifier), '1')
    c1 = loop.run_until_complete(v1.compute())
    self.assertEqual(c1.numpy(), 10)
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIsInstance(v2, caching_executor.CachedValue)
    self.assertEqual(str(v2.identifier), '1')
    self.assertIs(v2, v1)
    expected_trace = [('create_value', 10,
                       computation_types.TensorType(tf.int32), 1),
                      ('compute', 1, c1)]
    self.assertLen(tracer.trace, len(expected_trace))
    for x, y in zip(tracer.trace, expected_trace):
      self.assertCountEqual(_tensor_to_id(x), _tensor_to_id(y))

  def test_with_no_arg_tf_computation(self):
    ex, tracer = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()

    v1 = loop.run_until_complete(ex.create_value(foo))
    self.assertIsInstance(v1, caching_executor.CachedValue)
    self.assertEqual(str(v1.identifier), '1')
    v2 = loop.run_until_complete(ex.create_call(v1))
    self.assertIsInstance(v2, caching_executor.CachedValue)
    self.assertEqual(str(v2.identifier), '1()')
    c2 = loop.run_until_complete(v2.compute())
    self.assertEqual(c2.numpy(), 10)
    v3 = loop.run_until_complete(ex.create_value(foo))
    self.assertIsInstance(v3, caching_executor.CachedValue)
    self.assertEqual(str(v3.identifier), '1')
    self.assertIs(v3, v1)
    v4 = loop.run_until_complete(ex.create_call(v3))
    self.assertIsInstance(v4, caching_executor.CachedValue)
    self.assertEqual(str(v4.identifier), '1()')
    self.assertIs(v4, v2)
    c4 = loop.run_until_complete(v4.compute())
    self.assertEqual(c4.numpy(), 10)
    expected_trace = [('create_value',
                       computation_impl.ComputationImpl.get_proto(foo),
                       foo.type_signature, 1), ('create_call', 1, 2),
                      ('compute', 2, c4)]
    self.assertLen(tracer.trace, len(expected_trace))
    for x, y in zip(tracer.trace, expected_trace):
      self.assertCountEqual(_tensor_to_id(x), _tensor_to_id(y))

  def test_with_one_arg_tf_computation(self):
    ex, tracer = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return tf.add(x, 1)

    v1 = loop.run_until_complete(ex.create_value(add_one))
    self.assertEqual(str(v1.identifier), '1')
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertEqual(str(v2.identifier), '2')
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    self.assertEqual(str(v3.identifier), '1(2)')
    v4 = loop.run_until_complete(ex.create_value(add_one))
    self.assertIs(v4, v1)
    v5 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIs(v5, v2)
    v6 = loop.run_until_complete(ex.create_call(v4, v5))
    self.assertIs(v6, v3)
    c6 = loop.run_until_complete(v6.compute())
    self.assertEqual(c6.numpy(), 11)
    expected_trace = [
        ('create_value', computation_impl.ComputationImpl.get_proto(add_one),
         add_one.type_signature, 1),
        ('create_value', 10, computation_types.TensorType(tf.int32), 2),
        ('create_call', 1, 2, 3), ('compute', 3, c6)
    ]
    self.assertLen(tracer.trace, len(expected_trace))
    for x, y in zip(tracer.trace, expected_trace):
      self.assertCountEqual(_tensor_to_id(x), _tensor_to_id(y))

  def test_with_tuple_of_unnamed_elements(self):
    ex, _ = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()

    v1 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertEqual(str(v1.identifier), '1')
    v2 = loop.run_until_complete(ex.create_value(11, tf.int32))
    self.assertEqual(str(v2.identifier), '2')
    v3 = loop.run_until_complete(ex.create_tuple([v1, v2]))
    self.assertEqual(str(v3.identifier), '<1,2>')
    v4 = loop.run_until_complete(ex.create_tuple((v1, v2)))
    self.assertIs(v4, v3)
    c4 = loop.run_until_complete(v4.compute())
    self.assertEqual(
        str(anonymous_tuple.map_structure(lambda x: x.numpy(), c4)), '<10,11>')

  def test_with_tuple_of_named_elements(self):
    ex, _ = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()

    v1 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertEqual(str(v1.identifier), '1')
    v2 = loop.run_until_complete(ex.create_value(11, tf.int32))
    self.assertEqual(str(v2.identifier), '2')
    v3 = loop.run_until_complete(
        ex.create_tuple(collections.OrderedDict([('P', v1), ('Q', v2)])))
    self.assertEqual(str(v3.identifier), '<P=1,Q=2>')
    v4 = loop.run_until_complete(
        ex.create_tuple(collections.OrderedDict([('P', v1), ('Q', v2)])))
    self.assertIs(v4, v3)
    c4 = loop.run_until_complete(v4.compute())
    self.assertEqual(
        str(anonymous_tuple.map_structure(lambda x: x.numpy(), c4)),
        '<P=10,Q=11>')

  def test_with_selection_by_index(self):
    ex, _ = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()

    v1 = loop.run_until_complete(
        ex.create_value([10, 20],
                        computation_types.NamedTupleType([tf.int32, tf.int32])))
    self.assertEqual(str(v1.identifier), '1')
    v2 = loop.run_until_complete(ex.create_selection(v1, index=0))
    self.assertEqual(str(v2.identifier), '1[0]')
    v3 = loop.run_until_complete(ex.create_selection(v1, index=1))
    self.assertEqual(str(v3.identifier), '1[1]')
    v4 = loop.run_until_complete(ex.create_selection(v1, index=0))
    self.assertIs(v4, v2)
    v5 = loop.run_until_complete(ex.create_selection(v1, index=1))
    self.assertIs(v5, v3)
    c5 = loop.run_until_complete(v5.compute())
    self.assertEqual(c5.numpy(), 20)

  def test_with_numpy_array(self):
    ex, _ = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()
    v1 = loop.run_until_complete(
        ex.create_value(np.array([10]), (tf.int32, [1])))
    self.assertEqual(str(v1.identifier), '1')
    c1 = loop.run_until_complete(v1.compute())
    self.assertEqual(c1.numpy(), 10)
    v2 = loop.run_until_complete(
        ex.create_value(np.array([10]), (tf.int32, [1])))
    self.assertIs(v2, v1)

  def test_with_eager_dataset(self):
    ex, _ = _make_executor_and_tracer_for_test()
    loop = asyncio.get_event_loop()

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def ds_reduce(ds):
      return ds.reduce(np.int32(0), lambda x, y: x + y)

    v1 = loop.run_until_complete(ex.create_value(ds_reduce))
    self.assertEqual(str(v1.identifier), '1')
    ds = tf.data.Dataset.from_tensor_slices([10, 20, 30])
    v2 = loop.run_until_complete(
        ex.create_value(ds, computation_types.SequenceType(tf.int32)))
    self.assertEqual(str(v2.identifier), '2')
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    self.assertEqual(str(v3.identifier), '1(2)')
    c3 = loop.run_until_complete(v3.compute())
    self.assertEqual(c3.numpy(), 60)
    v4 = loop.run_until_complete(
        ex.create_value(ds, computation_types.SequenceType(tf.int32)))
    self.assertIs(v4, v2)

  def test_execution_of_tensorflow(self):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    executor = create_test_executor_factory()
    with executor_test_utils.install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)


if __name__ == '__main__':
  absltest.main()
