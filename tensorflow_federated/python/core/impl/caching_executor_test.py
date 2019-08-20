# Lint as: python3
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
"""Tests for caching_executor.py."""

import asyncio
import collections

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import caching_executor
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl import lambda_executor


def _make_executor_and_tracer_for_test(support_lambdas=False):
  tracer = executor_test_utils.TracingExecutor(eager_executor.EagerExecutor())
  ex = caching_executor.CachingExecutor(tracer)
  if support_lambdas:
    ex = lambda_executor.LambdaExecutor(caching_executor.CachingExecutor(ex))
  return ex, tracer


def _tensor_to_id(iterable):
  # Tensor is not hashable in TF 2.0 so we hash it using id().
  return [
      item if not isinstance(item, tf.Tensor) else id(item) for item in iterable
  ]


class CachingExecutorTest(absltest.TestCase):

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

    @computations.tf_computation
    def foo():
      return tf.constant(10)

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
    def foo(x):
      return tf.add(x, 1)

    v1 = loop.run_until_complete(ex.create_value(foo))
    self.assertEqual(str(v1.identifier), '1')
    v2 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertEqual(str(v2.identifier), '2')
    v3 = loop.run_until_complete(ex.create_call(v1, v2))
    self.assertEqual(str(v3.identifier), '1(2)')
    v4 = loop.run_until_complete(ex.create_value(foo))
    self.assertIs(v4, v1)
    v5 = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIs(v5, v2)
    v6 = loop.run_until_complete(ex.create_call(v4, v5))
    self.assertIs(v6, v3)
    c6 = loop.run_until_complete(v6.compute())
    self.assertEqual(c6.numpy(), 11)
    expected_trace = [
        ('create_value', computation_impl.ComputationImpl.get_proto(foo),
         foo.type_signature, 1),
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
    def foo(ds):
      return ds.reduce(np.int32(0), lambda x, y: x + y)

    v1 = loop.run_until_complete(ex.create_value(foo))
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

  def test_with_mnist_training_example(self):
    ex, _ = _make_executor_and_tracer_for_test(support_lambdas=True)
    executor_test_utils.test_mnist_training(self, ex)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
