# Copyright 2021, The TensorFlow Federated Authors.
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
"""Tests for the data executor."""

import asyncio
import collections
from absl.testing import absltest
import numpy as np
import tensorflow as tf
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import data_backend_base
from tensorflow_federated.python.core.impl.executors import data_executor
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.federated_context import data as tff_data
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


class TestDataBackend(data_backend_base.DataBackend):

  def __init__(self, test, uri, value, type_spec):
    self._test = test
    self._uri = uri
    self._value = value
    self._type_spec = computation_types.to_type(type_spec)

  async def materialize(self, data, type_spec):
    self._test.assertIsInstance(data, pb.Data)
    self._test.assertIsInstance(type_spec, computation_types.Type)
    self._test.assertEqual(data.uri, self._uri)
    self._test.assertEqual(str(type_spec), str(self._type_spec))
    return self._value


class DataExecutorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._loop = asyncio.get_event_loop()

  def test_data_proto_tensor(self):
    ex = data_executor.DataExecutor(
        eager_tf_executor.EagerTFExecutor(),
        TestDataBackend(self, 'foo://bar', 10, tf.int32))
    proto = pb.Computation(
        data=pb.Data(uri='foo://bar'),
        type=type_serialization.serialize_type(
            computation_types.TensorType(tf.int32)))
    val = self._loop.run_until_complete(ex.create_value(proto))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertEqual(self._loop.run_until_complete(val.compute()), 10)
    ex.close()

  def test_data_proto_dataset(self):
    type_spec = computation_types.SequenceType(tf.int64)
    ex = data_executor.DataExecutor(
        eager_tf_executor.EagerTFExecutor(),
        TestDataBackend(self, 'foo://bar', tf.data.Dataset.range(3), type_spec))
    proto = pb.Computation(
        data=pb.Data(uri='foo://bar'),
        type=type_serialization.serialize_type(type_spec))
    val = self._loop.run_until_complete(ex.create_value(proto))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), 'int64*')
    self.assertCountEqual(
        [x.numpy() for x in iter(self._loop.run_until_complete(val.compute()))],
        [0, 1, 2])
    ex.close()

  def test_pass_through_tensor(self):
    ex = data_executor.DataExecutor(eager_tf_executor.EagerTFExecutor(),
                                    TestDataBackend(self, 'none', None, None))
    val = self._loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertEqual(self._loop.run_until_complete(val.compute()), 10)
    ex.close()

  def test_pass_through_comp(self):
    ex = data_executor.DataExecutor(eager_tf_executor.EagerTFExecutor(),
                                    TestDataBackend(self, 'none', None, None))

    @computations.tf_computation
    def comp():
      return tf.constant(10, tf.int32)

    val = self._loop.run_until_complete(ex.create_value(comp))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '( -> int32)')
    val2 = self._loop.run_until_complete(ex.create_call(val))
    self.assertEqual(str(val2.type_signature), 'int32')
    self.assertEqual(self._loop.run_until_complete(val2.compute()), 10)
    ex.close()

  def test_combo_data_with_comp_and_tensor(self):
    type_spec = computation_types.SequenceType(tf.int64)
    ex = data_executor.DataExecutor(
        eager_tf_executor.EagerTFExecutor(),
        TestDataBackend(self, 'foo://bar', tf.data.Dataset.range(3), type_spec))
    proto = pb.Computation(
        data=pb.Data(uri='foo://bar'),
        type=type_serialization.serialize_type(type_spec))
    arg_val = self._loop.run_until_complete(
        ex.create_value(
            collections.OrderedDict([('x', proto), ('y', 10)]),
            computation_types.StructType([('x', type_spec), ('y', tf.int32)])))

    @computations.tf_computation(type_spec, tf.int32)
    def comp(x, y):
      return tf.cast(x.reduce(np.int64(0), lambda p, q: p + q), tf.int32) + y

    comp_val = self._loop.run_until_complete(ex.create_value(comp))
    ret_val = self._loop.run_until_complete(ex.create_call(comp_val, arg_val))
    self.assertIsInstance(ret_val, eager_tf_executor.EagerValue)
    self.assertEqual(str(ret_val.type_signature), 'int32')
    self.assertEqual(self._loop.run_until_complete(ret_val.compute()), 13)
    ex.close()

  def test_in_executor_stack(self):
    type_spec = computation_types.SequenceType(tf.int64)
    ex = data_executor.DataExecutor(
        eager_tf_executor.EagerTFExecutor(),
        TestDataBackend(self, 'foo://bar', tf.data.Dataset.range(5), type_spec))
    ex_fn = lambda device: ex
    factory = executor_stacks.local_executor_factory(leaf_executor_fn=ex_fn)
    context = execution_context.ExecutionContext(executor_fn=factory)

    @computations.tf_computation(type_spec)
    def foo(ds):
      return tf.cast(ds.reduce(np.int64(0), lambda p, q: p + q), tf.int32)

    @computations.federated_computation
    def bar():
      ds = tff_data.data('foo://bar', type_spec)
      return foo(ds)

    with context_stack_impl.context_stack.install(context):
      result = bar()

    self.assertEqual(result, 10)


if __name__ == '__main__':
  absltest.main()
