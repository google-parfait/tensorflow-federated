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

import asyncio
import collections
from absl.testing import absltest

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import sequence_executor
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization


def _run_sync(coroutine):
  return asyncio.get_event_loop().run_until_complete(coroutine)


def _make_sequence_reduce_type(element_type, accumulator_type):
  return computation_types.FunctionType(
      parameter=[
          computation_types.SequenceType(element_type), accumulator_type,
          type_factory.reduction_op(accumulator_type, element_type)
      ],
      result=accumulator_type)


def _make_sequence_map_type(source_type, target_type):
  return computation_types.FunctionType(
      parameter=[
          computation_types.FunctionType(source_type, target_type),
          computation_types.SequenceType(source_type)
      ],
      result=computation_types.SequenceType(target_type))


def _make_sequence_reduce_value(executor, element_type, accumulator_type):
  intrinsic_spec = pb.Intrinsic(uri=intrinsic_defs.SEQUENCE_REDUCE.uri)
  type_spec = _make_sequence_reduce_type(element_type, accumulator_type)
  comp_pb = pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      intrinsic=intrinsic_spec)
  return _run_sync(executor.create_value(comp_pb, type_spec))


def _make_sequence_map_value(executor, source_type, target_type):
  intrinsic_spec = pb.Intrinsic(uri=intrinsic_defs.SEQUENCE_MAP.uri)
  type_spec = _make_sequence_map_type(source_type, target_type)
  comp_pb = pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      intrinsic=intrinsic_spec)
  return _run_sync(executor.create_value(comp_pb, type_spec))


class SequenceExecutorTest(absltest.TestCase):

  def setUp(self):
    super(SequenceExecutorTest, self).setUp()
    self._target_executor = eager_tf_executor.EagerTFExecutor()
    self._sequence_executor = sequence_executor.SequenceExecutor(
        self._target_executor)

  def test_create_value_and_compute_with_int_const(self):
    int_const = 10
    type_spec = computation_types.TensorType(tf.int32)
    val = _run_sync(self._sequence_executor.create_value(int_const, type_spec))
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), str(type_spec))
    self.assertIsInstance(val.internal_representation,
                          eager_tf_executor.EagerValue)
    self.assertEqual(val.internal_representation.internal_representation,
                     int_const)
    result = _run_sync(val.compute())
    self.assertEqual(result, 10)

  def test_create_value_and_compute_with_struct(self):
    my_struct = collections.OrderedDict([('a', 10), ('b', 20)])
    type_spec = computation_types.StructType(
        collections.OrderedDict([('a', tf.int32), ('b', tf.int32)]))
    val = _run_sync(self._sequence_executor.create_value(my_struct, type_spec))
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), str(type_spec))
    self.assertIsInstance(val.internal_representation, structure.Struct)
    self.assertIsInstance(val.internal_representation.a,
                          eager_tf_executor.EagerValue)
    self.assertIsInstance(val.internal_representation.b,
                          eager_tf_executor.EagerValue)
    result = _run_sync(val.compute())
    self.assertEqual(str(result), '<a=10,b=20>')

  def test_create_struct(self):
    elements = []
    for x in [10, 20]:
      elements.append(('v{}'.format(x),
                       _run_sync(
                           self._sequence_executor.create_value(
                               x, computation_types.TensorType(tf.int32)))))
    elements = structure.Struct(elements)
    val = _run_sync(self._sequence_executor.create_struct(elements))
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), '<v10=int32,v20=int32>')
    self.assertIsInstance(val.internal_representation, structure.Struct)
    self.assertListEqual(
        structure.name_list(val.internal_representation), ['v10', 'v20'])
    self.assertIsInstance(val.internal_representation.v10,
                          eager_tf_executor.EagerValue)
    self.assertIsInstance(val.internal_representation.v20,
                          eager_tf_executor.EagerValue)
    self.assertEqual(_run_sync(val.internal_representation.v10.compute()), 10)
    self.assertEqual(_run_sync(val.internal_representation.v20.compute()), 20)

  def test_create_selection(self):
    my_struct = collections.OrderedDict([('a', 10), ('b', 20)])
    type_spec = computation_types.StructType(
        collections.OrderedDict([('a', tf.int32), ('b', tf.int32)]))
    struct_val = _run_sync(
        self._sequence_executor.create_value(my_struct, type_spec))
    el_0 = _run_sync(
        self._sequence_executor.create_selection(struct_val, index=0))
    self.assertEqual(_run_sync(el_0.compute()), 10)
    el_b = _run_sync(
        self._sequence_executor.create_selection(struct_val, name='b'))
    self.assertEqual(_run_sync(el_b.compute()), 20)

  def test_create_value_with_tf_computation(self):
    comp = computations.tf_computation(lambda x: x + 1, tf.int32)
    comp_pb = computation_impl.ComputationImpl.get_proto(comp)
    type_spec = comp.type_signature
    val = _run_sync(self._sequence_executor.create_value(comp_pb, type_spec))
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), str(type_spec))
    self.assertIsInstance(val.internal_representation,
                          eager_tf_executor.EagerValue)

    async def _fn():
      fn = val.internal_representation
      arg = await self._target_executor.create_value(10, tf.int32)
      result = await self._target_executor.create_call(fn, arg)
      return await result.compute()

    self.assertEqual(_run_sync(_fn()), 11)

  def test_call_tf_comp_with_int(self):
    comp = computations.tf_computation(lambda x: x + 1, tf.int32)
    comp_pb = computation_impl.ComputationImpl.get_proto(comp)
    comp_type = comp.type_signature
    comp_val = _run_sync(
        self._sequence_executor.create_value(comp_pb, comp_type))
    arg = 10
    arg_type = tf.int32
    arg_val = _run_sync(self._sequence_executor.create_value(arg, arg_type))
    result_val = _run_sync(
        self._sequence_executor.create_call(comp_val, arg_val))
    self.assertIsInstance(result_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_val.type_signature), 'int32')
    self.assertEqual(_run_sync(result_val.compute()), 11)

  def test_call_tf_comp_with_int_tuple(self):
    comp = computations.tf_computation(lambda x, y: x + y, tf.int32, tf.int32)
    comp_pb = computation_impl.ComputationImpl.get_proto(comp)
    comp_type = comp.type_signature
    comp_val = _run_sync(
        self._sequence_executor.create_value(comp_pb, comp_type))
    arg = collections.OrderedDict([('a', 10), ('b', 20)])
    arg_type = computation_types.StructType(
        collections.OrderedDict([('a', tf.int32), ('b', tf.int32)]))
    arg_val = _run_sync(self._sequence_executor.create_value(arg, arg_type))
    result_val = _run_sync(
        self._sequence_executor.create_call(comp_val, arg_val))
    self.assertIsInstance(result_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_val.type_signature), 'int32')
    self.assertEqual(_run_sync(result_val.compute()), 30)

  def test_create_value_with_eager_tf_dataset(self):
    ds = tf.data.Dataset.range(5)
    type_spec = computation_types.SequenceType(tf.int64)
    val = _run_sync(self._sequence_executor.create_value(ds, type_spec))
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), str(type_spec))
    self.assertIsInstance(val.internal_representation,
                          sequence_executor._Sequence)
    self.assertIs(_run_sync(val.internal_representation.compute()), ds)
    result = list(_run_sync(val.compute()))
    self.assertListEqual(result, list(range(5)))

  def test_call_tf_comp_with_eager_tf_dataset(self):
    comp = computations.tf_computation(
        (lambda x: x.reduce(np.int64(0), lambda x, y: x + y)),
        computation_types.SequenceType(tf.int64))
    comp_pb = computation_impl.ComputationImpl.get_proto(comp)
    comp_type = comp.type_signature
    comp_val = _run_sync(
        self._sequence_executor.create_value(comp_pb, comp_type))
    arg = tf.data.Dataset.range(5)
    arg_type = computation_types.SequenceType(tf.int64)
    arg_val = _run_sync(self._sequence_executor.create_value(arg, arg_type))
    result_val = _run_sync(
        self._sequence_executor.create_call(comp_val, arg_val))
    self.assertIsInstance(result_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_val.type_signature), 'int64')
    self.assertEqual(_run_sync(result_val.compute()), 10)

  def test_create_value_with_sequence_reduce_intrinsic_spec(self):
    type_spec = _make_sequence_reduce_type(tf.int32, tf.int32)
    val = _make_sequence_reduce_value(self._sequence_executor, tf.int32,
                                      tf.int32)
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), str(type_spec))
    self.assertIsInstance(val.internal_representation,
                          sequence_executor._SequenceReduceOp)

  def test_sequence_reduce_tf_dataset(self):
    ds = tf.data.Dataset.range(5)
    op = computations.tf_computation(lambda x, y: x + y, tf.int64, tf.int64)
    sequence_reduce_val = _make_sequence_reduce_value(self._sequence_executor,
                                                      tf.int64, tf.int64)
    ds_val = _run_sync(
        self._sequence_executor.create_value(
            ds, computation_types.SequenceType(tf.int64)))
    zero_val = _run_sync(self._sequence_executor.create_value(0, tf.int64))
    op_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(op), op.type_signature))
    arg_val = _run_sync(
        self._sequence_executor.create_struct([ds_val, zero_val, op_val]))
    result_val = _run_sync(
        self._sequence_executor.create_call(sequence_reduce_val, arg_val))
    self.assertEqual(str(result_val.type_signature), 'int64')
    self.assertEqual(_run_sync(result_val.compute()), 10)

  def test_sequence_reduce_list(self):
    ds = list(range(5))
    op = computations.tf_computation(lambda x, y: x + y, tf.int64, tf.int64)
    sequence_reduce_val = _make_sequence_reduce_value(self._sequence_executor,
                                                      tf.int64, tf.int64)
    ds_val = _run_sync(
        self._sequence_executor.create_value(
            ds, computation_types.SequenceType(tf.int64)))
    zero_val = _run_sync(self._sequence_executor.create_value(0, tf.int64))
    op_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(op), op.type_signature))
    arg_val = _run_sync(
        self._sequence_executor.create_struct([ds_val, zero_val, op_val]))
    result_val = _run_sync(
        self._sequence_executor.create_call(sequence_reduce_val, arg_val))
    self.assertEqual(str(result_val.type_signature), 'int64')
    self.assertEqual(_run_sync(result_val.compute()), 10)

  def test_create_value_with_sequence_map_intrinsic_spec(self):
    type_spec = _make_sequence_map_type(tf.int32, tf.int32)
    val = _make_sequence_map_value(self._sequence_executor, tf.int32, tf.int32)
    self.assertIsInstance(val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(val.type_signature), str(type_spec))
    self.assertIsInstance(val.internal_representation,
                          sequence_executor._SequenceMapOp)

  def test_sequence_map_tf_dataset(self):
    ds = tf.data.Dataset.range(3)
    map_fn = computations.tf_computation(lambda x: x + 2, tf.int64)
    sequence_map_val = _make_sequence_map_value(self._sequence_executor,
                                                tf.int64, tf.int64)
    ds_val = _run_sync(
        self._sequence_executor.create_value(
            ds, computation_types.SequenceType(tf.int64)))
    map_fn_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(map_fn),
            map_fn.type_signature))
    arg_val = _run_sync(
        self._sequence_executor.create_struct([map_fn_val, ds_val]))
    result_val = _run_sync(
        self._sequence_executor.create_call(sequence_map_val, arg_val))
    self.assertIsInstance(result_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_val.type_signature), 'int64*')
    self.assertIsInstance(result_val.internal_representation,
                          sequence_executor._SequenceFromMap)
    result = list(_run_sync(result_val.compute()))
    self.assertListEqual(result, [2, 3, 4])

  def test_cascading_sequence_map_tf_dataset(self):
    ds = tf.data.Dataset.range(3)
    ds_val = _run_sync(
        self._sequence_executor.create_value(
            ds, computation_types.SequenceType(tf.int64)))

    @computations.tf_computation(tf.int64)
    def map_fn_1(x):
      return collections.OrderedDict([('a', x + 2), ('b', x + 3)])

    @computations.tf_computation(
        collections.OrderedDict([('a', tf.int64), ('b', tf.int64)]))
    def map_fn_2(a, b):
      return a + b

    map_fn_1_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(map_fn_1),
            map_fn_1.type_signature))
    map_fn_2_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(map_fn_2),
            map_fn_2.type_signature))
    sequence_map_1_val = _make_sequence_map_value(
        self._sequence_executor, map_fn_1.type_signature.parameter,
        map_fn_1.type_signature.result)
    sequence_map_2_val = _make_sequence_map_value(
        self._sequence_executor, map_fn_2.type_signature.parameter,
        map_fn_2.type_signature.result)
    arg_1_val = _run_sync(
        self._sequence_executor.create_struct([map_fn_1_val, ds_val]))
    result_1_val = _run_sync(
        self._sequence_executor.create_call(sequence_map_1_val, arg_1_val))
    arg_2_val = _run_sync(
        self._sequence_executor.create_struct([map_fn_2_val, result_1_val]))
    result_2_val = _run_sync(
        self._sequence_executor.create_call(sequence_map_2_val, arg_2_val))
    self.assertIsInstance(result_2_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_2_val.type_signature), 'int64*')
    self.assertIsInstance(result_2_val.internal_representation,
                          sequence_executor._SequenceFromMap)
    result = list(_run_sync(result_2_val.compute()))
    self.assertListEqual(result, [5, 7, 9])

  def test_sequence_map_reduce_tf_dataset(self):
    ds = tf.data.Dataset.range(3)
    ds_val = _run_sync(
        self._sequence_executor.create_value(
            ds, computation_types.SequenceType(tf.int64)))
    map_fn = computations.tf_computation(lambda x: x + 2, tf.int64)
    map_fn_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(map_fn),
            map_fn.type_signature))
    sequence_map_val = _make_sequence_map_value(self._sequence_executor,
                                                tf.int64, tf.int64)
    arg_1_val = _run_sync(
        self._sequence_executor.create_struct([map_fn_val, ds_val]))
    result_1_val = _run_sync(
        self._sequence_executor.create_call(sequence_map_val, arg_1_val))
    zero_val = _run_sync(self._sequence_executor.create_value(0, tf.int64))
    op = computations.tf_computation(lambda x, y: x + y, tf.int64, tf.int64)
    op_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(op), op.type_signature))
    sequence_reduce_val = _make_sequence_reduce_value(self._sequence_executor,
                                                      tf.int64, tf.int64)
    arg_2_val = _run_sync(
        self._sequence_executor.create_struct([result_1_val, zero_val, op_val]))
    result_2_val = _run_sync(
        self._sequence_executor.create_call(sequence_reduce_val, arg_2_val))
    self.assertIsInstance(result_2_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_2_val.type_signature), 'int64')
    self.assertEqual(_run_sync(result_2_val.compute()), 9)

  def test_sequence_map_tf_reduce_tf_dataset(self):
    ds = tf.data.Dataset.range(3)
    ds_val = _run_sync(
        self._sequence_executor.create_value(
            ds, computation_types.SequenceType(tf.int64)))
    map_fn = computations.tf_computation(lambda x: x + 2, tf.int64)
    map_fn_val = _run_sync(
        self._sequence_executor.create_value(
            computation_impl.ComputationImpl.get_proto(map_fn),
            map_fn.type_signature))
    sequence_map_val = _make_sequence_map_value(self._sequence_executor,
                                                tf.int64, tf.int64)
    arg_1_val = _run_sync(
        self._sequence_executor.create_struct([map_fn_val, ds_val]))
    result_1_val = _run_sync(
        self._sequence_executor.create_call(sequence_map_val, arg_1_val))
    comp = computations.tf_computation(
        (lambda x: x.reduce(np.int64(0), lambda x, y: x + y)),
        computation_types.SequenceType(tf.int64))
    comp_pb = computation_impl.ComputationImpl.get_proto(comp)
    comp_type = comp.type_signature
    comp_val = _run_sync(
        self._sequence_executor.create_value(comp_pb, comp_type))
    result_2_val = _run_sync(
        self._sequence_executor.create_call(comp_val, result_1_val))
    self.assertIsInstance(result_2_val, sequence_executor.SequenceExecutorValue)
    self.assertEqual(str(result_2_val.type_signature), 'int64')
    self.assertEqual(_run_sync(result_2_val.compute()), 9)


if __name__ == '__main__':
  absltest.main()
