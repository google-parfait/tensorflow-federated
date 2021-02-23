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

import asyncio
import collections
from absl.testing import absltest
from jax.lib.xla_bridge import xla_client
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.backends.xla import executor
from tensorflow_federated.python.core.backends.xla import xla_serialization


class ExecutorTest(absltest.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._backend = xla_client.get_local_backend()

  def test_to_representation_for_type_with_int32_constant(self):
    rep = executor.to_representation_for_type(10, np.int32)
    self.assertEqual(rep, 10)

  def test_to_representation_for_type_with_2xint32_list_constant(self):
    rep = executor.to_representation_for_type([10, 20], (np.int32, np.int32))
    self.assertIsInstance(rep, structure.Struct)
    self.assertEqual(str(rep), '<10,20>')

  def test_to_representation_for_type_with_nested_odict_constant(self):
    rep = executor.to_representation_for_type(
        collections.OrderedDict([('a', 10), ('b', [20, 30])]),
        collections.OrderedDict([('a', np.int32), ('b', [np.int32, np.int32])]))
    self.assertIsInstance(rep, structure.Struct)
    self.assertEqual(str(rep), '<a=10,b=<20,30>>')

  def test_to_representation_for_type_with_noarg_to_int32_comp(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Parameter(builder, 0, xla_client.shape_from_pyval(tuple()))
    xla_client.ops.Constant(builder, np.int32(10))
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType(None, np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [], comp_type)
    rep = executor.to_representation_for_type(comp_pb, comp_type, self._backend)
    self.assertTrue(callable(rep))
    result = rep()
    self.assertEqual(result, 10)

  def test_to_representation_for_type_with_noarg_to_2xint32_comp(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Parameter(builder, 0, xla_client.shape_from_pyval(tuple()))
    xla_client.ops.Tuple(builder, [
        xla_client.ops.Constant(builder, np.int32(10)),
        xla_client.ops.Constant(builder, np.int32(20))
    ])
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType(
        None, computation_types.StructType([('a', np.int32), ('b', np.int32)]))
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [0, 1], comp_type)
    rep = executor.to_representation_for_type(comp_pb, comp_type, self._backend)
    self.assertTrue(callable(rep))
    result = rep()
    self.assertEqual(str(result), '<a=10,b=20>')

  def test_to_representation_for_type_with_2xint32_to_int32_comp(self):
    builder = xla_client.XlaBuilder('comp')
    param = xla_client.ops.Parameter(
        builder, 0,
        xla_client.shape_from_pyval(tuple([np.array(0, dtype=np.int32)] * 2)))
    xla_client.ops.Add(
        xla_client.ops.GetTupleElement(param, 0),
        xla_client.ops.GetTupleElement(param, 1))
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType((np.int32, np.int32), np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [0, 1], comp_type)
    rep = executor.to_representation_for_type(comp_pb, comp_type, self._backend)
    self.assertTrue(callable(rep))
    result = rep(structure.Struct([(None, np.int32(20)), (None, np.int32(30))]))
    self.assertEqual(result, 50)

  def test_create_compute_int32(self):
    ex = executor.XlaExecutor()
    int_val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(10, np.int32))
    self.assertIsInstance(int_val, executor.XlaValue)
    self.assertEqual(str(int_val.type_signature), 'int32')
    self.assertIsInstance(int_val.internal_representation, np.int32)
    self.assertEqual(int_val.internal_representation, 10)
    result = asyncio.get_event_loop().run_until_complete(int_val.compute())
    self.assertEqual(result, 10)

  def test_create_compute_2xint32_struct(self):
    ex = executor.XlaExecutor()
    x_val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(10, np.int32))
    y_val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(20, np.int32))
    struct_val = asyncio.get_event_loop().run_until_complete(
        ex.create_struct([x_val, y_val]))
    self.assertIsInstance(struct_val, executor.XlaValue)
    self.assertEqual(str(struct_val.type_signature), '<int32,int32>')
    self.assertIsInstance(struct_val.internal_representation, structure.Struct)
    self.assertEqual(str(struct_val.internal_representation), '<10,20>')
    result = asyncio.get_event_loop().run_until_complete(struct_val.compute())
    self.assertEqual(str(result), '<10,20>')

  def test_create_and_invoke_noarg_comp_returning_int32(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Parameter(builder, 0, xla_client.shape_from_pyval(tuple()))
    xla_client.ops.Constant(builder, np.int32(10))
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType(None, np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [], comp_type)
    ex = executor.XlaExecutor()
    comp_val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(comp_pb, comp_type))
    self.assertIsInstance(comp_val, executor.XlaValue)
    self.assertEqual(str(comp_val.type_signature), str(comp_type))
    self.assertTrue(callable(comp_val.internal_representation))
    result = comp_val.internal_representation()
    self.assertEqual(result, 10)
    call_val = asyncio.get_event_loop().run_until_complete(
        ex.create_call(comp_val))
    self.assertIsInstance(call_val, executor.XlaValue)
    self.assertEqual(str(call_val.type_signature), 'int32')
    result = asyncio.get_event_loop().run_until_complete(call_val.compute())
    self.assertEqual(result, 10)

  def test_add_numbers(self):
    builder = xla_client.XlaBuilder('comp')
    param = xla_client.ops.Parameter(
        builder, 0,
        xla_client.shape_from_pyval(tuple([np.array(0, dtype=np.int32)] * 2)))
    xla_client.ops.Add(
        xla_client.ops.GetTupleElement(param, 0),
        xla_client.ops.GetTupleElement(param, 1))
    xla_comp = builder.build()
    comp_type = computation_types.FunctionType((np.int32, np.int32), np.int32)
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [0, 1], comp_type)
    ex = executor.XlaExecutor()

    async def _compute_fn():
      comp_val = await ex.create_value(comp_pb, comp_type)
      x_val = await ex.create_value(20, np.int32)
      y_val = await ex.create_value(30, np.int32)
      arg_val = await ex.create_struct([x_val, y_val])
      call_val = await ex.create_call(comp_val, arg_val)
      return await call_val.compute()

    result = asyncio.get_event_loop().run_until_complete(_compute_fn())
    self.assertEqual(result, 50)

  def test_selection(self):
    ex = executor.XlaExecutor()
    struct_val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(
            collections.OrderedDict([('a', 10), ('b', 20)]),
            computation_types.StructType([('a', np.int32), ('b', np.int32)])))
    self.assertIsInstance(struct_val, executor.XlaValue)
    self.assertEqual(str(struct_val.type_signature), '<a=int32,b=int32>')
    by_index_val = asyncio.get_event_loop().run_until_complete(
        ex.create_selection(struct_val, index=0))
    self.assertEqual(by_index_val.internal_representation, 10)
    by_name_val = asyncio.get_event_loop().run_until_complete(
        ex.create_selection(struct_val, name='b'))
    self.assertEqual(by_name_val.internal_representation, 20)


if __name__ == '__main__':
  absltest.main()
