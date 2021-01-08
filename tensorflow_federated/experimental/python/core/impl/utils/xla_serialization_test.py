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

from absl.testing import absltest
from jax.lib.xla_bridge import xla_client
import numpy as np

from google.protobuf import any_pb2
from tensorflow_federated.experimental.python.core.impl.utils import xla_serialization
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


def _make_test_xla_comp_noarg_to_int32():
  builder = xla_client.XlaBuilder('comp')
  xla_client.ops.Constant(builder, np.int32(10))
  return builder.build()


def _make_test_xla_comp_int32x10_to_int32x10():
  builder = xla_client.XlaBuilder('comp')
  tensor_shape = xla_client.Shape.array_shape(
      xla_client.dtype_to_etype(np.int32), (10,))
  param = xla_client.ops.Parameter(builder, 0, tensor_shape)
  constant = xla_client.ops.Constant(builder, np.zeros((10,), dtype=np.int32))
  xla_client.ops.Add(param, constant)
  return builder.build()


class XlaUtilsTest(absltest.TestCase):

  def test_pack_xla_computation(self):
    xla_comp = _make_test_xla_comp_noarg_to_int32()
    any_pb = xla_serialization.pack_xla_computation(xla_comp)
    self.assertEqual(type(any_pb), any_pb2.Any)

  def test_pack_unpack_xla_computation_roundtrip(self):
    xla_comp = _make_test_xla_comp_noarg_to_int32()
    any_pb = xla_serialization.pack_xla_computation(xla_comp)
    new_comp = xla_serialization.unpack_xla_computation(any_pb)
    self.assertEqual(new_comp.as_hlo_text(), xla_comp.as_hlo_text())

  def test_create_xla_tff_computation_noarg(self):
    xla_comp = _make_test_xla_comp_noarg_to_int32()
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [], computation_types.FunctionType(None, np.int32))
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    self.assertEqual(str(type_spec), '( -> int32)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn('ROOT constant.1 = s32[] constant(10)',
                  xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.parameter), '')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n' '  index: 0\n' '}\n')

  def test_create_xla_tff_computation_raises_missing_arg_in_xla(self):
    xla_comp = _make_test_xla_comp_noarg_to_int32()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp, [0], computation_types.FunctionType(np.int32, np.int32))

  def test_create_xla_tff_computation_raises_missing_arg_in_type_spec(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp, [], computation_types.FunctionType(None, np.int32))

  def test_create_xla_tff_computation_raises_arg_type_mismatch(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp, [0],
          computation_types.FunctionType(np.int32, (np.int32, (10,))))

  def test_create_xla_tff_computation_raises_result_type_mismatch(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp, [0],
          computation_types.FunctionType((np.int32, (10,)), np.int32))

  def test_create_xla_tff_computation_int32x10_to_int32x10(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, [0],
        computation_types.FunctionType((np.int32, (10,)), (np.int32, (10,))))
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    self.assertEqual(str(type_spec), '(int32[10] -> int32[10])')

  def test_create_xla_tff_computation_with_reordered_tensor_indexes(self):
    builder = xla_client.XlaBuilder('comp')
    tensor_shape_1 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10, 1))
    param_1 = xla_client.ops.Parameter(builder, 0, tensor_shape_1)
    tensor_shape_2 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (1, 20))
    param_2 = xla_client.ops.Parameter(builder, 1, tensor_shape_2)
    xla_client.ops.Dot(param_1, param_2)
    xla_comp = builder.build()
    comp_pb_1 = xla_serialization.create_xla_tff_computation(
        xla_comp, [0, 1],
        computation_types.FunctionType(
            ((np.int32, (10, 1)), (np.int32, (1, 20))), (np.int32, (
                10,
                20,
            ))))
    self.assertIsInstance(comp_pb_1, pb.Computation)
    self.assertEqual(comp_pb_1.WhichOneof('computation'), 'xla')
    type_spec_1 = type_serialization.deserialize_type(comp_pb_1.type)
    self.assertEqual(
        str(type_spec_1), '(<int32[10,1],int32[1,20]> -> int32[10,20])')
    comp_pb_2 = xla_serialization.create_xla_tff_computation(
        xla_comp, [1, 0],
        computation_types.FunctionType(
            ((np.int32, (1, 20)), (np.int32, (10, 1))), (np.int32, (
                10,
                20,
            ))))
    self.assertIsInstance(comp_pb_2, pb.Computation)
    self.assertEqual(comp_pb_2.WhichOneof('computation'), 'xla')
    type_spec_2 = type_serialization.deserialize_type(comp_pb_2.type)
    self.assertEqual(
        str(type_spec_2), '(<int32[1,20],int32[10,1]> -> int32[10,20])')

  def test_flatten_xla_tensor_shape(self):
    tensor_shape = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    flattened = xla_serialization.flatten_xla_shape(tensor_shape)
    self.assertIsInstance(flattened, list)
    self.assertListEqual(flattened, [tensor_shape])

  def test_flatten_xla_tuple_shape(self):
    tensor_shape_1 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    tensor_shape_2 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.float32), (20,))
    tuple_shape = xla_client.Shape.tuple_shape([tensor_shape_1, tensor_shape_2])
    flattened = xla_serialization.flatten_xla_shape(tuple_shape)
    self.assertIsInstance(flattened, list)
    self.assertListEqual(flattened, [tensor_shape_1, tensor_shape_2])

  def test_xla_shapes_and_binding_to_tff_type_with_tensor(self):
    tensor_shape = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    xla_shapes = [tensor_shape]
    binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_shapes_and_binding_to_tff_type(
        xla_shapes, binding)
    self.assertEqual(str(tff_type), 'int32[10]')

  def test_xla_shapes_and_binding_to_tff_type_with_tuple(self):
    tensor_shape_1 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    tensor_shape_2 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.float32), (20,))
    xla_shapes = [tensor_shape_1, tensor_shape_2]
    binding = pb.Xla.Binding(
        struct=pb.Xla.StructBinding(element=[
            pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=1)),
            pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
        ]))
    tff_type = xla_serialization.xla_shapes_and_binding_to_tff_type(
        xla_shapes, binding)
    self.assertEqual(str(tff_type), '<float32[20],int32[10]>')

  def test_xla_shapes_and_binding_to_tff_type_raises_unused_tensor(self):
    tensor_shape = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    xla_shapes = [tensor_shape]
    binding = None
    with self.assertRaises(ValueError):
      xla_serialization.xla_shapes_and_binding_to_tff_type(xla_shapes, binding)

  def test_xla_shapes_and_binding_to_tff_type_raises_unused_element(self):
    tensor_shape_1 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    tensor_shape_2 = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.float32), (20,))
    xla_shapes = [tensor_shape_1, tensor_shape_2]
    binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=1))
    with self.assertRaises(ValueError):
      xla_serialization.xla_shapes_and_binding_to_tff_type(xla_shapes, binding)

  def test_xla_computation_and_bindings_to_tff_type_none_binding_to_int32(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Constant(builder, np.int32(10))
    xla_computation = builder.build()
    parameter_binding = None
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding)
    self.assertEqual(str(tff_type), '( -> int32)')

  def test_xla_computation_and_bindings_to_tff_type_noarg_to_int32(self):
    xla_computation = _make_test_xla_comp_noarg_to_int32()
    parameter_binding = pb.Xla.Binding()
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding)
    self.assertEqual(str(tff_type), '( -> int32)')

  def test_xla_computation_and_bindings_to_tff_type_empty_tuple_to_int32(self):
    builder = xla_client.XlaBuilder('comp')
    xla_client.ops.Parameter(builder, 0, xla_client.Shape.tuple_shape([]))
    xla_client.ops.Constant(builder, np.int32(10))
    xla_computation = builder.build()
    parameter_binding = pb.Xla.Binding()
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding)
    self.assertEqual(str(tff_type), '( -> int32)')

  def test_xla_computation_and_bindings_to_tff_type_int32_tuple_to_int32(self):
    builder = xla_client.XlaBuilder('comp')
    tensor_shape = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    tuple_shape = xla_client.Shape.tuple_shape([tensor_shape])
    param = xla_client.ops.Parameter(builder, 0, tuple_shape)
    constant = xla_client.ops.Constant(builder, np.zeros((10,), dtype=np.int32))
    xla_client.ops.Add(xla_client.ops.GetTupleElement(param, 0), constant)
    xla_computation = builder.build()
    parameter_binding = pb.Xla.Binding(
        struct=pb.Xla.StructBinding(
            element=[pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))]))
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding)
    self.assertEqual(str(tff_type), '(<int32[10]> -> int32[10])')

  def test_xla_computation_and_bindings_to_tff_type_raises_noarg(self):
    xla_computation = _make_test_xla_comp_int32x10_to_int32x10()
    parameter_binding = None
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    with self.assertRaises(ValueError):
      xla_serialization.xla_computation_and_bindings_to_tff_type(
          xla_computation, parameter_binding, result_binding)

  def test_xla_computation_and_bindings_to_tff_type_raises_unused_element(self):
    builder = xla_client.XlaBuilder('comp')
    tensor_shape = xla_client.Shape.array_shape(
        xla_client.dtype_to_etype(np.int32), (10,))
    tuple_shape = xla_client.Shape.tuple_shape([tensor_shape, tensor_shape])
    param = xla_client.ops.Parameter(builder, 0, tuple_shape)
    constant = xla_client.ops.Constant(builder, np.zeros((10,), dtype=np.int32))
    xla_client.ops.Add(xla_client.ops.GetTupleElement(param, 0), constant)
    xla_computation = builder.build()
    parameter_binding = pb.Xla.Binding(
        struct=pb.Xla.StructBinding(
            element=[pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))]))
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    with self.assertRaises(ValueError):
      xla_serialization.xla_computation_and_bindings_to_tff_type(
          xla_computation, parameter_binding, result_binding)


if __name__ == '__main__':
  absltest.main()
