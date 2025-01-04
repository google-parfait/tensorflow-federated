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
import federated_language
from federated_language.proto import computation_pb2 as pb
import jax
import jax.numpy as jnp
import numpy as np

from google.protobuf import any_pb2
from tensorflow_federated.python.core.environments.xla_backend import xla_serialization


def _make_xla_shape(shapes_and_dtypes_pytree):
  xla_computation = (
      jax.jit(
          lambda: jax.tree_util.tree_map(
              lambda shape_dtype: jnp.zeros(
                  shape_dtype.shape, shape_dtype.dtype
              ),
              shapes_and_dtypes_pytree,
          )
      )
      .lower()
      .compiler_ir('hlo')
  )
  return xla_computation.program_shape().result_shape()


def _make_test_xla_comp_noarg_to_int32():
  return jax.jit(lambda: jnp.int32(10)).lower().compiler_ir('hlo')


def _make_test_xla_comp_int32x10_to_int32x10():
  return (
      jax.jit(lambda x: jnp.zeros((10,), dtype=jnp.int32), keep_unused=True)
      .lower(jnp.zeros((10,), jnp.int32))
      .compiler_ir('hlo')
  )


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
        xla_comp, [], federated_language.FunctionType(None, np.int32)
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertEqual(str(type_spec), '( -> int32)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn(
        'ROOT constant.1 = s32[] constant(10)', xla_comp.as_hlo_text()
    )
    self.assertEqual(str(comp_pb.xla.parameter), '')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_create_xla_tff_computation_raises_missing_arg_in_xla(self):
    xla_comp = _make_test_xla_comp_noarg_to_int32()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp, [0], federated_language.FunctionType(np.int32, np.int32)
      )

  def test_create_xla_tff_computation_raises_missing_arg_in_type_spec(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp, [], federated_language.FunctionType(None, np.int32)
      )

  def test_create_xla_tff_computation_raises_arg_type_mismatch(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp,
          [0],
          federated_language.FunctionType(np.int32, (np.int32, (10,))),
      )

  def test_create_xla_tff_computation_raises_result_type_mismatch(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    with self.assertRaises(ValueError):
      xla_serialization.create_xla_tff_computation(
          xla_comp,
          [0],
          federated_language.FunctionType((np.int32, (10,)), np.int32),
      )

  def test_create_xla_tff_computation_int32x10_to_int32x10(self):
    xla_comp = _make_test_xla_comp_int32x10_to_int32x10()
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp,
        [0],
        federated_language.FunctionType((np.int32, (10,)), (np.int32, (10,))),
    )
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = federated_language.FunctionType.from_proto(comp_pb.type)
    self.assertEqual(str(type_spec), '(int32[10] -> int32[10])')

  def test_create_xla_tff_computation_with_reordered_tensor_indexes(self):

    def dot(x, y):
      return jnp.dot(x, y)

    xla_comp = (
        jax.jit(dot)
        .lower(jnp.zeros((10, 1), jnp.int32), jnp.zeros((1, 20), jnp.int32))
        .compiler_ir('hlo')
    )
    comp_pb_1 = xla_serialization.create_xla_tff_computation(
        xla_comp,
        [0, 1],
        federated_language.FunctionType(
            ((np.int32, (10, 1)), (np.int32, (1, 20))),
            (np.int32, (10, 20)),
        ),
    )
    self.assertIsInstance(comp_pb_1, pb.Computation)
    self.assertEqual(comp_pb_1.WhichOneof('computation'), 'xla')
    type_spec_1 = federated_language.FunctionType.from_proto(comp_pb_1.type)
    self.assertEqual(
        str(type_spec_1), '(<int32[10,1],int32[1,20]> -> int32[10,20])'
    )
    comp_pb_2 = xla_serialization.create_xla_tff_computation(
        xla_comp,
        [1, 0],
        federated_language.FunctionType(
            ((np.int32, (1, 20)), (np.int32, (10, 1))),
            (np.int32, (10, 20)),
        ),
    )
    self.assertIsInstance(comp_pb_2, pb.Computation)
    self.assertEqual(comp_pb_2.WhichOneof('computation'), 'xla')
    type_spec_2 = federated_language.FunctionType.from_proto(comp_pb_2.type)
    self.assertEqual(
        str(type_spec_2), '(<int32[1,20],int32[10,1]> -> int32[10,20])'
    )

  def test_flatten_xla_tensor_shape(self):
    tensor_shape = _make_xla_shape(
        jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.int32)
    )
    flattened = xla_serialization.flatten_xla_shape(tensor_shape)
    self.assertIsInstance(flattened, list)
    self.assertEqual(flattened, [tensor_shape])

  def test_flatten_xla_tuple_shape(self):
    tensor_shape_1 = jax.ShapeDtypeStruct(shape=(10,), dtype=np.int32)
    tensor_shape_2 = jax.ShapeDtypeStruct(shape=(20,), dtype=np.float32)
    tuple_shape = _make_xla_shape((tensor_shape_1, tensor_shape_2))
    flattened = xla_serialization.flatten_xla_shape(tuple_shape)
    self.assertIsInstance(flattened, list)
    self.assertEqual(
        flattened,
        [_make_xla_shape(tensor_shape_1), _make_xla_shape(tensor_shape_2)],
    )

  def test_xla_shapes_and_binding_to_tff_type_with_tensor(self):
    tensor_shape = jax.ShapeDtypeStruct(shape=(10,), dtype=np.int32)
    xla_shapes = [_make_xla_shape(tensor_shape)]
    binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_shapes_and_binding_to_tff_type(
        xla_shapes, binding
    )
    self.assertEqual(str(tff_type), 'int32[10]')

  def test_xla_shapes_and_binding_to_tff_type_with_tuple(self):
    tensor_shape_1 = jax.ShapeDtypeStruct(shape=(10,), dtype=np.int32)
    tensor_shape_2 = jax.ShapeDtypeStruct(shape=(20,), dtype=np.float32)
    xla_shapes = [
        _make_xla_shape(tensor_shape_1),
        _make_xla_shape(tensor_shape_2),
    ]
    binding = pb.Xla.Binding(
        struct=pb.Xla.StructBinding(
            element=[
                pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=1)),
                pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0)),
            ]
        )
    )
    tff_type = xla_serialization.xla_shapes_and_binding_to_tff_type(
        xla_shapes, binding
    )
    self.assertEqual(str(tff_type), '<float32[20],int32[10]>')

  def test_xla_shapes_and_binding_to_tff_type_raises_unused_tensor(self):
    tensor_shape = jax.ShapeDtypeStruct(shape=(10,), dtype=np.int32)
    xla_shapes = [_make_xla_shape(tensor_shape)]
    binding = None
    with self.assertRaises(ValueError):
      xla_serialization.xla_shapes_and_binding_to_tff_type(xla_shapes, binding)

  def test_xla_shapes_and_binding_to_tff_type_raises_unused_element(self):
    tensor_shape_1 = jax.ShapeDtypeStruct(shape=(10,), dtype=np.int32)
    tensor_shape_2 = jax.ShapeDtypeStruct(shape=(20,), dtype=np.float32)
    xla_shapes = [
        _make_xla_shape(tensor_shape_1),
        _make_xla_shape(tensor_shape_2),
    ]
    binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=1))
    with self.assertRaises(ValueError):
      xla_serialization.xla_shapes_and_binding_to_tff_type(xla_shapes, binding)

  def test_xla_computation_and_bindings_to_tff_type_none_binding_to_int32(self):
    xla_computation = _make_test_xla_comp_noarg_to_int32()
    parameter_binding = None
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding
    )
    self.assertEqual(str(tff_type), '( -> int32)')

  def test_xla_computation_and_bindings_to_tff_type_noarg_to_int32(self):
    xla_computation = _make_test_xla_comp_noarg_to_int32()
    parameter_binding = pb.Xla.Binding()
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding
    )
    self.assertEqual(str(tff_type), '( -> int32)')

  def test_xla_computation_and_bindings_to_tff_type_empty_tuple_to_int32(self):

    def empty_tuple_to_int32(x):
      del x  # Unused.
      return jnp.int32(10)

    xla_computation = (
        jax.jit(empty_tuple_to_int32, keep_unused=True)
        .lower(())
        .compiler_ir('hlo')
    )
    parameter_binding = pb.Xla.Binding()
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding
    )
    self.assertEqual(str(tff_type), '( -> int32)')

  def test_xla_computation_and_bindings_to_tff_type_int32_tuple_to_int32(self):

    def tuple_int_to_int(x):
      return x[0]

    xla_computation = (
        jax.jit(tuple_int_to_int)
        .lower((jax.ShapeDtypeStruct(shape=(10,), dtype=np.int32),))
        .compiler_ir('hlo')
    )
    parameter_binding = pb.Xla.Binding(
        struct=pb.Xla.StructBinding(
            element=[pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))]
        )
    )
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    tff_type = xla_serialization.xla_computation_and_bindings_to_tff_type(
        xla_computation, parameter_binding, result_binding
    )
    self.assertEqual(str(tff_type), '(<int32[10]> -> int32[10])')

  def test_xla_computation_and_bindings_to_tff_type_raises_noarg(self):
    xla_computation = _make_test_xla_comp_int32x10_to_int32x10()
    parameter_binding = None
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    with self.assertRaises(ValueError):
      xla_serialization.xla_computation_and_bindings_to_tff_type(
          xla_computation, parameter_binding, result_binding
      )

  def test_xla_computation_and_bindings_to_tff_type_raises_unused_element(self):

    def unused_element(x):
      return x[0] + jnp.int32(10)

    xla_computation = (
        jax.jit(unused_element, keep_unused=True)
        .lower((
            jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.int32),
            jax.ShapeDtypeStruct(shape=(10,), dtype=jnp.int32),
        ))
        .compiler_ir('hlo')
    )
    parameter_binding = pb.Xla.Binding(
        struct=pb.Xla.StructBinding(
            element=[pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))]
        )
    )
    result_binding = pb.Xla.Binding(tensor=pb.Xla.TensorBinding(index=0))
    with self.assertRaises(ValueError):
      xla_serialization.xla_computation_and_bindings_to_tff_type(
          xla_computation, parameter_binding, result_binding
      )


if __name__ == '__main__':
  absltest.main()
