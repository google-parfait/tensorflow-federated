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
import jax
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.jax_context import jax_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.impl.xla_context import xla_serialization


# Convenience aliases.
FunctionType = computation_types.FunctionType
StructType = computation_types.StructType
TensorType = computation_types.TensorType


class JaxSerializationTest(absltest.TestCase):

  def test_serialize_jax_with_noarg_to_int32(self):

    def traced_fn():
      return 10

    param_type = None
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=param_type, result=np.int32))
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertNotEmpty(xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.parameter), '')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_serialize_jax_with_int32_to_int32(self):

    def traced_fn(x):
      return x + 10

    param_type = computation_types.to_type(np.int32)
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=param_type, result=np.int32))
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertNotEmpty(xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.result), str(comp_pb.xla.parameter))
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n  index: 0\n}\n')

  def test_serialize_jax_with_2xint32_to_2xint32(self):

    def traced_fn(x):
      return collections.OrderedDict(
          sum=x['foo'] + x['bar'], difference=x['bar'] - x['foo'])

    param_type = computation_types.to_type(
        collections.OrderedDict(foo=np.int32, bar=np.int32))
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec,
        FunctionType(
            parameter=param_type,
            result=StructType([('sum', np.int32), ('difference', np.int32)])))
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertNotEmpty(xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.result), str(comp_pb.xla.parameter))
    self.assertEqual(
        str(comp_pb.xla.parameter), 'struct {\n'
        '  element {\n'
        '    tensor {\n'
        '      index: 0\n'
        '    }\n'
        '  }\n'
        '  element {\n'
        '    tensor {\n'
        '      index: 1\n'
        '    }\n'
        '  }\n'
        '}\n')

  def test_serialize_jax_with_two_args(self):

    def traced_fn(x, y):
      return x + y

    param_type = computation_types.to_type(
        collections.OrderedDict(x=np.int32, y=np.int32))
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=param_type, result=np.int32))
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertNotEmpty(xla_comp.as_hlo_text())
    self.assertEqual(
        str(comp_pb.xla.parameter), 'struct {\n'
        '  element {\n'
        '    tensor {\n'
        '      index: 0\n'
        '    }\n'
        '  }\n'
        '  element {\n'
        '    tensor {\n'
        '      index: 1\n'
        '    }\n'
        '  }\n'
        '}\n')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n' '  index: 0\n' '}\n')

  def test_serialize_jax_with_nested_struct_arg(self):

    def traced_fn(x, y):
      return x[0] + y

    param_type = computation_types.StructType([
        (None, computation_types.StructType([(None, np.int32)])),
        (None, np.int32)
    ])
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=param_type, result=np.int32))

  def test_nested_structure_type_signature_roundtrip(self):

    def traced_fn(x):
      return x[0][0]

    param_type = computation_types.to_type([(np.int32,)])
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=param_type, result=np.int32))

  def test_arg_ordering(self):
    param_type = computation_types.to_type(
        (computation_types.TensorType(np.int32, 10),
         computation_types.TensorType(np.int32)))

    def traced_fn(b, a):
      return jax.numpy.add(a, jax.numpy.sum(b))

    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=param_type, result=np.int32))

  def test_tracing_with_float64_input(self):
    self.skipTest('b/237566862')
    param_type = computation_types.TensorType(np.float64)
    identity_fn = lambda x: x
    arg_fn = function_utils.create_argument_unpacking_fn(
        identity_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        identity_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    type_test_utils.assert_types_equivalent(
        type_spec, FunctionType(parameter=np.float64, result=np.float64))


class StructPytreeTest(absltest.TestCase):

  def test_named_struct(self):
    struct = structure.Struct.named(a=1, b=2, c=3)
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 2, 3])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_unnamed_struct(self):
    struct = structure.Struct.unnamed(1, 'a', 2)
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 'a', 2])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_mixed_named_struct(self):
    struct = structure.Struct([('a', 1), (None, 2), ('b', 3), (None, 4)])
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 2, 3, 4])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_nested_structs(self):
    struct = structure.Struct.named(
        a=1,
        b=structure.Struct.named(c=4, d=structure.Struct.unnamed(5, 6), e=7))
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 4, 5, 6, 7])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)

  def test_mixed_nested_structs_and_python_containers(self):
    struct = structure.Struct.named(
        a=1, b=[2, structure.Struct.named(c=4, d=(5, 6), e=7), 3])
    children, aux_data = jax.tree_util.tree_flatten(struct)
    self.assertEqual(children, [1, 2, 4, 5, 6, 7, 3])
    new_struct = jax.tree_util.tree_unflatten(aux_data, children)
    self.assertEqual(struct, new_struct)


if __name__ == '__main__':
  absltest.main()
