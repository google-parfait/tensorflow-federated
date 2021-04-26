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

from tensorflow_federated.experimental.python.core.impl.jax_context import jax_serialization
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.backends.xla import xla_serialization
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


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
    self.assertEqual(str(type_spec), '( -> int32)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn('ROOT tuple.4 = (s32[]) tuple(constant.3)',
                  xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.parameter), '')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n' '  index: 0\n' '}\n')

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
    self.assertEqual(str(type_spec), '(int32 -> int32)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn('ROOT tuple.6 = (s32[]) tuple(add.5)', xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.result), str(comp_pb.xla.parameter))
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n' '  index: 0\n' '}\n')

  def test_serialize_jax_with_2xint32_to_2xint32(self):

    def traced_fn(x):
      return collections.OrderedDict([('sum', x['foo'] + x['bar']),
                                      ('difference', x['bar'] - x['foo'])])

    param_type = computation_types.to_type(
        collections.OrderedDict([('foo', np.int32), ('bar', np.int32)]))
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    self.assertEqual(
        str(type_spec),
        '(<foo=int32,bar=int32> -> <sum=int32,difference=int32>)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn(
        # pylint: disable=line-too-long
        '  constant.4 = pred[] constant(false)\n'
        '  parameter.1 = (s32[], s32[]) parameter(0)\n'
        '  get-tuple-element.2 = s32[] get-tuple-element(parameter.1), index=0\n'
        '  get-tuple-element.3 = s32[] get-tuple-element(parameter.1), index=1\n'
        '  add.5 = s32[] add(get-tuple-element.2, get-tuple-element.3)\n'
        '  subtract.6 = s32[] subtract(get-tuple-element.3, get-tuple-element.2)\n'
        '  ROOT tuple.7 = (s32[], s32[]) tuple(add.5, subtract.6)\n',
        xla_comp.as_hlo_text())
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
        collections.OrderedDict([('x', np.int32), ('y', np.int32)]))
    arg_fn = function_utils.create_argument_unpacking_fn(traced_fn, param_type)
    ctx_stack = context_stack_impl.context_stack
    comp_pb = jax_serialization.serialize_jax_computation(
        traced_fn, arg_fn, param_type, ctx_stack)
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    self.assertEqual(str(type_spec), '(<x=int32,y=int32> -> int32)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn(
        # pylint: disable=line-too-long
        '  constant.4 = pred[] constant(false)\n'
        '  parameter.1 = (s32[], s32[]) parameter(0)\n'
        '  get-tuple-element.2 = s32[] get-tuple-element(parameter.1), index=0\n'
        '  get-tuple-element.3 = s32[] get-tuple-element(parameter.1), index=1\n'
        '  add.5 = s32[] add(get-tuple-element.2, get-tuple-element.3)\n'
        '  ROOT tuple.6 = (s32[]) tuple(add.5)\n',
        xla_comp.as_hlo_text())
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
    self.assertEqual(str(type_spec), '(<<int32>> -> int32)')

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
    self.assertEqual(str(type_spec), '(<int32[10],int32> -> int32)')


if __name__ == '__main__':
  absltest.main()
