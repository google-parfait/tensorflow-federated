# Lint as: python3
# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for values.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import type_serialization


class ComputationBuildingBlocksTest(absltest.TestCase):

  def test_basic_functionality_of_reference_class(self):
    x = computation_building_blocks.Reference('foo', tf.int32)
    self.assertEqual(x.name, 'foo')
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), 'Reference(\'foo\', TensorType(tf.int32))')
    self.assertEqual(x.tff_repr, 'foo')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'reference')
    self.assertEqual(x_proto.reference.name, x.name)
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_selection_class(self):
    x = computation_building_blocks.Reference('foo', [('bar', tf.int32),
                                                      ('baz', tf.bool)])
    y = computation_building_blocks.Selection(x, name='bar')
    self.assertEqual(y.name, 'bar')
    self.assertEqual(y.index, None)
    self.assertEqual(str(y.type_signature), 'int32')
    self.assertEqual(
        repr(y), 'Selection(Reference(\'foo\', NamedTupleType(['
        '(\'bar\', TensorType(tf.int32)), (\'baz\', TensorType(tf.bool))]))'
        ', name=\'bar\')')
    self.assertEqual(y.tff_repr, 'foo.bar')
    z = computation_building_blocks.Selection(x, name='baz')
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(z.tff_repr, 'foo.baz')
    with self.assertRaises(ValueError):
      _ = computation_building_blocks.Selection(x, name='bak')
    x0 = computation_building_blocks.Selection(x, index=0)
    self.assertEqual(x0.name, None)
    self.assertEqual(x0.index, 0)
    self.assertEqual(str(x0.type_signature), 'int32')
    self.assertEqual(
        repr(x0), 'Selection(Reference(\'foo\', NamedTupleType(['
        '(\'bar\', TensorType(tf.int32)), (\'baz\', TensorType(tf.bool))]))'
        ', index=0)')
    self.assertEqual(x0.tff_repr, 'foo[0]')
    x1 = computation_building_blocks.Selection(x, index=1)
    self.assertEqual(str(x1.type_signature), 'bool')
    self.assertEqual(x1.tff_repr, 'foo[1]')
    with self.assertRaises(ValueError):
      _ = computation_building_blocks.Selection(x, index=2)
    with self.assertRaises(ValueError):
      _ = computation_building_blocks.Selection(x, index=-1)
    y_proto = y.proto
    self.assertEqual(
        type_serialization.deserialize_type(y_proto.type), y.type_signature)
    self.assertEqual(y_proto.WhichOneof('computation'), 'selection')
    self.assertEqual(str(y_proto.selection.source), str(x.proto))
    self.assertEqual(y_proto.selection.name, 'bar')
    self._serialize_deserialize_roundtrip_test(y)
    self._serialize_deserialize_roundtrip_test(z)
    self._serialize_deserialize_roundtrip_test(x0)
    self._serialize_deserialize_roundtrip_test(x1)

  def test_basic_functionality_of_tuple_class(self):
    x = computation_building_blocks.Reference('foo', tf.int32)
    y = computation_building_blocks.Reference('bar', tf.bool)
    z = computation_building_blocks.Tuple([x, ('y', y)])
    with self.assertRaises(ValueError):
      _ = computation_building_blocks.Tuple([('', y)])
    self.assertIsInstance(z, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(z.type_signature), '<int32,y=bool>')
    self.assertEqual(
        repr(z),
        'Tuple([(None, Reference(\'foo\', TensorType(tf.int32))), (\'y\', '
        'Reference(\'bar\', TensorType(tf.bool)))])')
    self.assertEqual(z.tff_repr, '<foo,y=bar>')
    self.assertEqual(dir(z), ['y'])
    self.assertIs(z.y, y)
    self.assertLen(z, 2)
    self.assertIs(z[0], x)
    self.assertIs(z[1], y)
    self.assertEqual(','.join(e.tff_repr for e in iter(z)), 'foo,bar')
    z_proto = z.proto
    self.assertEqual(
        type_serialization.deserialize_type(z_proto.type), z.type_signature)
    self.assertEqual(z_proto.WhichOneof('computation'), 'tuple')
    self.assertEqual([e.name for e in z_proto.tuple.element], ['', 'y'])
    self._serialize_deserialize_roundtrip_test(z)

  def test_basic_functionality_of_call_class(self):
    x = computation_building_blocks.Reference(
        'foo', computation_types.FunctionType(tf.int32, tf.bool))
    y = computation_building_blocks.Reference('bar', tf.int32)
    z = computation_building_blocks.Call(x, y)
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertIs(z.function, x)
    self.assertIs(z.argument, y)
    self.assertEqual(
        repr(z), 'Call(Reference(\'foo\', '
        'FunctionType(TensorType(tf.int32), TensorType(tf.bool))), '
        'Reference(\'bar\', TensorType(tf.int32)))')
    self.assertEqual(z.tff_repr, 'foo(bar)')
    with self.assertRaises(TypeError):
      computation_building_blocks.Call(x)
    w = computation_building_blocks.Reference('bak', tf.float32)
    with self.assertRaises(TypeError):
      computation_building_blocks.Call(x, w)
    z_proto = z.proto
    self.assertEqual(
        type_serialization.deserialize_type(z_proto.type), z.type_signature)
    self.assertEqual(z_proto.WhichOneof('computation'), 'call')
    self.assertEqual(str(z_proto.call.function), str(x.proto))
    self.assertEqual(str(z_proto.call.argument), str(y.proto))
    self._serialize_deserialize_roundtrip_test(z)

  def test_basic_functionality_of_lambda_class(self):
    arg_name = 'arg'
    arg_type = [('f', computation_types.FunctionType(tf.int32, tf.int32)),
                ('x', tf.int32)]
    arg = computation_building_blocks.Reference(arg_name, arg_type)
    arg_f = computation_building_blocks.Selection(arg, name='f')
    arg_x = computation_building_blocks.Selection(arg, name='x')
    x = computation_building_blocks.Lambda(
        arg_name, arg_type,
        computation_building_blocks.Call(
            arg_f, computation_building_blocks.Call(arg_f, arg_x)))
    self.assertEqual(
        str(x.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')
    self.assertEqual(x.parameter_name, arg_name)
    self.assertEqual(str(x.parameter_type), '<f=(int32 -> int32),x=int32>')
    self.assertEqual(x.result.tff_repr, 'arg.f(arg.f(arg.x))')
    arg_type_repr = (
        'NamedTupleType(['
        '(\'f\', FunctionType(TensorType(tf.int32), TensorType(tf.int32))), '
        '(\'x\', TensorType(tf.int32))])')
    self.assertEqual(
        repr(x), 'Lambda(\'arg\', {0}, '
        'Call(Selection(Reference(\'arg\', {0}), name=\'f\'), '
        'Call(Selection(Reference(\'arg\', {0}), name=\'f\'), '
        'Selection(Reference(\'arg\', {0}), name=\'x\'))))'.format(
            arg_type_repr))
    self.assertEqual(x.tff_repr, '(arg -> arg.f(arg.f(arg.x)))')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'lambda')
    self.assertEqual(getattr(x_proto, 'lambda').parameter_name, arg_name)
    self.assertEqual(
        str(getattr(x_proto, 'lambda').result), str(x.result.proto))
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_block_class(self):
    x = computation_building_blocks.Block(
        [('x', computation_building_blocks.Reference('arg',
                                                     (tf.int32, tf.int32))),
         ('y',
          computation_building_blocks.Selection(
              computation_building_blocks.Reference('x', (tf.int32, tf.int32)),
              index=0))], computation_building_blocks.Reference('y', tf.int32))
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual([(k, v.tff_repr) for k, v in x.locals], [('x', 'arg'),
                                                              ('y', 'x[0]')])
    self.assertEqual(x.result.tff_repr, 'y')
    self.assertEqual(
        repr(x), 'Block([(\'x\', Reference(\'arg\', '
        'NamedTupleType([TensorType(tf.int32), TensorType(tf.int32)]))), '
        '(\'y\', Selection(Reference(\'x\', '
        'NamedTupleType([TensorType(tf.int32), TensorType(tf.int32)])), '
        'index=0))], '
        'Reference(\'y\', TensorType(tf.int32)))')
    self.assertEqual(x.tff_repr, '(let x=arg,y=x[0] in y)')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'block')
    self.assertEqual(str(x_proto.block.result), str(x.result.proto))
    for idx, loc_proto in enumerate(x_proto.block.local):
      loc_name, loc_value = x.locals[idx]
      self.assertEqual(loc_proto.name, loc_name)
      self.assertEqual(str(loc_proto.value), str(loc_value.proto))
      self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_intrinsic_class(self):
    x = computation_building_blocks.Intrinsic(
        'add_one', computation_types.FunctionType(tf.int32, tf.int32))
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(x.uri, 'add_one')
    self.assertEqual(
        repr(x), 'Intrinsic(\'add_one\', '
        'FunctionType(TensorType(tf.int32), TensorType(tf.int32)))')
    self.assertEqual(x.tff_repr, 'add_one')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'intrinsic')
    self.assertEqual(x_proto.intrinsic.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_intrinsic_functionality_plus_canonical_typecheck(self):
    x = computation_building_blocks.Intrinsic(
        'generic_plus',
        computation_types.FunctionType([tf.int32, tf.int32], tf.int32))
    self.assertEqual(str(x.type_signature), '(<int32,int32> -> int32)')
    self.assertEqual(x.uri, 'generic_plus')
    self.assertEqual(x.tff_repr, 'generic_plus')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'intrinsic')
    self.assertEqual(x_proto.intrinsic.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_intrinsic_class_fails_bad_type(self):
    with self.assertRaises(TypeError):
      _ = computation_building_blocks.Intrinsic(
          intrinsic_defs.GENERIC_PLUS.uri,
          computation_types.FunctionType([tf.int32, tf.int32], tf.float32))

  def test_intrinsic_class_fails_named_tuple_type_with_names(self):
    with self.assertRaises(TypeError):
      _ = computation_building_blocks.Intrinsic(
          intrinsic_defs.GENERIC_PLUS.uri,
          computation_types.FunctionType([('a', tf.int32), ('b', tf.int32)],
                                         tf.int32))

  def test_intrinsic_class_succeeds_simple_federated_map(self):
    simple_function = computation_types.FunctionType(tf.int32, tf.float32)
    federated_arg = computation_types.FederatedType(simple_function.parameter,
                                                    placements.CLIENTS)
    federated_result = computation_types.FederatedType(simple_function.result,
                                                       placements.CLIENTS)
    federated_map_concrete_type = computation_types.FunctionType(
        [simple_function, federated_arg], federated_result)
    concrete_federated_map = computation_building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri, federated_map_concrete_type)
    self.assertIsInstance(concrete_federated_map,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(
        str(concrete_federated_map.type_signature),
        '(<(int32 -> float32),{int32}@CLIENTS> -> {float32}@CLIENTS)')
    self.assertEqual(concrete_federated_map.uri, 'federated_map')
    self.assertEqual(concrete_federated_map.tff_repr, 'federated_map')
    concrete_federated_map_proto = concrete_federated_map.proto
    self.assertEqual(
        type_serialization.deserialize_type(concrete_federated_map_proto.type),
        concrete_federated_map.type_signature)
    self.assertEqual(
        concrete_federated_map_proto.WhichOneof('computation'), 'intrinsic')
    self.assertEqual(concrete_federated_map_proto.intrinsic.uri,
                     concrete_federated_map.uri)
    self._serialize_deserialize_roundtrip_test(concrete_federated_map)

  def test_basic_functionality_of_data_class(self):
    x = computation_building_blocks.Data(
        '/tmp/mydata', computation_types.SequenceType(tf.int32))
    self.assertEqual(str(x.type_signature), 'int32*')
    self.assertEqual(x.uri, '/tmp/mydata')
    self.assertEqual(
        repr(x), 'Data(\'/tmp/mydata\', SequenceType(TensorType(tf.int32)))')
    self.assertEqual(x.tff_repr, '/tmp/mydata')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'data')
    self.assertEqual(x_proto.data.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_compiled_computation_class(self):
    comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        lambda x: x + 3, tf.int32, context_stack_impl.context_stack)
    x = computation_building_blocks.CompiledComputation(comp)
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(str(x.proto), str(comp))
    self.assertTrue(
        re.match(
            r'CompiledComputation\([0-9a-f]+, '
            r'FunctionType\(TensorType\(tf\.int32\), '
            r'TensorType\(tf\.int32\)\)\)', repr(x)))
    self.assertTrue(re.match(r'comp#[0-9a-f]+', x.tff_repr))
    y = computation_building_blocks.CompiledComputation(comp, name='foo')
    self.assertEqual(y.tff_repr, 'comp#foo')
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_placement_class(self):
    x = computation_building_blocks.Placement(placements.CLIENTS)
    self.assertEqual(str(x.type_signature), 'placement')
    self.assertEqual(x.uri, 'clients')
    self.assertEqual(repr(x), 'Placement(\'clients\')')
    self.assertEqual(x.tff_repr, 'CLIENTS')
    x_proto = x.proto
    self.assertEqual(
        type_serialization.deserialize_type(x_proto.type), x.type_signature)
    self.assertEqual(x_proto.WhichOneof('computation'), 'placement')
    self.assertEqual(x_proto.placement.uri, x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def _serialize_deserialize_roundtrip_test(self, target):
    """Performs roundtrip serialization/deserialization of the given target.

    Args:
      target: An instane of ComputationBuildingBlock to serialize-deserialize.
    """
    assert isinstance(target,
                      computation_building_blocks.ComputationBuildingBlock)
    proto = target.proto
    target2 = computation_building_blocks.ComputationBuildingBlock.from_proto(
        proto)
    proto2 = target2.proto
    self.assertEqual(target.tff_repr, target2.tff_repr)
    self.assertEqual(str(proto), str(proto2))


if __name__ == '__main__':
  absltest.main()
