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

# Dependency imports
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple

from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import computation_building_blocks as bb
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import type_serialization


class ComputationBuildingBlocksTest(absltest.TestCase):

  def test_basic_functionality_of_reference_class(self):
    x = bb.Reference('foo', tf.int32)
    self.assertEqual(x.name, 'foo')
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), 'Reference(\'foo\', TensorType(tf.int32))')
    self.assertEqual(str(x), 'foo')
    x_proto = x.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(x_proto.type)),
        str(x.type_signature))
    self.assertEqual(x_proto.WhichOneof('computation'), 'reference')
    self.assertEqual(x_proto.reference.name, x.name)
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_selection_class(self):
    x = bb.Reference('foo', [('bar', tf.int32), ('baz', tf.bool)])
    y = bb.Selection(x, name='bar')
    self.assertEqual(y.name, 'bar')
    self.assertEqual(y.index, None)
    self.assertEqual(str(y.type_signature), 'int32')
    self.assertEqual(
        repr(y), 'Selection(Reference(\'foo\', NamedTupleType(['
        '(\'bar\', TensorType(tf.int32)), (\'baz\', TensorType(tf.bool))]))'
        ', name=\'bar\')')
    self.assertEqual(str(y), 'foo.bar')
    z = bb.Selection(x, name='baz')
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(str(z), 'foo.baz')
    with self.assertRaises(ValueError):
      _ = bb.Selection(x, name='bak')
    x0 = bb.Selection(x, index=0)
    self.assertEqual(x0.name, None)
    self.assertEqual(x0.index, 0)
    self.assertEqual(str(x0.type_signature), 'int32')
    self.assertEqual(
        repr(x0), 'Selection(Reference(\'foo\', NamedTupleType(['
        '(\'bar\', TensorType(tf.int32)), (\'baz\', TensorType(tf.bool))]))'
        ', index=0)')
    self.assertEqual(str(x0), 'foo[0]')
    x1 = bb.Selection(x, index=1)
    self.assertEqual(str(x1.type_signature), 'bool')
    self.assertEqual(str(x1), 'foo[1]')
    with self.assertRaises(ValueError):
      _ = bb.Selection(x, index=2)
    with self.assertRaises(ValueError):
      _ = bb.Selection(x, index=-1)
    y_proto = y.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(y_proto.type)),
        str(y.type_signature))
    self.assertEqual(y_proto.WhichOneof('computation'), 'selection')
    self.assertEqual(str(y_proto.selection.source), str(x.proto))
    self.assertEqual(str(y_proto.selection.name), 'bar')
    self._serialize_deserialize_roundtrip_test(y)
    self._serialize_deserialize_roundtrip_test(z)
    self._serialize_deserialize_roundtrip_test(x0)
    self._serialize_deserialize_roundtrip_test(x1)

  def test_basic_functionality_of_tuple_class(self):
    x = bb.Reference('foo', tf.int32)
    y = bb.Reference('bar', tf.bool)
    z = bb.Tuple([x, ('y', y)])
    with self.assertRaises(ValueError):
      _ = bb.Tuple([('', y)])
    self.assertIsInstance(z, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(z.type_signature), '<int32,y=bool>')
    self.assertEqual(
        repr(z),
        'Tuple([(None, Reference(\'foo\', TensorType(tf.int32))), (\'y\', '
        'Reference(\'bar\', TensorType(tf.bool)))])')
    self.assertEqual(str(z), '<foo,y=bar>')
    self.assertEqual(dir(z), ['y'])
    self.assertIs(z.y, y)
    self.assertLen(z, 2)
    self.assertIs(z[0], x)
    self.assertIs(z[1], y)
    self.assertEqual(','.join(str(e) for e in iter(z)), 'foo,bar')
    z_proto = z.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(z_proto.type)),
        str(z.type_signature))
    self.assertEqual(z_proto.WhichOneof('computation'), 'tuple')
    self.assertEqual([str(e.name) for e in z_proto.tuple.element], ['', 'y'])
    self._serialize_deserialize_roundtrip_test(z)

  def test_basic_functionality_of_call_class(self):
    x = bb.Reference('foo', types.FunctionType(tf.int32, tf.bool))
    y = bb.Reference('bar', tf.int32)
    z = bb.Call(x, y)
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertIs(z.function, x)
    self.assertIs(z.argument, y)
    self.assertEqual(
        repr(z), 'Call(Reference(\'foo\', '
        'FunctionType(TensorType(tf.int32), TensorType(tf.bool))), '
        'Reference(\'bar\', TensorType(tf.int32)))')
    self.assertEqual(str(z), 'foo(bar)')
    with self.assertRaises(TypeError):
      bb.Call(x)
    w = bb.Reference('bak', tf.float32)
    with self.assertRaises(TypeError):
      bb.Call(x, w)
    z_proto = z.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(z_proto.type)),
        str(z.type_signature))
    self.assertEqual(z_proto.WhichOneof('computation'), 'call')
    self.assertEqual(str(z_proto.call.function), str(x.proto))
    self.assertEqual(str(z_proto.call.argument), str(y.proto))
    self._serialize_deserialize_roundtrip_test(z)

  def test_basic_functionality_of_lambda_class(self):
    arg_name = 'arg'
    arg_type = [('f', types.FunctionType(tf.int32, tf.int32)), ('x', tf.int32)]
    arg = bb.Reference(arg_name, arg_type)
    arg_f = bb.Selection(arg, name='f')
    arg_x = bb.Selection(arg, name='x')
    x = bb.Lambda(arg_name, arg_type, bb.Call(arg_f, bb.Call(arg_f, arg_x)))
    self.assertEqual(
        str(x.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')
    self.assertEqual(x.parameter_name, arg_name)
    self.assertEqual(str(x.parameter_type), '<f=(int32 -> int32),x=int32>')
    self.assertEqual(str(x.result), 'arg.f(arg.f(arg.x))')
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
    self.assertEqual(str(x), '(arg -> arg.f(arg.f(arg.x)))')
    x_proto = x.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(x_proto.type)),
        str(x.type_signature))
    self.assertEqual(x_proto.WhichOneof('computation'), 'lambda')
    self.assertEqual(str(getattr(x_proto, 'lambda').parameter_name), arg_name)
    self.assertEqual(
        str(getattr(x_proto, 'lambda').result), str(x.result.proto))
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_block_class(self):
    x = bb.Block(
        [('x', bb.Reference('arg', (tf.int32, tf.int32))),
         ('y', bb.Selection(bb.Reference('x', (tf.int32, tf.int32)), index=0))],
        bb.Reference('y', tf.int32))
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual([(k, str(v)) for k, v in x.locals], [('x', 'arg'),
                                                          ('y', 'x[0]')])
    self.assertEqual(str(x.result), 'y')
    self.assertEqual(
        repr(x), 'Block([(\'x\', Reference(\'arg\', '
        'NamedTupleType([TensorType(tf.int32), TensorType(tf.int32)]))), '
        '(\'y\', Selection(Reference(\'x\', '
        'NamedTupleType([TensorType(tf.int32), TensorType(tf.int32)])), '
        'index=0))], '
        'Reference(\'y\', TensorType(tf.int32)))')
    self.assertEqual(str(x), '(let x=arg,y=x[0] in y)')
    x_proto = x.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(x_proto.type)),
        str(x.type_signature))
    self.assertEqual(x_proto.WhichOneof('computation'), 'block')
    self.assertEqual(str(x_proto.block.result), str(x.result.proto))
    for idx, loc_proto in enumerate(x_proto.block.local):
      loc_name, loc_value = x.locals[idx]
      self.assertEqual(str(loc_proto.name), loc_name)
      self.assertEqual(str(loc_proto.value), str(loc_value.proto))
      self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_intrinsic_class(self):
    x = bb.Intrinsic('add_one', types.FunctionType(tf.int32, tf.int32))
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(x.uri, 'add_one')
    self.assertEqual(
        repr(x), 'Intrinsic(\'add_one\', '
        'FunctionType(TensorType(tf.int32), TensorType(tf.int32)))')
    self.assertEqual(str(x), 'add_one')
    x_proto = x.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(x_proto.type)),
        str(x.type_signature))
    self.assertEqual(x_proto.WhichOneof('computation'), 'intrinsic')
    self.assertEqual(str(x_proto.intrinsic.uri), x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_data_class(self):
    x = bb.Data('/tmp/mydata', types.SequenceType(tf.int32))
    self.assertEqual(str(x.type_signature), 'int32*')
    self.assertEqual(x.uri, '/tmp/mydata')
    self.assertEqual(
        repr(x), 'Data(\'/tmp/mydata\', SequenceType(TensorType(tf.int32)))')
    self.assertEqual(str(x), '/tmp/mydata')
    x_proto = x.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(x_proto.type)),
        str(x.type_signature))
    self.assertEqual(x_proto.WhichOneof('computation'), 'data')
    self.assertEqual(str(x_proto.data.uri), x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_compiled_computation_class(self):
    comp = tensorflow_serialization.serialize_py_func_as_tf_computation(
        lambda x: x + 3, tf.int32)
    x = bb.CompiledComputation(comp)
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(str(x.proto), str(comp))
    self.assertTrue(
        re.match(
            r'CompiledComputation\([0-9a-f]+, '
            r'FunctionType\(TensorType\(tf\.int32\), '
            r'TensorType\(tf\.int32\)\)\)', repr(x)))
    self.assertTrue(re.match(r'comp#[0-9a-f]+', str(x)))
    y = bb.CompiledComputation(comp, name='foo')
    self.assertEqual(str(y), 'comp#foo')
    self._serialize_deserialize_roundtrip_test(x)

  def test_basic_functionality_of_placement_class(self):
    x = bb.Placement(placements.CLIENTS)
    self.assertEqual(str(x.type_signature), 'placement')
    self.assertEqual(x.uri, 'clients')
    self.assertEqual(repr(x), 'Placement(\'clients\')')
    self.assertEqual(str(x), 'CLIENTS')
    x_proto = x.proto
    self.assertEqual(
        str(type_serialization.deserialize_type(x_proto.type)),
        str(x.type_signature))
    self.assertEqual(x_proto.WhichOneof('computation'), 'placement')
    self.assertEqual(str(x_proto.placement.uri), x.uri)
    self._serialize_deserialize_roundtrip_test(x)

  def _serialize_deserialize_roundtrip_test(self, target):
    """Performs roundtrip serialization/deserialization of the given target.

    Args:
      target: An instane of ComputationBuildingBlock to serialize-deserialize.
    """
    assert isinstance(target, bb.ComputationBuildingBlock)
    proto = target.proto
    target2 = bb.ComputationBuildingBlock.from_proto(proto)
    proto2 = target2.proto
    self.assertEqual(str(target), str(target2))
    self.assertEqual(str(proto), str(proto2))


if __name__ == '__main__':
  absltest.main()
