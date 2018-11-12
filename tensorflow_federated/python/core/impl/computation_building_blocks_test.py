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
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import anonymous_tuple
from tensorflow_federated.python.core.impl import computation_building_blocks as bb
from tensorflow_federated.python.core.impl import tensorflow_serialization


class ComputationBuildingBlocksTest(unittest.TestCase):

  def test_basic_functionality_of_reference_class(self):
    x = bb.Reference('foo', tf.int32)
    self.assertEqual(x.name, 'foo')
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), 'Reference(\'foo\', TensorType(tf.int32))')
    self.assertEqual(str(x), 'foo')

  def test_basic_functionality_of_selection_class(self):
    x = bb.Reference('foo', [('bar', tf.int32), ('baz', tf.bool)])
    y = bb.Selection(x, name='bar')
    self.assertEqual(y.name, 'bar')
    self.assertEqual(y.index, None)
    self.assertEqual(str(y.type_signature), 'int32')
    self.assertEqual(
        repr(y),
        'Selection(Reference(\'foo\', NamedTupleType(['
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
        repr(x0),
        'Selection(Reference(\'foo\', NamedTupleType(['
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

  def test_basic_functionality_of_tuple_class(self):
    x = bb.Reference('foo', tf.int32)
    y = bb.Reference('bar', tf.bool)
    z = bb.Tuple([x, ('y', y)])
    self.assertIsInstance(z, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(z.type_signature), '<int32,y=bool>')
    self.assertEqual(
        repr(z),
        'Tuple([(None, Reference(\'foo\', TensorType(tf.int32))), (\'y\', '
        'Reference(\'bar\', TensorType(tf.bool)))])')
    self.assertEqual(str(z), '<foo,y=bar>')
    self.assertEqual(dir(z), ['y'])
    self.assertIs(z.y, y)
    self.assertEqual(len(z), 2)
    self.assertIs(z[0], x)
    self.assertIs(z[1], y)
    self.assertEqual(','.join(str(e) for e in iter(z)), 'foo,bar')

  def test_basic_functionality_of_call_class(self):
    x = bb.Reference('foo', types.FunctionType(tf.int32, tf.bool))
    y = bb.Reference('bar', tf.int32)
    z = bb.Call(x, y)
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertIs(z.function, x)
    self.assertIs(z.argument, y)
    self.assertEqual(
        repr(z),
        'Call(Reference(\'foo\', '
        'FunctionType(TensorType(tf.int32), TensorType(tf.bool))), '
        'Reference(\'bar\', TensorType(tf.int32)))')
    self.assertEqual(str(z), 'foo(bar)')
    with self.assertRaises(TypeError):
      bb.Call(x)
    w = bb.Reference('bak', tf.float32)
    with self.assertRaises(TypeError):
      bb.Call(x, w)

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
        repr(x),
        'Lambda(\'arg\', {0}, '
        'Call(Selection(Reference(\'arg\', {0}), name=\'f\'), '
        'Call(Selection(Reference(\'arg\', {0}), name=\'f\'), '
        'Selection(Reference(\'arg\', {0}), name=\'x\'))))'.format(
            arg_type_repr))
    self.assertEqual(str(x), '(arg -> arg.f(arg.f(arg.x)))')

  def test_basic_functionality_of_block_class(self):
    x = bb.Block(
        [('x', bb.Reference('arg', (tf.int32, tf.int32))),
         ('y', bb.Selection(
             bb.Reference('x', (tf.int32, tf.int32)), index=0))],
        bb.Reference('y', tf.int32))
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(
        [(k, str(v)) for k, v in x.locals],
        [('x', 'arg'), ('y', 'x[0]')])
    self.assertEqual(str(x.result), 'y')
    self.assertEqual(
        repr(x),
        'Block([(\'x\', Reference(\'arg\', '
        'NamedTupleType([TensorType(tf.int32), TensorType(tf.int32)]))), '
        '(\'y\', Selection(Reference(\'x\', '
        'NamedTupleType([TensorType(tf.int32), TensorType(tf.int32)])), '
        'index=0))], '
        'Reference(\'y\', TensorType(tf.int32)))')
    self.assertEqual(str(x), '(let x=arg,y=x[0] in y)')

  def test_basic_functionality_of_intrinsic_class(self):
    x = bb.Intrinsic('add_one', types.FunctionType(tf.int32, tf.int32))
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(x.uri, 'add_one')
    self.assertEqual(
        repr(x),
        'Intrinsic(\'add_one\', '
        'FunctionType(TensorType(tf.int32), TensorType(tf.int32)))')
    self.assertEqual(str(x), 'add_one')

  def test_basic_functionality_of_data_class(self):
    x = bb.Data('/tmp/mydata', types.SequenceType(tf.int32))
    self.assertEqual(str(x.type_signature), 'int32*')
    self.assertEqual(x.uri, '/tmp/mydata')
    self.assertEqual(
        repr(x), 'Data(\'/tmp/mydata\', SequenceType(TensorType(tf.int32)))')
    self.assertEqual(str(x), '/tmp/mydata')

  def test_basic_functionality_of_compiled_computation_class(self):
    comp = tensorflow_serialization.serialize_py_func_as_tf_computation(
        lambda x: x + 3, tf.int32)
    x = bb.CompiledComputation(comp)
    self.assertEqual(str(x.type_signature), '(int32 -> int32)')
    self.assertEqual(str(x.proto), str(comp))
    self.assertTrue(re.match(
        r'CompiledComputation\([0-9a-f]+, '
        r'FunctionType\(TensorType\(tf\.int32\), TensorType\(tf\.int32\)\)\)',
        repr(x)))
    self.assertTrue(re.match(r'comp\([0-9a-f]+\)', str(x)))
    y = bb.CompiledComputation(comp, name='foo')
    self.assertEqual(str(y), 'comp(foo)')


if __name__ == '__main__':
  unittest.main()
