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

import collections

# Dependency imports
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import anonymous_tuple
from tensorflow_federated.python.core.impl import values


class ValuesTest(unittest.TestCase):

  def test_reference(self):
    x = values.Reference('foo', tf.int32)
    self.assertIsInstance(x, value_base.Value)
    self.assertEqual(x.name, 'foo')
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), 'Reference(\'foo\', TensorType(tf.int32))')
    self.assertEqual(str(x), 'foo')
    with self.assertRaises(SyntaxError):
      x(10)

  def test_selection(self):
    x = values.Reference('foo', [('bar', tf.int32), ('baz', tf.bool)])
    self.assertEqual(dir(x), ['bar', 'baz'])
    self.assertEqual(len(x), 2)
    y = x.bar
    self.assertIsInstance(y, values.Selection)
    self.assertIsInstance(y, value_base.Value)
    self.assertEqual(y.name, 'bar')
    self.assertEqual(y.index, None)
    self.assertEqual(str(y.type_signature), 'int32')
    self.assertEqual(
        repr(y),
        'Selection(Reference(\'foo\', NamedTupleType(['
        '(\'bar\', TensorType(tf.int32)), (\'baz\', TensorType(tf.bool))]))'
        ', name=\'bar\')')
    self.assertEqual(str(y), 'foo.bar')
    z = x.baz
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(str(z), 'foo.baz')
    with self.assertRaises(AttributeError):
      _ = x.bak
    x0 = x[0]
    self.assertIsInstance(x0, values.Selection)
    self.assertEqual(x0.name, None)
    self.assertEqual(x0.index, 0)
    self.assertEqual(str(x0.type_signature), 'int32')
    self.assertEqual(
        repr(x0),
        'Selection(Reference(\'foo\', NamedTupleType(['
        '(\'bar\', TensorType(tf.int32)), (\'baz\', TensorType(tf.bool))]))'
        ', index=0)')
    self.assertEqual(str(x0), 'foo[0]')
    x1 = x[1]
    self.assertEqual(str(x1.type_signature), 'bool')
    self.assertEqual(str(x1), 'foo[1]')
    with self.assertRaises(KeyError):
      _ = x[2]
    with self.assertRaises(KeyError):
      _ = x[-1]
    self.assertEqual(','.join(str(e) for e in iter(x)), 'foo[0],foo[1]')
    self.assertEqual(
        ','.join(str(e.type_signature) for e in iter(x)), 'int32,bool')
    with self.assertRaises(SyntaxError):
      x(10)

  def test_tuple(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    z = values.Tuple([x, ('y', y)])
    self.assertIsInstance(z, value_base.Value)
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
    with self.assertRaises(SyntaxError):
      z(10)

  def test_call(self):
    x = values.Reference('foo', types.FunctionType(tf.int32, tf.bool))
    y = values.Reference('bar', tf.int32)
    z = x(y)
    self.assertIsInstance(z, values.Call)
    self.assertIsInstance(z, value_base.Value)
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
      x()
    w = values.Reference('bak', tf.float32)
    with self.assertRaises(TypeError):
      x(w)

  def test_to_value_for_tuple(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    self.assertEqual(str(values.to_value((x, y))), '<foo,bar>')

  def test_to_value_for_list(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    self.assertEqual(str(values.to_value([x, y])), '<foo,bar>')

  def test_to_value_for_dict(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    self.assertIn(
        str(values.to_value({'a': x, 'b': y})),
        ['<a=foo,b=bar>', '<b=bar,a=foo>'])

  def test_to_value_for_ordered_dict(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    self.assertEqual(
        str(values.to_value(collections.OrderedDict([('a', x), ('b', y)]))),
        '<a=foo,b=bar>')

  def test_to_value_for_named_tuple(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    self.assertEqual(
        str(values.to_value(collections.namedtuple('_', 'a b')(x, y))),
        '<a=foo,b=bar>')

  def test_to_value_for_anonymous_tuple(self):
    x = values.Reference('foo', tf.int32)
    y = values.Reference('bar', tf.bool)
    self.assertEqual(
        str(values.to_value(anonymous_tuple.AnonymousTuple(
            [('a', x), ('b', y)]))),
        '<a=foo,b=bar>')


if __name__ == '__main__':
  unittest.main()
