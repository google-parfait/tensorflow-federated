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
from absl.testing import parameterized
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import types
from tensorflow_federated.python.core.api import value_base

from tensorflow_federated.python.core.impl import anonymous_tuple
from tensorflow_federated.python.core.impl import computation_building_blocks as bb
from tensorflow_federated.python.core.impl import value_impl


class ValueImplTest(parameterized.TestCase):

  def test_value_impl_with_reference(self):
    x_comp = bb.Reference('foo', tf.int32)
    x = value_impl.ValueImpl(x_comp)
    self.assertIs(value_impl.ValueImpl.get_comp(x), x_comp)
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), 'Reference(\'foo\', TensorType(tf.int32))')
    self.assertEqual(str(x), 'foo')
    with self.assertRaises(SyntaxError):
      x(10)

  def test_value_impl_with_selection(self):
    x = value_impl.ValueImpl(
        bb.Reference('foo', [('bar', tf.int32), ('baz', tf.bool)]))
    self.assertEqual(dir(x), ['bar', 'baz'])
    self.assertEqual(len(x), 2)
    y = x.bar
    self.assertIsInstance(y, value_base.Value)
    self.assertEqual(str(y.type_signature), 'int32')
    self.assertEqual(str(y), 'foo.bar')
    z = x.baz
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(str(z), 'foo.baz')
    with self.assertRaises(AttributeError):
      _ = x.bak
    x0 = x[0]
    self.assertIsInstance(x0, value_base.Value)
    self.assertEqual(str(x0.type_signature), 'int32')
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

  def test_value_impl_with_tuple(self):
    x_comp = bb.Reference('foo', tf.int32)
    y_comp = bb.Reference('bar', tf.bool)
    z = value_impl.ValueImpl(bb.Tuple([x_comp, ('y', y_comp)]))
    self.assertIsInstance(z, value_base.Value)
    self.assertEqual(str(z.type_signature), '<int32,y=bool>')
    self.assertEqual(str(z), '<foo,y=bar>')
    self.assertEqual(dir(z), ['y'])
    self.assertEqual(str(z.y), 'bar')
    self.assertIs(value_impl.ValueImpl.get_comp(z.y), y_comp)
    self.assertEqual(len(z), 2)
    self.assertEqual(str(z[0]), 'foo')
    self.assertIs(value_impl.ValueImpl.get_comp(z[0]), x_comp)
    self.assertEqual(str(z[1]), 'bar')
    self.assertIs(value_impl.ValueImpl.get_comp(z[1]), y_comp)
    self.assertEqual(','.join(str(e) for e in iter(z)), 'foo,bar')
    with self.assertRaises(SyntaxError):
      z(10)

  def test_value_impl_with_call(self):
    x = value_impl.ValueImpl(
        bb.Reference('foo', types.FunctionType(tf.int32, tf.bool)))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.int32))
    z = x(y)
    self.assertIsInstance(z, value_base.Value)
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(str(z), 'foo(bar)')
    with self.assertRaises(TypeError):
      x()
    w = value_impl.ValueImpl(bb.Reference('bak', tf.float32))
    with self.assertRaises(TypeError):
      x(w)

  def test_value_impl_with_lambda(self):
    arg_name = 'arg'
    arg_type = [('f', types.FunctionType(tf.int32, tf.int32)), ('x', tf.int32)]
    result_value = (lambda arg: arg.f(arg.f(arg.x)))(
        value_impl.ValueImpl(bb.Reference(arg_name, arg_type)))
    x = value_impl.ValueImpl(bb.Lambda(
        arg_name, arg_type, value_impl.ValueImpl.get_comp(result_value)))
    self.assertIsInstance(x, value_base.Value)
    self.assertEqual(
        str(x.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')
    self.assertEqual(str(x), '(arg -> arg.f(arg.f(arg.x)))')

  def test_to_value_for_tuple(self):
    x = value_impl.ValueImpl(bb.Reference('foo', tf.int32))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.bool))
    v = value_impl.to_value((x, y))
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<foo,bar>')

  def test_to_value_for_list(self):
    x = value_impl.ValueImpl(bb.Reference('foo', tf.int32))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.bool))
    v = value_impl.to_value([x, y])
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<foo,bar>')

  def test_to_value_for_dict(self):
    x = value_impl.ValueImpl(bb.Reference('foo', tf.int32))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.bool))
    v = value_impl.to_value({'a': x, 'b': y})
    self.assertIsInstance(v, value_base.Value)
    self.assertIn(str(v), ['<a=foo,b=bar>', '<b=bar,a=foo>'])

  def test_to_value_for_ordered_dict(self):
    x = value_impl.ValueImpl(bb.Reference('foo', tf.int32))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.bool))
    v = value_impl.to_value(collections.OrderedDict([('a', x), ('b', y)]))
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<a=foo,b=bar>')

  def test_to_value_for_named_tuple(self):
    x = value_impl.ValueImpl(bb.Reference('foo', tf.int32))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.bool))
    v = value_impl.to_value(collections.namedtuple('_', 'a b')(x, y))
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<a=foo,b=bar>')

  def test_to_value_for_anonymous_tuple(self):
    x = value_impl.ValueImpl(bb.Reference('foo', tf.int32))
    y = value_impl.ValueImpl(bb.Reference('bar', tf.bool))
    v = value_impl.to_value(
        anonymous_tuple.AnonymousTuple([('a', x), ('b', y)]))
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<a=foo,b=bar>')

  def test_to_value_for_placement_literals(self):
    clients = value_impl.to_value(placements.CLIENTS)
    self.assertIsInstance(clients, value_base.Value)
    self.assertEqual(str(clients.type_signature), 'placement')
    self.assertEqual(str(clients), 'CLIENTS')

  def test_to_value_for_computations(self):
    val = value_impl.to_value(
        computations.tf_computation(lambda: tf.constant(10)))
    self.assertIsInstance(val, value_base.Value)
    self.assertEqual(str(val.type_signature), '( -> int32)')


if __name__ == '__main__':
  unittest.main()
