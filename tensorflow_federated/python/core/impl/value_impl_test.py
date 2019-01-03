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

from absl.testing import absltest
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import value_impl


class ValueImplTest(absltest.TestCase):

  def test_value_impl_with_reference(self):
    x_comp = computation_building_blocks.Reference('foo', tf.int32)
    x = value_impl.ValueImpl(x_comp, context_stack_impl.context_stack)
    self.assertIs(value_impl.ValueImpl.get_comp(x), x_comp)
    self.assertEqual(str(x.type_signature), 'int32')
    self.assertEqual(repr(x), 'Reference(\'foo\', TensorType(tf.int32))')
    self.assertEqual(str(x), 'foo')
    with self.assertRaises(SyntaxError):
      x(10)

  def test_value_impl_with_selection(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', [('bar', tf.int32),
                                                      ('baz', tf.bool)]),
        context_stack_impl.context_stack)
    self.assertEqual(dir(x), ['bar', 'baz'])
    self.assertLen(x, 2)
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
    with self.assertRaises(IndexError):
      _ = x[2]
    with self.assertRaises(IndexError):
      _ = x[-1]
    self.assertEqual(','.join(str(e) for e in iter(x)), 'foo[0],foo[1]')
    self.assertEqual(','.join(str(e.type_signature) for e in iter(x)),
                     'int32,bool')
    with self.assertRaises(SyntaxError):
      x(10)

  def test_value_impl_with_tuple(self):
    x_comp = computation_building_blocks.Reference('foo', tf.int32)
    y_comp = computation_building_blocks.Reference('bar', tf.bool)
    z = value_impl.ValueImpl(
        computation_building_blocks.Tuple([x_comp, ('y', y_comp)]),
        context_stack_impl.context_stack)
    self.assertIsInstance(z, value_base.Value)
    self.assertEqual(str(z.type_signature), '<int32,y=bool>')
    self.assertEqual(str(z), '<foo,y=bar>')
    self.assertEqual(dir(z), ['y'])
    self.assertEqual(str(z.y), 'bar')
    self.assertIs(value_impl.ValueImpl.get_comp(z.y), y_comp)
    self.assertLen(z, 2)
    self.assertEqual(str(z[0]), 'foo')
    self.assertIs(value_impl.ValueImpl.get_comp(z[0]), x_comp)
    self.assertEqual(str(z[1]), 'bar')
    self.assertIs(value_impl.ValueImpl.get_comp(z[1]), y_comp)
    self.assertEqual(','.join(str(e) for e in iter(z)), 'foo,bar')
    with self.assertRaises(SyntaxError):
      z(10)

  def test_value_impl_with_call(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference(
            'foo', computation_types.FunctionType(tf.int32, tf.bool)),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.int32),
        context_stack_impl.context_stack)
    z = x(y)
    self.assertIsInstance(z, value_base.Value)
    self.assertEqual(str(z.type_signature), 'bool')
    self.assertEqual(str(z), 'foo(bar)')
    with self.assertRaises(TypeError):
      x()
    w = value_impl.ValueImpl(
        computation_building_blocks.Reference('bak', tf.float32),
        context_stack_impl.context_stack)
    with self.assertRaises(TypeError):
      x(w)

  def test_value_impl_with_lambda(self):
    arg_name = 'arg'
    arg_type = [('f', computation_types.FunctionType(tf.int32, tf.int32)),
                ('x', tf.int32)]
    result_value = (lambda arg: arg.f(arg.f(arg.x)))(
        value_impl.ValueImpl(
            computation_building_blocks.Reference(arg_name, arg_type),
            context_stack_impl.context_stack))
    x = value_impl.ValueImpl(
        computation_building_blocks.Lambda(
            arg_name, arg_type, value_impl.ValueImpl.get_comp(result_value)),
        context_stack_impl.context_stack)
    self.assertIsInstance(x, value_base.Value)
    self.assertEqual(
        str(x.type_signature), '(<f=(int32 -> int32),x=int32> -> int32)')
    self.assertEqual(str(x), '(arg -> arg.f(arg.f(arg.x)))')

  def test_value_impl_with_plus(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('x', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('y', tf.int32),
        context_stack_impl.context_stack)
    z = x + y
    self.assertIsInstance(z, value_base.Value)
    self.assertEqual(str(z.type_signature), 'int32')
    self.assertEqual(str(z), 'generic_plus(<x,y>)')

  def test_to_value_for_tuple(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value((x, y), None, context_stack_impl.context_stack)
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<foo,bar>')

  def test_to_value_for_list(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value([x, y], None, context_stack_impl.context_stack)
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<foo,bar>')

  def test_to_value_for_dict(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value({
        'a': x,
        'b': y
    }, None, context_stack_impl.context_stack)
    self.assertIsInstance(v, value_base.Value)
    self.assertIn(str(v), ['<a=foo,b=bar>', '<b=bar,a=foo>'])

  def test_to_value_for_ordered_dict(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value(
        collections.OrderedDict([('a', x), ('b', y)]), None,
        context_stack_impl.context_stack)
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<a=foo,b=bar>')

  def test_to_value_for_named_tuple(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value(
        collections.namedtuple('_', 'a b')(x, y), None,
        context_stack_impl.context_stack)
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<a=foo,b=bar>')

  def test_to_value_for_anonymous_tuple(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value(
        anonymous_tuple.AnonymousTuple([('a', x), ('b', y)]), None,
        context_stack_impl.context_stack)
    self.assertIsInstance(v, value_base.Value)
    self.assertEqual(str(v), '<a=foo,b=bar>')

  def test_to_value_for_placement_literals(self):
    clients = value_impl.to_value(placements.CLIENTS, None,
                                  context_stack_impl.context_stack)
    self.assertIsInstance(clients, value_base.Value)
    self.assertEqual(str(clients.type_signature), 'placement')
    self.assertEqual(str(clients), 'CLIENTS')

  def test_to_value_for_computations(self):
    val = value_impl.to_value(
        computations.tf_computation(lambda: tf.constant(10)), None,
        context_stack_impl.context_stack)
    self.assertIsInstance(val, value_base.Value)
    self.assertEqual(str(val.type_signature), '( -> int32)')

  def test_to_value_with_int_and_int_type_spec(self):
    val = value_impl.to_value(10, tf.int32, context_stack_impl.context_stack)
    self.assertIsInstance(val, value_base.Value)
    self.assertEqual(str(val.type_signature), 'int32')

  def test_to_value_with_int_and_bool_type_spec(self):
    with self.assertRaises(TypeError):
      value_impl.to_value(10, tf.bool, context_stack_impl.context_stack)

  def test_to_value_with_int_list_and_int_sequence_type_spec(self):
    val = value_impl.to_value([1, 2, 3], computation_types.SequenceType(
        tf.int32), context_stack_impl.context_stack)
    self.assertIsInstance(val, value_base.Value)
    self.assertEqual(str(val.type_signature), 'int32*')

  def test_constant_mapping(self):
    raw_int_val = value_impl.to_value(10, None,
                                      context_stack_impl.context_stack)
    self.assertIsInstance(raw_int_val, value_base.Value)
    self.assertEqual(str(raw_int_val.type_signature), 'int32')
    raw_float_val = value_impl.to_value(10.0, None,
                                        context_stack_impl.context_stack)
    self.assertIsInstance(raw_float_val, value_base.Value)
    self.assertEqual(str(raw_float_val.type_signature), 'float32')
    np_array_val = value_impl.to_value(
        np.array([10.0]), None, context_stack_impl.context_stack)
    self.assertIsInstance(np_array_val, value_base.Value)
    self.assertEqual(str(np_array_val.type_signature), 'float64[1]')
    lg_np_array_flt = value_impl.to_value(
        np.ones([10, 10, 10], dtype=np.float32), None,
        context_stack_impl.context_stack)
    self.assertIsInstance(lg_np_array_flt, value_base.Value)
    self.assertEqual(str(lg_np_array_flt.type_signature), 'float32[10,10,10]')
    lg_np_array_int = value_impl.to_value(
        np.ones([10, 10, 10], dtype=np.int32), None,
        context_stack_impl.context_stack)
    self.assertIsInstance(lg_np_array_int, value_base.Value)
    self.assertEqual(str(lg_np_array_int.type_signature), 'int32[10,10,10]')
    raw_string_val = value_impl.to_value('10', None,
                                         context_stack_impl.context_stack)
    self.assertIsInstance(raw_string_val, value_base.Value)
    self.assertEqual(str(raw_string_val.type_signature), 'string')

  def test_slicing_support_namedtuple(self):
    x = value_impl.ValueImpl(
        computation_building_blocks.Reference('foo', tf.int32),
        context_stack_impl.context_stack)
    y = value_impl.ValueImpl(
        computation_building_blocks.Reference('bar', tf.bool),
        context_stack_impl.context_stack)
    v = value_impl.to_value(
        collections.namedtuple('_', 'a b')(x, y), None,
        context_stack_impl.context_stack)
    sliced_v = v[:int(len(v) / 2)]
    self.assertIsInstance(sliced_v, value_base.Value)
    sliced_v = v[:4:2]
    self.assertEqual(str(sliced_v), '<foo>')
    self.assertIsInstance(sliced_v, value_base.Value)
    sliced_v = v[4::-1]
    self.assertEqual(str(sliced_v), '<bar,foo>')
    self.assertIsInstance(sliced_v, value_base.Value)
    with self.assertRaisesRegexp(IndexError, 'slice 0 elements'):
      _ = v[2:4]

  def test_slicing_fails_non_namedtuple(self):
    v = value_impl.to_value(
        np.ones([10, 10, 10], dtype=np.float32), None,
        context_stack_impl.context_stack)
    with self.assertRaisesRegexp(TypeError, 'only supported for named tuples'):
      _ = v[:1]

  def test_slicing_support_non_tuple_underlying_comp(self):
    test_computation_building_blocks = computation_building_blocks.Reference(
        'test', [tf.int32] * 5)
    v = value_impl.ValueImpl(test_computation_building_blocks,
                             context_stack_impl.context_stack)
    sliced_v = v[:4:2]
    self.assertIsInstance(sliced_v, value_base.Value)
    sliced_v = v[4:2:-1]
    self.assertIsInstance(sliced_v, value_base.Value)
    with self.assertRaisesRegexp(IndexError, 'slice 0 elements'):
      _ = v[2:4:-1]

  def test_slicing_tuple_values(self):
    for op in [list, tuple]:
      t = op(range(0, 50, 10))
      v = value_impl.to_value(t, None, context_stack_impl.context_stack)
      self.assertEqual((str(v.type_signature)),
                       '<int32,int32,int32,int32,int32>')
      self.assertEqual(
          str(v[:]),
          str(value_impl.to_value(t, None, context_stack_impl.context_stack)))
      sliced = v[:2]
      self.assertEqual((str(sliced.type_signature)), '<int32,int32>')
      self.assertEqual(
          str(sliced),
          str(
              value_impl.to_value(t[:2], None,
                                  context_stack_impl.context_stack)))
      sliced = v[-3:]
      self.assertEqual((str(sliced.type_signature)), '<int32,int32,int32>')
      self.assertEqual(
          str(sliced),
          str(
              value_impl.to_value(t[-3:], None,
                                  context_stack_impl.context_stack)))
      sliced = v[::2]
      self.assertEqual((str(sliced.type_signature)), '<int32,int32,int32>')
      self.assertEqual(
          str(sliced),
          str(
              value_impl.to_value(t[::2], None,
                                  context_stack_impl.context_stack)))


if __name__ == '__main__':
  absltest.main()
