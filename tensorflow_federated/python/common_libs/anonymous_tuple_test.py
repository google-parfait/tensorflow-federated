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
"""Tests for anonymous_tuple.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
from six.moves import range

from tensorflow_federated.python.common_libs import anonymous_tuple


class AnonymousTupleTest(absltest.TestCase):

  def test_empty(self):
    v = []
    x = anonymous_tuple.AnonymousTuple(v)
    # Explicitly test the implementation of __len__() here so use, assertLen()
    # instead of assertEmpty().
    self.assertLen(x, 0)  # pylint: disable=g-generic-assert
    self.assertRaises(IndexError, lambda _: x[0], None)
    self.assertEqual(list(iter(x)), [])
    self.assertEqual(dir(x), [])
    self.assertRaises(AttributeError, lambda _: x.foo, None)
    self.assertEqual(x, anonymous_tuple.AnonymousTuple([]))
    self.assertNotEqual(x, anonymous_tuple.AnonymousTuple([('foo', 10)]))
    self.assertEqual(anonymous_tuple.to_elements(x), v)
    self.assertEqual(repr(x), 'AnonymousTuple([])')
    self.assertEqual(str(x), '<>')

  def test_single_unnamed(self):
    v = [(None, 10)]
    x = anonymous_tuple.AnonymousTuple(v)
    self.assertLen(x, 1)
    self.assertRaises(IndexError, lambda _: x[1], None)
    self.assertEqual(x[0], 10)
    self.assertEqual(list(iter(x)), [10])
    self.assertEqual(dir(x), [])
    self.assertRaises(AttributeError, lambda _: x.foo, None)
    self.assertNotEqual(x, anonymous_tuple.AnonymousTuple([]))
    self.assertNotEqual(x, anonymous_tuple.AnonymousTuple([('foo', 10)]))
    self.assertEqual(x, anonymous_tuple.AnonymousTuple([(None, 10)]))
    self.assertNotEqual(
        x, anonymous_tuple.AnonymousTuple([(None, 10), ('foo', 20)]))
    self.assertEqual(anonymous_tuple.to_elements(x), v)
    self.assertEqual(repr(x), 'AnonymousTuple([(None, 10)])')
    self.assertEqual(str(x), '<10>')

  def test_single_named(self):
    v = [('foo', 20)]
    x = anonymous_tuple.AnonymousTuple(v)
    self.assertLen(x, 1)
    self.assertEqual(x[0], 20)
    self.assertRaises(IndexError, lambda _: x[1], None)
    self.assertEqual(list(iter(x)), [20])
    self.assertEqual(dir(x), ['foo'])
    self.assertEqual(x.foo, 20)
    self.assertRaises(AttributeError, lambda _: x.bar, None)
    self.assertNotEqual(x, anonymous_tuple.AnonymousTuple([]))
    self.assertNotEqual(x, anonymous_tuple.AnonymousTuple([('foo', 10)]))
    self.assertNotEqual(x, anonymous_tuple.AnonymousTuple([(None, 20)]))
    self.assertEqual(x, anonymous_tuple.AnonymousTuple([('foo', 20)]))
    self.assertNotEqual(
        x, anonymous_tuple.AnonymousTuple([('foo', 20), ('bar', 30)]))
    self.assertEqual(anonymous_tuple.to_elements(x), v)
    self.assertEqual(repr(x), 'AnonymousTuple([(foo, 20)])')
    self.assertEqual(str(x), '<foo=20>')

  def test_multiple_named_and_unnamed(self):
    v = [(None, 10), ('foo', 20), ('bar', 30)]
    x = anonymous_tuple.AnonymousTuple(v)
    self.assertLen(x, 3)
    self.assertEqual(x[0], 10)
    self.assertEqual(x[1], 20)
    self.assertEqual(x[2], 30)
    self.assertRaises(IndexError, lambda _: x[3], None)
    self.assertEqual(list(iter(x)), [10, 20, 30])
    self.assertEqual(dir(x), ['bar', 'foo'])
    self.assertEqual(x.foo, 20)
    self.assertEqual(x.bar, 30)
    self.assertRaises(AttributeError, lambda _: x.baz, None)
    self.assertEqual(
        x, anonymous_tuple.AnonymousTuple([(None, 10), ('foo', 20), ('bar',
                                                                     30)]))
    self.assertNotEqual(
        x, anonymous_tuple.AnonymousTuple([('foo', 10), ('bar', 20), (None,
                                                                      30)]))
    self.assertEqual(anonymous_tuple.to_elements(x), v)
    self.assertEqual(
        repr(x), 'AnonymousTuple([(None, 10), (foo, 20), (bar, 30)])')
    self.assertEqual(str(x), '<10,foo=20,bar=30>')

  def test_hash(self):
    v1 = [(str(i) if i > 30 else None, i) for i in range(0, 50, 10)]
    x1 = anonymous_tuple.AnonymousTuple(v1)
    self.assertNotEqual(x1, v1)
    self.assertNotEqual(hash(x1), hash(iter(v1)))
    v2 = [(None, i) for i in range(0, 50, 10)]
    x2 = anonymous_tuple.AnonymousTuple(v2)
    self.assertNotEqual(hash(x2), hash(iter(v2)))
    self.assertNotEqual(x1, x2)
    self.assertNotEqual(hash(x1), hash(x2))
    v3 = [(None, 0), (None, 10), (None, 20), (None, 30), (None, 40)]
    x3 = anonymous_tuple.AnonymousTuple(v3)
    self.assertEqual(v2, v3)
    self.assertEqual(x2, x3)
    self.assertEqual(hash(x2), hash(x3))

  def test_slicing_behavior(self):
    v = [(None, i) for i in range(0, 50, 10)]
    x = anonymous_tuple.AnonymousTuple(v)
    self.assertEqual(x[:], list(range(0, 50, 10)))
    self.assertEqual(x[::-1], list(reversed(range(0, 50, 10))))
    self.assertEqual(x[:-1], list(range(0, 40, 10)))
    self.assertEqual(x[1:], list(range(10, 50, 10)))
    self.assertEqual(x[-1:], [40])

  def test_getitem_bad_bounds(self):
    v = [(None, i) for i in range(0, 50, 10)]
    x = anonymous_tuple.AnonymousTuple(v)
    with self.assertRaises(IndexError):
      _ = x[10]

  def test_flatten_and_pack_sequence_as(self):
    x = anonymous_tuple.AnonymousTuple([
        ('a', 10),
        ('b',
         anonymous_tuple.AnonymousTuple([
             ('x', anonymous_tuple.AnonymousTuple([('p', 40)])),
             ('y', 30),
             ('z', anonymous_tuple.AnonymousTuple([('q', 50), ('r', 60)])),
         ])),
        ('c', 20),
    ])
    y = anonymous_tuple.flatten(x)
    self.assertEqual(y, [10, 40, 30, 50, 60, 20])
    z = anonymous_tuple.pack_sequence_as(x, y)
    self.assertEqual(str(z), '<a=10,b=<x=<p=40>,y=30,z=<q=50,r=60>>,c=20>')

  def test_is_same_structure_check_types(self):
    self.assertTrue(
        anonymous_tuple.is_same_structure(
            anonymous_tuple.AnonymousTuple([('a', 10)]),
            anonymous_tuple.AnonymousTuple([('a', 20)])))
    self.assertTrue(
        anonymous_tuple.is_same_structure(
            anonymous_tuple.AnonymousTuple([
                ('a', 10),
                ('b', anonymous_tuple.AnonymousTuple([('z', 5)])),
            ]),
            anonymous_tuple.AnonymousTuple([
                ('a', 20),
                ('b', anonymous_tuple.AnonymousTuple([('z', 50)])),
            ])))
    self.assertFalse(
        anonymous_tuple.is_same_structure(
            anonymous_tuple.AnonymousTuple([('x', {
                'y': 4
            })]), anonymous_tuple.AnonymousTuple([('x', {
                'y': 5,
                'z': 6
            })])))
    self.assertTrue(
        anonymous_tuple.is_same_structure(
            anonymous_tuple.AnonymousTuple([('x', {
                'y': 5
            })]), anonymous_tuple.AnonymousTuple([('x', {
                'y': 6
            })])))
    with self.assertRaises(TypeError):
      self.assertTrue(
          anonymous_tuple.is_same_structure(
              {'x': 5.0},  # not an AnonymousTuple
              anonymous_tuple.AnonymousTuple([('x', 5.0)])))

  def test_map_structure(self):
    x = anonymous_tuple.AnonymousTuple([
        ('a', 10),
        ('b',
         anonymous_tuple.AnonymousTuple([
             ('x', anonymous_tuple.AnonymousTuple([('p', 40)])),
             ('y', 30),
             ('z', anonymous_tuple.AnonymousTuple([('q', 50), ('r', 60)])),
         ])),
        ('c', 20),
    ])
    y = anonymous_tuple.AnonymousTuple([
        ('a', 1),
        ('b',
         anonymous_tuple.AnonymousTuple([
             ('x', anonymous_tuple.AnonymousTuple([('p', 4)])),
             ('y', 3),
             ('z', anonymous_tuple.AnonymousTuple([('q', 5), ('r', 6)])),
         ])),
        ('c', 2),
    ])

    self.assertEqual(
        anonymous_tuple.map_structure(lambda x, y: x + y, x, y),
        anonymous_tuple.AnonymousTuple([
            ('a', 11),
            ('b',
             anonymous_tuple.AnonymousTuple([
                 ('x', anonymous_tuple.AnonymousTuple([('p', 44)])),
                 ('y', 33),
                 ('z', anonymous_tuple.AnonymousTuple([('q', 55), ('r', 66)])),
             ])),
            ('c', 22),
        ]))


if __name__ == '__main__':
  absltest.main()
