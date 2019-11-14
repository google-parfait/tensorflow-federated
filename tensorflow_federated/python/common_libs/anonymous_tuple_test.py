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

import collections

from absl.testing import absltest
import attr

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
    self.assertEqual(anonymous_tuple.to_odict(x), collections.OrderedDict())
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
    with self.assertRaisesRegex(ValueError, 'unnamed'):
      anonymous_tuple.to_odict(x)

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
    self.assertEqual(repr(x), 'AnonymousTuple([(\'foo\', 20)])')
    self.assertEqual(str(x), '<foo=20>')
    self.assertEqual(anonymous_tuple.to_odict(x), collections.OrderedDict(v))

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
    self.assertEqual(anonymous_tuple.name_list(x), ['foo', 'bar'])
    self.assertEqual(x.foo, 20)
    self.assertEqual(x.bar, 30)
    self.assertRaises(AttributeError, lambda _: x.baz, None)
    self.assertEqual(
        x, anonymous_tuple.AnonymousTuple([(None, 10), ('foo', 20),
                                           ('bar', 30)]))
    self.assertNotEqual(
        x, anonymous_tuple.AnonymousTuple([('foo', 10), ('bar', 20),
                                           (None, 30)]))
    self.assertEqual(anonymous_tuple.to_elements(x), v)
    self.assertEqual(
        repr(x), 'AnonymousTuple([(None, 10), (\'foo\', 20), (\'bar\', 30)])')
    self.assertEqual(str(x), '<10,foo=20,bar=30>')
    with self.assertRaisesRegex(ValueError, 'unnamed'):
      anonymous_tuple.to_odict(x)

  def test_bad_names(self):
    with self.assertRaisesRegex(ValueError, 'duplicated.*foo'):
      anonymous_tuple.AnonymousTuple([('foo', 20), ('foo', 30)])

    with self.assertRaisesRegex(ValueError, '_asdict.*reserved'):
      anonymous_tuple.AnonymousTuple([('_asdict', 40)])

    with self.assertRaisesRegex(ValueError, '_element_array.*reserved'):
      anonymous_tuple.AnonymousTuple([('_element_array', 40)])

    with self.assertRaisesRegex(ValueError, '_name_to_index.*reserved'):
      anonymous_tuple.AnonymousTuple([('_name_to_index', 40)])

    with self.assertRaisesRegex(ValueError, '_name_array.*reserved'):
      anonymous_tuple.AnonymousTuple([('_name_array', 40)])

    with self.assertRaisesRegex(ValueError, '_hash.*reserved'):
      anonymous_tuple.AnonymousTuple([('_hash', 40)])

  def test_immutable(self):
    v = [('foo', 'a string'), ('bar', 1), ('baz', [1.0, 2.0, 3.0])]
    t = anonymous_tuple.AnonymousTuple(v)

    # Expect that we can read by name the values.
    self.assertEqual(t.foo, 'a string')
    self.assertEqual(t[0], 'a string')
    self.assertEqual(t.bar, 1)
    self.assertEqual(t[1], 1)
    self.assertEqual(t.baz, [1.0, 2.0, 3.0])
    self.assertEqual(t[2], [1.0, 2.0, 3.0])

    # But trying to set an attribute fails.

    # These raise "AttributeError" saying that the particular attribute is
    # unknown. This can look strange because the attribute was "known" above.
    with self.assertRaises(AttributeError):
      t.foo = 'a different string'
    with self.assertRaises(AttributeError):
      t.bar = 5
    with self.assertRaises(AttributeError):
      t.baz = [1, 2, 3]

    # These raise "TypeError" saying that tuples are immutable.
    with self.assertRaises(TypeError):
      t[0] = 'a different string'
    with self.assertRaises(TypeError):
      t[1] = 5
    with self.assertRaises(TypeError):
      t[2] = [1, 2, 3]

  def test_equality_unnamed(self):
    # identity
    t1 = anonymous_tuple.AnonymousTuple([(None, 1), (None, 2)])
    self.assertTrue(t1.__eq__(t1))
    self.assertFalse(t1.__ne__(t1))
    # different type
    self.assertFalse(t1.__eq__(None))
    self.assertTrue(t1.__ne__(None))
    # copy
    t2 = anonymous_tuple.AnonymousTuple([(None, 1), (None, 2)])
    self.assertTrue(t1.__eq__(t2))
    self.assertTrue(t2.__eq__(t1))
    self.assertFalse(t1.__ne__(t2))
    self.assertFalse(t2.__ne__(t1))
    # different ordering
    t3 = anonymous_tuple.AnonymousTuple([(None, 2), (None, 1)])
    self.assertFalse(t1.__eq__(t3))
    self.assertFalse(t3.__eq__(t1))
    self.assertTrue(t1.__ne__(t3))
    self.assertTrue(t3.__ne__(t1))
    # different names
    t4 = anonymous_tuple.AnonymousTuple([('a', 1), ('b', 2)])
    self.assertFalse(t1.__eq__(t4))
    self.assertFalse(t4.__eq__(t1))
    self.assertTrue(t1.__ne__(t4))
    self.assertTrue(t4.__ne__(t1))
    # different values
    t5 = anonymous_tuple.AnonymousTuple([(None, 10), (None, 10)])
    self.assertFalse(t1.__eq__(t5))
    self.assertFalse(t5.__eq__(t1))
    self.assertTrue(t1.__ne__(t5))
    self.assertTrue(t5.__ne__(t1))

  def test_equality_named(self):
    # identity
    t1 = anonymous_tuple.AnonymousTuple([('a', 1), ('b', 2)])
    self.assertTrue(t1.__eq__(t1))
    self.assertFalse(t1.__ne__(t1))
    # different type
    self.assertFalse(t1.__eq__(None))
    self.assertTrue(t1.__ne__(None))
    # copy
    t2 = anonymous_tuple.AnonymousTuple([('a', 1), ('b', 2)])
    self.assertTrue(t1.__eq__(t2))
    self.assertTrue(t2.__eq__(t1))
    self.assertFalse(t1.__ne__(t2))
    self.assertFalse(t2.__ne__(t1))
    # different ordering
    t3 = anonymous_tuple.AnonymousTuple([('b', 2), ('a', 1)])
    self.assertFalse(t1.__eq__(t3))
    self.assertFalse(t3.__eq__(t1))
    self.assertTrue(t1.__ne__(t3))
    self.assertTrue(t3.__ne__(t1))
    # different names
    t4 = anonymous_tuple.AnonymousTuple([('c', 1), ('d', 2)])
    self.assertFalse(t1.__eq__(t4))
    self.assertFalse(t4.__eq__(t1))
    self.assertTrue(t1.__ne__(t4))
    self.assertTrue(t4.__ne__(t1))
    # different values
    t5 = anonymous_tuple.AnonymousTuple([('a', 10), ('b', 10)])
    self.assertFalse(t1.__eq__(t5))
    self.assertFalse(t5.__eq__(t1))
    self.assertTrue(t1.__ne__(t5))
    self.assertTrue(t5.__ne__(t1))

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
    self.assertEqual(x[:], tuple(range(0, 50, 10)))
    self.assertEqual(x[::-1], tuple(reversed(range(0, 50, 10))))
    self.assertEqual(x[:-1], tuple(range(0, 40, 10)))
    self.assertEqual(x[1:], tuple(range(10, 50, 10)))
    self.assertEqual(x[-1:], (40,))

  def test_getitem_bad_bounds(self):
    v = [(None, i) for i in range(0, 50, 10)]
    x = anonymous_tuple.AnonymousTuple(v)
    with self.assertRaises(IndexError):
      _ = x[10]

  def test_pack_sequence_as_fails_non_anonymous_tuple(self):
    x = anonymous_tuple.AnonymousTuple([
        ('a', 10),
        ('b', {
            'd': 20
        }),
        ('c', 30),
    ])
    y = [10, 20, 30]
    with self.assertRaisesRegex(TypeError, 'Cannot pack sequence'):
      _ = anonymous_tuple.pack_sequence_as(x, y)

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
      anonymous_tuple.is_same_structure(
          {'x': 5.0},  # not an AnonymousTuple
          anonymous_tuple.AnonymousTuple([('x', 5.0)]))

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

  def test_from_container_with_none(self):
    with self.assertRaises(TypeError):
      anonymous_tuple.from_container(None)

  def test_from_container_with_int(self):
    with self.assertRaises(TypeError):
      anonymous_tuple.from_container(10)

  def test_from_container_with_list(self):
    x = anonymous_tuple.from_container([10, 20])
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(x), '<10,20>')

  def test_from_container_with_tuple(self):
    x = anonymous_tuple.from_container(tuple([10, 20]))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(x), '<10,20>')

  def test_from_container_with_dict(self):
    x = anonymous_tuple.from_container({'z': 10, 'y': 20, 'a': 30})
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(x), '<a=30,y=20,z=10>')

  def test_from_container_with_ordered_dict(self):
    x = anonymous_tuple.from_container(
        collections.OrderedDict([('z', 10), ('y', 20), ('a', 30)]))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(x), '<z=10,y=20,a=30>')

  def test_from_container_with_namedtuple(self):
    x = anonymous_tuple.from_container(collections.namedtuple('_', 'x y')(1, 2))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(x), '<x=1,y=2>')

  def test_from_container_with_attrs_class(self):

    @attr.s
    class TestFoo(object):
      x = attr.ib()
      y = attr.ib()

    x = anonymous_tuple.from_container(TestFoo(1, 2))
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(str(x), '<x=1,y=2>')

  def test_from_container_with_anonymous_tuple(self):
    x = anonymous_tuple.from_container(
        anonymous_tuple.AnonymousTuple([('a', 10), ('b', 20)]))
    self.assertIs(x, x)

  def test_from_container_with_namedtuple_of_odict_recursive(self):
    x = anonymous_tuple.from_container(
        collections.namedtuple('_',
                               'x y')(collections.OrderedDict([('a', 10),
                                                               ('b', 20)]),
                                      collections.OrderedDict([('c', 30),
                                                               ('d', 40)])),
        recursive=True)
    self.assertEqual(str(x), '<x=<a=10,b=20>,y=<c=30,d=40>>')

  def test_to_container_recursive(self):

    def odict(**kwargs):
      return collections.OrderedDict(sorted(list(kwargs.items())))

    # Nested OrderedDicts.
    s = odict(a=1, b=2, c=odict(d=3, e=odict(f=4, g=5)))
    x = anonymous_tuple.from_container(s, recursive=True)
    s2 = x._asdict(recursive=True)
    self.assertEqual(s, s2)

    # Single OrderedDict.
    s = odict(a=1, b=2)
    x = anonymous_tuple.from_container(s)
    self.assertEqual(x._asdict(recursive=True), s)

    # Single empty OrderedDict.
    s = odict()
    x = anonymous_tuple.from_container(s)
    self.assertEqual(x._asdict(recursive=True), s)

    # Invalid argument.
    with self.assertRaises(TypeError):
      anonymous_tuple.from_container(3)


if __name__ == '__main__':
  absltest.main()
