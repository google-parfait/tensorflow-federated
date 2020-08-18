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

from tensorflow_federated.python.common_libs import structure


class StructTest(absltest.TestCase):

  def test_construction_from_list(self):
    v = [('a', 1), ('b', 2), (None, 3)]
    x = structure.Struct(v)
    self.assertSequenceEqual(structure.to_elements(x), v)

  def test_construction_from_tuple(self):
    v = (('a', 1), ('b', 2), (None, 3))
    x = structure.Struct(v)
    self.assertSequenceEqual(structure.to_elements(x), v)

  def test_construction_from_ordereddict(self):
    v = collections.OrderedDict(a=1, b=2, c=3)
    x = structure.Struct(v.items())
    self.assertSequenceEqual(structure.to_elements(x), list(v.items()))

  def test_construction_from_generator_expression(self):
    x = structure.Struct((name, i) for i, name in enumerate(('a', 'b', None)))
    self.assertSequenceEqual(
        structure.to_elements(x), [('a', 0), ('b', 1), (None, 2)])

  def test_construction_from_iter_elements(self):
    x = structure.Struct((('a', 1), ('b', 2), (None, 3)))
    self.assertSequenceEqual(structure.Struct(structure.iter_elements(x)), x)

  def test_empty(self):
    v = []
    x = structure.Struct(v)
    # Explicitly test the implementation of __len__() here so use, assertLen()
    # instead of assertEmpty().
    self.assertLen(x, 0)  # pylint: disable=g-generic-assert
    self.assertRaises(IndexError, lambda _: x[0], None)
    self.assertEqual(list(iter(x)), [])
    self.assertEqual(dir(x), [])
    self.assertRaises(AttributeError, lambda _: x.foo, None)
    self.assertEqual(x, structure.Struct([]))
    self.assertNotEqual(x, structure.Struct([('foo', 10)]))
    self.assertEqual(structure.to_elements(x), v)
    self.assertEqual(structure.to_odict(x), collections.OrderedDict())
    self.assertEqual(repr(x), 'Struct([])')
    self.assertEqual(str(x), '<>')

  def test_single_unnamed(self):
    v = [(None, 10)]
    x = structure.Struct(v)
    self.assertLen(x, 1)
    self.assertRaises(IndexError, lambda _: x[1], None)
    self.assertEqual(x[0], 10)
    self.assertEqual(list(iter(x)), [10])
    self.assertEqual(dir(x), [])
    self.assertRaises(AttributeError, lambda _: x.foo, None)
    self.assertNotEqual(x, structure.Struct([]))
    self.assertNotEqual(x, structure.Struct([('foo', 10)]))
    self.assertEqual(x, structure.Struct([(None, 10)]))
    self.assertNotEqual(x, structure.Struct([(None, 10), ('foo', 20)]))
    self.assertEqual(structure.to_elements(x), v)
    self.assertEqual(repr(x), 'Struct([(None, 10)])')
    self.assertEqual(str(x), '<10>')
    with self.assertRaisesRegex(ValueError, 'unnamed'):
      structure.to_odict(x)

  def test_single_named(self):
    v = [('foo', 20)]
    x = structure.Struct(v)
    self.assertLen(x, 1)
    self.assertEqual(x[0], 20)
    self.assertRaises(IndexError, lambda _: x[1], None)
    self.assertEqual(list(iter(x)), [20])
    self.assertEqual(dir(x), ['foo'])
    self.assertEqual(x.foo, 20)
    self.assertRaises(AttributeError, lambda _: x.bar, None)
    self.assertNotEqual(x, structure.Struct([]))
    self.assertNotEqual(x, structure.Struct([('foo', 10)]))
    self.assertNotEqual(x, structure.Struct([(None, 20)]))
    self.assertEqual(x, structure.Struct([('foo', 20)]))
    self.assertNotEqual(x, structure.Struct([('foo', 20), ('bar', 30)]))
    self.assertEqual(structure.to_elements(x), v)
    self.assertEqual(repr(x), 'Struct([(\'foo\', 20)])')
    self.assertEqual(str(x), '<foo=20>')
    self.assertEqual(structure.to_odict(x), collections.OrderedDict(v))

  def test_multiple_named_and_unnamed(self):
    v = [(None, 10), ('foo', 20), ('bar', 30)]
    x = structure.Struct(v)
    self.assertLen(x, 3)
    self.assertEqual(x[0], 10)
    self.assertEqual(x[1], 20)
    self.assertEqual(x[2], 30)
    self.assertRaises(IndexError, lambda _: x[3], None)
    self.assertEqual(list(iter(x)), [10, 20, 30])
    self.assertEqual(dir(x), ['bar', 'foo'])
    self.assertEqual(structure.name_list(x), ['foo', 'bar'])
    self.assertEqual(x.foo, 20)
    self.assertEqual(x.bar, 30)
    self.assertRaises(AttributeError, lambda _: x.baz, None)
    self.assertEqual(x, structure.Struct([(None, 10), ('foo', 20),
                                          ('bar', 30)]))
    self.assertNotEqual(
        x, structure.Struct([('foo', 10), ('bar', 20), (None, 30)]))
    self.assertEqual(structure.to_elements(x), v)
    self.assertEqual(
        repr(x), 'Struct([(None, 10), (\'foo\', 20), (\'bar\', 30)])')
    self.assertEqual(str(x), '<10,foo=20,bar=30>')
    with self.assertRaisesRegex(ValueError, 'unnamed'):
      structure.to_odict(x)

  def test_bad_names(self):
    with self.assertRaisesRegex(ValueError, 'duplicated.*foo'):
      structure.Struct([('foo', 20), ('foo', 30)])

    with self.assertRaisesRegex(ValueError, '_asdict.*reserved'):
      structure.Struct([('_asdict', 40)])

    with self.assertRaisesRegex(ValueError, '_element_array.*reserved'):
      structure.Struct([('_element_array', 40)])

    with self.assertRaisesRegex(ValueError, '_name_to_index.*reserved'):
      structure.Struct([('_name_to_index', 40)])

    with self.assertRaisesRegex(ValueError, '_name_array.*reserved'):
      structure.Struct([('_name_array', 40)])

    with self.assertRaisesRegex(ValueError, '_hash.*reserved'):
      structure.Struct([('_hash', 40)])

  def test_immutable(self):
    v = [('foo', 'a string'), ('bar', 1), ('baz', [1.0, 2.0, 3.0])]
    t = structure.Struct(v)

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
    t1 = structure.Struct([(None, 1), (None, 2)])
    self.assertTrue(t1.__eq__(t1))
    self.assertFalse(t1.__ne__(t1))
    # different type
    self.assertFalse(t1.__eq__(None))
    self.assertTrue(t1.__ne__(None))
    # copy
    t2 = structure.Struct([(None, 1), (None, 2)])
    self.assertTrue(t1.__eq__(t2))
    self.assertTrue(t2.__eq__(t1))
    self.assertFalse(t1.__ne__(t2))
    self.assertFalse(t2.__ne__(t1))
    # different ordering
    t3 = structure.Struct([(None, 2), (None, 1)])
    self.assertFalse(t1.__eq__(t3))
    self.assertFalse(t3.__eq__(t1))
    self.assertTrue(t1.__ne__(t3))
    self.assertTrue(t3.__ne__(t1))
    # different names
    t4 = structure.Struct([('a', 1), ('b', 2)])
    self.assertFalse(t1.__eq__(t4))
    self.assertFalse(t4.__eq__(t1))
    self.assertTrue(t1.__ne__(t4))
    self.assertTrue(t4.__ne__(t1))
    # different values
    t5 = structure.Struct([(None, 10), (None, 10)])
    self.assertFalse(t1.__eq__(t5))
    self.assertFalse(t5.__eq__(t1))
    self.assertTrue(t1.__ne__(t5))
    self.assertTrue(t5.__ne__(t1))

  def test_equality_named(self):
    # identity
    t1 = structure.Struct([('a', 1), ('b', 2)])
    self.assertTrue(t1.__eq__(t1))
    self.assertFalse(t1.__ne__(t1))
    # different type
    self.assertFalse(t1.__eq__(None))
    self.assertTrue(t1.__ne__(None))
    # copy
    t2 = structure.Struct([('a', 1), ('b', 2)])
    self.assertTrue(t1.__eq__(t2))
    self.assertTrue(t2.__eq__(t1))
    self.assertFalse(t1.__ne__(t2))
    self.assertFalse(t2.__ne__(t1))
    # different ordering
    t3 = structure.Struct([('b', 2), ('a', 1)])
    self.assertFalse(t1.__eq__(t3))
    self.assertFalse(t3.__eq__(t1))
    self.assertTrue(t1.__ne__(t3))
    self.assertTrue(t3.__ne__(t1))
    # different names
    t4 = structure.Struct([('c', 1), ('d', 2)])
    self.assertFalse(t1.__eq__(t4))
    self.assertFalse(t4.__eq__(t1))
    self.assertTrue(t1.__ne__(t4))
    self.assertTrue(t4.__ne__(t1))
    # different values
    t5 = structure.Struct([('a', 10), ('b', 10)])
    self.assertFalse(t1.__eq__(t5))
    self.assertFalse(t5.__eq__(t1))
    self.assertTrue(t1.__ne__(t5))
    self.assertTrue(t5.__ne__(t1))

  def test_hash(self):
    v1 = [(str(i) if i > 30 else None, i) for i in range(0, 50, 10)]
    x1 = structure.Struct(v1)
    self.assertNotEqual(x1, v1)
    self.assertNotEqual(hash(x1), hash(iter(v1)))
    v2 = [(None, i) for i in range(0, 50, 10)]
    x2 = structure.Struct(v2)
    self.assertNotEqual(hash(x2), hash(iter(v2)))
    self.assertNotEqual(x1, x2)
    self.assertNotEqual(hash(x1), hash(x2))
    v3 = [(None, 0), (None, 10), (None, 20), (None, 30), (None, 40)]
    x3 = structure.Struct(v3)
    self.assertEqual(v2, v3)
    self.assertEqual(x2, x3)
    self.assertEqual(hash(x2), hash(x3))

  def test_slicing_behavior(self):
    v = [(None, i) for i in range(0, 50, 10)]
    x = structure.Struct(v)
    self.assertEqual(x[:], tuple(range(0, 50, 10)))
    self.assertEqual(x[::-1], tuple(reversed(range(0, 50, 10))))
    self.assertEqual(x[:-1], tuple(range(0, 40, 10)))
    self.assertEqual(x[1:], tuple(range(10, 50, 10)))
    self.assertEqual(x[-1:], (40,))

  def test_getitem_key(self):
    v = [('foo', 10), ('bar', 20)]
    x = structure.Struct(v)
    self.assertEqual(x['foo'], 10)
    self.assertEqual(x['bar'], 20)
    with self.assertRaises(AttributeError):
      _ = x['badkey']

  def test_getitem_key_builtin_attribute_raises(self):
    v = [('foo', 10), ('bar', 20)]
    x = structure.Struct(v)
    with self.assertRaises(AttributeError):
      _ = x['__getattr__']

  def test_getitem_bad_bounds(self):
    v = [(None, i) for i in range(0, 50, 10)]
    x = structure.Struct(v)
    with self.assertRaises(IndexError):
      _ = x[10]

  def test_pack_sequence_as_fails_non_struct(self):
    x = structure.Struct([
        ('a', 10),
        ('b', {
            'd': 20
        }),
        ('c', 30),
    ])
    y = [10, 20, 30]
    with self.assertRaisesRegex(TypeError, 'Cannot pack sequence'):
      _ = structure.pack_sequence_as(x, y)

  def test_flatten_and_pack_sequence_as(self):
    x = structure.Struct([
        ('a', 10),
        ('b',
         structure.Struct([
             ('x', structure.Struct([('p', 40)])),
             ('y', 30),
             ('z', structure.Struct([('q', 50), ('r', 60)])),
         ])),
        ('c', 20),
    ])
    y = structure.flatten(x)
    self.assertEqual(y, [10, 40, 30, 50, 60, 20])
    z = structure.pack_sequence_as(x, y)
    self.assertEqual(str(z), '<a=10,b=<x=<p=40>,y=30,z=<q=50,r=60>>,c=20>')

  def test_is_same_structure_check_types(self):
    self.assertTrue(
        structure.is_same_structure(
            structure.Struct([('a', 10)]), structure.Struct([('a', 20)])))
    self.assertTrue(
        structure.is_same_structure(
            structure.Struct([
                ('a', 10),
                ('b', structure.Struct([('z', 5)])),
            ]),
            structure.Struct([
                ('a', 20),
                ('b', structure.Struct([('z', 50)])),
            ])))
    self.assertFalse(
        structure.is_same_structure(
            structure.Struct([('x', {
                'y': 4
            })]), structure.Struct([('x', {
                'y': 5,
                'z': 6
            })])))
    self.assertTrue(
        structure.is_same_structure(
            structure.Struct([('x', {
                'y': 5
            })]), structure.Struct([('x', {
                'y': 6
            })])))
    with self.assertRaises(TypeError):
      structure.is_same_structure(
          {'x': 5.0},  # not an Struct
          structure.Struct([('x', 5.0)]))

  def test_map_structure(self):
    x = structure.Struct([
        ('a', 10),
        ('b',
         structure.Struct([
             ('x', structure.Struct([('p', 40)])),
             ('y', 30),
             ('z', structure.Struct([('q', 50), ('r', 60)])),
         ])),
        ('c', 20),
    ])
    y = structure.Struct([
        ('a', 1),
        ('b',
         structure.Struct([
             ('x', structure.Struct([('p', 4)])),
             ('y', 3),
             ('z', structure.Struct([('q', 5), ('r', 6)])),
         ])),
        ('c', 2),
    ])

    self.assertEqual(
        structure.map_structure(lambda x, y: x + y, x, y),
        structure.Struct([
            ('a', 11),
            ('b',
             structure.Struct([
                 ('x', structure.Struct([('p', 44)])),
                 ('y', 33),
                 ('z', structure.Struct([('q', 55), ('r', 66)])),
             ])),
            ('c', 22),
        ]))

  def test_from_container_with_none(self):
    with self.assertRaises(TypeError):
      structure.from_container(None)

  def test_from_container_with_int(self):
    with self.assertRaises(TypeError):
      structure.from_container(10)

  def test_from_container_with_list(self):
    x = structure.from_container([10, 20])
    self.assertIsInstance(x, structure.Struct)
    self.assertEqual(str(x), '<10,20>')

  def test_from_container_with_tuple(self):
    x = structure.from_container(tuple([10, 20]))
    self.assertIsInstance(x, structure.Struct)
    self.assertEqual(str(x), '<10,20>')

  def test_from_container_with_dict(self):
    x = structure.from_container({'z': 10, 'y': 20, 'a': 30})
    self.assertIsInstance(x, structure.Struct)
    self.assertEqual(str(x), '<a=30,y=20,z=10>')

  def test_from_container_with_ordered_dict(self):
    x = structure.from_container(
        collections.OrderedDict([('z', 10), ('y', 20), ('a', 30)]))
    self.assertIsInstance(x, structure.Struct)
    self.assertEqual(str(x), '<z=10,y=20,a=30>')

  def test_from_container_with_namedtuple(self):
    x = structure.from_container(collections.namedtuple('_', 'x y')(1, 2))
    self.assertIsInstance(x, structure.Struct)
    self.assertEqual(str(x), '<x=1,y=2>')

  def test_from_container_with_attrs_class(self):

    @attr.s
    class TestFoo(object):
      x = attr.ib()
      y = attr.ib()

    x = structure.from_container(TestFoo(1, 2))
    self.assertIsInstance(x, structure.Struct)
    self.assertEqual(str(x), '<x=1,y=2>')

  def test_from_container_with_struct(self):
    x = structure.from_container(structure.Struct([('a', 10), ('b', 20)]))
    self.assertIs(x, x)

  def test_from_container_with_namedtuple_of_odict_recursive(self):
    x = structure.from_container(
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
    x = structure.from_container(s, recursive=True)
    s2 = x._asdict(recursive=True)
    self.assertEqual(s, s2)

    # Single OrderedDict.
    s = odict(a=1, b=2)
    x = structure.from_container(s)
    self.assertEqual(x._asdict(recursive=True), s)

    # Single empty OrderedDict.
    s = odict()
    x = structure.from_container(s)
    self.assertEqual(x._asdict(recursive=True), s)

    # Invalid argument.
    with self.assertRaises(TypeError):
      structure.from_container(3)

  def test_name_to_index_map_empty_unnamed_struct(self):
    unnamed_struct = structure.Struct([(None, 10), (None, 20)])
    self.assertEmpty(structure.name_to_index_map(unnamed_struct))

  def test_name_to_index_map_partially_named_struct(self):
    partially_named_struct = structure.Struct([(None, 10), ('a', 20)])

    name_to_index_dict = structure.name_to_index_map(partially_named_struct)
    expected_name_to_index_map = {'a': 1}
    self.assertEqual(name_to_index_dict, expected_name_to_index_map)

  def test_name_to_index_map_fully_named_struct(self):
    partially_named_struct = structure.Struct([('b', 10), ('a', 20)])

    name_to_index_dict = structure.name_to_index_map(partially_named_struct)
    expected_name_to_index_map = {'b': 0, 'a': 1}
    self.assertEqual(name_to_index_dict, expected_name_to_index_map)


if __name__ == '__main__':
  absltest.main()
