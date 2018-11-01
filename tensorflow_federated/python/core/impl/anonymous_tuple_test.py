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

import unittest

from tensorflow_federated.python.core.impl.anonymous_tuple import AnonymousTuple
from tensorflow_federated.python.core.impl.anonymous_tuple import to_elements


class AnonymousTupleTest(unittest.TestCase):

  def test_empty(self):
    v = []
    x = AnonymousTuple(v)
    self.assertEqual(len(x), 0)
    self.assertRaises(IndexError, lambda _: x[0], None)
    self.assertEqual(list(iter(x)), [])
    self.assertEqual(dir(x), [])
    self.assertRaises(ValueError, lambda _: x.foo, None)
    self.assertEqual(x, AnonymousTuple([]))
    self.assertNotEqual(x, AnonymousTuple([('foo', 10)]))
    self.assertEqual(to_elements(x), v)
    self.assertEqual(repr(x), 'AnonymousTuple([])')
    self.assertEqual(str(x), '<>')

  def test_single_unnamed(self):
    v = [(None, 10)]
    x = AnonymousTuple(v)
    self.assertEqual(len(x), 1)
    self.assertRaises(IndexError, lambda _: x[1], None)
    self.assertEqual(x[0], 10)
    self.assertEqual(list(iter(x)), [10])
    self.assertEqual(dir(x), [])
    self.assertRaises(ValueError, lambda _: x.foo, None)
    self.assertNotEqual(x, AnonymousTuple([]))
    self.assertNotEqual(x, AnonymousTuple([('foo', 10)]))
    self.assertEqual(x, AnonymousTuple([(None, 10)]))
    self.assertNotEqual(x, AnonymousTuple([(None, 10), ('foo', 20)]))
    self.assertEqual(to_elements(x), v)
    self.assertEqual(repr(x), 'AnonymousTuple([(None, 10)])')
    self.assertEqual(str(x), '<10>')

  def test_single_named(self):
    v = [('foo', 20)]
    x = AnonymousTuple(v)
    self.assertEqual(len(x), 1)
    self.assertEqual(x[0], 20)
    self.assertRaises(IndexError, lambda _: x[1], None)
    self.assertEqual(list(iter(x)), [20])
    self.assertEqual(dir(x), ['foo'])
    self.assertEqual(x.foo, 20)
    self.assertRaises(ValueError, lambda _: x.bar, None)
    self.assertNotEqual(x, AnonymousTuple([]))
    self.assertNotEqual(x, AnonymousTuple([('foo', 10)]))
    self.assertNotEqual(x, AnonymousTuple([(None, 20)]))
    self.assertEqual(x, AnonymousTuple([('foo', 20)]))
    self.assertNotEqual(x, AnonymousTuple([('foo', 20), ('bar', 30)]))
    self.assertEqual(to_elements(x), v)
    self.assertEqual(repr(x), 'AnonymousTuple([(foo, 20)])')
    self.assertEqual(str(x), '<foo=20>')

  def test_multiple_named_and_unnamed(self):
    v = [(None, 10), ('foo', 20), ('bar', 30)]
    x = AnonymousTuple(v)
    self.assertEqual(len(x), 3)
    self.assertEqual(x[0], 10)
    self.assertEqual(x[1], 20)
    self.assertEqual(x[2], 30)
    self.assertRaises(IndexError, lambda _: x[3], None)
    self.assertEqual(list(iter(x)), [10, 20, 30])
    self.assertEqual(sorted(dir(x)), ['bar', 'foo'])
    self.assertEqual(x.foo, 20)
    self.assertEqual(x.bar, 30)
    self.assertRaises(ValueError, lambda _: x.baz, None)
    self.assertEqual(x, AnonymousTuple([(None, 10), ('foo', 20), ('bar', 30)]))
    self.assertNotEqual(
        x, AnonymousTuple([('foo', 10), ('bar', 20), (None, 30)]))
    self.assertEqual(to_elements(x), v)
    self.assertEqual(
        repr(x), 'AnonymousTuple([(None, 10), (foo, 20), (bar, 30)])')
    self.assertEqual(str(x), '<10,foo=20,bar=30>')


if __name__ == '__main__':
  unittest.main()
