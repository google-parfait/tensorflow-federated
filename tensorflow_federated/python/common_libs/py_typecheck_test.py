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
"""Tests for py_typecheck.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
import attr
import six

from tensorflow_federated.python.common_libs import py_typecheck


class PyTypeCheckTest(absltest.TestCase):

  def test_check_type(self):
    try:
      self.assertEqual('foo', py_typecheck.check_type('foo', str))
      py_typecheck.check_type('foo', six.string_types)
      py_typecheck.check_type(10, int)
      py_typecheck.check_type(10, (str, int))
      py_typecheck.check_type(10, (str, int, bool, float))
    except TypeError:
      self.fail('Function {} raised TypeError unexpectedly.'.format(
          py_typecheck.check_type.__name__))
    self.assertRaisesRegex(TypeError, 'Expected .*TestCase, found int.',
                           py_typecheck.check_type, 10, absltest.TestCase)
    self.assertRaisesRegex(
        TypeError,
        'Expected foo to be of type int, found __main__.PyTypeCheckTest.',
        py_typecheck.check_type,
        self,
        int,
        label='foo')
    self.assertRaisesRegex(TypeError, 'Expected int or bool, found str.',
                           py_typecheck.check_type, 'a', (int, bool))
    self.assertRaisesRegex(TypeError,
                           'Expected int, bool, or float, found str.',
                           py_typecheck.check_type, 'a', (int, bool, float))

  def test_check_subclass(self):
    py_typecheck.check_subclass(PyTypeCheckTest, absltest.TestCase)
    py_typecheck.check_subclass(PyTypeCheckTest, (absltest.TestCase, int))
    py_typecheck.check_subclass(int, (int, float))
    py_typecheck.check_subclass(float, float)
    with self.assertRaisesRegex(TypeError, 'Expected .* to subclass '):
      py_typecheck.check_subclass(int, float)
      py_typecheck.check_subclass(int, (float, float))
    with self.assertRaisesRegex(TypeError, 'Expected a class,'):
      py_typecheck.check_subclass(0, int)
      py_typecheck.check_subclass(int, 0)
      py_typecheck.check_subclass(int, (int, 0))

  def test_check_callable(self):
    try:
      f = lambda x: x + 10
      self.assertEqual(py_typecheck.check_callable(f), f)
    except TypeError:
      self.fail('Function {} raised TypeError unexpectedly.'.format(
          py_typecheck.check_callable.__name__))
    self.assertRaisesRegex(TypeError,
                           'Expected a callable, found non-callable int.',
                           py_typecheck.check_callable, 10)

  def test_is_named_tuple(self):
    T = collections.namedtuple('T', ['a', 'b'])  # pylint: disable=invalid-name

    class U(T):
      pass

    t = T(1, 2)
    self.assertIn('_asdict', vars(type(t)))
    self.assertTrue(py_typecheck.is_named_tuple(t))
    self.assertTrue(py_typecheck.is_named_tuple(T))
    u = U(3, 4)
    self.assertNotIn('_asdict', vars(type(u)))
    self.assertTrue(py_typecheck.is_named_tuple(u))
    self.assertTrue(py_typecheck.is_named_tuple(U))

  def test_is_name_value_pair(self):
    self.assertTrue(py_typecheck.is_name_value_pair(['a', 1]))
    self.assertTrue(py_typecheck.is_name_value_pair(['a', [1, 2]]))
    self.assertTrue(py_typecheck.is_name_value_pair(('a', 1)))
    self.assertTrue(py_typecheck.is_name_value_pair(('a', [1, 2])))

    self.assertFalse(py_typecheck.is_name_value_pair([0, 'a']))
    self.assertFalse(py_typecheck.is_name_value_pair((0, 'a')))
    self.assertFalse(py_typecheck.is_name_value_pair('a'))
    self.assertFalse(py_typecheck.is_name_value_pair('abc'))
    self.assertFalse(py_typecheck.is_name_value_pair(['abc']))
    self.assertFalse(py_typecheck.is_name_value_pair(('abc')))
    self.assertFalse(py_typecheck.is_name_value_pair((None, 0)))
    self.assertFalse(py_typecheck.is_name_value_pair([None, 0]))
    self.assertFalse(py_typecheck.is_name_value_pair({'a': 1}))

  def test_is_attr(self):

    @attr.s
    class TestAttrClass(object):
      a = attr.ib(default=0)

    class TestClass(object):
      a = 0

    self.assertTrue(py_typecheck.is_attrs(TestAttrClass))
    self.assertTrue(py_typecheck.is_attrs(TestAttrClass()))
    self.assertFalse(py_typecheck.is_attrs(0))
    self.assertFalse(py_typecheck.is_attrs(TestClass))
    self.assertFalse(py_typecheck.is_attrs(TestClass()))


if __name__ == '__main__':
  absltest.main()
