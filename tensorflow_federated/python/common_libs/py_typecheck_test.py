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
from tensorflow_federated.python.common_libs import py_typecheck


class PyTypeCheckTest(absltest.TestCase):

  def test_check_type(self):
    try:
      self.assertEqual('foo', py_typecheck.check_type('foo', str))
      py_typecheck.check_type('foo', str)
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

  def test_check_none(self):
    py_typecheck.check_none(None)
    with self.assertRaises(TypeError):
      py_typecheck.check_none(10)
    with self.assertRaisesRegex(TypeError, 'foo'):
      py_typecheck.check_none(10, 'foo')

  def test_check_not_none(self):
    py_typecheck.check_not_none(10)
    with self.assertRaises(TypeError):
      py_typecheck.check_not_none(None)
    with self.assertRaisesRegex(TypeError, 'foo'):
      py_typecheck.check_not_none(None, 'foo')

  def test_check_subclass(self):
    py_typecheck.check_subclass(PyTypeCheckTest, absltest.TestCase)
    py_typecheck.check_subclass(PyTypeCheckTest, (absltest.TestCase, int))
    py_typecheck.check_subclass(int, (int, float))
    py_typecheck.check_subclass(float, float)
    with self.assertRaisesRegex(TypeError, 'Expected .* to subclass '):
      py_typecheck.check_subclass(int, float)
      py_typecheck.check_subclass(int, (float, float))

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

    # Not named tuples
    self.assertFalse(
        py_typecheck.is_named_tuple(
            anonymous_tuple.AnonymousTuple([(None, 10)])))
    self.assertFalse(py_typecheck.is_named_tuple([]))
    self.assertFalse(py_typecheck.is_named_tuple(tuple()))

  def test_is_name_value_pair(self):
    self.assertTrue(py_typecheck.is_name_value_pair(('a', 1)))
    self.assertTrue(py_typecheck.is_name_value_pair(['a', 1]))
    self.assertTrue(py_typecheck.is_name_value_pair(('a', 'b')))
    self.assertFalse(py_typecheck.is_name_value_pair({'a': 1}))
    self.assertFalse(py_typecheck.is_name_value_pair({'a': 1, 'b': 2}))
    self.assertFalse(py_typecheck.is_name_value_pair(('a')))
    self.assertFalse(py_typecheck.is_name_value_pair(('a', 'b', 'c')))
    self.assertFalse(py_typecheck.is_name_value_pair((None, 1)))
    self.assertFalse(py_typecheck.is_name_value_pair((1, 1)))

  def test_is_name_value_pair_with_no_name_required(self):
    self.assertTrue(
        py_typecheck.is_name_value_pair(('a', 1), name_required=False))
    self.assertTrue(
        py_typecheck.is_name_value_pair(['a', 1], name_required=False))
    self.assertTrue(
        py_typecheck.is_name_value_pair(('a', 'b'), name_required=False))
    self.assertFalse(
        py_typecheck.is_name_value_pair({'a': 1}, name_required=False))
    self.assertFalse(
        py_typecheck.is_name_value_pair({
            'a': 1,
            'b': 2,
        }, name_required=False))
    self.assertFalse(
        py_typecheck.is_name_value_pair(('a'), name_required=False))
    self.assertFalse(
        py_typecheck.is_name_value_pair(('a', 'b', 'c'), name_required=False))
    self.assertTrue(
        py_typecheck.is_name_value_pair((None, 1), name_required=False))
    self.assertFalse(
        py_typecheck.is_name_value_pair((1, 1), name_required=False))

  def test_is_name_value_pair_with_value_type(self):
    self.assertTrue(py_typecheck.is_name_value_pair(('a', 1), value_type=int))
    self.assertTrue(py_typecheck.is_name_value_pair(['a', 1], value_type=int))
    self.assertFalse(
        py_typecheck.is_name_value_pair(('a', 'b'), value_type=int))
    self.assertFalse(py_typecheck.is_name_value_pair({'a': 1}, value_type=int))
    self.assertFalse(
        py_typecheck.is_name_value_pair({
            'a': 1,
            'b': 2,
        }, value_type=int))
    self.assertFalse(py_typecheck.is_name_value_pair(('a'), value_type=int))
    self.assertFalse(
        py_typecheck.is_name_value_pair(('a', 'b', 'c'), value_type=int))
    self.assertFalse(py_typecheck.is_name_value_pair((None, 1), value_type=int))
    self.assertFalse(py_typecheck.is_name_value_pair((1, 1), value_type=int))


if __name__ == '__main__':
  absltest.main()
