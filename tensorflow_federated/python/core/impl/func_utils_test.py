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
"""Tests for func_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import itertools

# Dependency imports

from absl.testing import parameterized

import tensorflow as tf

from tensorflow.python.framework import function as tf_function

from tensorflow_federated.python.core.api import types

from tensorflow_federated.python.core.impl import func_utils as fu

from tensorflow_federated.python.core.impl.anonymous_tuple import AnonymousTuple


class FuncUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_is_defun(self):
    self.assertTrue(fu.is_defun(tf_function.Defun()(lambda x: None)))
    self.assertTrue(fu.is_defun(tf_function.Defun(tf.int32)(lambda x: None)))
    self.assertFalse(fu.is_defun(tf_function.Defun))
    self.assertFalse(fu.is_defun(lambda x: None))
    self.assertFalse(fu.is_defun(None))

  def test_get_defun_argspec_with_typed_non_eager_defun(self):
    # In a non-eager defun with a defined input signature, **kwargs or default
    # values are not allowed, but *args are, and the input signature may
    # overlap with *args.
    self.assertEqual(
        fu.get_argspec(
            tf_function.Defun(tf.int32, tf.bool, tf.float32, tf.float32)(
                lambda x, y, *z: None)),
        inspect.ArgSpec(
            args=['x', 'y'],
            varargs='z',
            keywords=None,
            defaults=None))

  def test_get_defun_argspec_with_untyped_non_eager_defun(self):
    # In a non-eager defun with no input signature, the same restrictions as in
    # a typed defun apply.
    self.assertEqual(
        fu.get_argspec(
            tf_function.Defun()(lambda x, y, *z: None)),
        inspect.ArgSpec(
            args=['x', 'y'],
            varargs='z',
            keywords=None,
            defaults=None))

  @parameterized.parameters(
      itertools.product(
          # Values of 'func' to test.
          [lambda: None,
           lambda a: None,
           lambda a, b: None,
           lambda *a: None,
           lambda **a: None,
           lambda *a, **b: None,
           lambda a, *b: None,
           lambda a, **b: None,
           lambda a, b, **c: None,
           lambda a, b=10: None,
           lambda a, b=10, c=20: None,
           lambda a, b=10, *c: None,
           lambda a, b=10, **c: None,
           lambda a, b=10, *c, **d: None,
           lambda a, b, c=10, *d: None,
           lambda a=10, b=20, c=30, **d: None],
          # Values of 'args' to test.
          [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
          # Values of 'kwargs' to test.
          [{}, {'b': 100}, {'name': 'foo'}, {'b': 100, 'name': 'foo'}]))
  def test_get_callargs_for_argspec(self, func, args, kwargs):
    argspec = inspect.getargspec(func)
    try:
      expected_callargs = inspect.getcallargs(func, *args, **kwargs)
      expected_error = None
    except TypeError as expected_error:
      expected_callargs = None
    try:
      if not expected_error:
        result_callargs = fu.get_callargs_for_argspec(argspec, *args, **kwargs)
        self.assertEqual(result_callargs, expected_callargs)
      else:
        with self.assertRaises(TypeError):
          result_callargs = fu.get_callargs_for_argspec(
              argspec, *args, **kwargs)
    except (TypeError, AssertionError) as test_err:
      raise AssertionError(
          'With argspec {}, args {}, kwargs {}, expected callargs {} and '
          'error {}, tested function returned {} and the test has failed '
          'with message: {}'.format(
              str(argspec), str(args), str(kwargs),
              str(expected_callargs), str(expected_error),
              str(result_callargs), str(test_err)))

  @parameterized.parameters(
      (inspect.getargspec(params[0]),) + params[1:] for params in [
          (lambda a: None, [tf.int32], {}, True),
          (lambda a=True: None, [tf.int32], {}, False),
          (lambda a, b=True: None, [tf.int32, tf.bool], {}, True),
          (lambda a, b=True: None, [tf.int32], {'b': tf.bool}, True),
          (lambda a, b=True: None, [tf.bool], {'b': tf.bool}, True),
          (lambda a=10, b=True: None, [tf.int32], {'b': tf.bool}, True),
          (lambda a=10, b=True: None, [tf.bool], {'b': tf.bool}, False)])
  def test_is_argspec_compatible_with_types(
      self, argspec, args, kwargs, expected_result):
    self.assertEqual(
        fu.is_argspec_compatible_with_types(
            argspec,
            *[types.to_type(a) for a in args],
            **{k: types.to_type(v) for k, v in kwargs.iteritems()}),
        expected_result)

  @parameterized.parameters(
      (tf.int32, False),
      ([tf.int32, tf.int32], True),
      ([tf.int32, ('b', tf.int32)], True),
      ([('a', tf.int32), ('b', tf.int32)], True),
      ([('a', tf.int32), tf.int32], False),
      (AnonymousTuple([(None, 1), ('a', 2)]), True),
      (AnonymousTuple([('a', 1), (None, 2)]), False))
  def test_is_argument_tuple(self, arg, expected_result):
    self.assertEqual(fu.is_argument_tuple(arg), expected_result)

  @parameterized.parameters(
      (AnonymousTuple([(None, 1)]), [1], {}),
      (AnonymousTuple([(None, 1), ('a', 2)]), [1], {'a': 2}))
  def test_unpack_args_from_anonymous_tuple(
      self, tuple_with_args, expected_args, expected_kwargs):
    self.assertEqual(
        fu.unpack_args_from_tuple(tuple_with_args),
        (expected_args, expected_kwargs))

  @parameterized.parameters(
      ([tf.int32], [tf.int32], {}),
      ([('a', tf.int32)], [], {'a': tf.int32}),
      ([tf.int32, tf.bool], [tf.int32, tf.bool], {}),
      ([tf.int32, ('b', tf.bool)], [tf.int32], {'b': tf.bool}),
      ([('a', tf.int32), ('b', tf.bool)], [], {'a': tf.int32, 'b': tf.bool}))
  def test_unpack_args_from_tuple_type(
      self, tuple_with_args, expected_args, expected_kwargs):
    args, kwargs = fu.unpack_args_from_tuple(tuple_with_args)
    self.assertEqual(args, [types.to_type(a) for a in expected_args])
    self.assertEqual(
        kwargs, {k: types.to_type(v) for k, v in expected_kwargs.iteritems()})

  def test_pack_args_into_anonymous_tuple(self):
    self.assertEqual(
        fu.pack_args_into_anonymous_tuple(1, a=10),
        AnonymousTuple([(None, 1), ('a', 10)]))
    self.assertIn(
        fu.pack_args_into_anonymous_tuple(1, 2, a=10, b=20), [
            AnonymousTuple([(None, 1), (None, 2), ('a', 10), ('b', 20)]),
            AnonymousTuple([(None, 1), (None, 2), ('b', 20), ('a', 10)])])
    self.assertIn(
        fu.pack_args_into_anonymous_tuple(a=10, b=20), [
            AnonymousTuple([('a', 10), ('b', 20)]),
            AnonymousTuple([('b', 20), ('a', 10)])])
    self.assertEqual(
        fu.pack_args_into_anonymous_tuple(1),
        AnonymousTuple([(None, 1)]))

  @parameterized.parameters(
      (1, lambda: 10, None, None, None, 10),
      (2, lambda x=1: x + 10, None, None, None, 11),
      (3, lambda x=1: x + 10, tf.int32, None, 20, 30),
      (4, lambda x, y: x + y, [tf.int32, tf.int32], None,
       AnonymousTuple([('x', 5), ('y', 6)]), 11),
      (5, lambda *args: str(args), [tf.int32, tf.int32], True,
       AnonymousTuple([('x', 5), ('y', 6)]), '(5, 6)'),
      (6, lambda *args: str(args), [tf.int32, tf.int32], False,
       AnonymousTuple([('x', 5), ('y', 6)]),
       '(AnonymousTuple([(x, 5), (y, 6)]),)'))
  def test_wrap_as_zero_or_one_arg_callable(
      self, unused_index, func, parameter_type, unpack, arg, expected_result):
    wrapped_fn = fu.wrap_as_zero_or_one_arg_callable(
        func, parameter_type, unpack)
    actual_result = wrapped_fn(arg) if parameter_type else wrapped_fn()
    self.assertEqual(actual_result, expected_result)

  def test_polymorphic_function(self):

    class TestFunction(object):

      def __init__(self, name, parameter_type):
        self._name = name
        self._parameter_type = parameter_type

      def __call__(self, *args, **kwargs):
        return 'name={},type={},args=({})'.format(
            self._name, str(self._parameter_type), ','.join(
                [str(a) for a in args] +
                ['{}={}'.format(k, str(v)) for k, v in kwargs.iteritems()]))

    class TestFunctionFactory(object):

      def __init__(self):
        self._count = 0

      def __call__(self, parameter_type):
        self._count = self._count + 1
        return TestFunction(str(self._count), parameter_type)

    fn = fu.PolymorphicFunction(TestFunctionFactory())
    self.assertEqual(fn(10), 'name=1,type=<int32>,args=(10)')
    self.assertEqual(
        fn(20, x=True), 'name=2,type=<int32,x=bool>,args=(20,x=True)')
    self.assertEqual(fn(True), 'name=3,type=<bool>,args=(True)')
    self.assertEqual(fn(30, x=40), 'name=4,type=<int32,x=int32>,args=(30,x=40)')
    self.assertEqual(fn(50), 'name=1,type=<int32>,args=(50)')
    self.assertEqual(
        fn(0, x=False), 'name=2,type=<int32,x=bool>,args=(0,x=False)')
    self.assertEqual(fn(False), 'name=3,type=<bool>,args=(False)')
    self.assertEqual(fn(60, x=70), 'name=4,type=<int32,x=int32>,args=(60,x=70)')


if __name__ == '__main__':
  tf.test.main()
