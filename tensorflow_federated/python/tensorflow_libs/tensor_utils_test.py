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
"""Tests for tensor_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import tensor_utils

nest = tf.contrib.framework.nest


class TensorUtilsTest(tf.test.TestCase, absltest.TestCase):

  def test_check_nested_equal(self):
    nested_dict = {
        'KEY1': {
            'NESTED_KEY': 0
        },
        'KEY2': 1,
    }
    nested_list = [('KEY1', ('NESTED_KEY', 0)), ('KEY2', 1)]
    flat_dict = {
        'KEY1': 0,
        'KEY2': 1,
    }
    nested_dtypes = {
        'x': [tf.int32, tf.float32],
        'y': tf.float32,
    }
    nested_shapes = {
        'x': [tf.TensorShape([1]), tf.TensorShape([None])],
        'y': tf.TensorShape([1]),
    }

    # Should not raise an exception.
    tensor_utils.check_nested_equal(nested_dict, nested_dict)
    tensor_utils.check_nested_equal(nested_list, nested_list)
    tensor_utils.check_nested_equal(flat_dict, flat_dict)
    tensor_utils.check_nested_equal(nested_dtypes, nested_dtypes)
    tensor_utils.check_nested_equal(nested_shapes, nested_shapes)

    with self.assertRaises(TypeError):
      tensor_utils.check_nested_equal(nested_dict, nested_list)

    with self.assertRaises(ValueError):
      # Different nested structures.
      tensor_utils.check_nested_equal(nested_dict, flat_dict)

    # Same as nested_dict, but using float values. Equality still holds for
    # 0 == 0.0 despite different types.
    nested_dict_different_types = {
        'KEY1': {
            'NESTED_KEY': 0.0
        },
        'KEY2': 1.0,
    }
    nest.assert_same_structure(nested_dict, nested_dict_different_types)

    # Same as nested_dict but with one different value
    nested_dict_different_value = {
        'KEY1': {
            'NESTED_KEY': 0.5
        },
        'KEY2': 1.0,
    }
    with self.assertRaises(ValueError):
      tensor_utils.check_nested_equal(nested_dict, nested_dict_different_value)

    tensor_utils.check_nested_equal([tf.TensorShape(None)],
                                    [tf.TensorShape(None)])

    # Pretend unknown shape tensors cannot be equal.
    def eq_if_defined(x, y):
      if x.rank is None or y.rank is None:
        return False
      return x.value == y.value

    with self.assertRaises(ValueError):
      tensor_utils.check_nested_equal([tf.TensorShape(None)],
                                      [tf.TensorShape(None)], eq_if_defined)

  def test_to_var_dict(self):
    v1 = tf.Variable(0, name='v1')
    v2 = tf.Variable(0, name='v2')

    d0 = tensor_utils.to_var_dict([])
    self.assertIsInstance(d0, collections.OrderedDict)
    self.assertEmpty(d0)

    d1 = tensor_utils.to_var_dict([v1])
    self.assertIsInstance(d1, collections.OrderedDict)
    self.assertLen(d1, 1)
    self.assertEqual(d1['v1'], v1)

    d2 = tensor_utils.to_var_dict([v1, v2])
    self.assertIsInstance(d2, collections.OrderedDict)
    self.assertLen(d2, 2)
    self.assertEqual(d2['v1'], v1)
    self.assertEqual(d2['v2'], v2)

    with self.assertRaises(TypeError):
      tensor_utils.to_var_dict(v1)

    with self.assertRaises(TypeError):
      tensor_utils.to_var_dict([tf.constant(1)])

  def test_to_odict(self):
    d1 = {'b': 2, 'a': 1}
    odict1 = tensor_utils.to_odict(d1)
    self.assertIsInstance(odict1, collections.OrderedDict)
    self.assertCountEqual(d1, odict1)

    odict2 = tensor_utils.to_odict(odict1)
    self.assertEqual(odict1, odict2)

    with self.assertRaises(TypeError):
      tensor_utils.to_odict({1: 'a', 2: 'b'})

  def test_zero_all_if_any_non_finite(self):

    def expect_ok(structure):
      tf.reset_default_graph()
      result, error = tensor_utils.zero_all_if_any_non_finite(structure)
      with self.session() as sess:
        result, error = sess.run((result, error))
      try:
        nest.map_structure(np.testing.assert_allclose, result, structure)
      except AssertionError:
        self.fail('Expected to get input {} back, but instead got {}'.format(
            structure, result))
      self.assertEqual(error, 0)

    expect_ok([])
    expect_ok([(), {}])
    expect_ok(1.1)
    expect_ok([1.0, 0.0])
    expect_ok([1.0, 2.0, {'a': 0.0, 'b': -3.0}])

    def expect_zeros(structure, expected):
      tf.reset_default_graph()
      result, error = tensor_utils.zero_all_if_any_non_finite(structure)
      with self.session() as sess:
        result, error = sess.run((result, error))
      try:
        nest.map_structure(np.testing.assert_allclose, result, expected)
      except AssertionError:
        self.fail('Expected to get zeros, but instead got {}'.format(result))
      self.assertEqual(error, 1)

    expect_zeros(np.inf, 0.0)
    expect_zeros((1.0, (2.0, np.nan)), (0.0, (0.0, 0.0)))
    expect_zeros((1.0, (2.0, {
        'a': 3.0,
        'b': [[np.inf], [np.nan]]
    })), (0.0, (0.0, {
        'a': 0.0,
        'b': [[0.0], [0.0]]
    })))

  def test_is_scalar_with_list(self):
    self.assertRaises(TypeError, tensor_utils.is_scalar, [10])

  def test_is_scalar_with_bool(self):
    self.assertRaises(TypeError, tensor_utils.is_scalar, True)

  def test_is_scalar_with_tf_constant(self):
    self.assertTrue(tensor_utils.is_scalar(tf.constant(10)))

  def test_is_scalar_with_scalar_tf_variable(self):
    self.assertTrue(tensor_utils.is_scalar(tf.Variable(0.0, 'scalar')))

  def test_is_scalar_with_nonscalar_tf_variable(self):
    self.assertFalse(
        tensor_utils.is_scalar(tf.Variable([0.0, 1.0], 'notscalar')))

  def test_metrics_sum(self):
    with self.test_session() as sess:
      v = tf.placeholder(tf.float32)
      sum_tensor, update_op = tensor_utils.metrics_sum(v)
      sess.run(tf.local_variables_initializer())
      sess.run(update_op, feed_dict={v: [1.0, 2.0]})
      self.assertEqual(sess.run(sum_tensor), 3.0)
      sess.run(update_op, feed_dict={v: [3.0]})
      self.assertEqual(sess.run(sum_tensor), 6.0)

  def test_same_dimension(self):
    self.assertTrue(
        tensor_utils.same_dimension(tf.Dimension(None), tf.Dimension(None)))
    self.assertTrue(
        tensor_utils.same_dimension(tf.Dimension(1), tf.Dimension(1)))

    self.assertFalse(
        tensor_utils.same_dimension(tf.Dimension(None), tf.Dimension(1)))
    self.assertFalse(
        tensor_utils.same_dimension(tf.Dimension(1), tf.Dimension(None)))
    self.assertFalse(
        tensor_utils.same_dimension(tf.Dimension(1), tf.Dimension(2)))

  def test_same_shape(self):
    self.assertTrue(
        tensor_utils.same_shape(tf.TensorShape(None), tf.TensorShape(None)))
    self.assertTrue(
        tensor_utils.same_shape(tf.TensorShape([None]), tf.TensorShape([None])))
    self.assertTrue(
        tensor_utils.same_shape(tf.TensorShape([1]), tf.TensorShape([1])))
    self.assertTrue(
        tensor_utils.same_shape(
            tf.TensorShape([None, 1]), tf.TensorShape([None, 1])))
    self.assertTrue(
        tensor_utils.same_shape(
            tf.TensorShape([1, 2, 3]), tf.TensorShape([1, 2, 3])))

    self.assertFalse(
        tensor_utils.same_shape(tf.TensorShape(None), tf.TensorShape([1])))
    self.assertFalse(
        tensor_utils.same_shape(tf.TensorShape([1]), tf.TensorShape(None)))
    self.assertFalse(
        tensor_utils.same_shape(tf.TensorShape([1]), tf.TensorShape([None])))
    self.assertFalse(
        tensor_utils.same_shape(tf.TensorShape([1]), tf.TensorShape([2])))
    self.assertFalse(
        tensor_utils.same_shape(tf.TensorShape([1, 2]), tf.TensorShape([2, 1])))


if __name__ == '__main__':
  tf.test.main()
