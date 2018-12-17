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

# Dependency imports
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import tensor_utils as tu


class TensorUtilsTest(tf.test.TestCase, absltest.TestCase):

  def test_to_var_dict(self):
    v1 = tf.Variable(0, name='v1')
    v2 = tf.Variable(0, name='v2')

    d0 = tu.to_var_dict([])
    self.assertIsInstance(d0, collections.OrderedDict)
    self.assertEmpty(d0)

    d1 = tu.to_var_dict([v1])
    self.assertIsInstance(d1, collections.OrderedDict)
    self.assertLen(d1, 1)
    self.assertEqual(d1['v1'], v1)

    d2 = tu.to_var_dict([v1, v2])
    self.assertIsInstance(d2, collections.OrderedDict)
    self.assertLen(d2, 2)
    self.assertEqual(d2['v1'], v1)
    self.assertEqual(d2['v2'], v2)

    with self.assertRaises(TypeError):
      tu.to_var_dict(v1)

    with self.assertRaises(TypeError):
      tu.to_var_dict([tf.constant(1)])

  def test_to_odict(self):
    d1 = {'b': 2, 'a': 1}
    odict1 = tu.to_odict(d1)
    self.assertIsInstance(odict1, collections.OrderedDict)
    self.assertCountEqual(d1, odict1)

    odict2 = tu.to_odict(odict1)
    self.assertEqual(odict1, odict2)

    with self.assertRaises(TypeError):
      tu.to_odict({1: 'a', 2: 'b'})

  def test_is_scalar_with_list(self):
    self.assertRaises(TypeError, tu.is_scalar, [10])

  def test_is_scalar_with_bool(self):
    self.assertRaises(TypeError, tu.is_scalar, True)

  def test_is_scalar_with_tf_constant(self):
    self.assertTrue(tu.is_scalar(tf.constant(10)))

  def test_is_scalar_with_scalar_tf_variable(self):
    self.assertTrue(tu.is_scalar(tf.Variable(0.0, 'scalar')))

  def test_is_scalar_with_nonscalar_tf_variable(self):
    self.assertFalse(tu.is_scalar(tf.Variable([0.0, 1.0], 'notscalar')))

  def test_metrics_sum(self):
    with self.test_session() as sess:
      v = tf.placeholder(tf.float32)
      sum_tensor, update_op = tu.metrics_sum(v)
      sess.run(tf.local_variables_initializer())
      sess.run(update_op, feed_dict={v: [1.0, 2.0]})
      self.assertEqual(sess.run(sum_tensor), 3.0)
      sess.run(update_op, feed_dict={v: [3.0]})
      self.assertEqual(sess.run(sum_tensor), 6.0)


if __name__ == '__main__':
  tf.test.main()
