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

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.utils import tf_computation_utils


class TfComputationUtilsTest(test.TestCase):

  def _assertMatchesVariable(self, var, expected_name, expected_shape,
                             expected_dtype):
    self.assertEqual(var.name, expected_name)
    self.assertEqual(var.shape, expected_shape)
    self.assertEqual(var.dtype, expected_dtype)

  def test_create_variables_with_tensor_type(self):
    x = tf_computation_utils.create_variables('foo', tf.int32)
    self.assertIsInstance(x, tf.Variable)
    self.assertIs(x.dtype.base_dtype, tf.int32)
    self.assertEqual(x.shape, tf.TensorShape([]))
    self.assertEqual(str(x.name), 'foo:0')

  def test_create_variables_with_named_tuple_type(self):
    x = tf_computation_utils.create_variables('foo',
                                              [('x', tf.int32),
                                               ('y', tf.string), tf.bool])
    self.assertIsInstance(x, structure.Struct)
    self.assertLen(x, 3)
    self.assertEqual(dir(x), ['x', 'y'])
    self._assertMatchesVariable(x[0], 'foo/x:0', (), tf.int32)
    self._assertMatchesVariable(x[1], 'foo/y:0', (), tf.string)
    self._assertMatchesVariable(x[2], 'foo/2:0', (), tf.bool)

  def test_assign_with_unordered_dict(self):
    with tf.Graph().as_default() as graph:
      v = tf.Variable(0, name='foo', dtype=tf.int32, shape=[])
      c = tf.constant(10, dtype=tf.int32, shape=[])
      v_dict = {'bar': v}
      c_dict = {'bar': c}
      op = tf_computation_utils.assign(v_dict, c_dict)
    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(v.initializer)
      sess.run(op)
      result = sess.run(v)
    self.assertEqual(result, 10)

  def test_assign_with_structure(self):
    with tf.Graph().as_default() as graph:
      v = tf.Variable(0, name='foo', dtype=tf.int32, shape=[])
      c = tf.constant(10, dtype=tf.int32, shape=[])
      v_tuple = structure.Struct([('bar', v)])
      c_tuple = structure.Struct([('bar', c)])
      op = tf_computation_utils.assign(v_tuple, c_tuple)
    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(v.initializer)
      sess.run(op)
      result = sess.run(v)
    self.assertEqual(result, 10)

  def test_assign_with_no_nesting(self):
    with tf.Graph().as_default() as graph:
      v = tf.Variable(0, name='foo', dtype=tf.int32, shape=[])
      c = tf.constant(10, dtype=tf.int32, shape=[])
      op = tf_computation_utils.assign(v, c)
    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(v.initializer)
      sess.run(op)
      result = sess.run(v)
    self.assertEqual(result, 10)

  def test_identity_with_unordered_dict(self):
    with tf.Graph().as_default() as graph:
      c1 = {'foo': tf.constant(10, dtype=tf.int32, shape=[])}
      c2 = tf_computation_utils.identity(c1)
    self.assertIsNot(c2, c1)
    with tf.compat.v1.Session(graph=graph) as sess:
      result = sess.run(c2['foo'])
    self.assertEqual(result, 10)

  def test_identity_with_structure(self):
    with tf.Graph().as_default() as graph:
      c1 = structure.Struct([('foo', tf.constant(10, dtype=tf.int32,
                                                 shape=[]))])
      c2 = tf_computation_utils.identity(c1)
    self.assertIsNot(c2, c1)
    with tf.compat.v1.Session(graph=graph) as sess:
      result = sess.run(c2.foo)
    self.assertEqual(result, 10)

  def test_identity_with_no_nesting(self):
    with tf.Graph().as_default() as graph:
      c1 = tf.constant(10, dtype=tf.int32, shape=[])
      c2 = tf_computation_utils.identity(c1)
    self.assertIsNot(c2, c1)
    with tf.compat.v1.Session(graph=graph) as sess:
      result = sess.run(c2)
    self.assertEqual(result, 10)


if __name__ == '__main__':
  test.main()
