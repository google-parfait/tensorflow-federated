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
"""Tests for tf_computation_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple

from tensorflow_federated.python.core.utils import tf_computation_utils


class TfComputationUtilsTest(tf.test.TestCase):

  def test_get_variables_with_tensor_type(self):
    x = tf_computation_utils.get_variables('foo', tf.int32)
    self.assertIsInstance(x, tf.Variable)
    self.assertIs(x.dtype.base_dtype, tf.int32)
    self.assertEqual(x.shape, tf.TensorShape([]))
    self.assertEqual(str(x.name), 'foo:0')

  def test_get_variables_with_named_tuple_type(self):
    x = tf_computation_utils.get_variables(
        'foo', [('x', tf.int32), ('y', tf.string), tf.bool])
    self.assertIsInstance(x, anonymous_tuple.AnonymousTuple)
    self.assertEqual(
        str(x),
        '<x=<tf.Variable \'foo/x:0\' shape=() dtype=int32_ref>,'
        'y=<tf.Variable \'foo/y:0\' shape=() dtype=string_ref>,'
        '<tf.Variable \'foo/2:0\' shape=() dtype=bool_ref>>')


if __name__ == '__main__':
  tf.test.main()
