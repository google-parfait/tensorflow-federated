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
"""Tests for tensorflow_deserialization.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import tensorflow_deserialization
from tensorflow_federated.python.core.impl import tensorflow_serialization


class TensorFlowDeserializationTest(test_utils.TffTestCase):

  def test_deserialize_and_call_tf_computation_with_add_one(self):
    ctx_stack = context_stack_impl.context_stack
    add_one = tensorflow_serialization.serialize_py_func_as_tf_computation(
        lambda x: tf.add(x, 1, name='the_add'), tf.int32, ctx_stack)
    result = tensorflow_deserialization.deserialize_and_call_tf_computation(
        add_one, tf.constant(10, name='the_ten'), tf.get_default_graph())
    self.assertTrue(tf.contrib.framework.is_tensor(result))
    with tf.Session() as sess:
      result_val = sess.run(result)
    self.assertEqual(result_val, 11)


if __name__ == '__main__':
  tf.test.main()
