# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for tensorflow_federated.python.tensorflow_libs.variable_utils."""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.tensorflow_libs import variable_utils


class VariableUtilsTest(test_utils.TestCase):

  def test_variable_capture(self):
    with variable_utils.record_variable_creation_scope() as variable_list:
      v1 = tf.Variable(1.0)
      v2 = tf.Variable('abc', name='my_test_var')
      v3 = tf.compat.v1.get_variable(
          name='v1_var',
          shape=(),
          initializer=tf.compat.v1.initializers.constant)
      # Explicitly add a variable that is not added to any collections.
      v4 = tf.compat.v1.get_variable(
          name='v1_var_no_collections',
          shape=(),
          initializer=tf.compat.v1.initializers.constant,
          collections=[])
    self.assertEqual([v1, v2, v3, v4], variable_list)


if __name__ == '__main__':
  test_utils.main()
