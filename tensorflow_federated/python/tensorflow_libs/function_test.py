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

from tensorflow_federated.python.tensorflow_libs import function


class FunctionTest(tf.test.TestCase):

  def test_is_tf_function(self):
    self.assertTrue(function.is_tf_function(tf.function(lambda x: None)))
    concrete_fn = tf.function(
        lambda x: None, input_signature=(tf.TensorSpec(None, tf.int32),))
    self.assertTrue(function.is_tf_function(concrete_fn))
    self.assertFalse(function.is_tf_function(lambda x: None))
    self.assertFalse(function.is_tf_function(None))


if __name__ == '__main__':
  tf.test.main()
