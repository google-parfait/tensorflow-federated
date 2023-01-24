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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import function


class FunctionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('polymorphic_function', tf.function(lambda x: None)),
      (
          'concrete_function',
          tf.function(
              lambda x: None, input_signature=(tf.TensorSpec(None, tf.int32),)
          ),
      ),
  )
  def test_is_tf_function_true(self, fn):
    self.assertTrue(function.is_tf_function(fn))

  @parameterized.named_parameters(
      ('lambda', lambda x: None),
      ('none', None),
  )
  def test_is_tf_function_false(self, fn):
    self.assertFalse(function.is_tf_function(fn))


if __name__ == '__main__':
  tf.test.main()
