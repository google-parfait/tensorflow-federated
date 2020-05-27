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
"""Tests for model_utils.

These tests also serve as examples for users who are familiar with Keras.
"""

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils

tf.compat.v1.enable_v2_behavior()


class ModelUtilsTest(test.TestCase):

  def test_model_initializer(self):
    with tf.Graph().as_default() as g:
      model = model_utils.enhance(model_examples.LinearRegression(2))
      init = model_utils.model_initializer(model)
      with self.session(graph=g) as sess:
        sess.run(init)
        # Make sure we can read all the variables
        try:
          sess.run(model.local_variables)
          sess.run(model.weights)
        except tf.errors.FailedPreconditionError:
          self.fail('Expected variables to be initialized, but got '
                    'tf.errors.FailedPreconditionError')

  def test_enhance(self):
    model = model_utils.enhance(model_examples.LinearRegression(3))
    self.assertIsInstance(model, model_utils.EnhancedModel)

    with self.assertRaisesRegex(ValueError, 'another EnhancedModel'):
      model_utils.EnhancedModel(model)

  def test_enhanced_var_lists(self):

    class BadModel(model_examples.LinearRegression):

      @property
      def trainable_variables(self):
        return ['not_a_variable']

      @property
      def local_variables(self):
        return 1

      def forward_pass(self, batch, training=True):
        return 'Not BatchOutput'

    bad_model = model_utils.enhance(BadModel())
    self.assertRaisesRegex(TypeError, 'Variable',
                           lambda: bad_model.trainable_variables)
    self.assertRaisesRegex(TypeError, 'Iterable',
                           lambda: bad_model.local_variables)
    self.assertRaisesRegex(TypeError, 'BatchOutput',
                           lambda: bad_model.forward_pass(1))


if __name__ == '__main__':
  test.main()
