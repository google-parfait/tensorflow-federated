# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.learning.optimizers import optimizer


class OptimizerChecksTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('bad_shape', tf.zeros([2], tf.float32), tf.zeros([3], tf.float32)),
      ('bad_dtype', tf.zeros([2], tf.float32), tf.zeros([2], tf.float64)),
      ('bad_structure', [tf.zeros([2]), tf.zeros([3])
                        ], [tf.zeros([2]), [tf.zeros([3])]]),
  )
  def test_check_weights_gradients_match(self, weights, gradients):
    with self.assertRaises(ValueError):
      optimizer.check_weights_gradients_match(weights, gradients)

  @parameterized.named_parameters(
      ('bad_shape', tf.zeros([2], tf.float32), tf.zeros([3], tf.float32)),
      ('bad_dtype', tf.zeros([2], tf.float32), tf.zeros([2], tf.float64)),
      ('bad_structure', [tf.zeros([2]), tf.zeros([3])
                        ], [tf.zeros([2]), [tf.zeros([3])]]),
  )
  def test_check_weights_state_match(self, weights, state):
    with self.assertRaisesRegex(ValueError, 'foo'):
      optimizer.check_weights_state_match(weights, state, name='foo')

  def test_handle_indexed_slices_single_value(self):
    gradients = tf.IndexedSlices(
        values=tf.constant([[0.0, 1.0], [1.0, 3.0]]),
        indices=tf.constant([0, 2]),
        dense_shape=tf.constant([4, 2]))
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    self.assertIsInstance(gradients, tf.Tensor)
    self.assertAllClose([[0.0, 1.0], [0.0, 0.0], [1.0, 3.0], [0.0, 0.0]],
                        gradients)

  def test_handle_indexed_slices_struct(self):
    tensor = tf.constant([4.0, 5.5])
    slices = tf.IndexedSlices(
        values=tf.constant([[0.0, 1.0], [1.0, 3.0]]),
        indices=tf.constant([0, 2]),
        dense_shape=tf.constant([4, 2]))
    gradients = [tensor, slices]
    gradients = optimizer.handle_indexed_slices_gradients(gradients)
    self.assertIsInstance(gradients[0], tf.Tensor)
    self.assertIsInstance(gradients[1], tf.Tensor)
    self.assertAllClose(
        [[4.0, 5.5], [[0.0, 1.0], [0.0, 0.0], [1.0, 3.0], [0.0, 0.0]]],
        gradients)


if __name__ == '__main__':
  test_case.main()
