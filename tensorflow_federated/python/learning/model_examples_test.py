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
"""Tests for learning.model_examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import six
import tensorflow as tf
from tensorflow_federated.python.learning import model_examples


class ModelExamplesTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ModelExamplesTest, self).setUp()
    # Required since we use defuns.
    print('TF version', tf.VERSION)
    tf.enable_resource_variables()

  @parameterized.named_parameters(('', 1), ('_three_features', 3))
  def test_linear_regression(self, feature_dim):
    model = model_examples.LinearRegression(feature_dim=feature_dim)
    init_op = tf.variables_initializer(model.trainable_variables +
                                       model.non_trainable_variables +
                                       model.local_variables)
    batch = model.make_batch(
        x=tf.placeholder(tf.float32, shape=(None, feature_dim)),
        y=tf.placeholder(tf.float32, shape=(None, 1)))
    output_op = model.forward_pass(batch)
    metrics = model.aggregated_outputs()

    tf.get_default_graph().finalize()
    with self.session() as sess:
      sess.run(init_op)
      output = sess.run(
          output_op,
          feed_dict={
              batch.x: [np.zeros(feature_dim),
                        np.ones(feature_dim)],
              batch.y: [[0.0], [1.0]]
          })
      self.assertAllEqual(output.predictions, [[0.0], [0.0]])
      # The residuals are (0., 1.), so average loss is 0.5 * 0.5 * 1.
      self.assertEqual(output.loss, 0.25)
      m = sess.run(metrics)
      self.assertEqual(m['num_examples'], 2)
      self.assertEqual(m['num_batches'], 1)
      self.assertEqual(m['loss'], 0.25)

  def test_trainable_linear_regression(self):
    dim = 1
    model = model_examples.TrainableLinearRegression(feature_dim=dim)
    init_op = tf.variables_initializer(model.trainable_variables +
                                       model.non_trainable_variables +
                                       model.local_variables)
    batch = model.make_batch(
        x=tf.placeholder(tf.float32, shape=(None, dim)),
        y=tf.placeholder(tf.float32, shape=(None, 1)))

    train_op = model.train_on_batch(batch)
    metrics = model.aggregated_outputs()
    train_feed_dict = {batch.x: [[0.0], [5.0]], batch.y: [[0.0], [5.0]]}
    prior_loss = float('inf')
    with self.session() as sess:
      sess.run(init_op)
      num_iters = 10
      for _ in six.moves.range(num_iters):
        r = sess.run(train_op, feed_dict=train_feed_dict)
        # Loss should be decreasing.
        self.assertLess(r.loss, prior_loss)
        prior_loss = r.loss

      m = sess.run(metrics)
      self.assertEqual(m['num_batches'], num_iters)
      self.assertEqual(m['num_examples'], 2 * num_iters)
      self.assertLess(m['loss'], 1.0)


if __name__ == '__main__':
  tf.test.main()
