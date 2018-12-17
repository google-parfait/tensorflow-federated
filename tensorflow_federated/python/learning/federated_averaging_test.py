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
"""Tests for learning.federated_averaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils

from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import model_examples


class FederatedAveragingTest(test_utils.TffTestCase):

  def setUp(self):
    super(FederatedAveragingTest, self).setUp()
    # Required since we use defuns.
    tf.enable_resource_variables()

  def test_smoke(self):
    model = model_examples.TrainableLinearRegression(feature_dim=2)

    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.TrainableLinearRegression.make_batch(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[0.0, 0.0, 1.0, 1.0]))
    # Repeat the dataset 5 times with batches of 3 examples,
    # producing 7 minibatches (the last one with only 2 examples).
    # Note thta `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    dataset = dataset.repeat(5).batch(3)

    initial_model = federated_averaging.ModelVars(
        trainable_variables={'a': tf.constant([[0.0], [0.0]]),
                             'b': tf.constant(0.0)},
        non_trainable_variables={'c': 0.0})

    init_op = federated_averaging.model_initializer(model)
    client_outputs = federated_averaging.client_tf(model, dataset,
                                                   initial_model)

    tf.get_default_graph().finalize()
    with self.session() as sess:
      sess.run(init_op)
      out = sess.run(client_outputs)

      # Both trainable parameters should have been updated,
      # and we don't return the non-trainable 'c'.
      self.assertCountEqual(['a', 'b'], out.model_delta.keys())
      self.assertGreater(np.linalg.norm(out.model_delta['a']), 0.1)
      self.assertGreater(np.linalg.norm(out.model_delta['b']), 0.1)

      self.assertEqual(out.model_output['num_examples'], 20)
      self.assertEqual(out.model_output['num_batches'], 7)
      self.assertBetween(out.model_output['loss'],
                         np.finfo(np.float32).eps, 10.0)


if __name__ == '__main__':
  tf.test.main()
