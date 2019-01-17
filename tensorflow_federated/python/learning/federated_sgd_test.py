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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.learning import federated_sgd
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils

nest = tf.contrib.framework.nest


class FederatedSgdTest(test_utils.TffTestCase, parameterized.TestCase):

  def dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.LinearRegression.make_batch(
            x=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            y=[[1.0], [1.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(2).batch(3)

  def model(self):
    return model_examples.LinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable={
            'a': tf.constant([[0.0], [0.0]]),
            'b': tf.constant(0.0)
        },
        non_trainable={'c': 0.0})

  def test_client_tf(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(model)
    out = client_tf(dataset, self.initial_weights())
    out = nest.map_structure(lambda t: t.numpy(), out)

    # Both trainable parameters should have gradients,
    # and we don't return the non-trainable 'c'.
    self.assertCountEqual(['a', 'b'], out.weights_delta.keys())
    # Model deltas for squared error.
    self.assertAllClose(out.weights_delta['a'], [[1.0], [0.0]])
    self.assertAllClose(out.weights_delta['b'], 1.0)
    self.assertAllClose(out.weights_delta_weight, 8.0)

    self.assertEqual(out.model_output['num_examples'], 8)
    self.assertEqual(out.model_output['num_batches'], 3)
    self.assertAlmostEqual(out.model_output['loss'], 0.5)

    self.assertEqual(out.optimizer_output['client_weight'], 8.0)
    self.assertEqual(out.optimizer_output['has_non_finite_delta'], 0)

  def test_client_tf_custom_batch_weight(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(
        model, batch_weight_fn=lambda batch: 2.0 * tf.reduce_sum(batch.x))
    out = client_tf(dataset, self.initial_weights())
    self.assertEqual(out.weights_delta_weight.numpy(), 16.0)  # 2 * 8

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_sgd.ClientSgd(model)
    init_weights = self.initial_weights()
    init_weights.trainable['b'] = bad_value
    out = client_tf(dataset, init_weights)
    self.assertEqual(out.weights_delta_weight.numpy(), 0.0)
    self.assertAllClose(out.weights_delta['a'].numpy(), np.array([[0.0],
                                                                  [0.0]]))
    self.assertAllClose(out.weights_delta['b'].numpy(), 0.0)
    self.assertEqual(out.optimizer_output['has_non_finite_delta'].numpy(), 1)


if __name__ == '__main__':
  # We default to TF 2 with eager execution, and use the @graph_mode_test
  # annotation for graph-mode (sess.run) tests.
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
