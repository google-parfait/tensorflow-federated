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
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.learning import federated_sgd
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils


nest = tf.contrib.framework.nest


class FederatedSgdTest(test_utils.TffTestCase, parameterized.TestCase):

  def test_client_tf(self):
    model = model_examples.LinearRegression(feature_dim=2)

    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.LinearRegression.make_batch(
            x=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            y=[[1.0], [1.0], [1.0], [1.0]]))
    # Repeat the dataset 2 times with batches of 3 examples,
    # producing 3 minibatches (the last one with only 2 examples).
    # Note that `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    dataset = dataset.repeat(2).batch(3)

    initial_weights = model_utils.ModelWeights(
        trainable={
            'a': tf.constant([[0.0], [0.0]]),
            'b': tf.constant(0.0)
        },
        non_trainable={'c': 0.0})

    client_tf = federated_sgd.ClientSgd(model)
    out = client_tf(dataset, initial_weights)
    out = nest.map_structure(lambda t: t.numpy(), out)

    # Both trainable parameters should have gradients,
    # and we don't return the non-trainable 'c'.
    self.assertCountEqual(['a', 'b'], out.weights_delta.keys())
    # Model deltas for squared error.
    self.assertAllClose(out.weights_delta['a'], [[1.0], [0.0]])
    self.assertAllClose(out.weights_delta['b'], 1.0)

    self.assertEqual(out.model_output['num_examples'], 8)
    self.assertEqual(out.optimizer_output['client_weight'], 8)
    self.assertEqual(out.model_output['num_batches'], 3)
    self.assertAlmostEqual(out.model_output['loss'], 0.5)


if __name__ == '__main__':
  # We default to TF 2 with eager execution, and use the @graph_mode_test
  # annotation for graph-mode (sess.run) tests.
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
