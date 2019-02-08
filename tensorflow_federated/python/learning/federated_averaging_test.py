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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import federated_averaging
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils


class FederatedAveragingClientTest(test.TestCase, parameterized.TestCase):
  """Tests of ClientFedAvg that use a common model and data."""

  def dataset(self):
    # Create a dataset with 4 examples:
    dataset = tf.data.Dataset.from_tensor_slices(
        model_examples.TrainableLinearRegression.make_batch(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]]))
    # Repeat the dataset 5 times with batches of 3 examples,
    # producing 7 minibatches (the last one with only 2 examples).
    # Note thta `batch` is required for this dataset to be useable,
    # as it adds the batch dimension which is expected by the model.
    return dataset.repeat(5).batch(3)

  def model(self):
    return model_examples.TrainableLinearRegression(feature_dim=2)

  def initial_weights(self):
    return model_utils.ModelWeights(
        trainable={
            'a': tf.constant([[0.0], [0.0]]),
            'b': tf.constant(0.0)
        },
        non_trainable={'c': 0.0})

  @test.graph_mode_test
  def test_client_tf(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_averaging.ClientFedAvg(model)
    init_op = tf.group(
        model_utils.model_initializer(model),
        tf.variables_initializer(client_tf.variables),
        name='fedavg_initializer')
    client_outputs = client_tf(dataset, self.initial_weights())

    tf.get_default_graph().finalize()
    with self.session() as sess:
      sess.run(init_op)
      out = sess.run(client_outputs)

      # Both trainable parameters should have been updated,
      # and we don't return the non-trainable 'c'.
      self.assertCountEqual(['a', 'b'], out.weights_delta.keys())
      self.assertGreater(np.linalg.norm(out.weights_delta['a']), 0.1)
      self.assertGreater(np.linalg.norm(out.weights_delta['b']), 0.1)
      self.assertEqual(out.weights_delta_weight, 20.0)
      self.assertEqual(out.optimizer_output['num_examples'], 20)
      self.assertEqual(out.optimizer_output['has_non_finite_delta'], 0)

      self.assertEqual(out.model_output['num_examples'], 20)
      self.assertEqual(out.model_output['num_batches'], 7)
      self.assertBetween(out.model_output['loss'],
                         np.finfo(np.float32).eps, 10.0)

  def test_client_tf_custom_delta_weight(self):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_averaging.ClientFedAvg(
        model, client_weight_fn=lambda _: tf.constant(1.5))
    out = client_tf(dataset, self.initial_weights())
    self.assertEqual(out.weights_delta_weight.numpy(), 1.5)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    model = self.model()
    dataset = self.dataset()
    client_tf = federated_averaging.ClientFedAvg(model)
    init_weights = self.initial_weights()
    init_weights.trainable['b'] = bad_value
    out = client_tf(dataset, init_weights)
    self.assertEqual(out.weights_delta_weight.numpy(), 0.0)
    self.assertAllClose(out.weights_delta['a'].numpy(), np.array([[0.0],
                                                                  [0.0]]))
    self.assertAllClose(out.weights_delta['b'].numpy(), 0.0)
    self.assertEqual(out.optimizer_output['has_non_finite_delta'].numpy(), 1)


class FederatedAveragingTffTest(test.TestCase):

  def test_orchestration(self):
    iterative_process = federated_averaging.build_federated_averaging_process(
        model_fn=model_examples.TrainableLinearRegression)

    expected_model_weights_type = model_utils.ModelWeights(
        collections.OrderedDict([('a', tff.TensorType(tf.float32, [2, 1])),
                                 ('b', tf.float32)]),
        collections.OrderedDict([('c', tf.float32)]))

    # ServerState consists of a model and optimizer_state. The optimizer_state
    # is provided by TensorFlow, TFF doesn't care what the actual value is.
    expected_federated_server_state_type = tff.FederatedType(
        optimizer_utils.ServerState(expected_model_weights_type,
                                    test.AnyType()),
        placement=tff.SERVER,
        all_equal=True)

    expected_federated_dataset_type = tff.FederatedType(
        tff.SequenceType(
            model_examples.TrainableLinearRegression.make_batch(
                tff.TensorType(tf.float32, [None, 2]),
                tff.TensorType(tf.float32, [None, 1]))),
        tff.CLIENTS,
        all_equal=False)

    expected_model_output_types = tff.FederatedType(
        collections.OrderedDict([
            ('loss', tff.TensorType(tf.float32, [])),
            ('num_examples', tff.TensorType(tf.int32, [])),
        ]),
        tff.SERVER,
        all_equal=True)

    # `initialize` is expected to be a funcion of no arguments to a ServerState.
    self.assertEqual(
        tff.FunctionType(
            parameter=None, result=expected_federated_server_state_type),
        iterative_process.initialize.type_signature)

    # `next` is expected be a function of (ServerState, Datasets) to
    # ServerState.
    self.assertEqual(
        tff.FunctionType(
            parameter=[
                expected_federated_server_state_type,
                expected_federated_dataset_type
            ],
            result=(expected_federated_server_state_type,
                    expected_model_output_types)),
        iterative_process.next.type_signature)


if __name__ == '__main__':
  # We default to TF 2.0 behavior, including eager execution, and use the
  # @graph_mode_test annotation for graph-mode (sess.run) tests.
  tf.compat.v1.enable_v2_behavior()
  test.main()
