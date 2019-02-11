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
"""Tests for learning.framework.optimizer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/123578208): Remove deep keras imports after updating TF version.
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils

nest = tf.contrib.framework.nest


class DummyClientDeltaFn(optimizer_utils.ClientDeltaFn):

  def __init__(self, model_fn):
    self._model = model_fn()

  @property
  def variables(self):
    return []

  @tf.contrib.eager.function()
  def __call__(self, dataset, initial_weights):
    trainable_weights_delta = nest.map_structure(lambda x: -1.0 * x,
                                                 initial_weights.trainable)
    client_weight = tf.constant(3.0)
    return optimizer_utils.ClientOutput(
        trainable_weights_delta,
        weights_delta_weight=client_weight,
        model_output=self._model.report_local_outputs(),
        optimizer_output={'client_weight': client_weight})


class UtilsTest(test.TestCase):

  def test_state_with_new_model_weights(self):
    trainable = [('b', np.array([1.0, 2.0])), ('a', np.array([[1.0]]))]
    non_trainable = [('c', np.array(1))]
    state = anonymous_tuple.from_container(
        optimizer_utils.ServerState(
            model=model_utils.ModelWeights(
                trainable=collections.OrderedDict(trainable),
                non_trainable=collections.OrderedDict(non_trainable)),
            optimizer_state=[]),
        recursive=True)

    new_state = optimizer_utils.state_with_new_model_weights(
        state,
        trainable_weights=[np.array([3.0, 3.0]),
                           np.array([[3.0]])],
        non_trainable_weights=[np.array(3)])
    self.assertEqual(list(new_state.model.trainable.keys()), ['b', 'a'])
    self.assertEqual(list(new_state.model.non_trainable.keys()), ['c'])
    self.assertAllClose(new_state.model.trainable['b'], [3.0, 3.0])
    self.assertAllClose(new_state.model.trainable['a'], [[3.0]])
    self.assertAllClose(new_state.model.non_trainable['c'], 3)

    with self.assertRaisesRegexp(ValueError, 'dtype'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=[np.array([3.0, 3.0]),
                             np.array([[3]])],
          non_trainable_weights=[np.array(3.0)])

    with self.assertRaisesRegexp(ValueError, 'shape'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=[np.array([3.0, 3.0]),
                             np.array([3.0])],
          non_trainable_weights=[np.array(3)])

    with self.assertRaisesRegexp(ValueError, 'Lengths differ'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=[np.array([3.0, 3.0])],
          non_trainable_weights=[np.array(3)])


class ServerTest(test.TestCase, parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('_sgd', lambda: tf.train.GradientDescentOptimizer(learning_rate=0.1),
       0.1, 0),
      # It looks like Adam introduces 2 + 2*num_model_variables additional vars.
      ('_adam', lambda: tf.train.AdamOptimizer(  # pylint: disable=g-long-lambda
          learning_rate=0.1, beta1=0.0, beta2=0.0, epsilon=1.0), 0.05, 6))
  # pyformat: enable
  def test_server_eager_mode(self, optimizer_fn, updated_val,
                             num_optimizer_vars):
    model_fn = lambda: model_examples.TrainableLinearRegression(feature_dim=2)

    server_state = optimizer_utils.server_init(model_fn, optimizer_fn)
    train_vars = server_state.model.trainable
    self.assertAllClose(train_vars['a'].numpy(), np.array([[0.0], [0.0]]))
    self.assertEqual(train_vars['b'].numpy(), 0.0)
    self.assertEqual(server_state.model.non_trainable['c'].numpy(), 0.0)
    self.assertLen(server_state.optimizer_state, num_optimizer_vars)
    weights_delta = tensor_utils.to_odict({
        'a': tf.constant([[1.0], [0.0]]),
        'b': tf.constant(1.0)
    })
    server_state = optimizer_utils.server_update_model(
        server_state, weights_delta, model_fn, optimizer_fn)

    train_vars = server_state.model.trainable
    # For SGD: learning_Rate=0.1, update=[1.0, 0.0], initial model=[0.0, 0.0],
    # so updated_val=0.1
    self.assertAllClose(train_vars['a'].numpy(), [[updated_val], [0.0]])
    self.assertAllClose(train_vars['b'].numpy(), updated_val)
    self.assertEqual(server_state.model.non_trainable['c'].numpy(), 0.0)

  @test.graph_mode_test
  def test_server_graph_mode(self):
    optimizer_fn = lambda: gradient_descent.SGD(learning_rate=0.1)
    model_fn = lambda: model_examples.TrainableLinearRegression(feature_dim=2)

    # Explicitly entering a graph as a default enables graph-mode.
    with tf.Graph().as_default() as g:
      server_state_op = optimizer_utils.server_init(model_fn, optimizer_fn)
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      g.finalize()
      with self.session() as sess:
        sess.run(init_op)
        server_state = sess.run(server_state_op)
    train_vars = server_state.model.trainable
    self.assertAllClose(train_vars['a'], [[0.0], [0.0]])
    self.assertEqual(train_vars['b'], 0.0)
    self.assertEqual(server_state.model.non_trainable['c'], 0.0)
    self.assertEqual(server_state.optimizer_state, [0.0])

    with tf.Graph().as_default() as g:
      # N.B. Must use a fresh graph so variable names are the same.
      weights_delta = tensor_utils.to_odict({
          'a': tf.constant([[1.0], [0.0]]),
          'b': tf.constant(2.0)
      })
      update_op = optimizer_utils.server_update_model(
          server_state, weights_delta, model_fn, optimizer_fn)
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      g.finalize()
      with self.session() as sess:
        sess.run(init_op)
        server_state = sess.run(update_op)
    train_vars = server_state.model.trainable
    # learning_Rate=0.1, update is [1.0, 0.0], initial model is [0.0, 0.0].
    self.assertAllClose(train_vars['a'], [[0.1], [0.0]])
    self.assertAllClose(train_vars['b'], 0.2)
    self.assertEqual(server_state.model.non_trainable['c'], 0.0)

  # TODO(b/109733734): update these tests to actually execute the iterative
  # process, instead of only checkign the type signatures. Test
  # state_with_new_model_weights at the same time.
  def test_orchestration(self):
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.TrainableLinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn)

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
        tff.SequenceType(model_examples.TrainableLinearRegression().input_spec),
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
