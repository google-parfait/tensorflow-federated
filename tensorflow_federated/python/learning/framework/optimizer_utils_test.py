# Lint as: python3
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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils

tf.compat.v1.enable_v2_behavior()


class DummyClientDeltaFn(optimizer_utils.ClientDeltaFn):

  def __init__(self, model_fn):
    self._model = model_fn()
    self._optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset, initial_weights):
    # Iterate over the dataset to get new metric values.
    def reduce_fn(dummy, batch):
      with tf.GradientTape() as tape:
        output = self._model.forward_pass(batch)
      gradients = tape.gradient(output.loss, self._model.trainable_variables)
      self._optimizer.apply_gradients(
          zip(gradients, self._model.trainable_variables))
      return dummy

    dataset.reduce(tf.constant(0.0), reduce_fn)

    # Create some fake weight deltas to send back.
    trainable_weights_delta = tf.nest.map_structure(lambda x: -tf.ones_like(x),
                                                    initial_weights.trainable)
    client_weight = tf.constant(1.0)
    return optimizer_utils.ClientOutput(
        trainable_weights_delta,
        weights_delta_weight=client_weight,
        model_output=self._model.report_local_outputs(),
        optimizer_output=collections.OrderedDict([('client_weight',
                                                   client_weight)]))


def _state_incrementing_mean_next(server_state, client_value, weight=None):
  add_one = tff.tf_computation(lambda x: x + 1, tf.int32)
  new_state = tff.federated_map(add_one, server_state)
  return (new_state, tff.federated_mean(client_value, weight=weight))


state_incrementing_mean = tff.utils.StatefulAggregateFn(
    lambda: tf.constant(0), _state_incrementing_mean_next)


def _state_incrementing_broadcast_next(server_state, server_value):
  add_one = tff.tf_computation(lambda x: x + 1, tf.int32)
  new_state = tff.federated_map(add_one, server_state)
  return (new_state, tff.federated_broadcast(server_value))


state_incrementing_broadcaster = tff.utils.StatefulBroadcastFn(
    lambda: tf.constant(0), _state_incrementing_broadcast_next)


class UtilsTest(test.TestCase):

  def test_state_with_new_model_weights(self):
    trainable = [np.array([1.0, 2.0]), np.array([[1.0]])]
    non_trainable = [np.array(1)]
    state = anonymous_tuple.from_container(
        optimizer_utils.ServerState(
            model=model_utils.ModelWeights(
                trainable=trainable, non_trainable=non_trainable),
            optimizer_state=[],
            delta_aggregate_state=tf.constant(0),
            model_broadcast_state=tf.constant(0)),
        recursive=True)

    new_state = optimizer_utils.state_with_new_model_weights(
        state,
        trainable_weights=[np.array([3.0, 3.0]),
                           np.array([[3.0]])],
        non_trainable_weights=[np.array(3)])
    self.assertAllClose(
        new_state.model.trainable,
        [np.array([3.0, 3.0]), np.array([[3.0]])])
    self.assertAllClose(new_state.model.non_trainable, [3])

    with self.assertRaisesRegex(TypeError, 'tensor type'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=[np.array([3.0, 3.0]),
                             np.array([[3]])],
          non_trainable_weights=[np.array(3.0)])

    with self.assertRaisesRegex(TypeError, 'tensor type'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=[np.array([3.0, 3.0]),
                             np.array([3.0])],
          non_trainable_weights=[np.array(3)])

    with self.assertRaisesRegex(TypeError, 'different lengths'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=[np.array([3.0, 3.0])],
          non_trainable_weights=[np.array(3)])

    with self.assertRaisesRegex(TypeError, 'cannot be handled'):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights={'a': np.array([3.0, 3.0])},
          non_trainable_weights=[np.array(3)])


class ServerTest(test.TestCase, parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('_sgd', lambda: tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1),
       0.1, 0),
      # It looks like Adam introduces 2 + 2*num_model_variables additional vars.
      ('_adam', lambda: tf.compat.v1.train.AdamOptimizer(  # pylint: disable=g-long-lambda
          learning_rate=0.1, beta1=0.0, beta2=0.0, epsilon=1.0), 0.05, 6))
  # pyformat: enable
  def test_server_eager_mode(self, optimizer_fn, updated_val,
                             num_optimizer_vars):
    model_fn = lambda: model_examples.LinearRegression(feature_dim=2)

    server_state = optimizer_utils.server_init(model_fn, optimizer_fn, (), ())
    model_vars = self.evaluate(server_state.model)
    train_vars = model_vars.trainable
    self.assertLen(train_vars, 2)
    self.assertAllClose(train_vars, [np.zeros((2, 1)), 0.0])
    self.assertAllClose(model_vars.non_trainable, [0.0])
    self.assertLen(server_state.optimizer_state, num_optimizer_vars)
    weights_delta = [tf.constant([[1.0], [0.0]]), tf.constant(1.0)]
    server_state = optimizer_utils.server_update_model(server_state,
                                                       weights_delta, model_fn,
                                                       optimizer_fn)

    model_vars = self.evaluate(server_state.model)
    train_vars = model_vars.trainable
    # For SGD: learning_Rate=0.1, update=[1.0, 0.0], initial model=[0.0, 0.0],
    # so updated_val=0.1
    self.assertLen(train_vars, 2)
    self.assertAllClose(train_vars, [[[updated_val], [0.0]], updated_val])
    self.assertAllClose(model_vars.non_trainable, [0.0])

  def test_orchestration_execute(self):
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        # A federated_mean that maintains an int32 state equal to the
        # number of times the federated_mean has been executed,
        # allowing us to test that a stateful aggregator's state
        # is properly updated.
        stateful_delta_aggregate_fn=state_incrementing_mean,
        # Similarly, a broadcast with state that increments:
        stateful_model_broadcast_fn=state_incrementing_broadcaster)

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([
            ('x', [[1.0, 2.0], [3.0, 4.0]]),
            ('y', [[5.0], [6.0]]),
        ])).batch(2)
    federated_ds = [ds] * 3

    state = iterative_process.initialize()
    self.assertAllClose(list(state.model.trainable), [np.zeros((2, 1)), 0.0])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertEqual(state.delta_aggregate_state, 0)
    self.assertEqual(state.model_broadcast_state, 0)

    state, outputs = iterative_process.next(state, federated_ds)
    self.assertAllClose(list(state.model.trainable), [-np.ones((2, 1)), -1.0])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertEqual(state.delta_aggregate_state, 1)
    self.assertEqual(state.model_broadcast_state, 1)

    # Since all predictions are 0, loss is:
    #    (0.5 * (0-5)^2 + (0-6)^2) / 2 = 15.25
    self.assertAlmostEqual(outputs.loss, 15.25, places=4)
    # 3 clients * 2 examples per client = 6 examples.
    self.assertAlmostEqual(outputs.num_examples, 6.0, places=8)


if __name__ == '__main__':
  test.main()
