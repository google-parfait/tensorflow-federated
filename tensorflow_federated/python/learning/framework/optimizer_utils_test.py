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
import functools

import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import optimizer_utils


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


@tff.tf_computation(tf.int32)
def _add_one(x):
  return x + 1


def _state_incrementing_mean_next(server_state, client_value, weight=None):
  new_state = tff.federated_map(_add_one, server_state)
  return (new_state, tff.federated_mean(client_value, weight=weight))


state_incrementing_mean = tff.utils.StatefulAggregateFn(
    lambda: tf.constant(0), _state_incrementing_mean_next)


def _state_incrementing_broadcast_next(server_state, server_value):
  new_state = tff.federated_map(_add_one, server_state)
  return (new_state, tff.federated_broadcast(server_value))


state_incrementing_broadcaster = tff.utils.StatefulBroadcastFn(
    lambda: tf.constant(0), _state_incrementing_broadcast_next)


def _build_test_measured_broadcast(
    model_weights_type: tff.NamedTupleType) -> tff.templates.MeasuredProcess:
  """Builds a test `MeasuredProcess` that has state and metrics."""

  @tff.federated_computation()
  def initialize_comp():
    return tff.federated_value(0, tff.SERVER)

  @tff.federated_computation(
      tff.FederatedType(tf.int32, tff.SERVER),
      tff.FederatedType(model_weights_type, tff.SERVER))
  def next_comp(state, value):
    return collections.OrderedDict(
        state=tff.federated_map(_add_one, state),
        result=tff.federated_broadcast(value),
        # Arbitrary metrics for testing.
        measurements=tff.federated_map(
            tff.tf_computation(
                lambda v: tf.linalg.global_norm(tf.nest.flatten(v)) + 3.0),
            value))

  return tff.templates.MeasuredProcess(
      initialize_fn=initialize_comp, next_fn=next_comp)


def _build_test_measured_mean(
    model_update_type: tff.NamedTupleType) -> tff.templates.MeasuredProcess:
  """Builds a test `MeasuredProcess` that has state and metrics."""

  @tff.federated_computation()
  def initialize_comp():
    return tff.federated_value(0, tff.SERVER)

  @tff.federated_computation(
      tff.FederatedType(tf.int32, tff.SERVER),
      tff.FederatedType(model_update_type, tff.CLIENTS),
      tff.FederatedType(tf.float32, tff.CLIENTS))
  def next_comp(state, value, weight):
    return collections.OrderedDict(
        state=tff.federated_map(_add_one, state),
        result=tff.federated_mean(value, weight),
        measurements=tff.federated_zip(
            collections.OrderedDict(
                num_clients=tff.federated_sum(
                    tff.federated_value(1, tff.CLIENTS)))))

  return tff.templates.MeasuredProcess(
      initialize_fn=initialize_comp, next_fn=next_comp)


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


class ModelDeltaOptimizerTest(test.TestCase):

  def test_construction(self):
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    server_state_type = tff.FederatedType(
        optimizer_utils.ServerState(
            model=model_utils.ModelWeights(
                trainable=[
                    tff.TensorType(tf.float32, [2, 1]),
                    tff.TensorType(tf.float32)
                ],
                non_trainable=[tff.TensorType(tf.float32)]),
            optimizer_state=[tf.int64],
            delta_aggregate_state=(),
            model_broadcast_state=()), tff.SERVER)

    self.assertEqual(iterative_process.initialize.type_signature,
                     tff.FunctionType(parameter=None, result=server_state_type))

    dataset_type = tff.FederatedType(
        tff.SequenceType(
            collections.OrderedDict(
                x=tff.TensorType(tf.float32, [None, 2]),
                y=tff.TensorType(tf.float32, [None, 1]))), tff.CLIENTS)

    metrics_type = tff.FederatedType(
        collections.OrderedDict(
            broadcast=(),
            aggregation=(),
            train=collections.OrderedDict(
                loss=tff.TensorType(tf.float32),
                num_examples=tff.TensorType(tf.int32))), tff.SERVER)

    self.assertEqual(
        iterative_process.next.type_signature,
        tff.FunctionType(
            parameter=(server_state_type, dataset_type),
            result=(server_state_type, metrics_type)))

  def test_construction_with_adam_optimizer(self):
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.Adam)
    # Assert that the optimizer_state includes the 5 variables (scalar for
    # # of iterations, plus two copies of the kernel and bias in the model).
    initialize_type = iterative_process.initialize.type_signature
    self.assertLen(initialize_type.result.member.optimizer_state, 5)

    next_type = iterative_process.next.type_signature
    self.assertLen(next_type.parameter[0].member.optimizer_state, 5)
    self.assertLen(next_type.result[0].member.optimizer_state, 5)

  def test_construction_with_stateful_aggregate_fn(self):
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        stateful_delta_aggregate_fn=state_incrementing_mean)

    expected_aggregate_state_type = tff.TensorType(tf.int32)

    initialize_type = iterative_process.initialize.type_signature
    self.assertEqual(initialize_type.result.member.delta_aggregate_state,
                     expected_aggregate_state_type)

    next_type = iterative_process.next.type_signature
    self.assertEqual(next_type.parameter[0].member.delta_aggregate_state,
                     expected_aggregate_state_type)
    self.assertEqual(next_type.result[0].member.delta_aggregate_state,
                     expected_aggregate_state_type)

  def test_construction_with_aggregation_process(self):
    model_update_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression).trainable
    aggregation_process = _build_test_measured_mean(model_update_type)
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        aggregation_process=aggregation_process)

    aggregation_state_type = aggregation_process.initialize.type_signature.result
    initialize_type = iterative_process.initialize.type_signature
    self.assertEqual(
        tff.FederatedType(initialize_type.result.member.delta_aggregate_state,
                          tff.SERVER), aggregation_state_type)

    next_type = iterative_process.next.type_signature
    self.assertEqual(
        tff.FederatedType(next_type.parameter[0].member.delta_aggregate_state,
                          tff.SERVER), aggregation_state_type)
    self.assertEqual(
        tff.FederatedType(next_type.result[0].member.delta_aggregate_state,
                          tff.SERVER), aggregation_state_type)

    aggregation_metrics_type = aggregation_process.next.type_signature.result.measurements
    self.assertEqual(
        tff.FederatedType(next_type.result[1].member.aggregation, tff.SERVER),
        aggregation_metrics_type)

  def test_construction_with_broadcast_stateful_fn(self):
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        stateful_model_broadcast_fn=state_incrementing_broadcaster)

    expected_broadcast_state_type = tff.TensorType(tf.int32)

    initialize_type = iterative_process.initialize.type_signature
    self.assertEqual(initialize_type.result.member.model_broadcast_state,
                     expected_broadcast_state_type)

    next_type = iterative_process.next.type_signature
    self.assertEqual(next_type.parameter[0].member.model_broadcast_state,
                     expected_broadcast_state_type)
    self.assertEqual(next_type.result[0].member.model_broadcast_state,
                     expected_broadcast_state_type)

  def test_construction_with_broadcast_process(self):
    model_weights_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression)
    broadcast_process = _build_test_measured_broadcast(model_weights_type)
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        broadcast_process=broadcast_process)

    expected_broadcast_state_type = broadcast_process.initialize.type_signature.result
    initialize_type = iterative_process.initialize.type_signature
    self.assertEqual(
        tff.FederatedType(initialize_type.result.member.model_broadcast_state,
                          tff.SERVER), expected_broadcast_state_type)

    next_type = iterative_process.next.type_signature
    self.assertEqual(
        tff.FederatedType(next_type.parameter[0].member.model_broadcast_state,
                          tff.SERVER), expected_broadcast_state_type)
    self.assertEqual(
        tff.FederatedType(next_type.result[0].member.model_broadcast_state,
                          tff.SERVER), expected_broadcast_state_type)

  def test_fails_stateful_broadcast_and_process(self):
    model_weights_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression)
    with self.assertRaises(optimizer_utils.DisjointArgumentError):
      optimizer_utils.build_model_delta_optimizer_process(
          model_fn=model_examples.LinearRegression,
          model_to_client_delta_fn=DummyClientDeltaFn,
          server_optimizer_fn=tf.keras.optimizers.SGD,
          stateful_model_broadcast_fn=tff.utils.StatefulBroadcastFn(
              initialize_fn=lambda: (),
              next_fn=lambda state, weights:  # pylint: disable=g-long-lambda
              (state, tff.federated_broadcast(weights))),
          broadcast_process=optimizer_utils.build_stateless_broadcaster(
              model_weights_type=model_weights_type))

  def test_fails_stateful_aggregate_and_process(self):
    model_weights_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression)
    with self.assertRaises(optimizer_utils.DisjointArgumentError):
      optimizer_utils.build_model_delta_optimizer_process(
          model_fn=model_examples.LinearRegression,
          model_to_client_delta_fn=DummyClientDeltaFn,
          server_optimizer_fn=tf.keras.optimizers.SGD,
          stateful_delta_aggregate_fn=tff.utils.StatefulAggregateFn(
              initialize_fn=lambda: (),
              next_fn=lambda state, value, weight=None:  # pylint: disable=g-long-lambda
              (state, tff.federated_mean(value, weight))),
          aggregation_process=optimizer_utils.build_stateless_mean(
              model_delta_type=model_weights_type.trainable))

  def test_orchestration_execute_stateful_fn(self):
    learning_rate = 1.0
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=functools.partial(
            tf.keras.optimizers.SGD, learning_rate=learning_rate),
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
    # SGD keeps track of a single scalar for the number of iterations.
    self.assertAllEqual(state.optimizer_state, [0])
    self.assertAllClose(list(state.model.trainable), [np.zeros((2, 1)), 0.0])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertEqual(state.delta_aggregate_state, 0)
    self.assertEqual(state.model_broadcast_state, 0)

    state, outputs = iterative_process.next(state, federated_ds)
    self.assertAllClose(
        list(state.model.trainable), [-np.ones((2, 1)), -1.0 * learning_rate])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertAllEqual(state.optimizer_state, [1])
    self.assertEqual(state.delta_aggregate_state, 1)
    self.assertEqual(state.model_broadcast_state, 1)

    expected_outputs = anonymous_tuple.from_container(
        collections.OrderedDict(
            # The StatefulFn output no metrics.
            broadcast=(),
            aggregation=(),
            train=collections.OrderedDict(
                loss=15.25,
                num_examples=6,
            )),
        recursive=True)
    self.assertEqual(expected_outputs, outputs)

  def test_orchestration_execute_measured_process(self):
    model_weights_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression)
    learning_rate = 1.0
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=functools.partial(
            tf.keras.optimizers.SGD, learning_rate=learning_rate),
        broadcast_process=_build_test_measured_broadcast(model_weights_type),
        aggregation_process=_build_test_measured_mean(
            model_weights_type.trainable))

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([
            ('x', [[1.0, 2.0], [3.0, 4.0]]),
            ('y', [[5.0], [6.0]]),
        ])).batch(2)
    federated_ds = [ds] * 3

    state = iterative_process.initialize()
    # SGD keeps track of a single scalar for the number of iterations.
    self.assertAllEqual(state.optimizer_state, [0])
    self.assertAllClose(list(state.model.trainable), [np.zeros((2, 1)), 0.0])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertEqual(state.delta_aggregate_state, 0)
    self.assertEqual(state.model_broadcast_state, 0)

    state, outputs = iterative_process.next(state, federated_ds)
    self.assertAllClose(
        list(state.model.trainable), [-np.ones((2, 1)), -1.0 * learning_rate])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertAllEqual(state.optimizer_state, [1])
    self.assertEqual(state.delta_aggregate_state, 1)
    self.assertEqual(state.model_broadcast_state, 1)

    expected_outputs = anonymous_tuple.from_container(
        collections.OrderedDict(
            broadcast=3.0,
            aggregation=collections.OrderedDict(num_clients=3),
            train=collections.OrderedDict(
                loss=15.25,
                num_examples=6,
            )),
        recursive=True)
    self.assertEqual(expected_outputs, outputs)


if __name__ == '__main__':
  test.main()
