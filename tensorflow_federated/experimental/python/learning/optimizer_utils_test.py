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
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.experimental.python.learning import optimizer_utils
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import sampling
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.optimizers import sgdm


def _tff_optimizer(learning_rate):
  return sgdm.SGD(learning_rate=learning_rate)


def _keras_optimizer_fn(learning_rate):
  return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


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
    def reduce_fn(whimsy, batch):
      with tf.GradientTape() as tape:
        output = self._model.forward_pass(batch)
      gradients = tape.gradient(output.loss, self._model.trainable_variables)
      self._optimizer.apply_gradients(
          zip(gradients, self._model.trainable_variables))
      return whimsy

    dataset.reduce(tf.constant(0.0), reduce_fn)

    # Create some fake weight deltas to send back.
    trainable_weights_delta = tf.nest.map_structure(lambda x: -tf.ones_like(x),
                                                    initial_weights.trainable)
    client_weight = tf.constant(1.0)
    return optimizer_utils.ClientOutput(
        weights_delta=trainable_weights_delta,
        weights_delta_weight=client_weight,
        model_output=self._model.report_local_outputs(),
        optimizer_output=collections.OrderedDict([
            ('num_examples', client_weight),
        ]))


@computations.tf_computation(tf.int32)
def _add_one(x):
  return x + 1


def _build_test_measured_broadcast(
    model_weights_type: computation_types.StructType
) -> measured_process.MeasuredProcess:
  """Builds a test `MeasuredProcess` that has state and metrics."""

  @computations.federated_computation()
  def initialize_comp():
    return intrinsics.federated_value(0, placements.SERVER)

  @computations.federated_computation(
      computation_types.FederatedType(tf.int32, placements.SERVER),
      computation_types.FederatedType(model_weights_type, placements.SERVER))
  def next_comp(state, value):
    return measured_process.MeasuredProcessOutput(
        state=intrinsics.federated_map(_add_one, state),
        result=intrinsics.federated_broadcast(value),
        # Arbitrary metrics for testing.
        measurements=intrinsics.federated_map(
            computations.tf_computation(
                lambda v: tf.linalg.global_norm(tf.nest.flatten(v)) + 3.0),
            value))

  return measured_process.MeasuredProcess(
      initialize_fn=initialize_comp, next_fn=next_comp)


class TestMeasuredMeanFactory(factory.WeightedAggregationFactory):

  def create(self, value_type, weight_type):

    @computations.federated_computation()
    def initialize_comp():
      return intrinsics.federated_value(0, placements.SERVER)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER),
        computation_types.FederatedType(value_type, placements.CLIENTS),
        computation_types.FederatedType(weight_type, placements.CLIENTS))
    def next_comp(state, value, weight):
      return measured_process.MeasuredProcessOutput(
          state=intrinsics.federated_map(_add_one, state),
          result=intrinsics.federated_mean(value, weight),
          measurements=intrinsics.federated_zip(
              collections.OrderedDict(
                  num_clients=intrinsics.federated_sum(
                      intrinsics.federated_value(1, placements.CLIENTS)))))

    return aggregation_process.AggregationProcess(
        initialize_fn=initialize_comp, next_fn=next_comp)


class UtilsTest(test_case.TestCase):

  def test_state_with_new_model_weights(self):
    trainable = [np.array([1.0, 2.0]), np.array([[1.0]])]
    non_trainable = [np.array(1)]
    state = optimizer_utils.ServerState(
        model=model_utils.ModelWeights(
            trainable=trainable, non_trainable=non_trainable),
        optimizer_state=[],
        delta_aggregate_state=tf.constant(0),
        model_broadcast_state=tf.constant(0))

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


class ModelDeltaOptimizerTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('weighted', True),
      ('unweighted', False),
  )
  def test_construction(self, weighted):
    aggregation_factory = (
        mean.MeanFactory() if weighted else sum_factory.SumFactory())
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        model_update_aggregation_factory=aggregation_factory)

    if weighted:
      aggregate_state = collections.OrderedDict(
          value_sum_process=(), weight_sum_process=())
      aggregate_metrics = collections.OrderedDict(mean_value=(), mean_weight=())
    else:
      aggregate_state = ()
      aggregate_metrics = ()

    server_state_type = computation_types.FederatedType(
        optimizer_utils.ServerState(
            model=model_utils.ModelWeights(
                trainable=[
                    computation_types.TensorType(tf.float32, [2, 1]),
                    computation_types.TensorType(tf.float32)
                ],
                non_trainable=[computation_types.TensorType(tf.float32)]),
            optimizer_state=[tf.int64],
            delta_aggregate_state=aggregate_state,
            model_broadcast_state=()), placements.SERVER)
    self.assert_types_equivalent(
        computation_types.FunctionType(
            parameter=None, result=server_state_type),
        iterative_process.initialize.type_signature)

    dataset_type = computation_types.FederatedType(
        computation_types.SequenceType(
            collections.OrderedDict(
                x=computation_types.TensorType(tf.float32, [None, 2]),
                y=computation_types.TensorType(tf.float32, [None, 1]))),
        placements.CLIENTS)
    metrics_type = computation_types.FederatedType(
        collections.OrderedDict(
            broadcast=(),
            aggregation=aggregate_metrics,
            train=collections.OrderedDict(
                loss=computation_types.TensorType(tf.float32),
                num_examples=computation_types.TensorType(tf.int32)),
            stat=collections.OrderedDict(
                num_examples=computation_types.TensorType(tf.float32))),
        placements.SERVER)
    self.assert_types_equivalent(
        computation_types.FunctionType(
            parameter=collections.OrderedDict(
                server_state=server_state_type,
                federated_dataset=dataset_type,
            ),
            result=(server_state_type, metrics_type)),
        iterative_process.next.type_signature)

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=model_examples.LinearRegression)
    optimizer_utils.build_model_delta_optimizer_process(
        model_fn=mock_model_fn,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD)
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 4)

  def test_initial_weights_pulled_from_model(self):

    self.skipTest('b/184855264')

    def _model_fn_with_zero_weights():
      linear_regression_model = model_examples.LinearRegression()
      weights = model_utils.ModelWeights.from_model(linear_regression_model)
      zero_trainable = [tf.zeros_like(x) for x in weights.trainable]
      zero_non_trainable = [tf.zeros_like(x) for x in weights.non_trainable]
      zero_weights = model_utils.ModelWeights(
          trainable=zero_trainable, non_trainable=zero_non_trainable)
      zero_weights.assign_weights_to(linear_regression_model)
      return linear_regression_model

    def _model_fn_with_one_weights():
      linear_regression_model = model_examples.LinearRegression()
      weights = model_utils.ModelWeights.from_model(linear_regression_model)
      ones_trainable = [tf.ones_like(x) for x in weights.trainable]
      ones_non_trainable = [tf.ones_like(x) for x in weights.non_trainable]
      ones_weights = model_utils.ModelWeights(
          trainable=ones_trainable, non_trainable=ones_non_trainable)
      ones_weights.assign_weights_to(linear_regression_model)
      return linear_regression_model

    iterative_process_returning_zeros = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=_model_fn_with_zero_weights,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    iterative_process_returning_ones = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=_model_fn_with_one_weights,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    zero_weights_expected = iterative_process_returning_zeros.initialize().model
    one_weights_expected = iterative_process_returning_ones.initialize().model

    self.assertEqual(
        sum(tf.reduce_sum(x) for x in zero_weights_expected.trainable) +
        sum(tf.reduce_sum(x) for x in zero_weights_expected.non_trainable), 0)
    self.assertEqual(
        sum(tf.reduce_sum(x) for x in one_weights_expected.trainable) +
        sum(tf.reduce_sum(x) for x in one_weights_expected.non_trainable),
        type_analysis.count_tensors_in_type(
            iterative_process_returning_ones.initialize.type_signature.result
            .member.model)['parameters'])

  def test_construction_fails_with_invalid_aggregation_factory(self):
    aggregation_factory = sampling.UnweightedReservoirSamplingFactory(
        sample_size=1)
    with self.assertRaisesRegex(
        TypeError, 'does not produce a compatible `AggregationProcess`'):
      optimizer_utils.build_model_delta_optimizer_process(
          model_fn=model_examples.LinearRegression,
          model_to_client_delta_fn=DummyClientDeltaFn,
          server_optimizer_fn=tf.keras.optimizers.SGD,
          model_update_aggregation_factory=aggregation_factory)

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

  def test_construction_with_aggregation_process(self):
    model_update_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression).trainable
    aggregator = TestMeasuredMeanFactory()
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        model_update_aggregation_factory=aggregator)

    agg_process = aggregator.create(model_update_type,
                                    computation_types.TensorType(tf.float32))

    aggregation_state_type = agg_process.initialize.type_signature.result
    initialize_type = iterative_process.initialize.type_signature
    self.assertEqual(
        computation_types.FederatedType(
            initialize_type.result.member.delta_aggregate_state,
            placements.SERVER), aggregation_state_type)

    next_type = iterative_process.next.type_signature
    self.assertEqual(
        computation_types.FederatedType(
            next_type.parameter[0].member.delta_aggregate_state,
            placements.SERVER), aggregation_state_type)
    self.assertEqual(
        computation_types.FederatedType(
            next_type.result[0].member.delta_aggregate_state,
            placements.SERVER), aggregation_state_type)

    agg_metrics_type = agg_process.next.type_signature.result.measurements
    self.assertEqual(
        computation_types.FederatedType(next_type.result[1].member.aggregation,
                                        placements.SERVER), agg_metrics_type)

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
        computation_types.FederatedType(
            initialize_type.result.member.model_broadcast_state,
            placements.SERVER), expected_broadcast_state_type)

    next_type = iterative_process.next.type_signature
    self.assertEqual(
        computation_types.FederatedType(
            next_type.parameter[0].member.model_broadcast_state,
            placements.SERVER), expected_broadcast_state_type)
    self.assertEqual(
        computation_types.FederatedType(
            next_type.result[0].member.model_broadcast_state,
            placements.SERVER), expected_broadcast_state_type)

  @parameterized.named_parameters([('tff_optimizer', _tff_optimizer),
                                   ('keras_optimizer', _keras_optimizer_fn)])
  @test_utils.skip_test_for_multi_gpu
  def test_orchestration_execute_measured_process(self, server_optimizer):
    model_weights_type = model_utils.weights_type_from_model(
        model_examples.LinearRegression)
    learning_rate = 1.0
    server_optimizer_fn = server_optimizer(learning_rate)
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_examples.LinearRegression,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=server_optimizer_fn,
        broadcast_process=_build_test_measured_broadcast(model_weights_type),
        model_update_aggregation_factory=TestMeasuredMeanFactory())

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([
            ('x', [[1.0, 2.0], [3.0, 4.0]]),
            ('y', [[5.0], [6.0]]),
        ])).batch(2)
    federated_ds = [ds] * 3

    state = iterative_process.initialize()
    if callable(server_optimizer_fn):
      # Keras SGD keeps track of a single scalar for the number of iterations.
      self.assertAllEqual(state.optimizer_state, [0])
    self.assertAllClose(list(state.model.trainable), [np.zeros((2, 1)), 0.0])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    self.assertEqual(state.delta_aggregate_state, 0)
    self.assertEqual(state.model_broadcast_state, 0)

    state, outputs = iterative_process.next(state, federated_ds)
    self.assertAllClose(
        list(state.model.trainable), [-np.ones((2, 1)), -1.0 * learning_rate])
    self.assertAllClose(list(state.model.non_trainable), [0.0])
    if callable(server_optimizer_fn):
      self.assertAllEqual(state.optimizer_state, [1])
    self.assertEqual(state.delta_aggregate_state, 1)
    self.assertEqual(state.model_broadcast_state, 1)

    expected_outputs = collections.OrderedDict(
        broadcast=3.0,
        aggregation=collections.OrderedDict(num_clients=3),
        train={
            'loss': 15.25,
            'num_examples': 6,
        },
        stat=collections.OrderedDict(num_examples=3.0))
    self.assertAllEqual(expected_outputs, outputs)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
