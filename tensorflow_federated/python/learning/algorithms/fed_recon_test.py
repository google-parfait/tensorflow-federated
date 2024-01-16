# Copyright 2020, Google LLC.
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
from typing import Any
from unittest import mock

from absl.testing import parameterized
import attrs
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process as aggregation_process_lib
from tensorflow_federated.python.core.templates import iterative_process as iterative_process_lib
from tensorflow_federated.python.core.templates import measured_process as measured_process_lib
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning.algorithms import fed_recon
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import reconstruction_model
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import composers
from tensorflow_federated.python.learning.templates import distributors

# Convenience aliases.
ReconstructionModel = reconstruction_model.ReconstructionModel


def _create_input_spec():
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int32, shape=[None, 1]),
  )


def _create_keras_model():
  initializer = tf.keras.initializers.RandomNormal(seed=0)
  max_pool = tf.keras.layers.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format='channels_last'
  )
  model = tf.keras.Sequential([
      tf.keras.layers.Reshape(target_shape=[28, 28, 1], input_shape=(28 * 28,)),
      tf.keras.layers.Conv2D(
          32,
          5,
          padding='same',
          data_format='channels_last',
          activation=tf.nn.relu,
          kernel_initializer=initializer,
      ),
      max_pool,
      tf.keras.layers.Conv2D(
          64,
          5,
          padding='same',
          data_format='channels_last',
          activation=tf.nn.relu,
          kernel_initializer=initializer,
      ),
      max_pool,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          1024, activation=tf.nn.relu, kernel_initializer=initializer
      ),
      tf.keras.layers.Dropout(0.4, seed=1),
      tf.keras.layers.Dense(10, kernel_initializer=initializer),
  ])
  return model


def global_recon_model_fn():
  """Keras MNIST model with no local variables."""
  keras_model = _create_keras_model()
  input_spec = _create_input_spec()
  return ReconstructionModel.from_keras_model_and_layers(
      keras_model=keras_model,
      global_layers=keras_model.layers,
      local_layers=[],
      input_spec=input_spec,
  )


def local_recon_model_fn():
  """Keras MNIST model with final dense layer local."""
  keras_model = _create_keras_model()
  input_spec = _create_input_spec()
  return ReconstructionModel.from_keras_model_and_layers(
      keras_model=keras_model,
      global_layers=keras_model.layers[:-1],
      local_layers=keras_model.layers[-1:],
      input_spec=input_spec,
  )


@attrs.define(eq=False, frozen=True)
class MnistVariables:
  """Structure for variables in an MNIST model."""

  weights: float
  bias: float


class MnistModel(reconstruction_model.ReconstructionModel):
  """An MNIST `tff.learning.reconstruction.Model` implementation without Keras.

  Applies a single dense layer followed by softmax. The weights of the dense
  layer are global, and the biases are local.
  """

  def __init__(self):
    self._variables = MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True,
        ),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name='bias',
            trainable=True,
        ),
    )

  @property
  def global_trainable_variables(self):
    return [self._variables.weights]

  @property
  def global_non_trainable_variables(self):
    return []

  @property
  def local_trainable_variables(self):
    return [self._variables.bias]

  @property
  def local_non_trainable_variables(self):
    return []

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32),
    )

  @tf.function
  def forward_pass(self, batch, training=True):
    del training

    y = tf.nn.softmax(
        tf.matmul(batch['x'], self._variables.weights) + self._variables.bias
    )
    return reconstruction_model.ReconstructionBatchOutput(
        predictions=y, labels=batch['y'], num_examples=tf.size(batch['y'])
    )


def create_emnist_client_data():
  np.random.seed(42)
  emnist_data = collections.OrderedDict(
      x=[
          0.1 * np.random.randn(784).astype(np.float32),
          0.1 * np.random.randn(784).astype(np.float32),
          0.1 * np.random.randn(784).astype(np.float32),
      ],
      y=[[5], [5], [9]],
  )

  dataset = tf.data.Dataset.from_tensor_slices(emnist_data)

  def client_data(batch_size=2, max_examples=None):
    client_dataset = dataset
    if max_examples is not None:
      client_dataset = client_dataset.take(max_examples)
    client_dataset = client_dataset.batch(batch_size)
    return client_dataset

  return client_data


class _DPMean(factory.UnweightedAggregationFactory):

  def __init__(self, dp_sum_factory):
    self._dp_sum = dp_sum_factory
    self._clear_sum = sum_factory.SumFactory()

  def create(
      self, value_type: computation_types.Type
  ) -> aggregation_process_lib.AggregationProcess:
    self._dp_sum_process = self._dp_sum.create(value_type)

    @federated_computation.federated_computation()
    def init():
      # Invoke here to instantiate anything we need
      return self._dp_sum_process.initialize()

    @tensorflow_computation.tf_computation(value_type, np.int32)
    def div(x, y):
      # Opaque shape manipulations
      return [tf.squeeze(tf.math.divide_no_nan(x, tf.cast(y, tf.float32)), 0)]

    @federated_computation.federated_computation(
        init.type_signature.result,
        computation_types.FederatedType(value_type, placements.CLIENTS),
    )
    def next_fn(state, value):
      one_at_clients = intrinsics.federated_value(1, placements.CLIENTS)
      dp_sum = self._dp_sum_process.next(state, value)
      summed_one = intrinsics.federated_sum(one_at_clients)
      return measured_process_lib.MeasuredProcessOutput(
          state=dp_sum.state,
          result=intrinsics.federated_map(div, (dp_sum.result, summed_one)),
          measurements=dp_sum.measurements,
      )

    return aggregation_process_lib.AggregationProcess(
        initialize_fn=init, next_fn=next_fn
    )


def _get_tff_optimizer(learning_rate=0.1):
  return sgdm.build_sgdm(learning_rate=learning_rate, momentum=0.5)


def _get_keras_optimizer_fn(learning_rate=0.1):
  return lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate)


class TrainingProcessTest(tf.test.TestCase, parameterized.TestCase):

  def _run_rounds(self, learning_process, federated_data, num_rounds) -> tuple[
      composers.LearningAlgorithmState,
      list[collections.OrderedDict[str, Any]],
      composers.LearningAlgorithmState,
  ]:
    train_outputs: list[collections.OrderedDict[str, Any]] = []
    initial_state = learning_process.initialize()
    state = initial_state
    for _ in range(num_rounds):
      output = learning_process.next(state, federated_data)
      train_outputs.append(output.metrics['client_work'])
      state = output.state
    return state, train_outputs, initial_state

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_build_train_iterative_process(self, optimizer_fn):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=optimizer_fn(),
    )

    self.assertIsInstance(
        learning_process, iterative_process_lib.IterativeProcess
    )
    federated_data_type = learning_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,784],y=int32[?,1]>*}@CLIENTS'
    )

  def test_fed_recon_with_custom_client_weight_fn(self):
    self.skipTest('b/201413656')
    client_data = create_emnist_client_data()
    federated_data = [client_data()]

    def client_weight_fn(local_outputs):
      return 1.0 / (1.0 + local_outputs['loss'][-1])

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=_get_tff_optimizer(0.0001),
        reconstruction_optimizer_fn=_get_tff_optimizer(0.001),
        client_weighting=client_weight_fn,
    )

    _, train_outputs, _ = self._run_rounds(learning_process, federated_data, 5)
    self.assertLess(
        np.mean([train_outputs[-1]['loss'], train_outputs[-2]['loss']]),
        train_outputs[0]['loss'],
    )

  def test_server_update_with_inf_weight_is_noop(self):
    client_data = create_emnist_client_data()
    federated_data = [client_data()]
    client_weight_fn = lambda x: np.inf

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
        client_weighting=client_weight_fn,
    )

    state, _, initial_state = self._run_rounds(
        learning_process, federated_data, 1
    )
    self.assertAllClose(
        state.global_model_weights.trainable,
        initial_state.global_model_weights.trainable,
        1e-8,
    )
    self.assertAllClose(
        state.global_model_weights.trainable,
        initial_state.global_model_weights.trainable,
        1e-8,
    )

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_keras_global_model(self, optimizer_fn):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    learning_process = fed_recon.build_fed_recon(
        global_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=optimizer_fn(0.0001),
        reconstruction_optimizer_fn=optimizer_fn(0.001),
    )

    server_state = learning_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    metrics = []
    loss_list = []
    for _ in range(5):
      output = learning_process.next(server_state, federated_data)
      server_state = output.state
      server_states.append(server_state)
      metrics.append(output.metrics)
      loss_list.append(output.metrics['client_work']['train']['loss'])

    self.assertNotAllClose(
        server_states[0].global_model_weights.trainable,
        server_states[1].global_model_weights.trainable,
    )
    self.assertLess(np.mean(loss_list[-2:]), np.mean(loss_list[:2]))

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(metrics[0].keys(), expected_keys)

    expected_train_keys = [
        'sparse_categorical_accuracy',
        'loss',
        'num_examples',
        'num_batches',
    ]
    self.assertCountEqual(
        metrics[0]['client_work']['train'].keys(), expected_train_keys
    )

    # On both rounds, each client has 1 reconstruction batch with 2 examples,
    # and one post-reconstruction batch with 1 example.
    self.assertEqual(metrics[0]['client_work']['train']['num_examples'], 2)
    self.assertEqual(metrics[1]['client_work']['train']['num_batches'], 2)
    self.assertEqual(metrics[0]['client_work']['train']['num_examples'], 2)
    self.assertEqual(metrics[1]['client_work']['train']['num_batches'], 2)

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_keras_local_layer(self, optimizer_fn):
    self.skipTest('b/201413656')

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=optimizer_fn(0.001),
        reconstruction_optimizer_fn=optimizer_fn(0.001),
    )

    server_state = learning_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [
        client_data(batch_size=1, max_examples=2),
        client_data(batch_size=2),
    ]

    server_states = []
    metrics = []
    loss_list = []
    for _ in range(5):
      output = learning_process.next(server_state, federated_data)
      server_state = output.state
      server_states.append(server_state)
      metrics.append(output.metrics)
      loss_list.append(output.metrics['client_work']['train']['loss'])

    self.assertNotAllClose(
        server_states[0].global_model_weights.trainable,
        server_states[1].global_model_weights.trainable,
    )
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(metrics[0].keys(), expected_keys)

    expected_train_keys = [
        'sparse_categorical_accuracy',
        'loss',
        'num_examples',
        'num_batches',
    ]
    self.assertCountEqual(
        metrics[0]['client_work']['train'].keys(), expected_train_keys
    )

    # On both rounds, each client has one post-reconstruction batch with 1
    # example.
    self.assertEqual(metrics[0]['client_work']['train']['num_examples'], 2)
    self.assertEqual(metrics[0]['client_work']['train']['num_batches'], 2)
    self.assertEqual(metrics[1]['client_work']['train']['num_examples'], 2)
    self.assertEqual(metrics[1]['client_work']['train']['num_batches'], 2)

    expected_aggregation_keys = ['mean_weight', 'mean_value']
    self.assertCountEqual(
        output['aggregator'].keys(), expected_aggregation_keys
    )

  def test_keras_local_layer_client_weighting_enum_num_examples(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )

    server_state = learning_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(max_examples=2), client_data()]

    output = learning_process.next(server_state, federated_data)
    metrics = output.metrics

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(metrics.keys(), expected_keys)

    expected_train_keys = [
        'sparse_categorical_accuracy',
        'loss',
        'num_examples',
        'num_batches',
    ]
    self.assertCountEqual(
        metrics['client_work']['train'].keys(), expected_train_keys
    )

    # Only one client has a post-reconstruction batch, with one example.
    self.assertEqual(metrics['client_work']['train']['num_examples'], 1)
    self.assertEqual(metrics['client_work']['train']['num_batches'], 1)

    # Ensure we are using a weighted aggregator.
    expected_aggregation_keys = ['mean_weight', 'mean_value']
    self.assertCountEqual(
        metrics['aggregator'].keys(), expected_aggregation_keys
    )

  def test_keras_local_layer_client_weighting_enum_uniform(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
        client_weighting=client_weight_lib.ClientWeighting.UNIFORM,
        dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
    )

    server_state = learning_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(max_examples=2), client_data()]

    output = learning_process.next(server_state, federated_data)
    metrics = output.metrics

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(metrics.keys(), expected_keys)
    expected_train_keys = [
        'sparse_categorical_accuracy',
        'loss',
        'num_examples',
        'num_batches',
    ]
    self.assertCountEqual(
        metrics['client_work']['train'].keys(), expected_train_keys
    )

    self.assertEqual(metrics['client_work']['train']['num_examples'], 5)
    self.assertEqual(metrics['client_work']['train']['num_batches'], 3)

    # Ensure we are using a weighted aggregator.
    expected_aggregation_keys = ['mean_weight', 'mean_value']
    self.assertCountEqual(
        metrics['aggregator'].keys(), expected_aggregation_keys
    )

  def test_keras_local_layer_metrics_empty_list(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return []

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=_get_keras_optimizer_fn(0.0001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
    )

    server_state = learning_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    output = learning_process.next(server_state, federated_data)
    metrics = output.metrics

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(metrics.keys(), expected_keys)
    expected_client_work_keys = ['train']
    self.assertCountEqual(
        metrics['client_work'].keys(), expected_client_work_keys
    )
    expected_train_keys = ['loss']
    self.assertCountEqual(
        metrics['client_work']['train'].keys(), expected_train_keys
    )

  def test_keras_local_layer_metrics_none(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=None,
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
    )

    server_state = learning_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]
    output = learning_process.next(server_state, federated_data)

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(output.metrics.keys(), expected_keys)
    expected_client_work_keys = ['train']
    self.assertCountEqual(
        output.metrics['client_work'].keys(), expected_client_work_keys
    )
    expected_train_keys = ['loss']
    self.assertCountEqual(
        output.metrics['client_work']['train'].keys(), expected_train_keys
    )

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_custom_model_no_recon(self, optimizer_fn):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = fed_recon.build_fed_recon(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=optimizer_fn(0.01),
        client_optimizer_fn=optimizer_fn(0.001),
        reconstruction_optimizer_fn=optimizer_fn(0.0),
        dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
    )
    state = trainer.initialize()

    metrics = []
    states = []
    for _ in range(2):
      output = trainer.next(state, train_data)
      metrics.append(output.metrics)
      states.append(output.state)
      state = output.state

    train_metrics = (
        metrics[0]['client_work']['train'],
        metrics[1]['client_work']['train'],
    )
    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(train_metrics[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(train_metrics[1]['loss'], train_metrics[0]['loss'])
    self.assertNotAllClose(
        states[0].global_model_weights.trainable,
        states[1].global_model_weights.trainable,
    )

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_examples'], 6.0)
    self.assertEqual(train_metrics[1]['num_examples'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_batches'], 4.0)
    self.assertEqual(train_metrics[1]['num_batches'], 4.0)

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_custom_model_adagrad_server_optimizer(self, optimizer_fn):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = fed_recon.build_fed_recon(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=functools.partial(
            tf.keras.optimizers.Adagrad, 0.01
        ),
        client_optimizer_fn=optimizer_fn(0.001),
        reconstruction_optimizer_fn=optimizer_fn(0.0),
        dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
    )
    state = trainer.initialize()

    metrics = []
    states = []
    for _ in range(2):
      output = trainer.next(state, train_data)
      metrics.append(output.metrics)
      states.append(output.state)
      state = output.state

    train_metrics = (
        metrics[0]['client_work']['train'],
        metrics[1]['client_work']['train'],
    )
    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(train_metrics[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(train_metrics[1]['loss'], train_metrics[0]['loss'])
    self.assertNotAllClose(
        states[0].global_model_weights.trainable,
        states[1].global_model_weights.trainable,
    )

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_examples'], 6.0)
    self.assertEqual(train_metrics[1]['num_examples'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_batches'], 4.0)
    self.assertEqual(train_metrics[1]['num_batches'], 4.0)

  def test_custom_model_zeroing_clipping_aggregator_factory(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    # No values should be clipped and zeroed
    aggregation_factory = robust.zeroing_factory(
        zeroing_norm=float('inf'), inner_agg_factory=mean.MeanFactory()
    )

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = fed_recon.build_fed_recon(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=_get_keras_optimizer_fn(0.01),
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.0),
        model_aggregator_factory=aggregation_factory,
        dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
    )
    state = trainer.initialize()

    metrics = []
    states = []
    for _ in range(2):
      output = trainer.next(state, train_data)
      metrics.append(output.metrics)
      states.append(output.state)
      state = output.state

    train_metrics = (
        metrics[0]['client_work']['train'],
        metrics[1]['client_work']['train'],
    )
    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(train_metrics[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(train_metrics[1]['loss'], train_metrics[0]['loss'])
    self.assertNotAllClose(
        states[0].global_model_weights.trainable,
        states[1].global_model_weights.trainable,
    )

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_examples'], 6.0)
    self.assertEqual(train_metrics[1]['num_examples'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_batches'], 4.0)
    self.assertEqual(train_metrics[1]['num_batches'], 4.0)

  def test_iterative_process_fails_with_dp_agg_and_client_weight_fn(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    # No values should be changed, but working with inf directly zeroes out all
    # updates. Preferring very large value, but one that can be handled in
    # multiplication/division
    gaussian_sum_query = tfp.GaussianSumQuery(l2_norm_clip=1e10, stddev=0)
    dp_sum_factory = differential_privacy.DifferentiallyPrivateFactory(
        query=gaussian_sum_query,
        record_aggregation_factory=sum_factory.SumFactory(),
    )
    dp_mean_factory = _DPMean(dp_sum_factory)

    def client_weight_fn(local_outputs):
      del local_outputs  # Unused
      return 1.0

    with self.assertRaisesRegex(ValueError, 'unweighted aggregator'):
      fed_recon.build_fed_recon(
          MnistModel,
          loss_fn=loss_fn,
          metrics_fn=metrics_fn,
          server_optimizer_fn=_get_keras_optimizer_fn(0.01),
          client_optimizer_fn=_get_keras_optimizer_fn(0.001),
          reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.0),
          model_aggregator_factory=dp_mean_factory,
          client_weighting=client_weight_fn,
          dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
      )

  def test_iterative_process_fails_with_dp_agg_and_none_client_weighting(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    # No values should be changed, but working with inf directly zeroes out all
    # updates. Preferring very large value, but one that can be handled in
    # multiplication/division
    gaussian_sum_query = tfp.GaussianSumQuery(l2_norm_clip=1e10, stddev=0)
    dp_sum_factory = differential_privacy.DifferentiallyPrivateFactory(
        query=gaussian_sum_query,
        record_aggregation_factory=sum_factory.SumFactory(),
    )
    dp_mean_factory = _DPMean(dp_sum_factory)

    with self.assertRaisesRegex(ValueError, 'unweighted aggregator'):
      fed_recon.build_fed_recon(
          MnistModel,
          loss_fn=loss_fn,
          metrics_fn=metrics_fn,
          server_optimizer_fn=_get_keras_optimizer_fn(0.01),
          client_optimizer_fn=_get_keras_optimizer_fn(0.001),
          reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.0),
          model_aggregator_factory=dp_mean_factory,
          client_weighting=None,
          dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
      )

  def test_execution_with_custom_dp_query(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    # No values should be changed, but working with inf directly zeroes out all
    # updates. Preferring very large value, but one that can be handled in
    # multiplication/division
    gaussian_sum_query = tfp.GaussianSumQuery(l2_norm_clip=1e10, stddev=0)
    dp_sum_factory = differential_privacy.DifferentiallyPrivateFactory(
        query=gaussian_sum_query,
        record_aggregation_factory=sum_factory.SumFactory(),
    )
    dp_mean_factory = _DPMean(dp_sum_factory)

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = fed_recon.build_fed_recon(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=_get_keras_optimizer_fn(0.01),
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.0),
        model_aggregator_factory=dp_mean_factory,
        dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
        client_weighting=client_weight_lib.ClientWeighting.UNIFORM,
    )
    state = trainer.initialize()

    metrics = []
    states = []
    for _ in range(2):
      output = trainer.next(state, train_data)
      metrics.append(output.metrics)
      states.append(output.state)
      state = output.state

    train_metrics = (
        metrics[0]['client_work']['train'],
        metrics[1]['client_work']['train'],
    )
    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(train_metrics[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(train_metrics[1]['loss'], train_metrics[0]['loss'])
    self.assertNotAllClose(
        states[0].global_model_weights.trainable,
        states[1].global_model_weights.trainable,
    )

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_examples'], 6.0)
    self.assertEqual(train_metrics[1]['num_examples'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(train_metrics[0]['num_batches'], 4.0)
    self.assertEqual(train_metrics[1]['num_batches'], 4.0)

  def test_keras_local_layer_custom_broadcaster(self):
    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    model_weights_type = reconstruction_model.global_weights_type_from_model(
        local_recon_model_fn()
    )

    def build_custom_stateful_distributor(
        model_weights_type,
    ) -> distributors.DistributionProcess:
      """Builds a `MeasuredProcess` that wraps `tff.federated_broadcast`."""

      @federated_computation.federated_computation()
      def test_server_initialization():
        return intrinsics.federated_value(2.0, placements.SERVER)

      @federated_computation.federated_computation(
          computation_types.FederatedType(np.float32, placements.SERVER),
          computation_types.FederatedType(
              model_weights_type, placements.SERVER
          ),
      )
      def stateful_broadcast(state, value):
        empty_metrics = intrinsics.federated_value(1.0, placements.SERVER)
        return measured_process_lib.MeasuredProcessOutput(
            state=state,
            result=intrinsics.federated_broadcast(value),
            measurements=empty_metrics,
        )

      return distributors.DistributionProcess(
          initialize_fn=test_server_initialization, next_fn=stateful_broadcast
      )

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
        dataset_split_fn=ReconstructionModel.simple_dataset_split_fn,
        model_distributor=build_custom_stateful_distributor(
            model_weights_type=model_weights_type
        ),
    )

    algorithm_state = learning_process.initialize()

    # Ensure initialization of broadcaster produces expected metric.
    self.assertEqual(algorithm_state.distributor, 2.0)

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    output = learning_process.next(algorithm_state, federated_data)
    metrics = output.metrics

    expected_keys = ['distributor', 'client_work', 'aggregator', 'finalizer']
    self.assertCountEqual(metrics.keys(), expected_keys)
    expected_client_work_keys = ['train']
    self.assertCountEqual(
        metrics['client_work'].keys(), expected_client_work_keys
    )
    expected_train_keys = [
        'sparse_categorical_accuracy',
        'loss',
        'num_examples',
        'num_batches',
    ]
    self.assertCountEqual(
        metrics['client_work']['train'].keys(), expected_train_keys
    )
    self.assertEqual(metrics['distributor'], 1.0)

  @parameterized.named_parameters([
      ('keras_opt', _get_keras_optimizer_fn),
      ('tff_opt', _get_tff_optimizer),
  ])
  def test_custom_model_multiple_epochs(self, optimizer_fn):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          counters.NumExamplesCounter(),
          counters.NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy(),
      ]

    dataset_split_fn = ReconstructionModel.build_dataset_split_fn(
        recon_epochs=3, post_recon_epochs=4, post_recon_steps_max=3
    )
    trainer = fed_recon.build_fed_recon(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=optimizer_fn(0.001),
        reconstruction_optimizer_fn=optimizer_fn(0.001),
        dataset_split_fn=dataset_split_fn,
    )
    state = trainer.initialize()

    metrics = []
    states = []
    for _ in range(2):
      output = trainer.next(state, train_data)
      metrics.append(output.metrics)
      states.append(output.state)
      state = output.state

    train_metrics = (
        metrics[0]['client_work']['train'],
        metrics[1]['client_work']['train'],
    )
    self.assertLess(train_metrics[1]['loss'], train_metrics[0]['loss'])
    self.assertNotAllClose(
        states[0].global_model_weights.trainable,
        states[1].global_model_weights.trainable,
    )

    self.assertEqual(train_metrics[0]['num_examples'], 10.0)
    self.assertEqual(train_metrics[1]['num_examples'], 10.0)
    self.assertEqual(train_metrics[0]['num_batches'], 6.0)
    self.assertEqual(train_metrics[1]['num_batches'], 6.0)

  def test_get_model_weights(self):
    client_data = create_emnist_client_data()
    federated_data = [client_data()]

    learning_process = fed_recon.build_fed_recon(
        local_recon_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=_get_keras_optimizer_fn(0.001),
        reconstruction_optimizer_fn=_get_keras_optimizer_fn(0.001),
    )
    state = learning_process.initialize()

    self.assertIsInstance(
        learning_process.get_model_weights(state), model_weights.ModelWeights
    )
    self.assertAllClose(
        state.global_model_weights.trainable,
        learning_process.get_model_weights(state).trainable,
    )

    for _ in range(3):
      output = learning_process.next(state, federated_data)
      state = output.state
      self.assertIsInstance(
          learning_process.get_model_weights(state), model_weights.ModelWeights
      )
      self.assertAllClose(
          state.global_model_weights.trainable,
          learning_process.get_model_weights(state).trainable,
      )

  def test_process_construction_calls_model_fn(self):
    # Assert that the process building does not call `model_fn` too many times.
    # `model_fn` can potentially be expensive (loading weights, processing,
    # etc).
    mock_model_fn = mock.Mock(side_effect=local_recon_model_fn)
    fed_recon.build_fed_recon(
        model_fn=mock_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=_get_keras_optimizer_fn(),
    )
    # TODO: b/186451541 - Reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 3)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
