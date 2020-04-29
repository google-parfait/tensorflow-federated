# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""End-to-end example testing targeted attacks against the MNIST model."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.research.targeted_attack import aggregate_fn
from tensorflow_federated.python.research.targeted_attack import attacked_fedavg
from tensorflow_federated.python.research.targeted_attack.attacked_fedavg import build_federated_averaging_process_attacked

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


def _model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')


def create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))


def mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int64)
  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32))

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return loss, predictions


def get_local_mnist_metrics(variables):
  return collections.OrderedDict([
      ('num_examples', variables.num_examples),
      ('loss', variables.loss_sum / variables.num_examples),
      ('accuracy', variables.accuracy_sum / variables.num_examples)
  ])


@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return {
      'num_examples': tff.federated_sum(metrics.num_examples),
      'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
      'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples)
  }


class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int64))

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    return tff.learning.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0])

  @tf.function
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients


def create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[5], pixels=np.random.rand(28, 28))
  output_types = collections.OrderedDict(label=tf.int64, pixels=tf.float32)
  output_shapes = collections.OrderedDict(
      label=tf.TensorShape([1]), pixels=tf.TensorShape([28, 28]))
  dataset = tf.data.Dataset.from_generator(lambda: (yield emnist_batch),
                                           output_types, output_shapes)

  def client_data():
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2).batch(2)

  return client_data


class ClientAttackTest(tf.test.TestCase):

  def test_input_types(self):
    it_process = build_federated_averaging_process_attacked(_model_fn)
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,784],y=int64[?,1]>*}@CLIENTS')
    self.assertEqual(
        str(it_process.next.type_signature.parameter[1]),
        str(it_process.next.type_signature.parameter[2]))
    federated_bool_type = it_process.next.type_signature.parameter[3]
    self.assertEqual(str(federated_bool_type), '{bool}@CLIENTS')

  def test_self_contained_example_keras_model(self):
    client_data = create_client_data()
    batch = client_data()
    train_data = [batch]
    malicious_data = [batch]
    client_type_list = [tf.constant(False)]
    trainer = build_federated_averaging_process_attacked(_model_fn)
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, outputs = trainer.next(state, train_data, malicious_data,
                                    client_type_list)
      losses.append(outputs.loss)
    self.assertLess(losses[1], losses[0])

  def test_self_contained_example_custom_model(self):
    client_data = create_client_data()
    batch = client_data()
    train_data = [batch]
    malicious_data = [batch]
    client_type_list = [tf.constant(False)]
    trainer = build_federated_averaging_process_attacked(MnistModel)
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, outputs = trainer.next(state, train_data, malicious_data,
                                    client_type_list)
      losses.append(outputs.loss)
    self.assertLess(losses[1], losses[0])

  def test_attack(self):
    """Test whether an attacker is doing the right attack."""
    self.skipTest('b/150215351 This test became flaky after TF change which '
                  'removed variable reads from control_outputs.')
    client_data = create_client_data()
    batch = client_data()
    train_data = [batch]
    malicious_data = [batch]
    client_type_list = [tf.constant(True)]
    trainer = build_federated_averaging_process_attacked(
        _model_fn,
        client_update_tf=attacked_fedavg.ClientExplicitBoosting(
            boost_factor=-1.0))
    state = trainer.initialize()
    initial_weights = state.model.trainable
    for _ in range(2):
      state, _ = trainer.next(state, train_data, malicious_data,
                              client_type_list)

    self.assertAllClose(initial_weights, state.model.trainable)


def server_init(model, optimizer, delta_aggregate_state=()):
  """Returns initial `tff.learning.framework.ServerState`.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.
    delta_aggregate_state: A server state.

  Returns:
    A `tff.learning.framework.ServerState` namedtuple.
  """
  optimizer_vars = attacked_fedavg._create_optimizer_vars(model, optimizer)
  return (attacked_fedavg.ServerState(
      model=attacked_fedavg._get_weights(model),
      optimizer_state=optimizer_vars,
      delta_aggregate_state=delta_aggregate_state), optimizer_vars)


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])
    optimizer = optimizer_fn()
    state, optimizer_vars = server_init(model, optimizer)
    weights_delta = tf.nest.map_structure(
        tf.ones_like,
        attacked_fedavg._get_weights(model).trainable)

    for _ in range(2):
      state = attacked_fedavg.server_update(model, optimizer, optimizer_vars,
                                            state, weights_delta, ())

    model_vars = self.evaluate(state.model)
    train_vars = model_vars.trainable
    # weights are initialized with all-zeros, weights_delta is all ones,
    # SGD learning rate is 0.1. Updating server for 2 steps.
    values = list(train_vars.values())
    self.assertAllClose(
        values, [np.ones_like(values[0]) * 0.2,
                 np.ones_like(values[1]) * 0.2])

  def test_self_contained_example_keras_model(self):
    self._assert_server_update_with_all_ones(_model_fn)

  def test_self_contained_example_custom_model(self):
    self._assert_server_update_with_all_ones(MnistModel)


class ClientTest(tf.test.TestCase):

  # TODO(b/155198591): bring GPU test back after the fix for tf.function.
  @test.skip_test_for_gpu
  def test_self_contained_example(self):
    client_data = create_client_data()
    model = MnistModel()
    optimizer = tf.keras.optimizers.SGD(0.1)
    losses = []
    client_update = attacked_fedavg.ClientExplicitBoosting(boost_factor=1.0)
    for _ in range(2):
      outputs = client_update(model, optimizer, client_data(), client_data(),
                              tf.constant(False),
                              attacked_fedavg._get_weights(model))
      losses.append(outputs.model_output['loss'].numpy())

    self.assertAllEqual(outputs.optimizer_output['num_examples'].numpy(), 2)
    self.assertLess(losses[1], losses[0])


class AggregationTest(tf.test.TestCase):

  def test_dp_fed_mean(self):
    """Test whether the norm clipping is done successfully."""
    client_data = create_client_data()
    batch = client_data()
    train_data = [batch]
    malicious_data = [batch]
    client_type_list = [tf.constant(False)]
    l2_norm = 0.01
    dp_aggregate_fn = aggregate_fn.build_dp_aggregate(l2_norm, 0.0, 1.0)
    trainer = build_federated_averaging_process_attacked(
        _model_fn, stateful_delta_aggregate_fn=dp_aggregate_fn)
    state = trainer.initialize()
    initial_weights = state.model.trainable
    state, _ = trainer.next(state, train_data, malicious_data, client_type_list)
    weights_delta = tf.nest.map_structure(tf.subtract,
                                          state.model.trainable._asdict(),
                                          initial_weights._asdict())
    self.assertLess(attacked_fedavg._get_norm(weights_delta), l2_norm * 1.1)

  def test_aggregate_and_clip(self):
    """Test whether the norm clipping is done successfully."""
    client_data = create_client_data()
    batch = client_data()
    train_data = [batch]
    malicious_data = [batch]
    client_type_list = [tf.constant(False)]
    l2_norm = 0.01
    aggregate_clip = aggregate_fn.build_aggregate_and_clip(norm_bound=l2_norm)
    trainer = build_federated_averaging_process_attacked(
        _model_fn, stateful_delta_aggregate_fn=aggregate_clip)
    state = trainer.initialize()
    initial_weights = state.model.trainable
    state, _ = trainer.next(state, train_data, malicious_data, client_type_list)
    weights_delta = tf.nest.map_structure(tf.subtract,
                                          state.model.trainable._asdict(),
                                          initial_weights._asdict())
    self.assertLess(attacked_fedavg._get_norm(weights_delta), l2_norm * 1.01)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
