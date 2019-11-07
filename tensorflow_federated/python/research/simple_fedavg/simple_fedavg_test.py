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
"""End-to-end example testing Federated Averaging against the MNIST model."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.simple_fedavg import simple_fedavg

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _create_random_batch():
  return _Batch(
      x=tf.random.uniform(tf.TensorShape([1, 784]), dtype=tf.float32),
      y=tf.constant(1, dtype=tf.int64, shape=[1, 1]))


def _model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=True)
  batch = _create_random_batch()
  return tff.learning.from_compiled_keras_model(keras_model, batch)


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
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

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
    return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                        tf.float32)),
                                    ('y', tf.TensorSpec([None, 1], tf.int32))])

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


class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

  @tf.function
  def train_on_batch(self, batch):
    output = self.forward_pass(batch)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.02)
    optimizer.minimize(output.loss, var_list=self.trainable_variables)
    return output


def create_client_data():
  emnist_batch = collections.OrderedDict([('label', [5]),
                                          ('pixels', np.random.rand(28, 28))])

  output_types = collections.OrderedDict([('label', tf.int32),
                                          ('pixels', tf.float32)])

  output_shapes = collections.OrderedDict([
      ('label', tf.TensorShape([1])),
      ('pixels', tf.TensorShape([28, 28])),
  ])

  dataset = tf.data.Dataset.from_generator(lambda: (yield emnist_batch),
                                           output_types, output_shapes)

  def client_data():
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2).batch(2)

  return client_data


class SimpleFedAvgTest(tf.test.TestCase):

  def test_something(self):
    it_process = simple_fedavg.build_federated_averaging_process(_model_fn)
    self.assertIsInstance(it_process, tff.utils.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,784],y=int64[?,1]>*}@CLIENTS')

  def test_simple_training(self):
    it_process = simple_fedavg.build_federated_averaging_process(_model_fn)
    server_state = it_process.initialize()
    Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

    # Test out manually setting weights:
    keras_model = tff.simulation.models.mnist.create_keras_model(
        compile_model=True)

    def deterministic_batch():
      return Batch(
          x=np.ones([1, 784], dtype=np.float32),
          y=np.ones([1, 1], dtype=np.int64))

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    def keras_evaluate(state):
      tff.learning.assign_weights_to_keras_model(keras_model, state.model)
      # N.B. The loss computed here won't match the
      # loss computed by TFF because of the Dropout layer.
      keras_model.test_on_batch(batch.x, batch.y)

    loss_list = []
    for _ in range(3):
      keras_evaluate(server_state)
      server_state, loss = it_process.next(server_state, federated_data)
      loss_list.append(loss)
    keras_evaluate(server_state)

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_self_contained_example_keras_model(self):

    def model_fn():
      return tff.learning.from_compiled_keras_model(
          tff.simulation.models.mnist.create_simple_keras_model(), sample_batch)

    client_data = create_client_data()
    train_data = [client_data()]
    sample_batch = self.evaluate(next(iter(train_data[0])))

    trainer = simple_fedavg.build_federated_averaging_process(model_fn)
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, outputs = trainer.next(state, train_data)
      # Track the loss.
      losses.append(outputs.loss)
    self.assertLess(losses[1], losses[0])

  def test_self_contained_example_custom_model(self):

    client_data = create_client_data()
    train_data = [client_data()]

    trainer = simple_fedavg.build_federated_averaging_process(
        MnistTrainableModel)
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, outputs = trainer.next(state, train_data)
      # Track the loss.
      losses.append(outputs.loss)
    self.assertLess(losses[1], losses[0])


def server_init(model, optimizer):
  """Returns initial `tff.learning.framework.ServerState`.

  Args:
    model: A `tff.learning.Model`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A `tff.learning.framework.ServerState` namedtuple.
  """
  optimizer_vars = simple_fedavg._create_optimizer_vars(model, optimizer)
  return (simple_fedavg.ServerState(
      model=simple_fedavg._get_weights(model),
      optimizer_state=optimizer_vars), optimizer_vars)


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
    model = model_fn()
    optimizer = optimizer_fn()
    state, optimizer_vars = server_init(model, optimizer)
    weights_delta = tf.nest.map_structure(
        tf.ones_like,
        simple_fedavg._get_weights(model).trainable)

    for _ in range(2):
      state = simple_fedavg.server_update(model, optimizer, optimizer_vars,
                                          state, weights_delta)

    model_vars = self.evaluate(state.model)
    train_vars = model_vars.trainable
    self.assertLen(train_vars, 2)
    # weights are initialized with all-zeros, weights_delta is all ones,
    # SGD learning rate is 0.1. Updating server for 2 steps.
    self.assertAllClose(
        train_vars, {k: np.ones_like(v) * 0.2 for k, v in train_vars.items()})

  def test_self_contained_example_keras_model(self):

    def model_fn():
      return tff.learning.from_compiled_keras_model(
          tff.simulation.models.mnist.create_simple_keras_model(), sample_batch)

    client_data = create_client_data()
    sample_batch = self.evaluate(next(iter(client_data())))

    self._assert_server_update_with_all_ones(model_fn)

  def test_self_contained_example_custom_model(self):
    model_fn = MnistTrainableModel

    self._assert_server_update_with_all_ones(model_fn)


class ClientTest(tf.test.TestCase):

  def test_self_contained_example(self):

    client_data = create_client_data()

    model = MnistTrainableModel()
    losses = []
    for _ in range(2):
      outputs = simple_fedavg.client_update(model, client_data(),
                                            simple_fedavg._get_weights(model))
      losses.append(outputs.model_output['loss'].numpy())

    self.assertAllEqual(outputs.optimizer_output['num_examples'].numpy(), 2)
    self.assertLess(losses[1], losses[0])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
