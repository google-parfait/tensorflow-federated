# Copyright 2020, The TensorFlow Federated Authors.
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
"""End-to-end example testing the FedAvg algorithm with stateful clients."""

import collections
import functools

from absl.testing import parameterized
import keras
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from examplesstateful_clients import stateful_fedavg_tf
from examplesstateful_clients import stateful_fedavg_tff


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format,
  )
  conv2d = functools.partial(
      keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation='relu',
  )

  model = keras.models.Sequential([
      keras.Input(shape=input_shape),
      conv2d(filters=32),
      max_pool(),
      keras.layers.Flatten(),
      keras.layers.Dense(10),
      keras.layers.Activation('softmax'),
  ])

  return model


def _create_one_client_state():
  return stateful_fedavg_tf.ClientState(client_index=-1, iters_count=0)


def _simple_model_fn():
  keras_model = _create_test_cnn_model()
  loss = keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32),
  )
  return stateful_fedavg_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss
  )


def _tff_learning_model_fn():
  keras_model = _create_test_cnn_model()
  loss = keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32),
  )
  return tff.learning.models.functional_model_from_keras(
      keras_model=keras_model,
      input_spec=input_spec,
      loss_fn=loss,
      metrics_constructor=collections.OrderedDict(
          loss=keras.metrics.SparseCategoricalCrossentropy,
          num_examples=tff.learning.metrics.NumExamplesCounter,
          num_batches=tff.learning.metrics.NumBatchesCounter,
      ),
  )


def create_mnist_functional_model() -> tff.learning.models.FunctionalModel:
  """Creates a custom `FunctionalModel` for MNIST."""
  initial_weights = (
      (
          np.zeros(dtype=np.float32, shape=(784, 10)),
          np.zeros(dtype=np.float32, shape=(10,)),
      ),
      (),
  )

  @tf.function
  def predict_on_batch(model_weights, x, training=True):
    del training  # Unused.
    trainable_weights, _ = model_weights
    weights, bias = trainable_weights
    logits = tf.nn.softmax(tf.matmul(x, weights) + bias)
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    return logits, predictions

  @tf.function
  def loss_fn(output, label, sample_weight=None):
    del sample_weight  # Unused.
    logits, _ = output
    flat_labels = tf.reshape(label, [-1])
    return -tf.reduce_mean(
        tf.reduce_sum(
            tf.one_hot(flat_labels, 10) * tf.math.log(logits), axis=[1]
        )
    )

  @tf.function
  def initialize_metrics_state():
    return collections.OrderedDict(
        num_examples=(tf.constant(0.0, tf.float32),),
        loss=(tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)),
        accuracy=(tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32)),
    )

  @tf.function
  def update_metrics_state(state, labels, batch_output, sample_weight=None):
    del sample_weight  # Unused.
    _, predictions = batch_output.predictions
    flat_labels = tf.reshape(labels, [-1])
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predictions, flat_labels), tf.float32)
    )
    num_examples = tf.cast(batch_output.num_examples, tf.float32)
    return collections.OrderedDict(
        num_examples=(state['num_examples'][0] + num_examples,),
        loss=(
            state['loss'][0] + batch_output.loss * num_examples,
            state['loss'][1] + num_examples,
        ),
        accuracy=(
            state['accuracy'][0] + accuracy * num_examples,
            state['accuracy'][1] + num_examples,
        ),
    )

  @tf.function
  def finalize_metrics(state):
    return collections.OrderedDict(
        num_examples=state['num_examples'][0],
        loss=tf.math.divide_no_nan(state['loss'][0], state['loss'][1]),
        accuracy=tf.math.divide_no_nan(
            state['accuracy'][0], state['accuracy'][1]
        ),
    )

  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 784], tf.float32),
      y=tf.TensorSpec([None, 1], tf.int32),
  )
  return tff.learning.models.FunctionalModel(
      initial_weights=initial_weights,
      predict_on_batch_fn=predict_on_batch,
      loss_fn=loss_fn,
      metrics_fns=(
          initialize_metrics_state,
          update_metrics_state,
          finalize_metrics,
      ),
      input_spec=input_spec,
  )


def _create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[[5]], pixels=[np.random.rand(28, 28).astype(np.float32)]
  )
  dataset = tf.data.Dataset.from_tensor_slices(emnist_batch)

  def client_data(num_batch=1):
    return (
        tff.simulation.models.mnist.keras_dataset_from_emnist(dataset)
        .repeat(2 * num_batch)
        .batch(2)
    )

  return client_data


class FedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple_model_wrapper', _simple_model_fn),
      ('tff.learning.models.FunctionalModel', _tff_learning_model_fn),
  )
  def test_something(self, model_fn):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        model_fn, _create_one_client_state
    )
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,28,28,1],y=int32[?]>*}@CLIENTS'
    )

  @parameterized.named_parameters(
      ('simple_model_wrapper', _simple_model_fn),
      ('tff.learning.models.FunctionalModel', _tff_learning_model_fn),
  )
  def test_simple_training(self, model_fn):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        model_fn, _create_one_client_state
    )
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32),
      )

    batch = tff.tensorflow.computation(deterministic_batch)()
    federated_data = [[batch]]
    client_states = [_create_one_client_state()]

    loss_list = []
    for _ in range(3):
      server_state, loss, client_states = it_process.next(
          server_state, federated_data, client_states
      )
      loss_list.append(loss)

    self.assertEqual(
        server_state.total_iters_count, client_states[0].iters_count
    )
    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_self_contained_example_custom_model(self):
    client_data = _create_client_data()
    train_data = [client_data(), client_data(2)]
    client_states = [_create_one_client_state(), _create_one_client_state()]

    trainer = stateful_fedavg_tff.build_federated_averaging_process(
        create_mnist_functional_model, _create_one_client_state
    )
    state = trainer.initialize()
    losses = []
    for _ in range(2):
      state, loss, client_states = trainer.next(
          state, train_data, client_states
      )
      losses.append(loss)
    self.assertEqual(
        state.total_iters_count,
        client_states[0].iters_count + client_states[1].iters_count,
    )
    self.assertLess(losses[1], losses[0])

  def test_keras_evaluate(self):
    keras_model = _create_test_cnn_model()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32),
        )
    ]
    metric = keras.metrics.SparseCategoricalAccuracy()
    stateful_fedavg_tf.keras_evaluate(keras_model, sample_data, metric)
    accuracy = stateful_fedavg_tf.keras_evaluate(
        keras_model, sample_data, metric
    )
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)
    self.assertEqual(metric.count.numpy(), 1)

  def test_tff_learning_evaluate(self):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        _tff_learning_model_fn, _create_one_client_state
    )
    server_state = it_process.initialize()
    keras_model = _create_test_cnn_model()
    server_state.model_weights.assign_weights_to(keras_model)

    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32),
        )
    ]
    metric = keras.metrics.SparseCategoricalAccuracy()
    accuracy = stateful_fedavg_tf.keras_evaluate(
        keras_model, sample_data, metric
    )
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)


def _server_init(model, optimizer):
  """Returns initial `ServerState`.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A `ServerState` namedtuple.
  """
  model_weights = stateful_fedavg_tf.get_model_weights(model)
  stateful_fedavg_tff._initialize_optimizer_vars(model_weights, optimizer)
  return stateful_fedavg_tf.ServerState(
      model_weights=model_weights,
      optimizer_state=optimizer.variables(),
      round_num=0,
      total_iters_count=0,
  )


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
    model = model_fn()
    optimizer = optimizer_fn()
    state = _server_init(model, optimizer)
    model_weights = state.model_weights
    weights_delta = tf.nest.map_structure(tf.ones_like, model_weights.trainable)

    for _ in range(2):
      state = stateful_fedavg_tf.server_update(
          model,
          model_weights,
          optimizer,
          state,
          weights_delta,
          total_iters_count=0,
      )

    model_vars = self.evaluate(state.model_weights)
    train_vars = model_vars.trainable
    self.assertLen(train_vars, 2)
    self.assertEqual(state.round_num, 2)
    # weights are initialized with all-zeros, weights_delta is all ones,
    # SGD learning rate is 0.1. Updating server for 2 steps.
    self.assertAllClose(train_vars, [np.ones_like(v) * 0.2 for v in train_vars])

  def test_self_contained_example_custom_model(self):
    self._assert_server_update_with_all_ones(create_mnist_functional_model)


class ClientTest(tf.test.TestCase):

  def test_self_contained_example(self):
    client_data = _create_client_data()

    model = create_mnist_functional_model()
    model_weights = stateful_fedavg_tf.get_model_weights(model)
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
    losses = []
    for r in range(2):
      optimizer = optimizer_fn()
      stateful_fedavg_tff._initialize_optimizer_vars(model_weights, optimizer)
      server_message = stateful_fedavg_tf.BroadcastMessage(
          model_weights=model_weights, round_num=r
      )
      outputs = stateful_fedavg_tf.client_update(
          model,
          model_weights,
          client_data(),
          _create_one_client_state(),
          server_message,
          optimizer,
      )
      losses.append(outputs.model_output.numpy())

    self.assertEqual(outputs.client_state.iters_count, 1)
    self.assertAllEqual(int(outputs.client_weight.numpy()), 2)
    self.assertLess(losses[1], losses[0])


if __name__ == '__main__':
  tf.test.main()
