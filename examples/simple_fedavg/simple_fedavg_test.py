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
"""End-to-end example testing Federated Averaging with CNN and RNN."""

import collections
import functools

from absl.testing import parameterized
import federated_language
import keras
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from examplessimple_fedavg import simple_fedavg_tf
from examplessimple_fedavg import simple_fedavg_tff


def _create_test_cnn_model():
  """Creates a deterministic CNN model for testing."""
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
      keras.layers.Dense(10, kernel_initializer='zeros'),
      keras.layers.Activation('softmax'),
  ])

  return model


NUM_LOGICAL_DEVICES = 8


# Initialize logical devices only once for the module.
def setUpModule():
  devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(
      devices[0],
      [
          tf.config.LogicalDeviceConfiguration(),
      ]
      * NUM_LOGICAL_DEVICES,
  )


def _setup_local_context():
  tff.backends.native.set_sync_local_cpp_execution_context()


def _tff_learning_model_fn():
  """Constructs a test `tff.learning.models.FunctionalModel`."""
  keras_model = _create_test_cnn_model()
  loss = keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32),
  )
  return tff.learning.models.functional_model_from_keras(
      keras_model=keras_model,
      loss_fn=loss,
      input_spec=input_spec,
      metrics_constructor=collections.OrderedDict(
          loss=keras.metrics.SparseCategoricalCrossentropy,
          sparse_categorical_accuracy=keras.metrics.SparseCategoricalAccuracy,
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

  def client_data():
    return (
        tff.simulation.models.mnist.keras_dataset_from_emnist(dataset)
        .repeat(2)
        .batch(2)
    )

  return client_data


@parameterized.named_parameters(
    ('tensorflow_use_sequence_reduce', True),
    ('tensorflow_use_dataset_iteration', False),
)
class SimpleFedAvgTest(tf.test.TestCase, parameterized.TestCase):

  def test_process_construction(
      self,
      use_sequence_reduce,
  ):
    _setup_local_context()
    it_process = simple_fedavg_tff.build_federated_averaging_process(
        _tff_learning_model_fn,
        use_sequence_reduce=use_sequence_reduce,
    )
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        federated_data_type,
        federated_language.FederatedType(
            federated_language.SequenceType(
                collections.OrderedDict(
                    x=federated_language.TensorType(
                        np.float32, [None, 28, 28, 1]
                    ),
                    y=federated_language.TensorType(np.int32, [None]),
                )
            ),
            federated_language.CLIENTS,
        ),
    )

  def test_training_keras_model_converges(
      self,
      use_sequence_reduce,
  ):
    _setup_local_context()
    it_process = simple_fedavg_tff.build_federated_averaging_process(
        _tff_learning_model_fn, use_sequence_reduce=use_sequence_reduce
    )
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32),
      )

    batch = tff.tensorflow.computation(deterministic_batch)()
    federated_data = [tf.data.Dataset.from_tensor_slices(batch).batch(1)]

    loss = 1.0
    first_loss = None
    for _ in range(10):
      server_state, outputs = it_process.next(server_state, federated_data)
      loss = outputs['loss']
      if first_loss is None:
        first_loss = loss

    self.assertLess(loss, first_loss)

  def test_training_custom_model_converges(
      self,
      use_sequence_reduce,
  ):
    _setup_local_context()
    client_data = _create_client_data()
    train_data = [client_data()]

    trainer = simple_fedavg_tff.build_federated_averaging_process(
        create_mnist_functional_model, use_sequence_reduce=use_sequence_reduce
    )
    state = trainer.initialize()

    loss = 1.0
    first_loss = None
    for _ in range(10):
      state, outputs = trainer.next(state, train_data)
      loss = outputs['loss']
      if first_loss is None:
        first_loss = loss

    self.assertLess(loss, first_loss)


def _server_init(model, optimizer):
  """Returns initial `ServerState`.

  Args:
    model: A `tff.learning.models.FunctionalModel`.
    optimizer: A `tf.train.Optimizer`.

  Returns:
    A `ServerState` namedtuple.
  """
  model_weights = simple_fedavg_tf.get_model_weights(model)
  simple_fedavg_tff._initialize_optimizer_vars(model_weights, optimizer)
  return simple_fedavg_tf.ServerState(
      model=model_weights,
      optimizer_state=optimizer.variables(),
      round_num=0,
  )


class ServerTest(tf.test.TestCase):

  def _assert_server_update_with_all_ones(self, model_fn):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
    model = model_fn()
    optimizer = optimizer_fn()
    state = _server_init(model, optimizer)
    model_weights = state.model
    weights_delta = tf.nest.map_structure(tf.ones_like, model_weights.trainable)

    for _ in range(2):
      state = simple_fedavg_tf.server_update(
          model, model_weights, optimizer, state, weights_delta
      )

    model_vars = self.evaluate(state.model)
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
    model_weights = simple_fedavg_tf.get_model_weights(model)
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=0.01)
    losses = []
    for r in range(2):
      optimizer = optimizer_fn()
      simple_fedavg_tff._initialize_optimizer_vars(model_weights, optimizer)
      server_message = simple_fedavg_tf.BroadcastMessage(
          model_weights=model_weights, round_num=r
      )
      outputs = simple_fedavg_tf.client_update(
          model, model_weights, client_data(), server_message, optimizer
      )
      losses.append(
          tf.math.divide_no_nan(
              outputs.model_output['loss'][0], outputs.model_output['loss'][1]
          )
      )
      self.assertEqual(int(outputs.client_weight.numpy()), 2)
    self.assertLess(losses[1], losses[0])


def _create_test_rnn_model(
    vocab_size: int = 6, mask_zero: bool = True
) -> keras.Model:
  """A simple RNN model for test."""
  model = keras.Sequential()
  model.add(
      keras.layers.Embedding(
          input_dim=vocab_size,
          output_dim=8,
          mask_zero=mask_zero,
      )
  )
  model.add(
      keras.layers.LSTM(
          units=16,
          kernel_initializer='he_normal',
          return_sequences=True,
          stateful=False,
      )
  )
  model.add(keras.layers.Dense(vocab_size))
  return model


def _rnn_model_fn() -> tff.learning.models.FunctionalModel:
  keras_model = _create_test_rnn_model()
  loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 5], tf.int32), y=tf.TensorSpec([None, 5], tf.int32)
  )
  return tff.learning.models.functional_model_from_keras(
      keras_model=keras_model,
      loss_fn=loss,
      input_spec=input_spec,
      metrics_constructor=collections.OrderedDict(
          loss=functools.partial(
              keras.metrics.SparseCategoricalCrossentropy, from_logits=True
          ),
          num_examples=tff.learning.metrics.NumExamplesCounter,
          num_batches=tff.learning.metrics.NumBatchesCounter,
      ),
  )


class RNNTest(tf.test.TestCase):

  def test_build_fedavg_process(self):

    def server_optimizer_fn():
      return tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    def client_optimizer_fn():
      return tf.keras.optimizers.legacy.SGD(learning_rate=0.02)

    it_process = simple_fedavg_tff.build_federated_averaging_process(
        _rnn_model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_optimizer_fn=client_optimizer_fn,
    )
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    global_model_type, client_datasets_type = (
        it_process.next.type_signature.parameter
    )
    model_type = tff.learning.models.weights_type_from_model(_rnn_model_fn())
    self.assertEqual(
        global_model_type,
        federated_language.FederatedType(
            simple_fedavg_tf.ServerState(
                model=model_type,
                optimizer_state=[np.int64],
                round_num=np.int32,
            ),
            federated_language.SERVER,
        ),
    )
    self.assertEqual(
        client_datasets_type,
        federated_language.FederatedType(
            federated_language.SequenceType(
                collections.OrderedDict(
                    x=federated_language.TensorType(np.int32, [None, 5]),
                    y=federated_language.TensorType(np.int32, [None, 5]),
                )
            ),
            federated_language.CLIENTS,
        ),
    )

  def test_client_adagrad_train(self):
    _setup_local_context()
    it_process = simple_fedavg_tff.build_federated_averaging_process(
        _rnn_model_fn,
        client_optimizer_fn=functools.partial(
            tf.keras.optimizers.legacy.SGD, learning_rate=0.01
        ),
    )
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32),
      )

    batch = tff.tensorflow.computation(deterministic_batch)()
    federated_data = [[batch]]

    loss_list = []
    for _ in range(3):
      server_state, outputs = it_process.next(server_state, federated_data)
      loss_list.append(outputs['loss'])

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])


if __name__ == '__main__':
  np.random.seed(42)
  tf.test.main()
