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
"""An implementation of the FedAvg algorithm with stateful clients.

The driver file with EMNSIT as an example.
"""

import collections
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.examples.stateful_clients import stateful_fedavg_tf
from tensorflow_federated.python.examples.stateful_clients import stateful_fedavg_tff

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 256, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 2,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 100, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

FLAGS = flags.FLAGS


def get_emnist_dataset():
  """Loads and preprocesses the EMNIST dataset.

  Returns:
    A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
    `tff.simulation.datasets.ClientData` object representing the training data
    and `emnist_test` is a single `tf.data.Dataset` representing the test data
    of all clients.
  """
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=True)

  def element_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(element['pixels'], -1), y=element['label'])

  def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
        count=FLAGS.client_epochs_per_round).batch(
            FLAGS.batch_size, drop_remainder=False)

  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(
        FLAGS.test_batch_size, drop_remainder=False)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients())
  return emnist_train, emnist_test


def create_original_fedavg_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  This function is duplicated from research/optimization/emnist/models.py to
  make this example completely stand-alone.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10 if only_digits else 62),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data = get_emnist_dataset()

  def tff_model_fn():
    """Constructs a fully initialized model for use in federated averaging."""
    keras_model = create_original_fedavg_cnn_model(only_digits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    return stateful_fedavg_tf.KerasModelWrapper(keras_model,
                                                test_data.element_spec, loss)

  # Initialize client states.
  client_states = {
      client_id: stateful_fedavg_tf.ClientState(client_index=i, iters_count=0)
      for i, client_id in enumerate(train_data.client_ids)
  }

  def get_sample_client_state():
    # Return a sample client state to initialize TFF types.
    return stateful_fedavg_tf.ClientState(client_index=-1, iters_count=0)

  iterative_process = stateful_fedavg_tff.build_federated_averaging_process(
      tff_model_fn, get_sample_client_state, server_optimizer_fn,
      client_optimizer_fn)
  server_state = iterative_process.initialize()

  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  model = tff_model_fn()
  for round_num in range(FLAGS.total_rounds):
    sampled_clients = np.random.choice(
        train_data.client_ids[:3],
        size=FLAGS.train_clients_per_round,
        replace=False)
    sampled_train_data = [
        train_data.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]
    sampled_client_states = [
        client_states[client] for client in sampled_clients
    ]  # Sample corresponding client states.
    server_state, train_metrics, updated_client_states = iterative_process.next(
        server_state, sampled_train_data, sampled_client_states)
    print(f'Round {round_num} training loss: {train_metrics}')
    # Save updated client states back into the global `client_states` structure.
    for client_state in updated_client_states:
      client_id = train_data.client_ids[client_state.client_index]
      client_states[client_id] = client_state
      print(f'Round {round_num} iterations on client '
            f'{client_id}: {client_state .iters_count}')
    print(f'Round {round_num} total iterations on '
          f'sampled clients: {server_state.total_iters_count}')
    if round_num % FLAGS.rounds_per_eval == 0:
      model.from_weights(server_state.model_weights)
      accuracy = stateful_fedavg_tf.keras_evaluate(model.keras_model, test_data,
                                                   metric)
      print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')


if __name__ == '__main__':
  app.run(main)
