# Lint as: python3
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
"""Simple FedAvg to train EMNIST.

This is intended to be a minimal stand-alone experiment script built on top of
core TFF.
"""

import collections
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.simple_fedavg import simple_fedavg_tf
from tensorflow_federated.python.research.simple_fedavg import simple_fedavg_tff

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
  """Load the EMNIST dataset.

  Preprocess the training and testing dataset.
  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
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

  This function is duplicated from research/baselines/emnist/models.py to
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


def keras_evaluate(model, test_data, metric):
  metric.reset_states()
  for batch in test_data:
    preds = model(batch['x'], training=False)
    metric(batch['y'], preds)
  return metric.result()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.compat.v1.enable_v2_behavior()

  train_data, test_data = get_emnist_dataset()

  @tf.function
  def get_sample_batch():
    """Return a sample batch to prepare model for TFF.

    Sample batch is used to enforce construction of model variables,
    and define data type in TFF. The returned sampled batch can be eager
    or non-eager depending on the contex.
    """
    example_dataset = train_data.create_tf_dataset_for_client(
        train_data.client_ids[0])
    return next(iter(example_dataset))

  def tff_model_fn():
    """Construct a fully initialized model for use in federated averaging."""
    sample_batch = get_sample_batch()
    keras_model = create_original_fedavg_cnn_model(only_digits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    return simple_fedavg_tf.KerasModelWrapper(keras_model, sample_batch, loss)

  iterative_process = simple_fedavg_tff.build_federated_averaging_process(
      tff_model_fn, server_optimizer_fn, client_optimizer_fn)
  server_state = iterative_process.initialize()

  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  model = tff_model_fn()
  for round_num in range(FLAGS.total_rounds):
    sampled_clients = np.random.choice(
        train_data.client_ids,
        size=FLAGS.train_clients_per_round,
        replace=False)
    sampled_train_data = [
        train_data.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]
    server_state, train_metrics = iterative_process.next(
        server_state, sampled_train_data)
    print(f'Round {round_num} training loss: {train_metrics}')
    if round_num % FLAGS.rounds_per_eval == 0:
      model.from_weights(server_state.model_weights)
      accuracy = keras_evaluate(model.keras_model, test_data, metric)
      print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')


if __name__ == '__main__':
  app.run(main)
