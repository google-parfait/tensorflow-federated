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
"""Example showing how to run a multi-machine simulation.

In order to run this example, you must have a running instance of the
Executor Service, either locally or on Kubernetes.

The model trains EMNIST for a small number of rounds, but uses a RemoteExecutor
to distribute the work to the ExecutorService.
"""

import collections
import warnings

from absl import app
from absl import flags
import grpc
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('host', None, 'The host to connect to.')
flags.mark_flag_as_required('host')
flags.DEFINE_string('port', '8000', 'The port to connect to.')
flags.DEFINE_integer('n_clients', 10, 'Number of clients.')
flags.DEFINE_integer('n_rounds', 3, 'Number of rounds.')


def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).batch(BATCH_SIZE)


def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]


NUM_EPOCHS = 10
BATCH_SIZE = 20


def make_remote_executor(inferred_cardinalities):
  """Make remote executor."""

  def create_worker_stack_on(ex):
    return tff.framework.ReferenceResolvingExecutor(
        tff.framework.ThreadDelegatingExecutor(ex))

  client_ex = []
  num_clients = inferred_cardinalities.get(tff.CLIENTS, None)
  if num_clients:
    print('Inferred that there are {} clients'.format(num_clients))
  else:
    print('No CLIENTS placement provided')

  for _ in range(num_clients or 0):
    channel = grpc.insecure_channel('{}:{}'.format(FLAGS.host, FLAGS.port))
    client_ex.append(
        create_worker_stack_on(
            tff.framework.RemoteExecutor(channel, rpc_mode='STREAMING')))

  federated_ex = tff.framework.FederatingExecutor({
      None: create_worker_stack_on(tff.framework.EagerTFExecutor()),
      tff.SERVER: create_worker_stack_on(tff.framework.EagerTFExecutor()),
      tff.CLIENTS: client_ex,
  })

  return tff.framework.ReferenceResolvingExecutor(federated_ex)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  warnings.simplefilter('ignore')

  np.random.seed(0)

  emnist_train, _ = tff.simulation.datasets.emnist.load_data()

  sample_clients = emnist_train.client_ids[0:FLAGS.n_clients]

  federated_train_data = make_federated_data(emnist_train, sample_clients)

  example_dataset = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0])

  preprocessed_example_dataset = preprocess(example_dataset)

  sample_batch = tf.nest.map_structure(
      lambda x: x.numpy(),
      iter(preprocessed_example_dataset).next())

  def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])
    return tff.learning.from_keras_model(
        model,
        dummy_batch=sample_batch,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))

  # Set the default executor to be a RemoteExecutor
  tff.framework.set_default_executor(
      tff.framework.create_executor_factory(make_remote_executor))

  state = iterative_process.initialize()

  state, metrics = iterative_process.next(state, federated_train_data)
  print('round  1, metrics={}'.format(metrics))

  for round_num in range(2, FLAGS.n_rounds + 1):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))


if __name__ == '__main__':
  app.run(main)
