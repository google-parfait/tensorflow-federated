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

FLAGS = flags.FLAGS

flags.DEFINE_string('host', None, 'The host to connect to.')
flags.mark_flag_as_required('host')
flags.DEFINE_string('port', '8000', 'The port to connect to.')
flags.DEFINE_integer('n_clients', 10, 'Number of clients.')
flags.DEFINE_integer('n_rounds', 3, 'Number of rounds.')


def build_synthetic_emnist():
  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=np.random.normal(size=[100, 28 * 28]),
          y=np.random.uniform(low=0, high=9, size=[100]).astype(np.int32),
      )).repeat(NUM_EPOCHS).batch(BATCH_SIZE)


NUM_EPOCHS = 10
BATCH_SIZE = 20
GRPC_OPTIONS = [('grpc.max_message_length', 20 * 1024 * 1024),
                ('grpc.max_receive_message_length', 20 * 1024 * 1024)]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  warnings.simplefilter('ignore')

  np.random.seed(0)

  federated_train_data = [
      build_synthetic_emnist() for _ in range(FLAGS.n_clients)
  ]
  example_dataset = federated_train_data[0]
  input_spec = example_dataset.element_spec

  def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])
    return tff.learning.from_keras_model(
        model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02))

  context = tff.backends.native.create_remote_execution_context(channels=[  # pytype: disable=module-attr  # gen-stub-imports
      grpc.insecure_channel(f'{FLAGS.host}:{FLAGS.port}', options=GRPC_OPTIONS)
  ])
  tff.framework.set_default_context(context)
  print('Set default context.')

  state = iterative_process.initialize()

  state, metrics = iterative_process.next(state, federated_train_data)
  print('round  1, metrics={}'.format(metrics))

  for round_num in range(2, FLAGS.n_rounds + 1):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))


if __name__ == '__main__':
  app.run(main)
