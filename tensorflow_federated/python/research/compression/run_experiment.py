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
"""An example training loop lossily compressing the server/client communication.

Example command line flags to use to run an experiment:
--client_optimizer=sgd
--client_learning_rate=0.2
--server_optimizer=sgd
--server_learning_rate=1.0
--use_compression=True
--broadcast_quantization_bits=8
--aggregation_quantization_bits=8
--use_sparsity_in_aggregation=True
"""

import collections

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.compression import metrics_hook
from tensorflow_federated.python.research.compression import sparsity
from tensorflow_federated.python.research.utils import training_loops
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


with utils_impl.record_new_flags() as hparam_flags:
  # Metadata
  flags.DEFINE_string(
      'exp_name', 'emnist', 'Unique name for the experiment, suitable for use '
      'in filenames.')

  # Training hyperparameters
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
  flags.DEFINE_integer('train_clients_per_round', 2,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

  # Compression hyperparameters.
  flags.DEFINE_boolean('use_compression', True,
                       'Whether to use compression code path.')
  flags.DEFINE_integer(
      'broadcast_quantization_bits', 8,
      'Number of quatnization bits for server to client '
      'compression.')
  flags.DEFINE_integer(
      'aggregation_quantization_bits', 8,
      'Number of quatnization bits for client to server '
      'compression.')
  flags.DEFINE_boolean('use_sparsity_in_aggregation', True,
                       'Whether to add sparsity to the aggregation. This will '
                       'only be used for client to server compression.')

# End of hyperparameter flags.

# Root output directories.
flags.DEFINE_string('root_output_dir', '/tmp/emnist_fedavg/',
                    'Root directory for writing experiment output.')

FLAGS = flags.FLAGS


def create_compiled_keras_model(only_digits=True):
  """Create compiled keras model based on the original FedAvg CNN."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28 * 28,), target_shape=input_shape),
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape,
          data_format=data_format),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=utils_impl.create_optimizer_from_flags('client'),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model


def _broadcast_enocder_fn(value):
  """Function for building encoded broadcast.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in server to client communication.

  Returns:
    A `te.core.SimpleEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    return te.encoders.as_simple_encoder(
        te.encoders.uniform_quantization(FLAGS.broadcast_quantization_bits),
        spec)
  else:
    return te.encoders.as_simple_encoder(te.encoders.identity(), spec)


def _mean_enocder_fn(value):
  """Function for building encoded mean.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in client to server communication.

  Returns:
    A `te.core.GatherEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    if FLAGS.use_sparsity_in_aggregation:
      return te.encoders.as_gather_encoder(
          sparsity.sparse_quantizing_encoder(
              FLAGS.aggregation_quantization_bits), spec)
    else:
      return te.encoders.as_gather_encoder(
          te.encoders.uniform_quantization(FLAGS.aggregation_quantization_bits),
          spec)
  else:
    return te.encoders.as_gather_encoder(te.encoders.identity(), spec)


def run_experiment():
  """Data preprocessing and experiment execution."""
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

  example_tuple = collections.namedtuple('Example', ['x', 'y'])

  def element_fn(element):
    return example_tuple(
        x=tf.reshape(element['pixels'], [-1]),
        y=tf.reshape(element['label'], [1]))

  def preprocess_train_dataset(dataset):
    """Preprocess training dataset."""
    return dataset.map(element_fn).apply(
        tf.data.experimental.shuffle_and_repeat(
            buffer_size=10000,
            count=FLAGS.client_epochs_per_round)).batch(FLAGS.batch_size)

  def preprocess_test_dataset(dataset):
    """Preprocess testing dataset."""
    return dataset.map(element_fn).batch(100, drop_remainder=False)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients())

  example_dataset = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0])
  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       next(iter(example_dataset)))

  def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

  def client_datasets_fn(round_num):
    """Returns a list of client datasets."""
    del round_num  # Unused.
    sampled_clients = np.random.choice(
        emnist_train.client_ids,
        size=FLAGS.train_clients_per_round,
        replace=False)
    return [
        emnist_train.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]

  tf.io.gfile.makedirs(FLAGS.root_output_dir)
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  the_metrics_hook = metrics_hook.MetricsHook.build(
      FLAGS.exp_name, FLAGS.root_output_dir, emnist_test, hparam_dict,
      create_compiled_keras_model())

  optimizer_fn = lambda: utils_impl.create_optimizer_from_flags('server')

  if FLAGS.use_compression:
    # We create a `StatefulBroadcastFn` and `StatefulAggregateFn` by providing
    # the `_broadcast_enocder_fn` and `_mean_enocder_fn` to corresponding
    # utilities. The fns are called once for each of the model weights created
    # by model_fn, and return instances of appropriate encoders.
    encoded_broadcast_fn = (
        tff.learning.framework.build_encoded_broadcast_from_model(
            model_fn, _broadcast_enocder_fn))
    encoded_mean_fn = tff.learning.framework.build_encoded_mean_from_model(
        model_fn, _mean_enocder_fn)
  else:
    encoded_broadcast_fn = None
    encoded_mean_fn = None

  training_loops.federated_averaging_training_loop(
      model_fn,
      optimizer_fn,
      client_datasets_fn,
      total_rounds=FLAGS.total_rounds,
      rounds_per_eval=FLAGS.rounds_per_eval,
      metrics_hook=the_metrics_hook,
      stateful_model_broadcast_fn=encoded_broadcast_fn,
      stateful_delta_aggregate_fn=encoded_mean_fn)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  tf.compat.v1.enable_v2_behavior()
  tff.framework.set_default_executor(
      tff.framework.create_local_executor(max_fanout=25))

  run_experiment()


if __name__ == '__main__':
  app.run(main)
