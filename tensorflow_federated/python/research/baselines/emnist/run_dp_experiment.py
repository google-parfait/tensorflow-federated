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
"""Trains InfEMNIST model with TFF using differential privacy."""

import collections

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.baselines.emnist import metrics_hook
from tensorflow_federated.python.research.baselines.emnist import models
from tensorflow_federated.python.research.utils import training_loops
from tensorflow_federated.python.research.utils import utils_impl

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
  flags.DEFINE_integer('num_pseudo_clients', 1, 'Number of pseudo-clients.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server', defaults=dict(learning_rate=1.0))
  utils_impl.define_optimizer_flags('client', defaults=dict(learning_rate=0.2))

  # Differential privacy hyperparameters
  flags.DEFINE_float('clip', 0.05, 'Initial clip.')
  flags.DEFINE_float('noise_multiplier', 1.0, 'Noise multiplier.')
  flags.DEFINE_float('adaptive_clip_learning_rate', 0,
                     'Adaptive clip learning rate.')
  flags.DEFINE_float('target_unclipped_quantile', 0.5,
                     'Target unclipped quantile.')
  flags.DEFINE_float(
      'clipped_count_budget_allocation', 0.1,
      'Fraction of privacy budget to allocate for clipped counts.')
  flags.DEFINE_boolean('use_per_vector', False, 'Use per-vector clipping.')


# End of hyperparameter flags.

# Root output directories.
flags.DEFINE_string('root_output_dir', '/tmp/emnist_fedavg/',
                    'Root directory for writing experiment output.')

FLAGS = flags.FLAGS


def create_compiled_keras_model():
  """Create compiled keras model based on the original FedAvg CNN."""
  model = models.create_original_fedavg_cnn_model()

  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=utils_impl.get_optimizer_from_flags('client'),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model


def run_experiment():
  """Data preprocessing and experiment execution."""
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
  if FLAGS.num_pseudo_clients > 1:
    emnist_train = tff.simulation.datasets.emnist.get_infinite(
        emnist_train, FLAGS.num_pseudo_clients)
    emnist_test = tff.simulation.datasets.emnist.get_infinite(
        emnist_test, FLAGS.num_pseudo_clients)

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

  keras_model = create_compiled_keras_model()
  the_metrics_hook = metrics_hook.MetricsHook.build(
      FLAGS.exp_name, FLAGS.root_output_dir, emnist_test, hparam_dict,
      keras_model)

  optimizer_fn = lambda: utils_impl.get_optimizer_from_flags('server')

  model = tff.learning.from_compiled_keras_model(keras_model, sample_batch)
  dp_query = tff.utils.build_dp_query(
      FLAGS.clip, FLAGS.noise_multiplier, FLAGS.train_clients_per_round,
      FLAGS.adaptive_clip_learning_rate, FLAGS.target_unclipped_quantile,
      FLAGS.clipped_count_budget_allocation, FLAGS.train_clients_per_round,
      FLAGS.use_per_vector, model)

  # Uniform weighting.
  def client_weight_fn(outputs):
    del outputs  # unused.
    return 1.0

  dp_aggregate_fn, _ = tff.utils.build_dp_aggregate(dp_query)

  def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

  training_loops.federated_averaging_training_loop(
      model_fn,
      optimizer_fn,
      client_datasets_fn,
      total_rounds=FLAGS.total_rounds,
      rounds_per_eval=FLAGS.rounds_per_eval,
      metrics_hook=the_metrics_hook,
      client_weight_fn=client_weight_fn,
      stateful_delta_aggregate_fn=dp_aggregate_fn)


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
