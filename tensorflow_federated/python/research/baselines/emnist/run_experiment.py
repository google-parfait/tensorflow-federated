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
"""Trains and evaluates EMNIST classification model using TFF."""

import collections
import logging
import os
import sys

from absl import app
from absl import flags
import attr
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from tensorboard.plugins.hparams import api as hp
from tensorflow_federated.python.research.baselines.emnist import models
from tensorflow_federated.python.research.utils import training_loops
from tensorflow_federated.python.research.utils import utils_impl

nest = tf.contrib.framework.nest

with utils_impl.record_new_flags() as hparam_flags:
  # Metadata
  flags.DEFINE_string(
      'exp_name', 'emnist', 'Unique name for the experiment, suitable for use '
      'in filenames.')
  flags.DEFINE_integer('random_seed', 0, 'Random seed for the experiment.')

  # Training hyperparameters
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
  flags.DEFINE_integer('train_clients_per_round', 2,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server', defaults=dict(learning_rate=1.0))
  utils_impl.define_optimizer_flags('client', defaults=dict(learning_rate=0.2))

# End of hyperparameter flags.

# Root output directories.
flags.DEFINE_string('root_output_dir', '/tmp/emnist_fedavg/',
                    'Root directory for writing experiment output.')

FLAGS = flags.FLAGS


def _check_not_exists(f):
  """Checks if file `f` exists."""
  if tf.io.gfile.exists(f):
    print('ERROR, {} already exists.\n'
          'Please ensure only a single worker is executing each experiment '
          'in the grid.\n'
          'When re-running a grid, please use a fresh output directory.'.format(
              f))
    sys.exit(1)


@attr.s(cmp=False, frozen=False)
class MetricsHook(object):
  """A callback for evaluation.

  This class holds all the logic for evaluating the FedAvg on EMNIST
  classification and writing output for later analysis (to .csv files and
  tensorboard). Hyperparameters are also recorded.

  This class should be constructed via the `MetricsHook.build` classmethod.
  """

  # Derived, conceptually post-init or in constructor, used in methods.
  results_file = attr.ib()
  summary_writer = attr.ib()

  eval_dataset = attr.ib()
  model = attr.ib()

  results = attr.ib(factory=pd.DataFrame)

  @classmethod
  def build(cls, exp_name, output_dir, eval_dataset, hparam_dict):
    """Constructs the MetricsHook.

    Args:
      exp_name: A unique filesystem-friendly name for the experiment.
      output_dir: A root output directory used for all experiment runs in a
        grid. The MetricsHook will combine this with exp_name to form suitable
        output directories for this run.
      eval_dataset: Evaluation dataset.
      hparam_dict: A dictionary of hyperparameters to be recorded to .csv and
        exported to TensorBoard.

    Returns:
      The `MetricsHook` object.
    """

    summary_logdir = os.path.join(output_dir, 'logdir/{}'.format(exp_name))
    _check_not_exists(summary_logdir)
    tf.io.gfile.makedirs(summary_logdir)

    summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_logdir, name=exp_name)
    with summary_writer.as_default():
      hp.hparams(hparam_dict)

    # Using .bz2 rather than .zip due to
    # https://github.com/pandas-dev/pandas/issues/26023
    results_file = os.path.join(output_dir,
                                '{}.results.csv.bz2'.format(exp_name))

    # Also write the hparam_dict to a CSV:
    hparam_dict['results_file'] = results_file
    hparams_file = os.path.join(output_dir, '{}.hparams.csv'.format(exp_name))
    utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

    model = create_compiled_keras_model()

    logging.info('Writing ...')
    logging.info('   result csv to: %s', results_file)
    logging.info('    summaries to: %s', summary_logdir)

    return cls(
        results_file=results_file,
        summary_writer=summary_writer,
        eval_dataset=eval_dataset,
        model=model)

  def __attrs_post_init__(self):
    _check_not_exists(self.results_file)

  def __call__(self, server_state, metrics, round_num):
    """A function suitable for passing as an eval hook to the training_loop.

    Args:
      server_state: A `ServerState`.
      metrics: A dict of metrics computed in TFF.
      round_num: The current round number.
    """
    tff.learning.assign_weights_to_keras_model(self.model, server_state.model)
    eval_metrics = self.model.evaluate(self.eval_dataset, verbose=0)

    metrics['eval'] = collections.OrderedDict(
        zip(['loss', 'sparse_categorical_accuracy'], eval_metrics))

    flat_metrics = collections.OrderedDict(
        nest.flatten_with_joined_string_paths(metrics))

    # Use a DataFrame just to get nice formatting.
    df = pd.DataFrame.from_dict(flat_metrics, orient='index', columns=['value'])
    print(df)

    # Also write metrics to a tf.summary logdir
    with self.summary_writer.as_default():
      for name, value in flat_metrics.items():
        tf.compat.v2.summary.scalar(name, value, step=round_num)

    self.results = self.results.append(flat_metrics, ignore_index=True)
    utils_impl.atomic_write_to_csv(self.results, self.results_file)


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
  np.random.seed(FLAGS.random_seed)
  tf.random.set_random_seed(FLAGS.random_seed)

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

  metrics_hook = MetricsHook.build(FLAGS.exp_name, FLAGS.root_output_dir,
                                   emnist_test, hparam_dict)

  optimizer_fn = lambda: utils_impl.get_optimizer_from_flags('server')

  training_loops.federated_averaging_training_loop(
      model_fn,
      optimizer_fn,
      client_datasets_fn,
      total_rounds=FLAGS.total_rounds,
      rounds_per_eval=FLAGS.rounds_per_eval,
      metrics_hook=metrics_hook)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  tf.compat.v1.enable_v2_behavior()

  run_experiment()


if __name__ == '__main__':
  app.run(main)
