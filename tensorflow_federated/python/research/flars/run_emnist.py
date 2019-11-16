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
import functools
import io
import logging
import os
import pprint
import random
import sys
import time

from absl import app
from absl import flags
import attr
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_federated as tff
import tree

from tensorboard.plugins.hparams import api as hp
from tensorflow_federated.python.research.baselines.emnist import models
from tensorflow_federated.python.research.flars import flars_fedavg
from tensorflow_federated.python.research.flars import flars_optimizer
from tensorflow_federated.python.research.simple_fedavg import simple_fedavg
from tensorflow_federated.python.research.utils import checkpoint_utils
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  # Metadata
  flags.DEFINE_string(
      'exp_name', 'emnist', 'Unique name for the experiment, suitable for use '
      'in filenames.')

  # Training hyperparameters
  flags.DEFINE_boolean(
      'digit_only_emnist', True,
      'Whether to train on the digits only (10 classes) data '
      'or the full data (62 classes).')
  flags.DEFINE_integer('total_rounds', 500, 'Number of total training rounds.')
  flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
  flags.DEFINE_integer(
      'rounds_per_checkpoint', 25,
      'How often to emit a state checkpoint. Higher numbers '
      'mean more lost work in case of failure, lower numbers '
      'mean more overhead per round.')
  flags.DEFINE_integer('train_clients_per_round', 2,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')

  # Client optimizer configuration (it defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('client')

  # Server optimizer configuration (it defines one or more flags per optimizer).
  flags.DEFINE_enum('server_optimizer', 'flars', ['sgd', 'flars'],
                    'Server optimizer')
  flags.DEFINE_float('server_learning_rate', 1., 'Server learning rate.')
  flags.DEFINE_float(
      'server_momentum', 0.9,
      'Server momentum. This is also the `beta1` parameter for '
      'the Yogi optimizer.')

  # Parameter for FLARS.
  flags.DEFINE_float('max_ratio', 0.1, 'max_ratio for optimizer FLARS.')
  # Parameter for Yogi.
  flags.DEFINE_float('initial_accumulator_value', 1e-6,
                     'initial_accumulator_value for optimizer Yogi.')

# End of hyperparameter flags.

# Root output directories.
flags.DEFINE_string(
    'root_output_dir', '/tmp/emnist_fedavg/',
    'Root directory for writing experiment output. This will '
    'be the destination for metrics CSV files, Tensorboard log '
    'directory, and checkpoint files.')
flags.DEFINE_boolean(
    'disable_check_exists', True, 'Disable checking the '
    'existence of root_output_dir. If False, code will exit '
    'without running the experiment if root_output_dir '
    'exists.')

FLAGS = flags.FLAGS
CHECKPOINT_PREFIX = 'ckpt_'


def federated_averaging_training_loop(model_fn,
                                      server_optimizer_fn,
                                      client_datasets_fn,
                                      total_rounds=500,
                                      rounds_per_eval=1,
                                      metrics_hook=lambda *args: None):
  """A simple example of training loop for the Federated Averaging algorithm.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    client_datasets_fn: A function that takes the round number, and returns a
      list of `tf.data.Datset`, one per client.
    total_rounds: Number of rounds to train.
    rounds_per_eval: How often to call the  `metrics_hook` function.
    metrics_hook: A function taking arguments (server_state, train_metrics,
      round_num) and performs evaluation. Optional.

  Returns:
    Final `ServerState`.
  """
  logging.info('Starting federated_training_loop')
  checkpoint_dir = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name)

  if FLAGS.server_optimizer != 'flars':
    iterative_process = simple_fedavg.build_federated_averaging_process(
        model_fn, server_optimizer_fn=server_optimizer_fn)
    ServerState = simple_fedavg.ServerState  # pylint: disable=invalid-name
  else:
    iterative_process = flars_fedavg.build_federated_averaging_process(
        model_fn, server_optimizer_fn=server_optimizer_fn)
    ServerState = flars_fedavg.ServerState  # pylint: disable=invalid-name

  server_state = ServerState.from_anon_tuple(iterative_process.initialize())
  round_num = None
  train_metrics = {}

  latest_checkpoint_dir = checkpoint_utils.latest_checkpoint(
      checkpoint_dir, CHECKPOINT_PREFIX)
  logging.info('Looking for checkpoints in [%s/%s]', checkpoint_dir,
               CHECKPOINT_PREFIX)
  while latest_checkpoint_dir is not None:
    # Restart from a previous round.
    logging.info('Loading a previous checkpoint')
    try:
      server_state, metrics_hook.results, round_num = read_checkpoint(
          latest_checkpoint_dir, server_state)
      break
    except OSError as e:
      # Likely corrupted checkpoint, possibly job died while writing. Delete the
      # checkpoint directory and try again.
      logging.error('Exception [%s]', e)
      logging.warning('Deleteing likely corrupted checkpoint at [%s]',
                      latest_checkpoint_dir)
      tf.io.gfile.rmtree(latest_checkpoint_dir)
      latest_checkpoint_dir = checkpoint_utils.latest_checkpoint(
          checkpoint_dir, CHECKPOINT_PREFIX)

  if round_num is not None:
    logging.info('Restarted from checkpoint round %d', round_num)
  else:
    # Write the initial checkpoint
    logging.info('No previous checkpoints, initializing experiment')
    round_num = 0
    metrics_hook(server_state, train_metrics, round_num)
    write_checkpoint(checkpoint_dir, server_state, metrics_hook.results,
                     round_num)

  while round_num < total_rounds:
    round_num += 1
    # Reset the executor to clear the cache, and clear the default graph to
    # garbage collect tf.Functions that will no longer be used.
    tff.framework.set_default_executor(
        tff.framework.create_local_executor(max_fanout=25))
    tf.compat.v1.reset_default_graph()

    round_start_time = time.time()
    data_prep_start_time = time.time()
    federated_train_data = client_datasets_fn(round_num)
    train_metrics['prepare_datasets_secs'] = time.time() - data_prep_start_time

    training_start_time = time.time()
    anon_tuple_server_state, tff_train_metrics = iterative_process.next(
        server_state, federated_train_data)
    server_state = ServerState.from_anon_tuple(anon_tuple_server_state)
    train_metrics.update(tff_train_metrics._asdict(recursive=True))
    train_metrics['training_secs'] = time.time() - training_start_time

    logging.info('Round {:2d} elapsed time: {:.2f}s .'.format(
        round_num, (time.time() - round_start_time)))
    train_metrics['total_round_secs'] = time.time() - round_start_time

    if round_num % FLAGS.rounds_per_checkpoint == 0:
      write_checkpoint_start_time = time.time()
      write_checkpoint(checkpoint_dir, server_state, metrics_hook.results,
                       round_num)
      train_metrics['write_checkpoint_secs'] = (
          time.time() - write_checkpoint_start_time)

    if round_num % rounds_per_eval == 0:
      metrics_hook(server_state, train_metrics, round_num)

  metrics_hook(server_state, train_metrics, total_rounds)
  write_checkpoint(checkpoint_dir, server_state, metrics_hook.results,
                   round_num)

  return server_state


def _check_not_exists(f, disable_check_exists=False):
  """Checks if file `f` exists."""
  if disable_check_exists:
    return
  if tf.io.gfile.exists(f):
    print('{} already exists.\n'
          'Please ensure only a single worker is executing each experiment '
          'in the grid.\n'
          'When re-running a grid, please use a fresh output directory.'.format(
              f))
    sys.exit(1)


@attr.s(frozen=False)
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
    _check_not_exists(summary_logdir, FLAGS.disable_check_exists)
    tf.io.gfile.makedirs(summary_logdir)

    summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_logdir, name=exp_name)
    with summary_writer.as_default():
      hp.hparams(hparam_dict)

    # Using .bz2 rather than .zip due to
    # https://github.com/pandas-dev/pandas/issues/26023
    results_file = os.path.join(output_dir, exp_name, 'results.csv.bz2')

    # Also write the hparam_dict to a CSV:
    hparam_dict['results_file'] = results_file
    hparams_file = os.path.join(output_dir, exp_name, 'hparams.csv')
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
    _check_not_exists(self.results_file, FLAGS.disable_check_exists)

  def __call__(self, server_state, train_metrics, round_num):
    """A function suitable for passing as an eval hook to the training_loop.

    Args:
      server_state: A `ServerState`.
      train_metrics: A `dict` of training metrics computed in TFF.
      round_num: The current round number.
    """
    tff.learning.assign_weights_to_keras_model(self.model, server_state.model)
    eval_metrics = self.model.evaluate(self.eval_dataset, verbose=0)

    metrics = {
        'train':
            train_metrics,
        'eval':
            collections.OrderedDict(
                zip(['loss', 'sparse_categorical_accuracy'], eval_metrics))
    }
    flat_metrics = tree.flatten_with_path(metrics)
    flat_metrics = [
        ('/'.join(map(str, path)), item) for path, item in flat_metrics
    ]
    flat_metrics = collections.OrderedDict(flat_metrics)
    flat_metrics['round'] = round_num

    logging.info('Evaluation at round {:d}:\n{!s}'.format(
        round_num, pprint.pformat(flat_metrics)))

    # Also write metrics to a tf.summary logdir
    with self.summary_writer.as_default():
      for name, value in flat_metrics.items():
        tf.compat.v2.summary.scalar(name, value, step=round_num)

    self.results = self.results.append(flat_metrics, ignore_index=True)
    utils_impl.atomic_write_to_csv(self.results, self.results_file)


@attr.s(frozen=False)
class ExperimentState(object):
  round_num = attr.ib()
  metrics_csv_string = attr.ib()
  server_state = attr.ib()


def write_checkpoint(checkpoint_dir, server_state, metrics_dataframe,
                     round_num):
  """Write the current experiment state to disk."""
  # Clean-up old checkpoints if more than 5 exist; but not the initialization
  # checkpoint.
  checkpoints = sorted(tf.io.gfile.glob(os.path.join(checkpoint_dir, 'ckpt_*')))
  for checkpoint in checkpoints[1:-3]:
    tf.io.gfile.rmtree(checkpoint)

  # We must flatten the pd.Dataframe to a single string, otherwise we don't
  # know the nested structure (how many rounds have passed) to unpack
  # in `checkpoint_utils.load()` during `read_checkpoint`.
  csv_string = io.StringIO()
  metrics_dataframe.to_csv(csv_string, header=True)

  experiment_state = ExperimentState(
      round_num=round_num,
      metrics_csv_string=csv_string.getvalue(),
      server_state=server_state)
  checkpoint_utils.save(
      experiment_state,
      os.path.join(checkpoint_dir, 'ckpt_{:03d}'.format(round_num)))


def read_checkpoint(filepath, example_server_state):
  """Read a previously saved experiment state to memory."""
  experiment_state = ExperimentState(
      round_num=0, metrics_csv_string='', server_state=example_server_state)
  experiment_state = checkpoint_utils.load(filepath, experiment_state)
  metrics_dict = pd.read_csv(
      io.BytesIO(experiment_state.metrics_csv_string.numpy()),
      header=0,
      index_col=0,
      engine='c')
  return (experiment_state.server_state, metrics_dict,
          experiment_state.round_num.numpy())


def create_compiled_keras_model():
  """Create compiled keras model."""
  model = models.create_original_fedavg_cnn_model(
      only_digits=FLAGS.digit_only_emnist)

  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=utils_impl.create_optimizer_from_flags('client'),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model


def run_experiment():
  """Data preprocessing and experiment execution."""
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=FLAGS.digit_only_emnist)

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
    sampled_clients = random.sample(
        population=emnist_train.client_ids, k=FLAGS.train_clients_per_round)
    return [
        emnist_train.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]

  tf.io.gfile.makedirs(FLAGS.root_output_dir)
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  hparam_dict = utils_impl.remove_unused_flags('client', hparam_dict)

  metrics_hook = MetricsHook.build(FLAGS.exp_name, FLAGS.root_output_dir,
                                   emnist_test, hparam_dict)

  if FLAGS.server_optimizer == 'sgd':
    optimizer_fn = functools.partial(
        tf.keras.optimizers.SGD,
        learning_rate=FLAGS.server_learning_rate,
        momentum=FLAGS.server_momentum)
  elif FLAGS.server_optimizer == 'flars':
    optimizer_fn = functools.partial(
        flars_optimizer.FLARSOptimizer,
        learning_rate=FLAGS.server_learning_rate,
        momentum=FLAGS.server_momentum,
        max_ratio=FLAGS.max_ratio)
  else:
    raise ValueError('Optimizer %s is not supported.' % FLAGS.server_optimizer)

  federated_averaging_training_loop(
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
  try:
    tf.io.gfile.makedirs(os.path.join(FLAGS.root_output_dir, FLAGS.exp_name))
  except tf.errors.OpError:
    pass
  run_experiment()


if __name__ == '__main__':
  app.run(main)
