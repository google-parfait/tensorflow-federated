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
"""Internal dispatcher for training loops."""

import collections
import os.path
import pprint
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.utils import adapters
from tensorflow_federated.python.research.utils import checkpoint_manager
from tensorflow_federated.python.research.utils import metrics_manager
from tensorflow_federated.python.research.utils import utils_impl

# Defining training loop flags
with utils_impl.record_hparam_flags():
  # Training rounds
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')

  # Root output directory.
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')

  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')

  # Checkpoint and evaluation flags.
  flags.DEFINE_integer('rounds_per_eval', 1,
                       'How often to evaluate the global model.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

FLAGS = flags.FLAGS


def create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _setup_outputs(root_output_dir, experiment_name, hparam_dict):
  """Set up directories for experiment loops, write hyperparameters to disk."""

  create_if_not_exists(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  create_if_not_exists(checkpoint_dir)
  checkpoint_mngr = checkpoint_manager.FileCheckpointManager(checkpoint_dir)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  create_if_not_exists(results_dir)
  metrics_mngr = metrics_manager.ScalarMetricsManager(results_dir)

  summary_logdir = os.path.join(root_output_dir, 'logdir', experiment_name)
  create_if_not_exists(summary_logdir)
  summary_writer = tf.compat.v2.summary.create_file_writer(summary_logdir)

  hparam_dict['metrics_file'] = metrics_mngr.metrics_filename
  hparams_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    metrics csv to: %s', metrics_mngr.metrics_filename)
  logging.info('    summaries to: %s', summary_logdir)

  return checkpoint_mngr, metrics_mngr, summary_writer


def _write_metrics(metrics_mngr, summary_writer, metrics, round_num):
  """Atomic metrics writer which inlines logic from MetricsHook class."""
  if not isinstance(metrics, dict):
    raise TypeError('metrics should be type `dict`.')
  if not isinstance(round_num, int):
    raise TypeError('round_num should be type `int`.')

  flat_metrics = metrics_mngr.update_metrics(round_num, metrics)
  logging.info('Evaluation at round {:d}:\n{!s}'.format(
      round_num, pprint.pformat(flat_metrics)))

  # Also write metrics to a tf.summary logdir
  with summary_writer.as_default():
    for name, val in flat_metrics.items():
      tf.compat.v2.summary.scalar(name, val, step=round_num)


def _compute_numpy_l2_difference(model, previous_model):
  squared_norms = tf.nest.map_structure(lambda x, y: tf.linalg.norm(x - y)**2,
                                        model, previous_model)
  l2_total_tensor = tf.reduce_sum(tf.nest.flatten(squared_norms))**0.5
  return l2_total_tensor.numpy()


def run(iterative_process: adapters.IterativeProcessPythonAdapter,
        client_datasets_fn: Callable[[int], Tuple[List[tf.data.Dataset],
                                                  List[str]]],
        evaluate_fn: Callable[[Any, Optional[bool]], Dict[str, float]]):
  """Runs federated training for the given TFF `IterativeProcess` instance.

  Args:
    iterative_process: An `IterativeProcessPythonAdapter` instance to run.
    client_datasets_fn: Function accepting an integer argument (the round
      number) and returning a list of client datasets to use as federated data
      for that round, and a list of the corresponding client ids.
    evaluate_fn: Callable accepting a server state (the `state` of the
      `IterationResult`) and an optional `bool` and returning a dict of
      evaluation metrics.

  Returns:
    The `state` of the `IterationResult` representing the result of the training
      loop.
  """
  if not isinstance(iterative_process, adapters.IterativeProcessPythonAdapter):
    raise TypeError('iterative_process should be type '
                    '`adapters.IterativeProcessPythonAdapter`.')
  if not callable(client_datasets_fn):
    raise TypeError('client_datasets_fn should be callable.')
  if not callable(evaluate_fn):
    raise TypeError('evaluate_fn should be callable.')
  total_rounds = FLAGS.total_rounds

  logging.info('Starting iterative_process_training_loop')
  initial_state = iterative_process.initialize()

  hparam_flags = utils_impl.get_hparam_flags()
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  checkpoint_mngr, metrics_mngr, summary_writer = _setup_outputs(
      FLAGS.root_output_dir, FLAGS.experiment_name, hparam_dict)

  logging.info('Asking checkpoint manager to load checkpoint.')
  state, round_num = checkpoint_mngr.load_latest_checkpoint(initial_state)

  if state is None:
    logging.info('Initializing experiment from scratch.')
    state = initial_state
    round_num = 0
    metrics_mngr.clear_all_rounds()
  else:
    logging.info('Restarted from checkpoint round %d', round_num)
    round_num += 1  # Increment to avoid overwriting current checkpoint
    metrics_mngr.clear_rounds_after(last_valid_round_num=round_num - 1)

  unique_clients = set()
  loop_start_time = time.time()
  while round_num < total_rounds:
    data_prep_start_time = time.time()
    federated_train_data, sampled_clients = client_datasets_fn(round_num)
    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }

    training_start_time = time.time()
    prev_model = state.model
    # TODO(b/145604851): This try/except is used to circumvent ambiguous TF
    # errors during training, and should be removed once the root cause is
    # determined (and possibly fixed).
    try:
      iteration_result = iterative_process.next(state, federated_train_data)
    except (tf.errors.FailedPreconditionError, tf.errors.NotFoundError,
            tf.errors.InternalError) as e:
      logging.warning('Caught %s exception while running round %d:\n\t%s',
                      type(e), round_num, e)
      logging.info('Rebuilding executor stack and retrying...')
      tff.framework.set_default_executor()
      continue  # restart the loop without incrementing the round number

    state = iteration_result.state
    round_metrics = iteration_result.metrics

    train_metrics['training_secs'] = time.time() - training_start_time
    train_metrics['model_delta_l2_norm'] = _compute_numpy_l2_difference(
        state.model, prev_model)
    unique_clients.update(sampled_clients)
    train_metrics['num_unique_clients'] = len(unique_clients)
    train_metrics.update(round_metrics)
    # TODO(b/148576550): Wire in client training time metrics into custom
    # training loops.
    train_metrics.pop('keras_training_time_client_sum_sec')

    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - loop_start_time) / (round_num + 1)))

    if (round_num % FLAGS.rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      checkpoint_mngr.save_checkpoint(state, round_num)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {
        'train': train_metrics,
        'round': round_num,
    }

    if round_num % FLAGS.rounds_per_eval == 0:
      evaluate_start_time = time.time()
      eval_metrics = evaluate_fn(state, use_test_dataset=False)  # pytype: disable=wrong-keyword-args
      eval_metrics['evaluate_secs'] = time.time() - evaluate_start_time

      metrics['eval'] = eval_metrics

    _write_metrics(metrics_mngr, summary_writer, metrics, round_num)
    round_num += 1

  test_start_time = time.time()
  test_metrics = evaluate_fn(state, use_test_dataset=True)  # pytype: disable=wrong-keyword-args
  test_metrics['evaluate_secs'] = time.time() - test_start_time

  metrics = {'test': test_metrics}
  _write_metrics(metrics_mngr, summary_writer, metrics, total_rounds)

  return state
