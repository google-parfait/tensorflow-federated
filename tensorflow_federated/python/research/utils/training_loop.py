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

import contextlib
import os.path
import pprint
import time
from typing import Any, Callable, Dict, List, Optional

from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.utils import checkpoint_manager
from tensorflow_federated.python.research.utils import metrics_manager
from tensorflow_federated.python.research.utils import utils_impl


def create_if_not_exists(path):
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _setup_outputs(root_output_dir,
                   experiment_name,
                   hparam_dict,
                   write_metrics_with_bz2=True,
                   rounds_per_profile=0):
  """Set up directories for experiment loops, write hyperparameters to disk."""

  if not experiment_name:
    raise ValueError('experiment_name must be specified.')

  create_if_not_exists(root_output_dir)

  checkpoint_dir = os.path.join(root_output_dir, 'checkpoints', experiment_name)
  create_if_not_exists(checkpoint_dir)
  checkpoint_mngr = checkpoint_manager.FileCheckpointManager(checkpoint_dir)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  create_if_not_exists(results_dir)
  metrics_mngr = metrics_manager.ScalarMetricsManager(
      results_dir, use_bz2=write_metrics_with_bz2)

  summary_logdir = os.path.join(root_output_dir, 'logdir', experiment_name)
  create_if_not_exists(summary_logdir)
  summary_writer = tf.summary.create_file_writer(summary_logdir)

  if hparam_dict:
    hparam_dict['metrics_file'] = metrics_mngr.metrics_filename
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', checkpoint_dir)
  logging.info('    metrics csv to: %s', metrics_mngr.metrics_filename)
  logging.info('    summaries to: %s', summary_logdir)

  @contextlib.contextmanager
  def profiler(round_num):
    if (rounds_per_profile > 0 and round_num % rounds_per_profile == 0):
      with tf.profiler.experimental.Profile(summary_logdir):
        yield
    else:
      yield

  return checkpoint_mngr, metrics_mngr, summary_writer, profiler


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
      tf.summary.scalar(name, val, step=round_num)


def _compute_numpy_l2_difference(model, previous_model):
  squared_norms = tf.nest.map_structure(lambda x, y: tf.linalg.norm(x - y)**2,
                                        model, previous_model)
  l2_total_tensor = tf.reduce_sum(tf.nest.flatten(squared_norms))**0.5
  return l2_total_tensor.numpy()


def run(iterative_process: tff.templates.IterativeProcess,
        client_datasets_fn: Callable[[int], List[tf.data.Dataset]],
        validation_fn: Callable[[Any], Dict[str, float]],
        total_rounds: int,
        experiment_name: str,
        train_eval_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
        test_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
        root_output_dir: Optional[str] = '/tmp/fed_opt',
        hparam_dict: Optional[Dict[str, Any]] = None,
        write_metrics_with_bz2: Optional[bool] = True,
        rounds_per_eval: Optional[int] = 1,
        rounds_per_checkpoint: Optional[int] = 50,
        rounds_per_train_eval: Optional[int] = 100,
        rounds_per_profile: Optional[int] = 0):
  """Runs federated training for a given `tff.templates.IterativeProcess`.

  Args:
    iterative_process: A `tff.templates.IterativeProcess` instance to run.
    client_datasets_fn: Function accepting an integer argument (the round
      number) and returning a list of client datasets to use as federated data
      for that round, and a list of the corresponding client ids.
    validation_fn: A callable accepting the `model` attribute of the iterative
      process state and returning a dict of evaluation metrics. Used to compute
      validation metrics throughout the training process.
    total_rounds: The number of federated training rounds to perform.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    train_eval_fn: An optional callable accepting the `model` attribute of the
      iterative process state and returning a dict of evaluation metrics. Used
      to compute training metrics over the entire training dataset throughout
      the course of the iterative process. If set to `None`, no such evaluation
      is done.
    test_fn: An optional callable accepting the `model` attribute of the
      iterative process state and returning a dict of test metrics. Used to
      compute test metrics at the end of the training process.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    hparam_dict: An optional dictionary specifying hyperparameters of the
      experiment. If provided, the hyperparameters will be written to CSV.
    write_metrics_with_bz2: Whether to use `bz2` compression when writing
      metrics to CSV.
    rounds_per_eval: How often to compute validation metrics.
    rounds_per_checkpoint: How often to checkpoint the iterative process state.
      If you expect the job to restart frequently, this should be small. If no
      interruptions are expected, this can be made larger.
    rounds_per_train_eval: How often to compute metrics over the entire training
      dataset. Note that this is only done if a `train_eval_fn` argument is
      supplied.
    rounds_per_profile: Experimental setting. If set to a value greater than 0,
      this dictates how often a TensorFlow profiler is run.

  Returns:
    The final `state` of the iterative process after training.
  """
  if not isinstance(iterative_process, tff.templates.IterativeProcess):
    raise TypeError('iterative_process should be type '
                    '`tff.templates.IterativeProcess`.')
  if not callable(client_datasets_fn):
    raise TypeError('client_datasets_fn should be callable.')
  if not callable(validation_fn):
    raise TypeError('validation_fn should be callable.')
  if train_eval_fn is not None and not callable(train_eval_fn):
    raise TypeError('train_eval_fn should be callable.')
  if test_fn is not None and not callable(test_fn):
    raise TypeError('test_fn should be callable.')

  logging.info('Starting iterative_process training loop...')
  initial_state = iterative_process.initialize()

  if not hasattr(initial_state, 'model'):
    raise TypeError('The server state must have a model attribute.')

  checkpoint_mngr, metrics_mngr, summary_writer, profiler = _setup_outputs(
      root_output_dir, experiment_name, hparam_dict, write_metrics_with_bz2,
      rounds_per_profile)

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

  loop_start_time = time.time()
  while round_num < total_rounds:
    data_prep_start_time = time.time()
    federated_train_data = client_datasets_fn(round_num)
    train_metrics = {
        'prepare_datasets_secs': time.time() - data_prep_start_time
    }

    training_start_time = time.time()
    prev_model = state.model
    # TODO(b/145604851): This try/except is used to circumvent ambiguous TF
    # errors during training, and should be removed once the root cause is
    # determined (and possibly fixed).
    try:
      with profiler(round_num):
        state, round_metrics = iterative_process.next(state,
                                                      federated_train_data)
    except (tf.errors.FailedPreconditionError, tf.errors.NotFoundError,
            tf.errors.InternalError) as e:
      logging.warning('Caught %s exception while running round %d:\n\t%s',
                      type(e), round_num, e)
      continue  # restart the loop without incrementing the round number

    train_metrics['training_secs'] = time.time() - training_start_time
    train_metrics['model_delta_l2_norm'] = _compute_numpy_l2_difference(
        state.model, prev_model)
    train_metrics.update(round_metrics)

    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - loop_start_time) / (round_num + 1)))

    if (round_num % rounds_per_checkpoint == 0 or
        round_num == total_rounds - 1):
      save_checkpoint_start_time = time.time()
      checkpoint_mngr.save_checkpoint(state, round_num)
      train_metrics['save_checkpoint_secs'] = (
          time.time() - save_checkpoint_start_time)

    metrics = {'train': train_metrics}

    if round_num % rounds_per_eval == 0:
      # Compute validation metrics
      evaluate_start_time = time.time()
      validation_metrics = validation_fn(state.model)
      validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
      metrics['eval'] = validation_metrics

    if train_eval_fn and round_num % rounds_per_train_eval == 0:
      # Compute metrics over the entire training dataset
      train_eval_start = time.time()
      train_eval_metrics = train_eval_fn(state.model)
      train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
      metrics['train_eval'] = train_eval_metrics

    _write_metrics(metrics_mngr, summary_writer, metrics, round_num)
    round_num += 1

  # Final metrics evaluation once the training has completed
  metrics = {}

  # Validation metrics
  evaluate_start_time = time.time()
  validation_metrics = validation_fn(state.model)
  validation_metrics['evaluate_secs'] = time.time() - evaluate_start_time
  metrics['eval'] = validation_metrics

  # Training set metrics
  if train_eval_fn:
    train_eval_start = time.time()
    train_eval_metrics = train_eval_fn(state.model)
    train_eval_metrics['evaluate_secs'] = time.time() - train_eval_start
    metrics['train_eval'] = train_eval_metrics

  # Test set metrics
  if test_fn:
    test_start_time = time.time()
    test_metrics = test_fn(state.model)
    test_metrics['evaluate_secs'] = time.time() - test_start_time
    metrics['test'] = test_metrics
  _write_metrics(metrics_mngr, summary_writer, metrics, total_rounds)

  return state
