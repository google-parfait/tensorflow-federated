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
"""A callback for evaluation of EMNIST models."""

import collections
import logging
import os.path
import sys

import attr
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import tree

from tensorboard.plugins.hparams import api as hp
from tensorflow_federated.python.research.utils import utils_impl


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
  def build(cls, exp_name, output_dir, eval_dataset, hparam_dict, model):
    """Constructs the MetricsHook.

    Args:
      exp_name: A unique filesystem-friendly name for the experiment.
      output_dir: A root output directory used for all experiment runs in a
        grid. The MetricsHook will combine this with exp_name to form suitable
        output directories for this run.
      eval_dataset: Evaluation dataset.
      hparam_dict: A dictionary of hyperparameters to be recorded to .csv and
        exported to TensorBoard.
      model: The model for evaluation.

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

    flat_metrics = tree.flatten_with_path(metrics)
    flat_metrics = [
        ('/'.join(map(str, path)), item) for path, item in flat_metrics
    ]
    flat_metrics = collections.OrderedDict(flat_metrics)

    # Use a DataFrame just to get nice formatting.
    df = pd.DataFrame.from_dict(flat_metrics, orient='index', columns=['value'])
    print(df)

    # Also write metrics to a tf.summary logdir
    with self.summary_writer.as_default():
      for name, value in flat_metrics.items():
        tf.compat.v2.summary.scalar(name, value, step=round_num)

    self.results = self.results.append(flat_metrics, ignore_index=True)
    utils_impl.atomic_write_to_csv(self.results, self.results_file)
