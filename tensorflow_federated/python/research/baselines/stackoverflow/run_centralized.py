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
"""Baseline experiment on centralized data."""

import collections
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.baselines.stackoverflow import dataset
from tensorflow_federated.python.research.baselines.stackoverflow import models
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_string(
      'exp_name', 'centralized_stackoverflow',
      'Unique name for the experiment, suitable for '
      'use in filenames.')

  # Training hyperparameters
  flags.DEFINE_integer('batch_size', 8, 'Batch size used.')
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('epochs', 3, 'Number of epochs to train for.')
  flags.DEFINE_integer('shuffle_buffer_size', 1000,
                       'Buffer size for data shuffling.')
  flags.DEFINE_integer('num_validation_examples', 10000,
                       'Number of examples to take for validation set.')
  flags.DEFINE_integer('num_test_examples', 10000,
                       'Number of examples to take for test set.')
  flags.DEFINE_integer('tensorboard_update_frequency', 100 * 1000,
                       'Number of steps between tensorboard logging calls.')
  flags.DEFINE_string('root_output_dir', '/tmp/centralized_stackoverflow/',
                      'Root directory for writing experiment output.')
  utils_impl.define_optimizer_flags('centralized')

  # Modeling flags
  flags.DEFINE_integer(
      'vocab_size', 10000,
      'Size of the vocab to use; results in most `vocab_size` number of most '
      'common words used as vocabulary.')
  flags.DEFINE_integer('embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 512,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'lstm', True,
      'Boolean indicating LSTM recurrent cell. If False, GRU is used.')
  flags.DEFINE_boolean(
      'shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

FLAGS = flags.FLAGS


class AtomicCSVLogger(tf.keras.callbacks.Callback):

  def __init__(self, path):
    self._path = path

  def on_epoch_end(self, epoch, logs=None):
    epoch_path = os.path.join(self._path, 'results.{:02d}.csv'.format(epoch))
    utils_impl.atomic_write_to_csv(pd.Series(logs), epoch_path)


def run_experiment():
  """Runs the training experiment."""
  try:
    tf.io.gfile.makedirs(os.path.join(FLAGS.root_output_dir, FLAGS.exp_name))
  except tf.errors.OpError:
    pass

  train_set, validation_set, test_set = (
      dataset.construct_word_level_datasets(
          vocab_size=FLAGS.vocab_size,
          batch_size=FLAGS.batch_size,
          client_epochs_per_round=1,
          max_seq_len=FLAGS.sequence_length,
          max_training_elements_per_user=-1,
          shuffle_buffer_size=None,
          num_validation_examples=FLAGS.num_validation_examples,
          num_test_examples=FLAGS.num_test_examples))
  train_set = (
      train_set.create_tf_dataset_from_all_clients().shuffle(
          FLAGS.shuffle_buffer_size))

  recurrent_model = tf.keras.layers.LSTM if FLAGS.lstm else tf.keras.layers.GRU

  def _layer_fn():
    return recurrent_model(FLAGS.latent_size, return_sequences=True)

  model = models.create_recurrent_model(
      FLAGS.vocab_size,
      FLAGS.embedding_size,
      FLAGS.num_layers,
      _layer_fn,
      'stackoverflow-recurrent',
      shared_embedding=FLAGS.shared_embedding)
  logging.info('Training model: %s', model.summary())
  optimizer = utils_impl.create_optimizer_from_flags('centralized')
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=optimizer,
      weighted_metrics=['acc'])

  train_results_path = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name,
                                    'train_results')
  test_results_path = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name,
                                   'test_results')

  train_csv_logger = AtomicCSVLogger(train_results_path)
  test_csv_logger = AtomicCSVLogger(test_results_path)

  log_dir = os.path.join(FLAGS.root_output_dir, 'logdir', FLAGS.exp_name)
  try:
    tf.io.gfile.makedirs(log_dir)
    tf.io.gfile.makedirs(train_results_path)
    tf.io.gfile.makedirs(test_results_path)
  except tf.errors.OpError:
    pass  # log_dir already exists.

  train_tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      write_graph=True,
      update_freq=FLAGS.tensorboard_update_frequency)

  test_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

  results_file = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name,
                              'results.csv.bz2')

  # Write the hyperparameters to a CSV:
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  hparam_dict['results_file'] = results_file
  hparams_file = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name,
                              'hparams.csv')
  utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

  class_weight = dataset.get_class_weight(FLAGS.vocab_size)

  model.fit(
      train_set,
      epochs=FLAGS.epochs,
      verbose=1,
      class_weight=class_weight,
      validation_data=validation_set,
      callbacks=[train_csv_logger, train_tensorboard_callback])
  score = model.evaluate(
      test_set,
      verbose=1,
      callbacks=[test_csv_logger, test_tensorboard_callback])
  logging.info('Final test loss: %.4f', score[0])
  logging.info('Final test accuracy: %.4f', score[1])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()
  run_experiment()


if __name__ == '__main__':
  app.run(main)
