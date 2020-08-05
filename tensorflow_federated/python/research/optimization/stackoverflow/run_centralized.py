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

from tensorflow_federated.python.research.optimization.shared import keras_callbacks
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import stackoverflow_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_models

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_string(
      'experiment_name', 'centralized_stackoverflow',
      'Unique name for the experiment, suitable for '
      'use in filenames.')
  flags.DEFINE_integer('batch_size', 128, 'Batch size used.')

  # Modeling flags
  flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

  # TODO(b/141867576): TFF currently needs a concrete maximum sequence length.
  # Follow up when this restriction is lifted.
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('epochs', 3, 'Number of epochs to train for.')
  flags.DEFINE_integer('num_validation_examples', 10000,
                       'Number of examples to take for validation set.')
  flags.DEFINE_integer('tensorboard_update_frequency', 10 * 1000,
                       'Number of steps between tensorboard logging calls.')
  flags.DEFINE_string('root_output_dir', '/tmp/centralized_stackoverflow/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'shuffle_buffer_size', 10000, 'Shuffle buffer size to '
      'use for centralized training.')
  optimizer_utils.define_optimizer_flags('centralized')

FLAGS = flags.FLAGS


def run_experiment():
  """Runs the training experiment."""
  _, validation_dataset, test_dataset = stackoverflow_dataset.construct_word_level_datasets(
      vocab_size=FLAGS.vocab_size,
      client_batch_size=FLAGS.batch_size,
      client_epochs_per_round=1,
      max_seq_len=FLAGS.sequence_length,
      max_training_elements_per_user=-1,
      num_validation_examples=FLAGS.num_validation_examples,
      num_oov_buckets=FLAGS.num_oov_buckets)
  train_dataset = stackoverflow_dataset.get_centralized_train_dataset(
      vocab_size=FLAGS.vocab_size,
      num_oov_buckets=FLAGS.num_oov_buckets,
      batch_size=FLAGS.batch_size,
      max_seq_len=FLAGS.sequence_length,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size)

  model = stackoverflow_models.create_recurrent_model(
      vocab_size=FLAGS.vocab_size,
      num_oov_buckets=FLAGS.num_oov_buckets,
      name='stackoverflow-lstm',
      embedding_size=FLAGS.embedding_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      shared_embedding=FLAGS.shared_embedding)

  logging.info('Training model: %s', model.summary())
  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()
  special_tokens = stackoverflow_dataset.get_special_tokens(
      vocab_size=FLAGS.vocab_size, num_oov_buckets=FLAGS.num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=[
          keras_metrics.MaskedCategoricalAccuracy(
              name='accuracy_with_oov', masked_tokens=[pad_token]),
          keras_metrics.MaskedCategoricalAccuracy(
              name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
          keras_metrics.MaskedCategoricalAccuracy(
              name='accuracy_no_oov_or_eos',
              masked_tokens=[pad_token, eos_token] + oov_tokens),
      ])

  train_results_path = os.path.join(FLAGS.root_output_dir, 'train_results',
                                    FLAGS.experiment_name)
  test_results_path = os.path.join(FLAGS.root_output_dir, 'test_results',
                                   FLAGS.experiment_name)

  train_csv_logger = keras_callbacks.AtomicCSVLogger(train_results_path)
  test_csv_logger = keras_callbacks.AtomicCSVLogger(test_results_path)

  log_dir = os.path.join(FLAGS.root_output_dir, 'logdir', FLAGS.experiment_name)
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

  # Write the hyperparameters to a CSV:
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  hparams_file = os.path.join(FLAGS.root_output_dir, FLAGS.experiment_name,
                              'hparams.csv')
  utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

  model.fit(
      train_dataset,
      epochs=FLAGS.epochs,
      verbose=0,
      validation_data=validation_dataset,
      callbacks=[train_csv_logger, train_tensorboard_callback])
  score = model.evaluate(
      test_dataset,
      verbose=0,
      callbacks=[test_csv_logger, test_tensorboard_callback])
  logging.info('Final test loss: %.4f', score[0])
  logging.info('Final test accuracy: %.4f', score[1])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  try:
    tf.io.gfile.makedirs(
        os.path.join(FLAGS.root_output_dir, FLAGS.experiment_name))
  except tf.errors.OpError:
    pass
  run_experiment()


if __name__ == '__main__':
  app.run(main)
