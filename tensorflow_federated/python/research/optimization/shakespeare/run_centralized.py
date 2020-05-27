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
"""Train a Char-RNN on the Shakespeare dataset in the centralized setting."""

import collections
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shakespeare import dataset
from tensorflow_federated.python.research.optimization.shakespeare import models
from tensorflow_federated.python.research.optimization.shared import keras_callbacks
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import utils_impl

FLAGS = flags.FLAGS

with utils_impl.record_new_flags() as hparam_flags:
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output '
      'directory.')
  flags.DEFINE_integer('num_epochs', 60, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 10,
                       'Size of batches for training and eval.')
  flags.DEFINE_boolean('shuffle_train_data', True,
                       'Whether to shuffle the training data.')

flags.DEFINE_string(
    'root_output_dir', '/tmp/tff/optimization/shakespeare/centralized',
    'The top-level output directory experiment runs. --experiment_name will '
    'be append, and the directory will contain tensorboard logs, metrics CSVs '
    'and other output.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()

  experiment_output_dir = FLAGS.root_output_dir
  tensorboard_dir = os.path.join(experiment_output_dir, 'logdir',
                                 FLAGS.experiment_name)
  results_dir = os.path.join(experiment_output_dir, 'results',
                             FLAGS.experiment_name)

  for path in [experiment_output_dir, tensorboard_dir, results_dir]:
    try:
      tf.io.gfile.makedirs(path)
    except tf.errors.OpError:
      pass  # Directory already exists.

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  hparam_dict['results_file'] = results_dir
  hparams_file = os.path.join(results_dir, 'hparams.csv')
  logging.info('Saving hyper parameters to: [%s]', hparams_file)
  utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

  train_client_data, test_client_data = (
      tff.simulation.datasets.shakespeare.load_data())

  def preprocess(ds):
    return dataset.convert_snippets_to_character_sequence_examples(
        ds, FLAGS.batch_size, epochs=1).cache()

  train_dataset = train_client_data.create_tf_dataset_from_all_clients()
  if FLAGS.shuffle_train_data:
    train_dataset = train_dataset.shuffle(buffer_size=10000)
  train_dataset = preprocess(train_dataset)

  eval_dataset = preprocess(
      test_client_data.create_tf_dataset_from_all_clients())

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()

  pad_token, _, _, _ = dataset.get_special_tokens()

  # Vocabulary with one OOV ID and zero for the mask.
  vocab_size = len(dataset.CHAR_VOCAB) + 2
  model = models.create_recurrent_model(
      vocab_size=vocab_size, batch_size=FLAGS.batch_size)
  model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token])
      ])

  logging.info('Training model:')
  logging.info(model.summary())

  csv_logger_callback = keras_callbacks.AtomicCSVLogger(results_dir)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
  # Reduce the learning rate every 20 epochs.
  def decay_lr(epoch, lr):
    if (epoch + 1) % 20 == 0:
      return lr * 0.1
    else:
      return lr

  lr_callback = tf.keras.callbacks.LearningRateScheduler(decay_lr, verbose=1)

  history = model.fit(
      train_dataset,
      validation_data=eval_dataset,
      epochs=FLAGS.num_epochs,
      callbacks=[lr_callback, tensorboard_callback, csv_logger_callback])

  logging.info('Final metrics:')
  for name in ['loss', 'accuracy']:
    metric = history.history['val_{}'.format(name)][-1]
    logging.info('\t%s: %.4f', name, metric)


if __name__ == '__main__':
  app.run(main)
