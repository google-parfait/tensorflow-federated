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
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.baselines.stackoverflow import models
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_string(
      'exp_name', 'centralized_keras_stackoverflow',
      'Unique name for the experiment, suitable for '
      'use in filenames.')
  flags.DEFINE_integer('batch_size', 256, 'Batch size used.')
  flags.DEFINE_integer(
      'vocab_size', 30000,
      'Size of the vocab to use; results in most `vocab_size` number of most '
      'common words used as vocabulary.')
  flags.DEFINE_integer('embedding_size', 256,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 512,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_float('learning_rate', 0.01,
                     'Learning rate to use for centralized SGD optimizer.')
  flags.DEFINE_float(
      'momentum', 0.0, 'Momentum value to use fo SGD optimizer. A value of 0.0 '
      'corresponds to no momentum.')
  # TODO(b/141867576): TFF currently needs a concrete maximum sequence length.
  # Follow up when this restriction is lifted.
  flags.DEFINE_integer('sequence_length', 100, 'Max sequence length to use')
  # There are over 100 million sentences in this dataset; this flag caps the
  # epoch size for speed. For comparison: EMNIST contains roughly 300,000
  # examples, so we set that as default here.
  flags.DEFINE_integer('num_training_examples', 300 * 1000,
                       'Number of training examples to process per epoch.')
  flags.DEFINE_integer('num_val_examples', 1000,
                       'Number of examples to take for validation set.')
  flags.DEFINE_integer('tensorboard_update_frequency', 1000,
                       'Number of steps between tensorboard logging calls.')
  flags.DEFINE_string('root_output_dir', '/tmp/non_federated_stackoverflow/',
                      'Root directory for writing experiment output.')

FLAGS = flags.FLAGS


def _create_vocab():
  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
  sorted_pairs = sorted(
      vocab_dict.items(), key=lambda x: -x[1])[:FLAGS.vocab_size]
  return list(x[0] for x in sorted_pairs)


def construct_word_level_datasets(vocab):
  """Preprocesses train and test datasets for stackoverflow."""
  (stackoverflow_train, _,
   stackoverflow_test) = tff.simulation.datasets.stackoverflow.load_data()
  # Mix all clients for training and testing in the centralized setting.
  raw_test_dataset = stackoverflow_test.create_tf_dataset_from_all_clients()
  raw_train_dataset = stackoverflow_train.create_tf_dataset_from_all_clients()

  BatchType = collections.namedtuple('BatchType', ['x', 'y'])  # pylint: disable=invalid-name

  table_values = tf.constant(list(range(FLAGS.vocab_size)), dtype=tf.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=1)

  def to_ids(x):
    """Splits a string into word IDs."""
    s = tf.reshape(x['tokens'], shape=[1])
    words = tf.string_split(s, sep=' ').values
    truncated_words = words[:FLAGS.sequence_length]
    ids = table.lookup(truncated_words)
    return ids

  def split_input_target(chunk):
    """Generate input and target data.

    The task of language model is to predict the next word.

    Args:
      chunk: A Tensor of text data.

    Returns:
      A namedtuple of input and target data.
    """
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    target_text = tf.map_fn(lambda x: x[1:], chunk)
    return BatchType(input_text, target_text)

  def preprocess(dataset):
    """Notice that this preprocess function repeats forever."""
    return (dataset.map(to_ids).padded_batch(
        FLAGS.batch_size, padded_shapes=[FLAGS.sequence_length
                                        ]).map(split_input_target).repeat(None))

  stackoverflow_train = preprocess(raw_train_dataset)
  stackoverflow_val = preprocess(raw_test_dataset).take(1000)
  stackoverflow_test = preprocess(raw_test_dataset)
  return stackoverflow_train, stackoverflow_val, stackoverflow_test


class AtomicCSVLogger(tf.keras.callbacks.Callback):

  def __init__(self, path):
    self._path = path

  def on_epoch_end(self, epoch, logs=None):
    epoch_path = os.path.join(self._path, 'epoch{}'.format(epoch))
    utils_impl.atomic_write_to_csv(pd.Series(logs), epoch_path)


def run_experiment():
  """Runs the training experiment."""
  vocab = _create_vocab()
  (stackoverflow_train, stackoverflow_val,
   stackoverflow_test) = construct_word_level_datasets(vocab)

  num_training_steps = FLAGS.num_training_examples / FLAGS.batch_size

  def _lstm_fn():
    return tf.keras.layers.LSTM(FLAGS.latent_size, return_sequences=True)

  model = models.create_recurrent_model(FLAGS.vocab_size, FLAGS.embedding_size,
                                        FLAGS.num_layers, _lstm_fn,
                                        'stackoverflow-lstm')
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(
          learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

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

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      write_graph=True,
      update_freq=FLAGS.tensorboard_update_frequency)

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

  model.fit(
      stackoverflow_train,
      steps_per_epoch=num_training_steps,
      epochs=25,
      verbose=1,
      validation_data=stackoverflow_val,
      callbacks=[train_csv_logger, tensorboard_callback])
  score = model.evaluate_generator(
      stackoverflow_test,
      verbose=1,
      callbacks=[test_csv_logger, tensorboard_callback])
  print('Final test loss: %.4f' % score[0])
  print('Final test accuracy: %.4f' % score[1])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()
  try:
    tf.io.gfile.makedirs(os.path.join(FLAGS.root_output_dir, FLAGS.exp_name))
  except tf.errors.OpError:
    pass
  run_experiment()


if __name__ == '__main__':
  app.run(main)
