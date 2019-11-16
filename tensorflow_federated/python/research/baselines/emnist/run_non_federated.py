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
"""Baseline experiment for non-federated EMNIST."""

import collections
import os

from absl import app
from absl import flags
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.baselines.emnist import models
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  # Metadata
  flags.DEFINE_string(
      'exp_name', 'nonfed', 'Unique name for the experiment, suitable for use '
      'in filenames.')

  # Model configuration
  flags.DEFINE_enum(
      'training_model', 'orig_cnn', ['cnn', 'orig_cnn', '2nn', 'resnet'],
      'The identifier of the model from models.py to use for '
      'training.')

  # Training hyperparameters
  flags.DEFINE_boolean(
      'only_digits', True,
      'Whether to train on the digits only (10 classes) data '
      'or the full data (62 classes).')
  flags.DEFINE_integer('epochs', 10, 'Number of total epochs.')
  flags.DEFINE_integer('batch_size', 100, 'Batch size for training.')

  # Server optimizer configuration (it defines one or more flags per optimizer).
  flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')
  flags.DEFINE_float('momentum', 0.0, 'Training optimizer momentum.')

# End of hyperparameter flags.

# Root output directories.
flags.DEFINE_string('root_output_dir', '/tmp/non_federated_emnist/',
                    'Root directory for writing experiment output.')

FLAGS = flags.FLAGS


def create_compiled_keras_model():
  """Create compiled keras model."""
  if FLAGS.training_model == 'cnn':
    model = models.create_conv_dropout_model(only_digits=FLAGS.only_digits)
  elif FLAGS.training_model == 'orig_cnn':
    model = models.create_original_fedavg_cnn_model(
        only_digits=FLAGS.only_digits)
  elif FLAGS.training_model == '2nn':
    model = models.create_two_hidden_layer_model(only_digits=FLAGS.only_digits)
  elif FLAGS.training_model == 'resnet':
    model = models.create_resnet(num_blocks=9, only_digits=FLAGS.only_digits)
  else:
    raise ValueError('Model {} is not supported.'.format(FLAGS.training_model))

  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(
          learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model


def create_nonfed_emnist(total_examples):
  """Creates non-federated EMNIST training and testing sets."""
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=FLAGS.only_digits)

  example_tuple = collections.namedtuple('Example', ['x', 'y'])

  def element_fn(element):
    return example_tuple(
        x=tf.reshape(element['pixels'], [-1]),
        y=tf.reshape(element['label'], [1]))

  all_train = emnist_train.create_tf_dataset_from_all_clients().map(element_fn)
  all_train = all_train.shuffle(total_examples).repeat().batch(FLAGS.batch_size)

  all_test = emnist_test.create_tf_dataset_from_all_clients().map(element_fn)
  all_test = all_test.batch(FLAGS.batch_size)

  return all_train, all_test


def run_experiment():
  """Runs the training experiment."""
  total_examples = 341873 if FLAGS.only_digits else 671585
  emnist_train, emnist_test = create_nonfed_emnist(total_examples)
  steps_per_epoch = int(total_examples / FLAGS.batch_size)

  model = create_compiled_keras_model()

  # Define TensorBoard callback
  log_dir = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name, 'log_dir')
  try:
    tf.io.gfile.makedirs(log_dir)
  except tf.errors.OpError:
    pass  # Directory already exists, we'll simply reuse.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

  # Define CSV callback
  results_path = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name,
                              'results.csv')
  csv_logger = tf.keras.callbacks.CSVLogger(results_path)

  # Write the hyperparameters to a CSV:
  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  hparam_dict['results_file'] = results_path
  hparams_file = os.path.join(FLAGS.root_output_dir, FLAGS.exp_name,
                              'hparams.csv')
  utils_impl.atomic_write_to_csv(pd.Series(hparam_dict), hparams_file)

  model.fit(
      emnist_train,
      steps_per_epoch=steps_per_epoch,
      epochs=FLAGS.epochs,
      verbose=1,
      validation_data=emnist_test,
      callbacks=[tensorboard_callback, csv_logger])
  score = model.evaluate(emnist_test, verbose=0)
  print('Final test loss: %.4f' % score[0])
  print('Final test accuracy: %.4f' % score[1])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.compat.v1.enable_v2_behavior()

  try:
    tf.io.gfile.makedirs(os.path.join(FLAGS.root_output_dir, FLAGS.exp_name))
  except tf.errors.OpError:
    pass  # Directory already exists, we'll simply reuse.
  run_experiment()


if __name__ == '__main__':
  app.run(main)
