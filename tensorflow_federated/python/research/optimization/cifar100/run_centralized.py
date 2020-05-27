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
"""Baseline experiment on centralized CIFAR-100 data."""

import collections
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf

from tensorflow_federated.python.research.optimization.cifar100 import dataset
from tensorflow_federated.python.research.optimization.shared import keras_callbacks
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.optimization.shared import resnet_models
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output '
      'directory.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 20,
                       'Size of batches for training and eval.')
  flags.DEFINE_integer('decay_epochs', 25, 'Number of epochs before decaying '
                       'the learning rate.')
  flags.DEFINE_float('lr_decay', 0.1, 'How much to decay the learning rate by'
                     ' at each stage.')

flags.DEFINE_string(
    'root_output_dir', '/tmp/tff/optimization/cifar100/centralized',
    'The top-level output directory experiment runs. --experiment_name will '
    'be append, and the directory will contain tensorboard logs, metrics CSVs '
    'and other output.')

FLAGS = flags.FLAGS

TEST_BATCH_SIZE = 100
CIFAR_SHAPE = (32, 32, 3)
CROP_SHAPE = (24, 24, 3)
NUM_CLASSES = 100


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

  cifar_train, cifar_test = dataset.get_centralized_cifar100(
      train_batch_size=FLAGS.batch_size, crop_shape=CROP_SHAPE)

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()
  model = resnet_models.create_resnet18(
      input_shape=CROP_SHAPE, num_classes=NUM_CLASSES)
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  logging.info('Training model:')
  logging.info(model.summary())

  csv_logger_callback = keras_callbacks.AtomicCSVLogger(results_dir)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

  # Reduce the learning rate after a fixed number of epochs.
  def decay_lr(epoch, learning_rate):
    if (epoch + 1) % FLAGS.decay_epochs == 0:
      return learning_rate * FLAGS.lr_decay
    else:
      return learning_rate

  lr_callback = tf.keras.callbacks.LearningRateScheduler(decay_lr, verbose=1)

  history = model.fit(
      cifar_train,
      validation_data=cifar_test,
      epochs=FLAGS.num_epochs,
      callbacks=[lr_callback, tensorboard_callback, csv_logger_callback])

  logging.info('Final metrics:')
  for name in ['loss', 'sparse_categorical_accuracy']:
    metric = history.history['val_{}'.format(name)][-1]
    logging.info('\t%s: %.4f', name, metric)


if __name__ == '__main__':
  app.run(main)
