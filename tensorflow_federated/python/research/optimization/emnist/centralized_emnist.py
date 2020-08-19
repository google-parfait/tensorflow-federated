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
"""Baseline experiment on centralized EMNIST data."""

import collections

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import centralized_training_loop
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import emnist_dataset
from tensorflow_federated.python.research.utils.models import emnist_models

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
  flags.DEFINE_enum('model', 'cnn', ['cnn', '2nn'],
                    'Which model to use for classification.')

flags.DEFINE_string(
    'root_output_dir', '/tmp/centralized/emnist',
    'The top-level output directory experiment runs. --experiment_name will '
    'be append, and the directory will contain tensorboard logs, metrics CSVs '
    'and other output.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_dataset, eval_dataset = emnist_dataset.get_centralized_emnist_datasets(
      batch_size=FLAGS.batch_size, only_digits=False)

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()

  if FLAGS.model == 'cnn':
    model = emnist_models.create_conv_dropout_model(only_digits=False)
  elif FLAGS.model == '2nn':
    model = emnist_models.create_two_hidden_layer_model(only_digits=False)
  else:
    raise ValueError('Cannot handle model flag [{!s}].'.format(FLAGS.model))

  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  hparams_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  centralized_training_loop.run(
      keras_model=model,
      train_dataset=train_dataset,
      validation_dataset=eval_dataset,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      num_epochs=FLAGS.num_epochs,
      hparams_dict=hparams_dict,
      decay_epochs=FLAGS.decay_epochs,
      lr_decay=FLAGS.lr_decay)


if __name__ == '__main__':
  app.run(main)
