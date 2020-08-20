# Copyright 2020, The TensorFlow Federated Authors.
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
"""Baseline experiment on centralized Stack Overflow LR data."""

import collections

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import centralized_training_loop
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import stackoverflow_lr_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_lr_models

with utils_impl.record_new_flags() as hparam_flags:
  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/centralized/stackoverflow_lr',
      'The top-level output directory experiment runs. --experiment_name will '
      'be appended, and the directory will contain tensorboard logs, metrics '
      'written as CSVs, and a CSV of hyperparameter choices.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 20,
                       'Size of batches for training and eval.')
  flags.DEFINE_integer('decay_epochs', 25, 'Number of epochs before decaying '
                       'the learning rate.')
  flags.DEFINE_float('lr_decay', 0.1, 'How much to decay the learning rate by'
                     ' at each stage.')
  flags.DEFINE_integer(
      'shuffle_buffer_size', 10000, 'Shuffle buffer size to '
      'use for centralized training.')

  # Stack Overflow LR flags
  flags.DEFINE_integer('so_lr_vocab_tokens_size', 10000,
                       'Vocab tokens size used.')
  flags.DEFINE_integer('so_lr_vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer(
      'so_lr_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_dataset, validation_dataset, test_dataset = stackoverflow_lr_dataset.get_centralized_stackoverflow_datasets(
      batch_size=FLAGS.batch_size,
      vocab_tokens_size=FLAGS.so_lr_vocab_tokens_size,
      vocab_tags_size=FLAGS.so_lr_vocab_tags_size,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      num_validation_examples=FLAGS.so_lr_num_validation_examples)

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()

  model = stackoverflow_lr_models.create_logistic_model(
      vocab_tokens_size=FLAGS.so_lr_vocab_tokens_size,
      vocab_tags_size=FLAGS.so_lr_vocab_tags_size)

  model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(
          from_logits=False, reduction=tf.keras.losses.Reduction.SUM),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall(top_k=5)])

  hparams_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  centralized_training_loop.run(
      keras_model=model,
      train_dataset=train_dataset,
      validation_dataset=validation_dataset,
      test_dataset=test_dataset,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      num_epochs=FLAGS.num_epochs,
      hparams_dict=hparams_dict,
      decay_epochs=FLAGS.decay_epochs,
      lr_decay=FLAGS.lr_decay)


if __name__ == '__main__':
  app.run(main)
