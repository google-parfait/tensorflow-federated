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
"""Baseline experiment on centralized Stack Overflow NWP data."""

import collections

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import centralized_training_loop
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import stackoverflow_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_models

with utils_impl.record_new_flags() as hparam_flags:
  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/centralized/stackoverflow_nwp',
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

  # Stack Overflow NWP flags
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer(
      'so_nwp_num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('so_nwp_embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('so_nwp_latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('so_nwp_num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'so_nwp_shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  _, validation_dataset, test_dataset = stackoverflow_dataset.construct_word_level_datasets(
      vocab_size=FLAGS.so_nwp_vocab_size,
      client_batch_size=FLAGS.batch_size,
      client_epochs_per_round=1,
      max_seq_len=FLAGS.so_nwp_sequence_length,
      max_training_elements_per_user=-1,
      num_validation_examples=FLAGS.so_nwp_num_validation_examples,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets)
  train_dataset = stackoverflow_dataset.get_centralized_train_dataset(
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      batch_size=FLAGS.batch_size,
      max_seq_len=FLAGS.so_nwp_sequence_length,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size)

  model = stackoverflow_models.create_recurrent_model(
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets,
      name='stackoverflow-lstm',
      embedding_size=FLAGS.so_nwp_embedding_size,
      latent_size=FLAGS.so_nwp_latent_size,
      num_layers=FLAGS.so_nwp_num_layers,
      shared_embedding=FLAGS.so_nwp_shared_embedding)

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()
  special_tokens = stackoverflow_dataset.get_special_tokens(
      vocab_size=FLAGS.so_nwp_vocab_size,
      num_oov_buckets=FLAGS.so_nwp_num_oov_buckets)
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
