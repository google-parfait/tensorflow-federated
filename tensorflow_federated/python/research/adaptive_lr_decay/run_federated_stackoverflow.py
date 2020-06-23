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
"""Trains and evaluates on Stackoverflow NWP with adaptive LR decay."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import decay_iterative_process_builder
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import stackoverflow_dataset
from tensorflow_federated.python.research.utils.models import stackoverflow_models

with utils_impl.record_new_flags() as hparam_flags:
  # Training hyperparameters
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 16,
                       'Batch size used on the client.')
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('max_elements_per_user', 160, 'Max number of training '
                       'sentences to use per user.')
  flags.DEFINE_integer(
      'max_batches_per_client', -1, 'Maximum number of batches to process at '
      'each client in a given round. If set to -1, we take the full dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_enum(
      'client_weight', 'uniform', ['num_tokens', 'uniform'],
      'Weighting scheme for the client model deltas. Currently, this can '
      'either weight according to the number of tokens on a client '
      '(num_tokens) or uniformly (uniform).')

  # Modeling flags
  flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

  flags.DEFINE_integer(
      'client_datasets_random_seed', 1, 'The random seed '
      'governing the selection of clients that participate in each training '
      'round. The seed is used to generate the starting point for a Lehmer '
      'pseudo-random number generator, the outputs of which are used as seeds '
      'for the client sampling.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  model_builder = functools.partial(
      stackoverflow_models.create_recurrent_model,
      vocab_size=FLAGS.vocab_size,
      embedding_size=FLAGS.embedding_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      shared_embedding=FLAGS.shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  pad_token, oov_token, _, eos_token = stackoverflow_dataset.get_special_tokens(
      FLAGS.vocab_size)

  def metrics_builder():
    return [
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_with_oov', masked_tokens=[pad_token]),
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov', masked_tokens=[pad_token, oov_token]),
        # Notice BOS never appears in ground truth.
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov_or_eos',
            masked_tokens=[pad_token, oov_token, eos_token]),
        keras_metrics.NumBatchesCounter(),
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token])
    ]

  train_set, validation_set, test_set = stackoverflow_dataset.construct_word_level_datasets(
      FLAGS.vocab_size,
      FLAGS.client_batch_size,
      FLAGS.client_epochs_per_round,
      FLAGS.sequence_length,
      FLAGS.max_elements_per_user,
      FLAGS.num_validation_examples,
      max_batches_per_user=FLAGS.max_batches_per_client)

  input_spec = validation_set.element_spec

  if FLAGS.client_weight == 'uniform':

    def client_weight_fn(local_outputs):
      del local_outputs
      return 1.0

  elif FLAGS.client_weight == 'num_tokens':

    def client_weight_fn(local_outputs):
      # Num_tokens is a tensor with type int64[1], to use as a weight need
      # a float32 scalar.
      return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  else:
    raise ValueError('Unsupported client_weight flag [{!s}]. Currently only '
                     '`uniform` and `num_tokens` are supported.'.format(
                         FLAGS.client_weight))

  training_process = decay_iterative_process_builder.from_flags(
      input_spec=input_spec,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_set,
      FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)

  assign_weights_fn = adaptive_fed_avg.ServerState.assign_weights_to_keras_model

  evaluate_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=validation_set,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  test_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=validation_set.concatenate(test_set),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      training_process, client_datasets_fn, evaluate_fn, test_fn=test_fn)


if __name__ == '__main__':
  app.run(main)
