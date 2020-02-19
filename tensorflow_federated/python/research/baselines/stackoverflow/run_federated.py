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
"""Trains and evaluates Stackoverflow NWP model using TFF."""

import functools

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.baselines.stackoverflow import dataset
from tensorflow_federated.python.research.baselines.stackoverflow import metrics
from tensorflow_federated.python.research.baselines.stackoverflow import models
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  # Training hyperparameters
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 8, 'Batch size used on the client.')
  flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                       'sentences to use per user.')
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('num_validation_examples', 10000,
                       'Number of examples to take for validation set.')
  flags.DEFINE_integer('num_test_examples', 10000,
                       'Number of examples to take for test set.')
  flags.DEFINE_integer('shuffle_buffer_size', 1000,
                       'Buffer size for data shuffling.')
  flags.DEFINE_boolean('uniform_weighting', False,
                       'Whether to weigh clients uniformly. If false, clients '
                       'are weighted by the number of tokens.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

  # Modeling flags
  flags.DEFINE_integer(
      'vocab_size', 10000,
      'Size of the vocab to use; results in most `vocab_size` number of most '
      'common words used as vocabulary.')
  flags.DEFINE_integer('embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 670,
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


def run_experiment():
  """Runs the training experiment."""
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=10))

  def _layer_fn():
    layer_type = tf.keras.layers.LSTM if FLAGS.lstm else tf.keras.layers.GRU
    return layer_type(FLAGS.latent_size, return_sequences=True)

  model_builder = functools.partial(
      models.create_recurrent_model,
      vocab_size=FLAGS.vocab_size,
      embedding_size=FLAGS.embedding_size,
      num_layers=FLAGS.num_layers,
      recurrent_layer_fn=_layer_fn,
      name='stackoverflow-recurrent',
      shared_embedding=FLAGS.shared_embedding)

  pad, oov, _, eos = dataset.get_special_tokens(FLAGS.vocab_size)

  train_set, validation_set, _ = (
      dataset.construct_word_level_datasets(
          FLAGS.vocab_size,
          FLAGS.client_batch_size,
          FLAGS.client_epochs_per_round,
          FLAGS.sequence_length,
          FLAGS.max_elements_per_user,
          False,
          FLAGS.shuffle_buffer_size,
          FLAGS.num_validation_examples,
          FLAGS.num_test_examples))

  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       next(iter(validation_set)))

  def model_fn():
    """Defines the model."""
    keras_model = model_builder()
    train_metrics = [
        metrics.NumTokensCounter(name='num_tokens', masked_tokens=[pad]),
        metrics.NumTokensCounter(
            name='num_tokens_no_oov', masked_tokens=[pad, oov]),
        metrics.NumBatchesCounter(),
        metrics.NumExamplesCounter(),
        metrics.MaskedCategoricalAccuracy(name='accuracy', masked_tokens=[pad]),
        metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov', masked_tokens=[pad, oov]),
        metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov_no_eos', masked_tokens=[pad, oov, eos]),
    ]
    return tff.learning.from_keras_model(
        keras_model,
        sample_batch,
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=train_metrics)

  def server_optimizer_fn():
    return utils_impl.create_optimizer_from_flags('server')

  def client_weight_fn(local_outputs):
    num_tokens = tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    return 1.0 if FLAGS.uniform_weighting else num_tokens

  client_optimizer_fn = lambda: utils_impl.create_optimizer_from_flags('client')
  iterative_process = (
      tff.learning.federated_averaging.build_federated_averaging_process(
          model_fn=model_fn,
          server_optimizer_fn=server_optimizer_fn,
          client_weight_fn=client_weight_fn,
          client_optimizer_fn=client_optimizer_fn))

  server_state = iterative_process.initialize()
  for round_num in range(1, FLAGS.total_rounds+1):
    sampled_clients = np.random.choice(
        train_set.client_ids,
        size=FLAGS.clients_per_round,
        replace=False)
    client_data = [
        train_set.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]
    server_state, server_metrics = iterative_process.next(
        server_state, client_data)
    print('Round: {}'.format(round_num))
    print('   Loss: {:.8f}'.format(server_metrics.loss))
    print('   num_batches: {}'.format(server_metrics.num_batches))
    print('   num_examples: {}'.format(server_metrics.num_examples))
    print('   num_tokens: {}'.format(server_metrics.num_tokens))
    print('   num_tokens_no_oov: {}'.format(server_metrics.num_tokens_no_oov))
    print('   accuracy: {:.5f}'.format(server_metrics.accuracy))
    print('   accuracy_no_oov: {:.5f}'.format(server_metrics.accuracy_no_oov))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()
  run_experiment()


if __name__ == '__main__':
  app.run(main)
