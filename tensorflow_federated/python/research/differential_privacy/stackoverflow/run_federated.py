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
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.differential_privacy import dp_utils
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
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
  flags.DEFINE_integer('client_batch_size', 8, 'Batch size used on the client.')
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                       'sentences to use per user.')
  flags.DEFINE_integer('num_validation_examples', 10000, 'Number of examples '
                       'to use from test set for per-round validation.')
  flags.DEFINE_boolean('uniform_weighting', False,
                       'Whether to weigh clients uniformly. If false, clients '
                       'are weighted by the number of tokens.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

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

  # Differential privacy flags
  flags.DEFINE_float('clip', 0.05, 'Initial clip.')
  flags.DEFINE_float('noise_multiplier', None,
                     'Noise multiplier. If None, no DP is used.')
  flags.DEFINE_float('adaptive_clip_learning_rate', 0,
                     'Adaptive clip learning rate.')
  flags.DEFINE_float('target_unclipped_quantile', 0.5,
                     'Target unclipped quantile.')
  flags.DEFINE_float(
      'clipped_count_budget_allocation', 0.1,
      'Fraction of privacy budget to allocate for clipped counts.')
  flags.DEFINE_boolean(
      'per_vector_clipping', False, 'Use per-vector clipping'
      'to indepednelty clip each weight tensor instead of the'
      'entire model.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=10))

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
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
    ]

  datasets = stackoverflow_dataset.construct_word_level_datasets(
      FLAGS.vocab_size, FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
      FLAGS.sequence_length, FLAGS.max_elements_per_user,
      FLAGS.num_validation_examples)
  train_dataset, validation_dataset, test_dataset = datasets

  if FLAGS.uniform_weighting:
    def client_weight_fn(local_outputs):
      del local_outputs
      return 1.0
  else:
    def client_weight_fn(local_outputs):
      return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  def model_fn():
    return tff.learning.from_keras_model(
        model_builder(),
        loss_builder(),
        input_spec=validation_dataset.element_spec,
        metrics=metrics_builder())

  if FLAGS.noise_multiplier is not None:
    if not FLAGS.uniform_weighting:
      raise ValueError(
          'Differential privacy is only implemented for uniform weighting.')

    dp_query = tff.utils.build_dp_query(
        clip=FLAGS.clip,
        noise_multiplier=FLAGS.noise_multiplier,
        expected_total_weight=FLAGS.clients_per_round,
        adaptive_clip_learning_rate=FLAGS.adaptive_clip_learning_rate,
        target_unclipped_quantile=FLAGS.target_unclipped_quantile,
        clipped_count_budget_allocation=FLAGS.clipped_count_budget_allocation,
        expected_num_clients=FLAGS.clients_per_round,
        per_vector_clipping=FLAGS.per_vector_clipping,
        model=model_fn())

    dp_aggregate_fn, _ = tff.utils.build_dp_aggregate(dp_query)
  else:
    dp_aggregate_fn = None

  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  training_process = (
      tff.learning.federated_averaging.build_federated_averaging_process(
          model_fn=model_fn,
          server_optimizer_fn=server_optimizer_fn,
          client_weight_fn=client_weight_fn,
          client_optimizer_fn=client_optimizer_fn,
          stateful_delta_aggregate_fn=dp_aggregate_fn))

  adaptive_clipping = (FLAGS.adaptive_clip_learning_rate > 0)
  training_process = dp_utils.DPFedAvgProcessAdapter(training_process,
                                                     FLAGS.per_vector_clipping,
                                                     adaptive_clipping)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_dataset, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=validation_dataset,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=dp_utils.assign_weights_to_keras_model)

  test_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=validation_dataset.concatenate(test_dataset),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=dp_utils.assign_weights_to_keras_model)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      training_process, client_datasets_fn, evaluate_fn, test_fn=test_fn)


if __name__ == '__main__':
  app.run(main)
