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
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import iterative_process_builder
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.optimization.stackoverflow import dataset
from tensorflow_federated.python.research.optimization.stackoverflow import models
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl

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
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

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
  flags.DEFINE_boolean('use_per_vector', False, 'Use per-vector clipping.')
  flags.DEFINE_boolean('geometric_clip_update', True,
                       'Use geometric updating of adaptive clip norm.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  tf.compat.v1.enable_v2_behavior()
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=10))

  model_builder = functools.partial(
      models.create_recurrent_model,
      vocab_size=FLAGS.vocab_size,
      embedding_size=FLAGS.embedding_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      shared_embedding=FLAGS.shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  pad_token, oov_token, _, eos_token = dataset.get_special_tokens(
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

  train_set, validation_set, test_set = dataset.construct_word_level_datasets(
      FLAGS.vocab_size, FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
      FLAGS.sequence_length, FLAGS.max_elements_per_user,
      FLAGS.num_validation_examples)

  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       next(iter(validation_set)))

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  training_process = iterative_process_builder.from_flags(
      dummy_batch=sample_batch,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_set, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=validation_set,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  test_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=validation_set.concatenate(test_set),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      training_process, client_datasets_fn, evaluate_fn, test_fn=test_fn)


if __name__ == '__main__':
  app.run(main)
