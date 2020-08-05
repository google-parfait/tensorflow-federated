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

from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule
from tensorflow_federated.python.research.optimization.shared import iterative_process_builder
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
  flags.DEFINE_integer('client_batch_size', 8, 'Batch size used on the client.')
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                       'sentences to use per user.')
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')

  flags.DEFINE_integer(
      'client_datasets_random_seed', 1, 'The random seed '
      'governing the client dataset selection.')

  # Modeling flags
  flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  model_builder = functools.partial(
      stackoverflow_models.create_recurrent_model,
      vocab_size=FLAGS.vocab_size,
      num_oov_buckets=FLAGS.num_oov_buckets,
      embedding_size=FLAGS.embedding_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      shared_embedding=FLAGS.shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  special_tokens = stackoverflow_dataset.get_special_tokens(
      FLAGS.vocab_size, FLAGS.num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  def metrics_builder():
    return [
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_with_oov', masked_tokens=[pad_token]),
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
        # Notice BOS never appears in ground truth.
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov_or_eos',
            masked_tokens=[pad_token, eos_token] + oov_tokens),
        keras_metrics.NumBatchesCounter(),
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token])
    ]

  dataset_vocab = stackoverflow_dataset.create_vocab(FLAGS.vocab_size)

  train_clientdata, _, test_clientdata = (
      tff.simulation.datasets.stackoverflow.load_data())

  # Split the test data into test and validation sets.
  # TODO(b/161914546): consider moving evaluation to use
  # `tff.learning.build_federated_evaluation` to get metrics over client
  # distributions, as well as the example weight means from this centralized
  # evaluation.
  base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
  preprocess_val_and_test = stackoverflow_dataset.create_test_dataset_preprocess_fn(
      vocab=dataset_vocab,
      num_oov_buckets=FLAGS.num_oov_buckets,
      max_seq_len=FLAGS.sequence_length)
  test_set = preprocess_val_and_test(
      base_test_dataset.skip(FLAGS.num_validation_examples))
  validation_set = preprocess_val_and_test(
      base_test_dataset.take(FLAGS.num_validation_examples))

  train_dataset_preprocess_comp = stackoverflow_dataset.create_train_dataset_preprocess_fn(
      vocab=stackoverflow_dataset.create_vocab(FLAGS.vocab_size),
      num_oov_buckets=FLAGS.num_oov_buckets,
      client_batch_size=FLAGS.client_batch_size,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      max_seq_len=FLAGS.sequence_length,
      max_training_elements_per_user=FLAGS.max_elements_per_user)

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  training_process = iterative_process_builder.from_flags(
      input_spec=None,  # type pulled from train_dataset_preproces_comp.
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn,
      dataset_preprocess_comp=train_dataset_preprocess_comp)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_dataset=train_clientdata,
      train_clients_per_round=FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)

  assign_weights_fn = fed_avg_schedule.ServerState.assign_weights_to_keras_model

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
