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
"""Trains and evaluates on Shakespeare with adaptive LR decay."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import decay_iterative_process_builder
from tensorflow_federated.python.research.optimization.shakespeare import models
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import shakespeare_dataset

FLAGS = flags.FLAGS

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_integer(
      'clients_per_round', 10,
      'Number of clients that participant in training each federated round.')
  flags.DEFINE_integer(
      'client_batch_size', 10,
      'Number of examples per batch for client (inner optimizer) training.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_integer(
      'sequence_length', 80,
      'Length of character sequences to use for the RNN model.')
  flags.DEFINE_integer(
      'max_batches_per_client', -1, 'Maximum number of batches to process at '
      'each client in a given round. If set to -1, we take the full dataset.')
  flags.DEFINE_enum(
      'client_weight', 'uniform', ['num_tokens', 'uniform'],
      'Weighting scheme for the client model deltas. Currently, this can '
      'either weight according to the number of tokens on a client '
      '(num_tokens) or uniformly (uniform).')
  flags.DEFINE_integer(
      'client_datasets_random_seed', 1, 'The random seed '
      'governing the selection of clients that participate in each training '
      'round. The seed is used to generate the starting point for a Lehmer '
      'pseudo-random number generator, the outputs of which are used as seeds '
      'for the client sampling.')

# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
VOCAB_SIZE = len(shakespeare_dataset.CHAR_VOCAB) + 4


def model_builder():
  """Constructs a `tf.keras.Model` to train."""
  return models.create_recurrent_model(
      vocab_size=VOCAB_SIZE, sequence_length=FLAGS.sequence_length)


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  pad_token, _, _, _ = shakespeare_dataset.get_special_tokens()

  return [
      keras_metrics.NumBatchesCounter(),
      keras_metrics.NumExamplesCounter(),
      keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
  ]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_clientdata = shakespeare_dataset.construct_character_level_datasets(
      FLAGS.client_batch_size,
      FLAGS.client_epochs_per_round,
      sequence_length=FLAGS.sequence_length,
      max_batches_per_client=FLAGS.max_batches_per_client,
      shuffle_buffer_size=0)
  eval_train_dataset, eval_test_dataset = (
      shakespeare_dataset.construct_centralized_datasets())

  loss_fn_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  input_spec = train_clientdata.create_tf_dataset_for_client(
      train_clientdata.client_ids[0]).element_spec

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
      loss_builder=loss_fn_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_clientdata,
      FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)

  assign_weights_fn = adaptive_fed_avg.ServerState.assign_weights_to_keras_model

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=eval_test_dataset,
      model_builder=model_builder,
      loss_builder=loss_fn_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  train_evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=eval_train_dataset,
      model_builder=model_builder,
      loss_builder=loss_fn_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=evaluate_fn,
      train_eval_fn=train_evaluate_fn,
  )


if __name__ == '__main__':
  app.run(main)
