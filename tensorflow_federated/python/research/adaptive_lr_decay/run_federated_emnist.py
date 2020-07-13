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
"""Trains and evaluates on FEMNIST with adaptive LR decay."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import decay_iterative_process_builder
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl
from tensorflow_federated.python.research.utils.datasets import emnist_dataset
from tensorflow_federated.python.research.utils.models import emnist_models

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum(
      'model', 'cnn', ['cnn', 'orig_cnn', '2nn'], 'Which model to '
      'use. This can be a convolutional model with dropout (cnn),'
      'a convolutional model without dropout (orig_cnn), or a two'
      'hidden-layer densely connected network (2nn).')
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_integer(
      'max_batches_per_client', -1, 'Maximum number of batches to process at '
      'each client in a given round. If set to -1, we take the full dataset.')
  flags.DEFINE_boolean(
      'only_digits', False, 'If True, we use the digits-only '
      'EMNIST (10 classes), otherwise we use the full EMNIST '
      '(62 classes).')
  flags.DEFINE_enum(
      'client_weight', 'uniform', ['num_samples', 'uniform'],
      'Weighting scheme for the client model deltas. Currently, this can '
      'either weight according to the number of samples on a client '
      '(num_samples) or uniformly (uniform).')
  flags.DEFINE_integer(
      'client_datasets_random_seed', 1, 'The random seed '
      'governing the selection of clients that participate in each training '
      'round. The seed is used to generate the starting point for a Lehmer '
      'pseudo-random number generator, the outputs of which are used as seeds '
      'for the client sampling.')

FLAGS = flags.FLAGS

# End of hyperparameter flags.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  emnist_train, emnist_test = emnist_dataset.get_emnist_datasets(
      FLAGS.client_batch_size,
      FLAGS.client_epochs_per_round,
      max_batches_per_client=FLAGS.max_batches_per_client,
      only_digits=FLAGS.only_digits)

  central_emnist_train, _ = emnist_dataset.get_centralized_emnist_datasets(
      batch_size=100, only_digits=FLAGS.only_digits, shuffle_train=False)

  input_spec = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0]).element_spec

  if FLAGS.model == 'cnn':
    model_builder = functools.partial(
        emnist_models.create_conv_dropout_model, only_digits=FLAGS.only_digits)
  elif FLAGS.model == 'orig_cnn':
    model_builder = functools.partial(
        emnist_models.create_original_fedavg_cnn_model,
        only_digits=FLAGS.only_digits)
  elif FLAGS.model == '2nn':
    model_builder = functools.partial(
        emnist_models.create_two_hidden_layer_model,
        only_digits=FLAGS.only_digits)
  else:
    raise ValueError('Cannot handle model flag [{!s}].'.format(FLAGS.model))

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  if FLAGS.client_weight == 'uniform':
    def client_weight_fn(local_outputs):
      del local_outputs
      return 1.0

  elif FLAGS.client_weight == 'num_samples':
    client_weight_fn = None
  else:
    raise ValueError('Unsupported client_weight flag [{!s}]. Currently only '
                     '`uniform` and `num_samples` are supported.'.format(
                         FLAGS.client_weight))

  training_process = decay_iterative_process_builder.from_flags(
      input_spec=input_spec,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      emnist_train,
      FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)

  assign_weights_fn = adaptive_fed_avg.ServerState.assign_weights_to_keras_model

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  train_evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=central_emnist_train,
      model_builder=model_builder,
      loss_builder=loss_builder,
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
