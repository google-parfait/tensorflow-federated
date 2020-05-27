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
"""Trains and evaluates an EMNIST classification model with DP-FedAvg."""

import functools
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.differential_privacy import dp_utils
from tensorflow_federated.python.research.optimization.emnist import dataset
from tensorflow_federated.python.research.optimization.emnist import models
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum(
      'model', 'cnn', ['cnn', '2nn'], 'Which model to use. This '
      'can be a convolutional model (cnn) or a two hidden-layer '
      'densely connected network (2nn).')
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_boolean(
      'uniform_weighting', False,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of samples.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

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

# End of hyperparameter flags.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  tf.compat.v1.enable_v2_behavior()

  emnist_train, emnist_test = dataset.get_emnist_datasets(
      FLAGS.client_batch_size, FLAGS.client_epochs_per_round, only_digits=False)

  if FLAGS.model == 'cnn':
    model_builder = functools.partial(
        models.create_conv_dropout_model, only_digits=False)
  elif FLAGS.model == '2nn':
    model_builder = functools.partial(
        models.create_two_hidden_layer_model, only_digits=False)
  else:
    raise ValueError('Cannot handle model flag [{!s}].'.format(FLAGS.model))

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  if FLAGS.uniform_weighting:

    def client_weight_fn(local_outputs):
      del local_outputs
      return 1.0

  else:
    client_weight_fn = None  #  Defaults to the number of examples per client.

  def model_fn():
    return tff.learning.from_keras_model(
        model_builder(),
        loss_builder(),
        input_spec=emnist_test.element_spec,
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
      emnist_train, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=dp_utils.assign_weights_to_keras_model)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      evaluate_fn=evaluate_fn,
  )


if __name__ == '__main__':
  app.run(main)
