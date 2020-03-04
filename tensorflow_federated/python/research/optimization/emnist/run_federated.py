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
"""Trains and evaluates an EMNIST classification model using TFF."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_federated.python.research.optimization.emnist import dataset
from tensorflow_federated.python.research.optimization.emnist import models
from tensorflow_federated.python.research.optimization.shared import iterative_process_builder
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl


with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum('model', 'cnn', ['cnn', '2nn'], 'Which model to use. This '
                    'can be a convolutional model (cnn) or a two hidden-layer '
                    'densely connected network (2nn).')
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')

FLAGS = flags.FLAGS

# End of hyperparameter flags.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  tf.compat.v1.enable_v2_behavior()

  emnist_train, emnist_test = dataset.get_emnist_datasets(
      FLAGS.client_batch_size, FLAGS.client_epochs_per_round, only_digits=False)

  sample_client_dataset = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0])
  # TODO(b/144382142): Sample batches cannot be eager tensors, since they are
  # passed (implicitly) to tff.learning.build_federated_averaging_process.
  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       next(iter(sample_client_dataset)))

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

  training_process = iterative_process_builder.from_flags(
      dummy_batch=sample_batch,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      emnist_train, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      evaluate_fn=evaluate_fn,
  )


if __name__ == '__main__':
  app.run(main)
