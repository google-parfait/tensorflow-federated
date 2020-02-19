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
from tensorflow_federated.python.research.optimization.shared import training_utils
from tensorflow_federated.python.research.optimization.stackoverflow_lr import dataset
from tensorflow_federated.python.research.optimization.stackoverflow_lr import models
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import utils_impl


with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_integer('vocab_tokens_size', 10000, 'Vocab tokens size used.')
  flags.DEFINE_integer('vocab_tags_size', 500, 'Vocab tags size used.')
  flags.DEFINE_integer('client_batch_size', 100,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                       'sentences to use per user.')


FLAGS = flags.FLAGS


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  return [
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
  ]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()
  # TODO(b/139129100): Remove this once the local executor is the default.
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=25))

  stackoverflow_train, stackoverflow_validation, stackoverflow_test = dataset.get_stackoverflow_datasets(
      vocab_tokens_size=FLAGS.vocab_tokens_size,
      vocab_tags_size=FLAGS.vocab_tags_size,
      client_batch_size=FLAGS.client_batch_size,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      max_training_elements_per_user=FLAGS.max_elements_per_user,
      num_validation_examples=FLAGS.num_validation_examples)

  sample_client_dataset = stackoverflow_train.create_tf_dataset_for_client(
      stackoverflow_train.client_ids[0])
  # TODO(b/144382142): Sample batches cannot be eager tensors, since they are
  # passed (implicitly) to tff.learning.build_federated_averaging_process.
  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       next(iter(sample_client_dataset)))

  model_builder = functools.partial(
      models.create_logistic_model,
      vocab_tokens_size=FLAGS.vocab_tokens_size,
      vocab_tags_size=FLAGS.vocab_tags_size)

  loss_builder = functools.partial(
      tf.keras.losses.BinaryCrossentropy,
      from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM)

  training_process = iterative_process_builder.from_flags(
      dummy_batch=sample_batch,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      stackoverflow_train, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=stackoverflow_validation,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      test_dataset=stackoverflow_validation.concatenate(stackoverflow_test))

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      evaluate_fn=evaluate_fn,
  )


if __name__ == '__main__':
  app.run(main)
