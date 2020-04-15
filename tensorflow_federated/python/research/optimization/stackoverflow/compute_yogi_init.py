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
"""Computes an estimate for the Yogi initial accumulator using TFF."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.optimization.stackoverflow import dataset
from tensorflow_federated.python.research.optimization.stackoverflow import models

# Training hyperparameters
flags.DEFINE_integer('client_batch_size', 8, 'Batch size on the clients.')
flags.DEFINE_integer('num_clients', 10,
                     'Number of clients to use for estimating the L2-norm'
                     'squared of the batch gradients.')
flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                     'sentences to use per user.')

# Modeling flags
flags.DEFINE_boolean(
    'shared_embedding', False,
    'Boolean indicating whether to tie input and output embeddings.')
flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')

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
      shared_embedding=FLAGS.shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  (stackoverflow_train, _, _) = dataset.construct_word_level_datasets(
      FLAGS.vocab_size,
      FLAGS.client_batch_size,
      client_epochs_per_round=1,
      max_seq_len=FLAGS.sequence_length,
      max_training_elements_per_user=FLAGS.max_elements_per_user,
      num_validation_examples=1)

  input_spec = stackoverflow_train.create_tf_dataset_for_client(
      stackoverflow_train.client_ids[0]).element_spec

  tff_model = tff.learning.from_keras_model(
      keras_model=model_builder(), input_spec=input_spec, loss=loss_builder())

  yogi_init_accum_estimate = optimizer_utils.compute_yogi_init(
      stackoverflow_train, tff_model, num_clients=FLAGS.num_clients)
  logging.info('Yogi initializer: {:s}'.format(
      format(yogi_init_accum_estimate, '10.6E')))


if __name__ == '__main__':
  app.run(main)
