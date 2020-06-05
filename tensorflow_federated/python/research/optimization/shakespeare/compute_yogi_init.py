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

from tensorflow_federated.python.research.optimization.shakespeare import dataset
from tensorflow_federated.python.research.optimization.shakespeare import models
from tensorflow_federated.python.research.optimization.shared import optimizer_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('client_batch_size', 10, 'Batch size on the clients.')
flags.DEFINE_integer('num_clients', 10,
                     'Number of clients to use for estimating the L2-norm'
                     'squared of the batch gradients.')


# Vocabulary with one OOV ID and zero for the padding.
VOCAB_SIZE = len(dataset.CHAR_VOCAB) + 2


def model_builder():
  """Constructs a `tf.keras.Model` to train."""
  return models.create_recurrent_model(
      vocab_size=VOCAB_SIZE, batch_size=FLAGS.client_batch_size)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=25))

  train_clientdata, _ = dataset.construct_character_level_datasets(
      FLAGS.client_batch_size, epochs=1)
  loss_fn_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  input_spec = train_clientdata.create_tf_dataset_for_client(
      train_clientdata.client_ids[0])

  tff_model = tff.learning.from_keras_model(
      keras_model=model_builder(),
      input_spec=input_spec,
      loss=loss_fn_builder())

  yogi_init_accum_estimate = optimizer_utils.compute_yogi_init(
      train_clientdata, tff_model, num_clients=FLAGS.num_clients)
  logging.info('Yogi initializer: {:s}'.format(
      format(yogi_init_accum_estimate, '10.6E')))

if __name__ == '__main__':
  app.run(main)
