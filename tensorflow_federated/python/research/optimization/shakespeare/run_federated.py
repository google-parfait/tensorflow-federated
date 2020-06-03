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
"""Train a Char-RNN based on the Shakespeare dataset in the federated setting."""

import functools

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow_federated.python.research.optimization.shakespeare import dataset
from tensorflow_federated.python.research.optimization.shakespeare import models
from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule
from tensorflow_federated.python.research.optimization.shared import iterative_process_builder
from tensorflow_federated.python.research.optimization.shared import keras_metrics
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import training_utils
from tensorflow_federated.python.research.utils import utils_impl

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
      'client_datasets_random_seed', 1, 'The random seed '
      'governing the client dataset selection.')

# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
VOCAB_SIZE = len(dataset.CHAR_VOCAB) + 4


def model_builder():
  """Constructs a `tf.keras.Model` to train."""
  return models.create_recurrent_model(
      vocab_size=VOCAB_SIZE, sequence_length=FLAGS.sequence_length)


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  pad_token, _, _, _ = dataset.get_special_tokens()

  return [
      keras_metrics.NumBatchesCounter(),
      keras_metrics.NumExamplesCounter(),
      keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
  ]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()

  train_clientdata, test_dataset = dataset.construct_character_level_datasets(
      FLAGS.client_batch_size, FLAGS.client_epochs_per_round,
      FLAGS.sequence_length)
  test_dataset = test_dataset.cache()

  loss_fn_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  input_spec = train_clientdata.create_tf_dataset_for_client(
      train_clientdata.client_ids[0]).element_spec

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  training_process = iterative_process_builder.from_flags(
      input_spec=input_spec,
      model_builder=model_builder,
      loss_builder=loss_fn_builder,
      metrics_builder=metrics_builder,
      client_weight_fn=client_weight_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_dataset=train_clientdata,
      train_clients_per_round=FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)

  assign_weights_fn = fed_avg_schedule.ServerState.assign_weights_to_keras_model

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=test_dataset,
      model_builder=model_builder,
      loss_builder=loss_fn_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      evaluate_fn=evaluate_fn)


if __name__ == '__main__':
  app.run(main)
