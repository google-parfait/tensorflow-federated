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

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.emnist_ae import dataset
from tensorflow_federated.python.research.optimization.emnist_ae import models
from tensorflow_federated.python.research.optimization.shared import iterative_process_builder
from tensorflow_federated.python.research.optimization.shared import training_utils
from tensorflow_federated.python.research.utils import training_loop
from tensorflow_federated.python.research.utils import utils_impl


with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')

FLAGS = flags.FLAGS

# End of hyperparameter flags.


def main(_):

  tf.enable_v2_behavior()
  # TODO(b/139129100): Remove this once the local executor is the default.
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=25))

  emnist_train, emnist_test = dataset.get_emnist_datasets(
      FLAGS.client_batch_size, FLAGS.client_epochs_per_round, only_digits=False)

  sample_client_dataset = emnist_train.create_tf_dataset_for_client(
      emnist_train.client_ids[0])
  # TODO(b/144382142): Sample batches cannot be eager tensors, since they are
  # passed (implicitly) to tff.learning.build_federated_averaging_process.
  sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                       next(iter(sample_client_dataset)))

  model_builder = models.create_autoencoder_model

  loss_builder = tf.keras.losses.MeanSquaredError
  metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

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
