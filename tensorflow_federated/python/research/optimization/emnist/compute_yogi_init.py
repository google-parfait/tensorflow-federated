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

from tensorflow_federated.python.research.optimization.emnist import dataset
from tensorflow_federated.python.research.optimization.emnist import models
from tensorflow_federated.python.research.optimization.shared import optimizer_utils


# Experiment hyperparameters
flags.DEFINE_enum('model', 'cnn', ['cnn', '2nn'], 'Which model to use. This '
                  'can be a convolutional model (cnn) or a two hidden-layer '
                  'densely connected network (2nn).')
flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the client.')
flags.DEFINE_integer('num_clients', 10,
                     'Number of clients to use for estimating the L2-norm'
                     'squared of the batch gradients.')

FLAGS = flags.FLAGS

# End of hyperparameter flags.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  tf.compat.v1.enable_v2_behavior()
  # TODO(b/139129100): Remove this once the local executor is the default.
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=25))

  emnist_train, _ = dataset.get_emnist_datasets(only_digits=False)
  emnist_train = emnist_train.preprocess(lambda x: x.batch(20))
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

  tff_model = tff.learning.from_keras_model(
      keras_model=model_builder(),
      dummy_batch=sample_batch,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())

  yogi_init_accum_estimate = optimizer_utils.compute_yogi_init(
      emnist_train, tff_model, num_clients=FLAGS.num_clients)
  logging.info('Yogi initializer: {:s}'.format(
      format(yogi_init_accum_estimate, '10.6E')))

if __name__ == '__main__':
  app.run(main)
