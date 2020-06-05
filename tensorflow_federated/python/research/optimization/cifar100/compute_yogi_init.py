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

from tensorflow_federated.python.research.optimization.cifar100 import dataset
from tensorflow_federated.python.research.optimization.shared import optimizer_utils
from tensorflow_federated.python.research.optimization.shared import resnet_models

# Experiment hyperparameters
flags.DEFINE_integer('client_batch_size', 32, 'Batch size on the clients.')
flags.DEFINE_integer('num_clients', 10,
                     'Number of clients to use for estimating the L2-norm'
                     'squared of the batch gradients.')

# End of hyperparameter flags.

FLAGS = flags.FLAGS

CIFAR_SHAPE = (32, 32, 3)
CROP_SHAPE = (24, 24, 3)
NUM_CLASSES = 100


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  # TODO(b/139129100): Remove this once the local executor is the default.
  tff.framework.set_default_executor(
      tff.framework.local_executor_factory(max_fanout=25))

  cifar_train, _ = dataset.get_federated_cifar100(
      client_epochs_per_round=1,
      train_batch_size=FLAGS.client_batch_size,
      crop_shape=CROP_SHAPE)
  input_spec = cifar_train.create_tf_dataset_for_client(
      cifar_train.client_ids[0]).element_spec

  model_builder = functools.partial(
      resnet_models.create_resnet18,
      input_shape=CROP_SHAPE,
      num_classes=NUM_CLASSES)
  tff_model = tff.learning.from_keras_model(
      keras_model=model_builder(),
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())

  yogi_init_accum_estimate = optimizer_utils.compute_yogi_init(
      cifar_train, tff_model, num_clients=FLAGS.num_clients)
  logging.info('Yogi initializer: {:s}'.format(
      format(yogi_init_accum_estimate, '10.6E')))


if __name__ == '__main__':
  app.run(main)
