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
"""Baseline experiment on centralized data.

The objective is to demonstrate what is possible if the EMNIST-62 data was
available in a central location, where all common optimization techniques apply.
In practice, federated learning is typically applied in cases where the
on-device data cannot be centralized, and so this ideal goal need not be
realized for FL to be effective. However, when studying optimization algorithms,
it is interesting to study this comparison.

In an example run, the model achieved:
* 85.33% accuracy after 2 passes through data.
* 87.20% accuracy after 10 passes through data.
* 87.83% accuracy after 25 passes through data.
* 88.22% accuracy after 75 passes through data.
"""

import collections
from absl import app

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.baselines.emnist import models

BATCH_SIZE = 100
# The total number of examples in
# emnist_train.create_tf_dataset_from_all_clients()
TOTAL_EXAMPLES = 671585


def run_experiment():
  """Runs the training experiment."""
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=False)

  example_tuple = collections.namedtuple('Example', ['x', 'y'])

  def element_fn(element):
    return example_tuple(
        # The expand_dims adds a channel dimension.
        x=tf.expand_dims(element['pixels'], -1),
        y=element['label'])

  all_train = emnist_train.create_tf_dataset_from_all_clients().map(element_fn)
  all_train = all_train.shuffle(TOTAL_EXAMPLES).repeat().batch(BATCH_SIZE)

  all_test = emnist_test.create_tf_dataset_from_all_clients().map(element_fn)
  all_test = all_test.batch(BATCH_SIZE)

  train_data_elements = int(TOTAL_EXAMPLES / BATCH_SIZE)

  model = models.create_conv_dropout_model(only_digits=False)
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(
          learning_rate=0.01,
          momentum=0.9,
          decay=0.2 / train_data_elements,
          nesterov=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  model.fit_generator(
      all_train,
      steps_per_epoch=train_data_elements,
      epochs=75,
      verbose=1,
      validation_data=all_test)
  score = model.evaluate_generator(all_test, verbose=0)
  print('Final test loss: %.4f' % score[0])
  print('Final test accuracy: %.4f' % score[1])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.compat.v1.enable_v2_behavior()
  run_experiment()


if __name__ == '__main__':
  app.run(main)
