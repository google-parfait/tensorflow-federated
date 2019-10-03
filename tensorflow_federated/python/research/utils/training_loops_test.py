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

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.examples.mnist import models
from tensorflow_federated.python.research.utils import training_loops


class TrainingLoopsTest(tf.test.TestCase):

  def test_federated_training_loop(self):
    Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

    batch = Batch(
        x=np.ones([1, 784], dtype=np.float32),
        y=np.ones([1, 1], dtype=np.int64))
    federated_data = [[batch]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data

    loss_list = []

    def metrics_hook(state, metrics, round_num):
      del round_num
      del metrics
      keras_model = models.create_keras_model(compile_model=True)
      tff.learning.assign_weights_to_keras_model(keras_model, state.model)
      loss_list.append(keras_model.test_on_batch(batch.x, batch.y))

    server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)

    training_loops.federated_averaging_training_loop(
        models.model_fn,
        server_optimizer_fn,
        client_datasets_fn,
        total_rounds=3,
        metrics_hook=metrics_hook)

    self.assertLess(np.mean(loss_list[1:]), loss_list[0])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
