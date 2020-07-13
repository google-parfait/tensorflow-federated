# Copyright 2020, The TensorFlow Federated Authors.
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
"""Integration tests for federated averaging with learning rate callbacks."""

import collections

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import callbacks


def _create_client_data(num_batches=2):
  # Create data for y = 3 * x + 1
  x = [[0.0], [1.0]]
  y = [[1.0], [4.0]]
  # Create a dataset of 4 examples (2 batches of two examples).
  return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
      x=x, y=y)).repeat().batch(2).take(num_batches)


def _uncompiled_model_builder():
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1,)),
      tf.keras.layers.Dense(
          units=1, kernel_initializer='zeros', bias_initializer='zeros')
  ])
  loss_fn = tf.keras.losses.MeanSquaredError()
  input_spec = _create_client_data().element_spec
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss_fn)


class AdaptiveFedAvgTest(tf.test.TestCase):

  def _run_rounds(self, iterative_process, num_rounds):
    client_datasets = [
        _create_client_data(num_batches=1),
        _create_client_data(num_batches=2)
    ]
    train_outputs = []
    state = iterative_process.initialize()
    for round_num in range(num_rounds):
      iteration_result = iterative_process.next(state, client_datasets)
      if hasattr(iteration_result.metrics, 'train'):
        # tff.learning returns a nested tuple of metrics, we only compare
        # against `train`.
        train_outputs.append(iteration_result.metrics.train)
      else:
        train_outputs.append(iteration_result.metrics)
      logging.info('Round %d: %s', round_num, iteration_result.metrics)
      logging.info('Model: %s', iteration_result.state.model)
      state = iteration_result.state
    return state, train_outputs

  def _run_rounds_tff_fedavg(self, iterative_process, num_rounds):
    client_datasets = [
        _create_client_data(num_batches=1),
        _create_client_data(num_batches=2)
    ]
    train_outputs = []
    state = iterative_process.initialize()
    for round_num in range(num_rounds):
      state, outputs = iterative_process.next(state, client_datasets)
      logging.info('Round %d: %s', round_num, outputs)
      logging.info('Model: %s', state.model)
      train_outputs.append(outputs)
    return state, train_outputs

  def test_comparable_to_fed_avg(self):
    client_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)
    server_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)

    iterative_process = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    reference_iterative_process = tff.learning.build_federated_averaging_process(
        _uncompiled_model_builder,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0))

    _, train_outputs = self._run_rounds(iterative_process, 5)
    _, reference_train_outputs = self._run_rounds_tff_fedavg(
        reference_iterative_process, 5)

    for i in range(5):
      self.assertAllClose(train_outputs[i]['during_training']['loss'],
                          reference_train_outputs[i].train['loss'], 1e-4)

  def test_fed_avg_without_decay_decreases_loss(self):
    client_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)
    server_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)

    iterative_process = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, train_outputs = self._run_rounds(iterative_process, 5)
    self.assertLess(train_outputs[-1]['before_training']['loss'],
                    train_outputs[0]['before_training']['loss'])
    self.assertLess(train_outputs[-1]['during_training']['loss'],
                    train_outputs[0]['during_training']['loss'])
    self.assertNear(state.client_lr_callback.learning_rate, 0.1, 1e-8)
    self.assertNear(state.server_lr_callback.learning_rate, 0.1, 1e-8)

  def test_fed_avg_with_client_decay_decreases_loss(self):
    client_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        window_size=1,
        min_delta=0.5,
        min_lr=0.05,
        decay_factor=0.5,
        patience=1,
        cooldown=0)
    server_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1, window_size=1, decay_factor=1.0, cooldown=0)

    iterative_process = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, train_outputs = self._run_rounds(iterative_process, 10)
    self.assertLess(train_outputs[-1]['before_training']['loss'],
                    train_outputs[0]['before_training']['loss'])
    self.assertLess(train_outputs[-1]['during_training']['loss'],
                    train_outputs[0]['during_training']['loss'])
    self.assertNear(state.client_lr_callback.learning_rate, 0.05, 1e-8)
    self.assertNear(state.server_lr_callback.learning_rate, 0.1, 1e-8)

  def test_fed_avg_with_server_decay_decreases_loss(self):
    client_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        window_size=1,
        patience=1,
        decay_factor=1.0,
        cooldown=0)

    server_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        window_size=1,
        patience=1,
        decay_factor=0.5,
        min_delta=0.5,
        min_lr=0.05,
        cooldown=0)

    iterative_process = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, train_outputs = self._run_rounds(iterative_process, 10)
    self.assertLess(train_outputs[-1]['before_training']['loss'],
                    train_outputs[0]['before_training']['loss'])
    self.assertLess(train_outputs[-1]['during_training']['loss'],
                    train_outputs[0]['during_training']['loss'])
    self.assertNear(state.client_lr_callback.learning_rate, 0.1, 1e-8)
    self.assertNear(state.server_lr_callback.learning_rate, 0.05, 1e-8)

  def test_fed_sgd_without_decay_decreases_loss(self):
    client_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.0,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)
    server_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)

    iterative_process = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, train_outputs = self._run_rounds(iterative_process, 5)
    self.assertLess(train_outputs[-1]['before_training']['loss'],
                    train_outputs[0]['before_training']['loss'])
    self.assertLess(train_outputs[-1]['during_training']['loss'],
                    train_outputs[0]['during_training']['loss'])
    self.assertNear(state.client_lr_callback.learning_rate, 0.0, 1e-8)
    self.assertNear(state.server_lr_callback.learning_rate, 0.1, 1e-8)

  def test_small_lr_comparable_zero_lr(self):
    client_lr_callback1 = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.0,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)
    client_lr_callback2 = callbacks.create_reduce_lr_on_plateau(
        learning_rate=1e-8,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)

    server_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=0.1,
        min_delta=0.5,
        window_size=2,
        decay_factor=1.0,
        cooldown=0)

    iterative_process1 = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback1,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)
    iterative_process2 = adaptive_fed_avg.build_fed_avg_process(
        _uncompiled_model_builder,
        client_lr_callback2,
        callbacks.update_reduce_lr_on_plateau,
        server_lr_callback,
        callbacks.update_reduce_lr_on_plateau,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state1, train_outputs1 = self._run_rounds(iterative_process1, 5)
    state2, train_outputs2 = self._run_rounds(iterative_process2, 5)

    self.assertAllClose(state1.model.trainable, state2.model.trainable, 1e-4)
    self.assertAllClose(train_outputs1, train_outputs2, 1e-4)


if __name__ == '__main__':
  tf.test.main()
