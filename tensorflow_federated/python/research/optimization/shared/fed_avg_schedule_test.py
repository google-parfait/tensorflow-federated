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
"""End-to-end example testing Federated Averaging against the MNIST model."""

import collections

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import fed_avg_schedule

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _batch_fn(has_nan=False):
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  if has_nan:
    batch[0][0, 0] = np.nan
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


def _uncompiled_model_builder():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=_create_input_spec(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


class ModelDeltaProcessTest(tf.test.TestCase):

  def _run_rounds(self, iterative_process, federated_data, num_rounds):
    train_outputs = []
    initial_state = iterative_process.initialize()
    state = initial_state
    for round_num in range(num_rounds):
      iteration_result = iterative_process.next(state, federated_data)
      train_outputs.append(iteration_result.metrics)
      logging.info('Round %d: %s', round_num, iteration_result.metrics)
      state = iteration_result.state
    return state, train_outputs, initial_state

  def test_fed_avg_without_schedule_decreases_loss(self):
    federated_data = [[_batch_fn()]]

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    _, train_outputs, _ = self._run_rounds(iterative_process, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_fed_avg_with_custom_client_weight_fn(self):
    federated_data = [[_batch_fn()]]

    def client_weight_fn(local_outputs):
      return 1.0/(1.0 + local_outputs['loss'][-1])

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_weight_fn=client_weight_fn)

    _, train_outputs, _ = self._run_rounds(iterative_process, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_client_update_with_finite_delta(self):
    federated_data = [_batch_fn()]
    model = _uncompiled_model_builder()
    client_optimizer = tf.keras.optimizers.SGD(0.1)
    client_update = fed_avg_schedule.create_client_update_fn()
    outputs = client_update(model, federated_data,
                            fed_avg_schedule._get_weights(model),
                            client_optimizer)
    self.assertAllEqual(self.evaluate(outputs.client_weight), 1)
    self.assertAllEqual(
        self.evaluate(outputs.optimizer_output['num_examples']), 1)

  def test_client_update_with_non_finite_delta(self):
    federated_data = [_batch_fn(has_nan=True)]
    model = _uncompiled_model_builder()
    client_optimizer = tf.keras.optimizers.SGD(0.1)
    client_update = fed_avg_schedule.create_client_update_fn()
    outputs = client_update(model, federated_data,
                            fed_avg_schedule._get_weights(model),
                            client_optimizer)
    self.assertAllEqual(self.evaluate(outputs.client_weight), 0)

  def test_server_update_with_nan_data_is_noop(self):
    federated_data = [[_batch_fn(has_nan=True)]]

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, _, initial_state = self._run_rounds(iterative_process,
                                               federated_data, 1)
    self.assertAllClose(state.model, initial_state.model, 1e-8)

  def test_server_update_with_inf_weight_is_noop(self):
    federated_data = [[_batch_fn()]]
    client_weight_fn = lambda x: np.inf

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_weight_fn=client_weight_fn)

    state, _, initial_state = self._run_rounds(iterative_process,
                                               federated_data, 1)
    self.assertAllClose(state.model, initial_state.model, 1e-8)

  def test_fed_avg_with_client_schedule(self):
    federated_data = [[_batch_fn()]]

    @tf.function
    def lr_schedule(x):
      return 0.1 if x < 1.5 else 0.0

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=lr_schedule,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    _, train_outputs, _ = self._run_rounds(iterative_process, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-4)

  def test_fed_avg_with_server_schedule(self):
    federated_data = [[_batch_fn()]]

    @tf.function
    def lr_schedule(x):
      return 1.0 if x < 1.5 else 0.0

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        server_lr=lr_schedule)

    _, train_outputs, _ = self._run_rounds(iterative_process, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-4)

  def test_fed_avg_with_client_and_server_schedules(self):
    federated_data = [[_batch_fn()]]

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=lambda x: 0.1 / (x + 1)**2,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        server_lr=lambda x: 1.0 / (x + 1)**2)

    _, train_outputs, _ = self._run_rounds(iterative_process, federated_data, 6)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])
    train_gap_first_half = train_outputs[0]['loss'] - train_outputs[2]['loss']
    train_gap_second_half = train_outputs[3]['loss'] - train_outputs[5]['loss']
    self.assertLess(train_gap_second_half, train_gap_first_half)

  def test_conversion_from_tff_result(self):
    federated_data = [[_batch_fn()]]

    iterative_process = fed_avg_schedule.build_fed_avg_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, _, _ = self._run_rounds(iterative_process, federated_data, 1)
    converted_state = fed_avg_schedule.ServerState.from_tff_result(state)
    self.assertIsInstance(converted_state, fed_avg_schedule.ServerState)
    self.assertIsInstance(converted_state.model, fed_avg_schedule.ModelWeights)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
