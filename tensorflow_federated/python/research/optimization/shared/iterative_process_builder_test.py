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
"""Tests for shared iterative process builder."""

import collections

from absl import flags
from absl import logging
from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.optimization.shared import iterative_process_builder

FLAGS = flags.FLAGS

FLAGS.client_optimizer = 'sgd'
FLAGS.client_learning_rate = 0.1
FLAGS.server_optimizer = 'sgd'
FLAGS.server_learning_rate = 1.0


_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _batch_fn():
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  return batch


def _get_input_spec():
  input_spec = _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec([None, 1], dtype=tf.int64))
  return input_spec


def model_builder():
  return tff.simulation.models.mnist.create_keras_model(compile_model=False)


def loss_builder():
  return tf.keras.losses.SparseCategoricalCrossentropy()


def metrics_builder():
  return [tf.keras.metrics.SparseCategoricalAccuracy()]


class IterativeProcessBuilderTest(tf.test.TestCase, parameterized.TestCase):

  def _run_rounds(self, iterproc, client_datasets, num_rounds):
    train_outputs = []
    state = iterproc.initialize()
    for round_num in range(num_rounds):
      state, metrics = iterproc.next(state, client_datasets)
      train_outputs.append(metrics)
      logging.info('Round %d: %s', round_num, metrics)
    return state, train_outputs

  def test_iterative_process_no_schedule_decreases_loss(self):
    FLAGS.client_lr_schedule = 'constant'
    FLAGS.server_lr_schedule = 'constant'
    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()
    iterproc = iterative_process_builder.from_flags(input_spec, model_builder,
                                                    loss_builder,
                                                    metrics_builder)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_iterative_process_with_custom_client_weight_fn_decreases_loss(self):
    FLAGS.client_lr_schedule = 'constant'
    FLAGS.server_lr_schedule = 'constant'
    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()

    def client_weight_fn(local_outputs):
      return 1.0 / (1.0 + local_outputs['loss'][-1])

    iterproc = iterative_process_builder.from_flags(
        input_spec,
        model_builder,
        loss_builder,
        metrics_builder,
        client_weight_fn=client_weight_fn)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  @parameterized.named_parameters(('inv_lin_decay', 'inv_lin_decay'),
                                  ('inv_sqrt_decay', 'inv_sqrt_decay'))
  def test_iterative_process_with_inv_time_client_schedule(self, sched_type):
    FLAGS.client_lr_schedule = 'constant'
    FLAGS.client_lr_decay_steps = 1
    FLAGS.client_lr_decay_rate = 1.0
    FLAGS.client_lr_staircase = False
    FLAGS.server_lr_schedule = 'constant'
    FLAGS.client_lr_schedule = sched_type
    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()
    iterproc = iterative_process_builder.from_flags(input_spec, model_builder,
                                                    loss_builder,
                                                    metrics_builder)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_iterative_process_with_exp_decay_client_schedule(self):
    FLAGS.client_lr_schedule = 'exp_decay'
    FLAGS.client_lr_decay_steps = 1
    FLAGS.client_lr_decay_rate = 0.5
    FLAGS.client_lr_staircase = False
    FLAGS.server_lr_schedule = 'constant'

    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()
    iterproc = iterative_process_builder.from_flags(input_spec, model_builder,
                                                    loss_builder,
                                                    metrics_builder)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  @parameterized.named_parameters(('inv_lin_decay', 'inv_lin_decay'),
                                  ('inv_sqrt_decay', 'inv_sqrt_decay'))
  def test_iterative_process_with_inv_time_server_schedule(self, sched_type):
    FLAGS.client_lr_schedule = 'constant'
    FLAGS.server_lr_decay_steps = 1
    FLAGS.server_lr_decay_rate = 1.0
    FLAGS.server_lr_staircase = False
    FLAGS.server_lr_schedule = sched_type
    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()
    iterproc = iterative_process_builder.from_flags(input_spec, model_builder,
                                                    loss_builder,
                                                    metrics_builder)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_iterative_process_with_exp_decay_server_schedule(self):
    FLAGS.client_lr_schedule = 'constant'
    FLAGS.server_lr_schedule = 'exp_decay'
    FLAGS.server_lr_decay_steps = 1
    FLAGS.server_lr_decay_rate = 0.5
    FLAGS.server_lr_staircase = False

    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()
    iterproc = iterative_process_builder.from_flags(input_spec, model_builder,
                                                    loss_builder,
                                                    metrics_builder)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_decay_factor_0_does_not_decrease_loss(self):
    FLAGS.client_lr_schedule = 'exp_decay'
    FLAGS.client_lr_decay_steps = 2
    FLAGS.client_lr_decay_rate = 0.0
    FLAGS.client_lr_staircase = True
    FLAGS.server_lr_schedule = 'constant'

    federated_data = [[_batch_fn()]]
    input_spec = _get_input_spec()
    iterproc = iterative_process_builder.from_flags(input_spec, model_builder,
                                                    loss_builder,
                                                    metrics_builder)
    _, train_outputs = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-5)


if __name__ == '__main__':
  tff.backends.native.set_local_execution_context(max_fanout=25)
  tf.test.main()
