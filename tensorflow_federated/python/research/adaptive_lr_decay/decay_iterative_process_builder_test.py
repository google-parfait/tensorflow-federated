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

import collections

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.adaptive_lr_decay import adaptive_fed_avg
from tensorflow_federated.python.research.adaptive_lr_decay import callbacks
from tensorflow_federated.python.research.adaptive_lr_decay import decay_iterative_process_builder

FLAGS = flags.FLAGS


def _create_client_data(num_batches=2):
  # Create data for y = 3 * x + 1
  x = [[0.0], [1.0]]
  y = [[1.0], [4.0]]
  # Create a dataset of 4 examples (2 batches of two examples).
  return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
      x=x, y=y)).repeat().batch(2).take(num_batches)


def get_input_spec():
  return _create_client_data().element_spec


def model_builder():
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1,)),
      tf.keras.layers.Dense(
          units=1, kernel_initializer='zeros', bias_initializer='zeros')
  ])
  return keras_model


def loss_builder():
  return tf.keras.losses.MeanSquaredError()


def metrics_builder():
  return [tf.keras.metrics.MeanSquaredError()]


class DecayIterativeProcessBuilderTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    FLAGS.client_optimizer = 'sgd'
    FLAGS.client_learning_rate = 0.1
    FLAGS.server_optimizer = 'sgd'
    FLAGS.server_learning_rate = 0.1
    FLAGS.client_decay_factor = 1.0
    FLAGS.server_decay_factor = 1.0
    FLAGS.min_delta = 0.001
    FLAGS.min_lr = 0.0
    FLAGS.window_size = 1
    FLAGS.patience = 1

  def _run_rounds(self, iterative_process, num_rounds):
    client_datasets = [
        _create_client_data(num_batches=3),
        _create_client_data(num_batches=2)
    ]
    train_outputs = []
    state = iterative_process.initialize()
    for round_num in range(num_rounds):
      state, metrics = iterative_process.next(state, client_datasets)
      train_outputs.append(metrics)
      logging.info('Round %d: %s', round_num, metrics)
      logging.info('Model: %s', state.model)
    return state, train_outputs

  def test_iterative_process_type_signature(self):
    iterative_process = decay_iterative_process_builder.from_flags(
        input_spec=get_input_spec(),
        model_builder=model_builder,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder)

    dummy_lr_callback = callbacks.create_reduce_lr_on_plateau(
        learning_rate=FLAGS.client_learning_rate,
        decay_factor=FLAGS.client_decay_factor,
        min_delta=FLAGS.min_delta,
        min_lr=FLAGS.min_lr,
        window_size=FLAGS.window_size,
        patience=FLAGS.patience)
    lr_callback_type = tff.framework.type_from_tensors(dummy_lr_callback)

    server_state_type = tff.FederatedType(
        adaptive_fed_avg.ServerState(
            model=tff.learning.ModelWeights(
                trainable=(tff.TensorType(tf.float32, [1, 1]),
                           tff.TensorType(tf.float32, [1])),
                non_trainable=()),
            optimizer_state=[tf.int64],
            client_lr_callback=lr_callback_type,
            server_lr_callback=lr_callback_type), tff.SERVER)

    self.assertEqual(iterative_process.initialize.type_signature,
                     tff.FunctionType(parameter=None, result=server_state_type))

    dataset_type = tff.FederatedType(
        tff.SequenceType(
            collections.OrderedDict(
                x=tff.TensorType(tf.float32, [None, 1]),
                y=tff.TensorType(tf.float32, [None, 1]))), tff.CLIENTS)

    metrics_type = tff.FederatedType(
        collections.OrderedDict(
            mean_squared_error=tff.TensorType(tf.float32),
            loss=tff.TensorType(tf.float32)), tff.SERVER)
    output_type = collections.OrderedDict(
        before_training=metrics_type, during_training=metrics_type)

    expected_result_type = (server_state_type, output_type)
    expected_type = tff.FunctionType(
        parameter=(server_state_type, dataset_type),
        result=expected_result_type)

    actual_type = iterative_process.next.type_signature
    self.assertTrue(actual_type.is_equivalent_to(expected_type))

  def test_iterative_process_decreases_loss(self):
    iterative_process = decay_iterative_process_builder.from_flags(
        input_spec=get_input_spec(),
        model_builder=model_builder,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder)

    state, train_outputs = self._run_rounds(iterative_process, 4)
    self.assertLess(train_outputs[-1]['before_training']['loss'],
                    train_outputs[0]['before_training']['loss'])
    self.assertLess(train_outputs[-1]['during_training']['loss'],
                    train_outputs[0]['during_training']['loss'])
    self.assertNear(state.client_lr_callback.learning_rate, 0.1, 1e-8)
    self.assertNear(state.server_lr_callback.learning_rate, 0.1, 1e-8)

  def test_client_decay_schedule(self):
    FLAGS.client_decay_factor = 0.5
    FLAGS.server_decay_factor = 1.0
    FLAGS.min_delta = 0.5
    FLAGS.min_lr = 0.05
    FLAGS.window_size = 1
    FLAGS.patience = 1

    iterative_process = decay_iterative_process_builder.from_flags(
        input_spec=get_input_spec(),
        model_builder=model_builder,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder)

    state, train_outputs = self._run_rounds(iterative_process, 10)
    self.assertLess(train_outputs[-1]['before_training']['loss'],
                    train_outputs[0]['before_training']['loss'])
    self.assertLess(train_outputs[-1]['during_training']['loss'],
                    train_outputs[0]['during_training']['loss'])
    self.assertNear(state.client_lr_callback.learning_rate, 0.05, 1e-8)
    self.assertNear(state.server_lr_callback.learning_rate, 0.1, 1e-8)


if __name__ == '__main__':
  tf.test.main()
