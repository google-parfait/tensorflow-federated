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
"""End-to-end example testing the FedAvg algorithm with stateful clients."""

import collections
import functools

from absl.testing import parameterized
import keras
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from examplesstateful_clients import stateful_fedavg_tf
from examplesstateful_clients import stateful_fedavg_tff


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format,
  )
  conv2d = functools.partial(
      keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
  )

  model = keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      keras.layers.Flatten(),
      keras.layers.Dense(10),
      keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def _create_one_client_state():
  return stateful_fedavg_tf.ClientState(client_index=-1, iters_count=0)


def _simple_model_fn():
  keras_model = _create_test_cnn_model()
  loss = keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32),
  )
  return stateful_fedavg_tf.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss
  )


class FedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple_model_wrapper', _simple_model_fn),
  )
  def test_something(self, model_fn):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        model_fn, _create_one_client_state
    )
    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,28,28,1],y=int32[?]>*}@CLIENTS'
    )

  @parameterized.named_parameters(
      ('simple_model_wrapper', _simple_model_fn),
  )
  def test_simple_training(self, model_fn):
    it_process = stateful_fedavg_tff.build_federated_averaging_process(
        model_fn, _create_one_client_state
    )
    server_state = it_process.initialize()

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.ones([1, 28, 28, 1], dtype=np.float32),
          y=np.ones([1], dtype=np.int32),
      )

    batch = tff.tensorflow.computation(deterministic_batch)()
    federated_data = [[batch]]
    client_states = [_create_one_client_state()]

    loss_list = []
    for _ in range(3):
      server_state, loss, client_states = it_process.next(
          server_state, federated_data, client_states
      )
      loss_list.append(loss)

    self.assertEqual(
        server_state.total_iters_count, client_states[0].iters_count
    )
    self.assertLess(np.mean(loss_list[1:]), loss_list[0])

  def test_keras_evaluate(self):
    keras_model = _create_test_cnn_model()
    sample_data = [
        collections.OrderedDict(
            x=np.ones([1, 28, 28, 1], dtype=np.float32),
            y=np.ones([1], dtype=np.int32),
        )
    ]
    metric = keras.metrics.SparseCategoricalAccuracy()
    accuracy = stateful_fedavg_tf.keras_evaluate(
        keras_model, sample_data, metric
    )
    self.assertIsInstance(accuracy, tf.Tensor)
    self.assertBetween(accuracy, 0.0, 1.0)


if __name__ == '__main__':
  tf.test.main()
