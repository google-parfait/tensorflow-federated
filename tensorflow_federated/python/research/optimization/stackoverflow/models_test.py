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
"""Tests for stackoverflow models."""

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.research.optimization.stackoverflow import models


class KerasSequenceModelsTest(absltest.TestCase):

  def test_dense_fn_raises(self):

    def _dense_layer_fn(x):
      return tf.keras.layers.Dense(x)

    with self.assertRaisesRegex(ValueError, 'tf.keras.layers.RNN'):
      models.create_recurrent_model(10, _dense_layer_fn, 'dense')

  def test_lstm_constructs(self):

    def _recurrent_layer_fn(x):
      return tf.keras.layers.LSTM(x, return_sequences=True)

    model = models.create_recurrent_model(10, _recurrent_layer_fn, 'rnn-lstm')
    self.assertIsInstance(model, tf.keras.Model)
    self.assertEqual('rnn-lstm', model.name)

  def test_gru_constructs(self):

    def _recurrent_layer_fn(x):
      return tf.keras.layers.GRU(x, return_sequences=True)

    model = models.create_recurrent_model(10, _recurrent_layer_fn, 'rnn-gru')
    self.assertIsInstance(model, tf.keras.Model)
    self.assertEqual('rnn-gru', model.name)

  def test_gru_fewer_parameters_than_lstm(self):

    def _gru_fn(x):
      return tf.keras.layers.GRU(x, return_sequences=True)

    def _lstm_fn(x):
      return tf.keras.layers.LSTM(x, return_sequences=True)

    gru_model = models.create_recurrent_model(10, _gru_fn, 'gru')
    lstm_model = models.create_recurrent_model(10, _lstm_fn, 'lstm')
    self.assertLess(gru_model.count_params(), lstm_model.count_params())


if __name__ == '__main__':
  absltest.main()
