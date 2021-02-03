# Copyright 2019, Google LLC.
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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.simulation.baselines.stackoverflow import word_prediction_models


class CreateRecurrentModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_nonpositive_vocab_size_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'vocab_size must be a positive integer'):
      word_prediction_models.create_recurrent_model(vocab_size=-2)

  def test_nonpositive_embedding_size_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'embedding_size must be a positive integer'):
      word_prediction_models.create_recurrent_model(
          vocab_size=3, embedding_size=-3)

  def test_nonpositive_num_lstm_layers_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_lstm_layers must be a positive integer'):
      word_prediction_models.create_recurrent_model(
          vocab_size=3, num_lstm_layers=-2)

  def test_nonpositive_lstm_size_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'lstm_size must be a positive integer'):
      word_prediction_models.create_recurrent_model(vocab_size=3, lstm_size=0)

  @parameterized.named_parameters(('vocab_size1', 1), ('vocab_size2', 801),
                                  ('vocab_size3', 10000))
  def test_output_shape_matches_vocab_size(self, vocab_size):
    model = word_prediction_models.create_recurrent_model(vocab_size=vocab_size)
    self.assertEqual(model.output_shape, (None, None, vocab_size))

  @parameterized.named_parameters(('param1', 1, True), ('param2', 1, False),
                                  ('param3', 3, True), ('param4', 3, False),
                                  ('param5', 7, True), ('param6', 7, False))
  def test_num_layers(self, num_lstm_layers, shared_embedding):
    model = word_prediction_models.create_recurrent_model(
        vocab_size=3,
        num_lstm_layers=num_lstm_layers,
        shared_embedding=shared_embedding)
    # There is an input layer, an embedding layer, a final dense layer, and
    # `num_lstm_layers` pairs of LSTM/Dense layers.
    expected_num_layers = 2 + 2 * num_lstm_layers + 1
    self.assertLen(model.layers, expected_num_layers)

  def test_shared_embedding_matches_input_and_output_weights(self):
    model = word_prediction_models.create_recurrent_model(
        vocab_size=3, shared_embedding=True)
    embedding_weights = model.layers[1].weights
    output_weights = model.layers[-1].weights
    self.assertAllClose(embedding_weights, output_weights)

  def test_shared_embedding_returns_dense_gradient_in_graph_mode(self):
    batch_size = 2
    sequence_length = 20
    graph = tf.Graph()
    with graph.as_default():
      batch_x = tf.ones((batch_size, sequence_length), dtype=tf.int32)
      batch_y = tf.ones((batch_size, sequence_length), dtype=tf.int32)
      model = word_prediction_models.create_recurrent_model(
          vocab_size=3, shared_embedding=True)
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        loss = loss_fn(y_true=batch_y, y_pred=predictions)
      embedding_gradient = tape.gradient(loss, model.trainable_variables[0])
      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      embedding_grad = sess.run(embedding_gradient)

    self.assertTrue(tf.reduce_all(tf.norm(embedding_grad, axis=1) > 0.0))


if __name__ == '__main__':
  tf.test.main()
