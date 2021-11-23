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

from tensorflow_federated.python.simulation.baselines.emnist import emnist_models


class ModelCollectionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cnn_dropout', emnist_models.create_conv_dropout_model),
      ('cnn', emnist_models.create_original_fedavg_cnn_model),
      ('deterministic_cnn', emnist_models.create_determnistic_cnn_model),
      ('two_hidden_layer_model', emnist_models.create_two_hidden_layer_model),
  )
  def test_char_prediction_models_input_shape(self, model_builder):
    model = model_builder(only_digits=False)
    self.assertEqual(model.input_shape, (None, 28, 28, 1))

  @parameterized.named_parameters(
      ('cnn_dropout', emnist_models.create_conv_dropout_model),
      ('cnn', emnist_models.create_original_fedavg_cnn_model),
      ('deterministic_cnn', emnist_models.create_determnistic_cnn_model),
      ('two_hidden_layer_model', emnist_models.create_two_hidden_layer_model),
  )
  def test_char_prediction_models_input_shape_only_digits(self, model_builder):
    model = model_builder(only_digits=True)
    self.assertEqual(model.input_shape, (None, 28, 28, 1))

  @parameterized.named_parameters(
      ('cnn_dropout', emnist_models.create_conv_dropout_model),
      ('cnn', emnist_models.create_original_fedavg_cnn_model),
      ('deterministic_cnn', emnist_models.create_determnistic_cnn_model),
      ('two_hidden_layer_model', emnist_models.create_two_hidden_layer_model),
  )
  def test_char_prediction_models_output_shape(self, model_builder):
    image = tf.random.normal([3, 28, 28, 1])
    model = model_builder(only_digits=False)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [3, 62])

  @parameterized.named_parameters(
      ('cnn_dropout', emnist_models.create_conv_dropout_model),
      ('cnn', emnist_models.create_original_fedavg_cnn_model),
      ('deterministic_cnn', emnist_models.create_determnistic_cnn_model),
      ('two_hidden_layer_model', emnist_models.create_two_hidden_layer_model),
  )
  def test_char_prediction_models_output_shape_only_digits(self, model_builder):
    image = tf.random.normal([4, 28, 28, 1])
    model = model_builder(only_digits=True)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [4, 10])

  @parameterized.named_parameters(
      ('only_digits_true', True),
      ('only_digits_false', False),
  )
  def test_non_dropout_cnn_models_have_same_shape(self, only_digits):
    model1 = emnist_models.create_original_fedavg_cnn_model(
        only_digits=only_digits)
    model2 = emnist_models.create_determnistic_cnn_model(
        only_digits=only_digits)
    for x, y in zip(model1.variables, model2.variables):
      self.assertEqual(x.shape, y.shape)

  @parameterized.named_parameters(
      ('only_digits_true', True),
      ('only_digits_false', False),
  )
  def test_deterministic_model_has_all_zero_variables(self, only_digits):
    model = emnist_models.create_determnistic_cnn_model(only_digits=only_digits)
    all_zero_structure = tf.nest.map_structure(tf.zeros_like, model.variables)
    self.assertAllClose(all_zero_structure, model.variables)

  def test_2nn_raises_on_nonpositive_hidden_units(self):
    with self.assertRaisesRegex(ValueError,
                                'hidden_units must be a positive integer'):
      emnist_models.create_two_hidden_layer_model(
          only_digits=True, hidden_units=0)

  def test_2nn_number_of_parameters(self):
    model = emnist_models.create_two_hidden_layer_model(
        only_digits=True, hidden_units=200)

    # We calculate the number of parameters based on the fact that given densely
    # connected layers of size n and m with bias units, there are (n+1)m
    # parameters between these layers. The network above should have layers of
    # size 28*28, 200, 200, and 10.
    num_model_params = (28 * 28 + 1) * 200 + 201 * 200 + 201 * 10
    self.assertEqual(model.count_params(), num_model_params)

  def test_autoencoder_model_input_shape(self):
    model = emnist_models.create_autoencoder_model()
    self.assertEqual(model.input_shape, (None, 784))

  def test_autoencoder_model_output_shape(self):
    image = tf.random.normal([4, 784])
    model = emnist_models.create_autoencoder_model()
    reconstructed_image = model(image)
    num_model_params = 2837314
    self.assertIsNotNone(reconstructed_image)
    self.assertEqual(reconstructed_image.shape, [4, 784])
    self.assertEqual(model.count_params(), num_model_params)


if __name__ == '__main__':
  tf.test.main()
