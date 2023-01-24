# Copyright 2020, Google LLC.
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

import tensorflow as tf

from tensorflow_federated.python.simulation.models import mobilenet_v2


class CreateMobilenetInputValidationTest(tf.test.TestCase):

  def test_non_tuple_input_shape_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        'input_shape must be a tuple of length 3 containing positive integers',
    ):
      mobilenet_v2.create_mobilenet_v2(input_shape=100)

  def test_input_shape_with_negative_value_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        'input_shape must be a tuple of length 3 containing positive integers',
    ):
      mobilenet_v2.create_mobilenet_v2(input_shape=(10, -1, 2))

  def test_non_length_3_input_shape_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        'input_shape must be a tuple of length 3 containing positive integers',
    ):
      mobilenet_v2.create_mobilenet_v2(input_shape=(10, 2))

  def test_nonpositive_alpha_raises(self):
    with self.assertRaisesRegex(ValueError, 'alpha must be positive'):
      mobilenet_v2.create_mobilenet_v2(input_shape=(32, 32, 3), alpha=-1.0)

  def test_unsupported_pooling_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'pooling must be one of avg or max'
    ):
      mobilenet_v2.create_mobilenet_v2(input_shape=(32, 32, 3), pooling='min')

  def test_negative_num_groups_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'num_groups must be a positive integer'
    ):
      mobilenet_v2.create_mobilenet_v2(input_shape=(32, 32, 3), num_groups=-1)

  def test_negative_dropout_prob_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'dropout_prob must be `None` or a float between 0 and 1'
    ):
      mobilenet_v2.create_mobilenet_v2(
          input_shape=(32, 32, 3), dropout_prob=-0.5
      )

  def test_negative_num_classes_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'num_classes must be a positive integer'
    ):
      mobilenet_v2.create_mobilenet_v2(input_shape=(32, 32, 3), num_classes=-1)


class MobileNetModelTest(tf.test.TestCase):

  def test_constructs_keras_model(self):
    model = mobilenet_v2.create_mobilenet_v2(input_shape=(100, 100, 50))
    self.assertIsInstance(model, tf.keras.Model)

  def test_alpha_changes_num_model_parameters(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), num_classes=1000
    )
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), alpha=0.5, num_classes=1000
    )
    model3 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), alpha=2.0, num_classes=1000
    )
    self.assertLess(model2.count_params(), model1.count_params())
    self.assertLess(model1.count_params(), model3.count_params())

  def test_num_groups_does_not_change_num_model_parameters(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), num_classes=1000
    )
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), num_groups=4, num_classes=1000
    )
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertEqual(model1.count_params(), model2.count_params())

  def test_pooling_does_not_change_num_model_parameters(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), pooling='avg', num_classes=1000
    )
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), pooling='max', num_classes=1000
    )
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertEqual(model1.count_params(), model2.count_params())

  def test_dropout_increases_num_model_layers(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), dropout_prob=0.5
    )
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), dropout_prob=0.2
    )
    model3 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), dropout_prob=None
    )
    self.assertEqual(len(model1.layers), len(model2.layers))
    self.assertGreater(len(model1.layers), len(model3.layers))


if __name__ == '__main__':
  tf.test.main()
