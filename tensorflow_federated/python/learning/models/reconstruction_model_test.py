# Copyright 2023, The TensorFlow Federated Authors.
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
"""Tests for reconstruction_model.py."""

import collections
from typing import Optional

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.learning.models import reconstruction_model


def _build_two_layer_model() -> tf.keras.Model:
  return tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          5,
          input_shape=(5,),
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      tf.keras.layers.Dense(
          2,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
  ])


def _build_encapsulated_layer_model() -> tf.keras.Model:
  class _EncapsulatingLayer(tf.keras.layers.Layer):

    def __init__(self, layer_to_encapsulate: tf.keras.layers.Layer):
      super().__init__()
      self._layer_to_encapsulate = layer_to_encapsulate

    def call(self, x: tf.Tensor) -> tf.Tensor:
      return self._layer_to_encapsulate(x)

  class _EncapsulatedLayerModel(tf.keras.Model):
    """A Keras model with more complex layer interaction, useful for testing."""

    def __init__(self):
      super().__init__(name='encapsulated_layer_model')
      # Note: By making the following Layer a class variable of this Keras
      # model, it will be added to the layers being tracked by this Keras model.
      # If the Layer is used in other Layers of this Keras model, the variables
      # will be shared. This all works from the Keras perspective, but it does
      # illuminate complications in separating Reconstruction global and local
      # variables purely via layer. These complications are shown via tests
      # below.
      self.dense_to_encapsulate = tf.keras.layers.Dense(
          5,
          input_shape=(5,),
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      )
      self.encapsulated_dense = _EncapsulatingLayer(self.dense_to_encapsulate)
      self.regular_dense = tf.keras.layers.Dense(
          2,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
      del training  # Unused.
      del mask  # Unused.
      intermediate = self.encapsulated_dense(inputs)
      return self.regular_dense(intermediate)

  return _EncapsulatedLayerModel()


def _get_input_spec() -> collections.OrderedDict[str, tf.TensorSpec]:
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 5], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
  )


class ReconstructionModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_from_keras_model_with_global_and_local_layers(self):
    keras_model = _build_two_layer_model()
    local_layers = keras_model.layers[:1]
    global_layers = keras_model.layers[1:]
    input_spec = _get_input_spec()

    expected_local_trainable_vars = []
    for local_layer in local_layers:
      expected_local_trainable_vars.extend(local_layer.trainable_variables)

    expected_global_trainable_vars = []
    for global_layer in global_layers:
      expected_global_trainable_vars.extend(global_layer.trainable_variables)

    recon_model = reconstruction_model.ReconstructionModel.from_keras_model(
        keras_model,
        global_layers=global_layers,
        local_layers=local_layers,
        input_spec=input_spec,
    )

    self.assertFalse(
        reconstruction_model.ReconstructionModel.has_only_global_variables(
            recon_model
        )
    )

    self.assertLen(
        reconstruction_model.ReconstructionModel.get_local_variables(
            recon_model
        ).trainable,
        len(expected_local_trainable_vars),
    )
    for var, expected_var in zip(
        reconstruction_model.ReconstructionModel.get_local_variables(
            recon_model
        ).trainable,
        expected_local_trainable_vars,
    ):
      self.assertIs(var, expected_var)

    self.assertLen(
        reconstruction_model.ReconstructionModel.get_global_variables(
            recon_model
        ).trainable,
        len(expected_global_trainable_vars),
    )
    for var, expected_var in zip(
        reconstruction_model.ReconstructionModel.get_global_variables(
            recon_model
        ).trainable,
        expected_global_trainable_vars,
    ):
      self.assertIs(var, expected_var)

  def test_from_keras_model_with_only_global_layers(self):
    keras_model = _build_two_layer_model()
    local_layers = []
    global_layers = keras_model.layers
    input_spec = _get_input_spec()

    expected_global_trainable_vars = []
    for global_layer in global_layers:
      expected_global_trainable_vars.extend(global_layer.trainable_variables)

    recon_model = reconstruction_model.ReconstructionModel.from_keras_model(
        keras_model,
        global_layers=global_layers,
        local_layers=local_layers,
        input_spec=input_spec,
    )

    self.assertTrue(
        reconstruction_model.ReconstructionModel.has_only_global_variables(
            recon_model
        )
    )

    self.assertEmpty(
        reconstruction_model.ReconstructionModel.get_local_variables(
            recon_model
        ).trainable
    )

    self.assertLen(
        reconstruction_model.ReconstructionModel.get_global_variables(
            recon_model
        ).trainable,
        len(expected_global_trainable_vars),
    )
    for var, expected_var in zip(
        reconstruction_model.ReconstructionModel.get_global_variables(
            recon_model
        ).trainable,
        expected_global_trainable_vars,
    ):
      self.assertIs(var, expected_var)

  def test_forward_pass_is_same_regardless_of_global_local_layer_split(self):
    keras_model = _build_two_layer_model()
    input_spec = _get_input_spec()

    model_global_local = (
        reconstruction_model.ReconstructionModel.from_keras_model(
            keras_model,
            global_layers=keras_model.layers[1:],
            local_layers=keras_model.layers[:1],
            input_spec=input_spec,
        )
    )

    model_only_global = (
        reconstruction_model.ReconstructionModel.from_keras_model(
            keras_model,
            global_layers=keras_model.layers[1:],
            local_layers=keras_model.layers[:1],
            input_spec=input_spec,
        )
    )

    batch_input = {
        'x': tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        'y': tf.constant([[6.0, 7.0]]),
    }
    batch_output_global_local = model_global_local.forward_pass(batch_input)
    batch_output_only_global = model_only_global.forward_pass(batch_input)
    self.assertAllEqual(
        batch_output_global_local.predictions,
        batch_output_only_global.predictions,
    )
    self.assertAllEqual(
        batch_output_global_local.labels, batch_output_only_global.labels
    )
    self.assertEqual(
        batch_output_global_local.num_examples,
        batch_output_only_global.num_examples,
    )

  @parameterized.named_parameters(
      (
          'more_than_two_elements',
          [
              tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
              tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
              tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          ],
      ),
      (
          'dict_with_key_not_named_x',
          collections.OrderedDict(
              foo=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
              y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          ),
      ),
      (
          'dict_with_key_not_named_y',
          collections.OrderedDict(
              x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
              bar=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          ),
      ),
  )
  def test_bad_input_spec_raises_error(self, input_spec):
    keras_model = _build_two_layer_model()

    with self.assertRaises(ValueError):
      reconstruction_model.ReconstructionModel.from_keras_model(
          keras_model,
          global_layers=keras_model.layers[1:],
          local_layers=keras_model.layers[:1],
          input_spec=input_spec,
      )

  def test_from_keras_model_with_local_layer_not_in_model_raises_error(self):
    keras_model = _build_two_layer_model()
    input_spec = _get_input_spec()

    # A layer that is *not* in the above Keras model.
    random_other_layer = tf.keras.layers.Dense(42)
    random_other_layer.build((42))

    with self.assertRaises(ValueError):
      reconstruction_model.ReconstructionModel.from_keras_model(
          keras_model,
          global_layers=keras_model.layers[1:],
          local_layers=[random_other_layer] + keras_model.layers[:1],
          input_spec=input_spec,
      )

  def test_from_keras_model_with_global_layer_not_in_model_raises_error(self):
    keras_model = _build_two_layer_model()
    input_spec = _get_input_spec()

    # A layer that is *not* in the above Keras model.
    random_other_layer = tf.keras.layers.Dense(42)
    random_other_layer.build((42))

    with self.assertRaises(ValueError):
      reconstruction_model.ReconstructionModel.from_keras_model(
          keras_model,
          global_layers=[random_other_layer] + keras_model.layers[1:],
          local_layers=keras_model.layers[:1],
          input_spec=input_spec,
      )

  def test_from_keras_model_non_disjoint_global_and_local_vars_raises_error(
      self,
  ):
    keras_model = _build_encapsulated_layer_model()
    input_spec = _get_input_spec()

    # Confirm that first two layers of the model have same trainable variables.
    for layer_0_var, layer_1_var in zip(
        keras_model.layers[0].trainable_variables,
        keras_model.layers[1].trainable_variables,
    ):
      self.assertIs(layer_0_var, layer_1_var)

    # This will cause problems, because the global layers and local layers have
    # overlapping variables. Test that this is caught and raises a ValueError.
    with self.assertRaises(ValueError):
      reconstruction_model.ReconstructionModel.from_keras_model(
          keras_model,
          global_layers=keras_model.layers[1:],
          local_layers=keras_model.layers[:1],
          input_spec=input_spec,
      )


if __name__ == '__main__':
  tf.test.main()
