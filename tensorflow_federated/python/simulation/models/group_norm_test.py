# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.simulation.models import group_norm


class GroupNormTest(tf.test.TestCase, parameterized.TestCase):

  def test_layer_has_no_weights(self):
    # Check if weights get initialized correctly
    group_norm_layer = group_norm.GroupNormalization(groups=1)
    group_norm_layer.build((None, 3, 4))
    self.assertEmpty(group_norm_layer.trainable_weights)
    self.assertEmpty(group_norm_layer.weights)

  def test_layer_applies_normalization_correctly(self):
    input_shape = (1, 4)
    reshaped_inputs = tf.constant([[[2.0, 2.0], [3.0, 3.0]]])
    layer = group_norm.GroupNormalization(groups=2, axis=1)
    normalized_input = layer._apply_normalization(reshaped_inputs, input_shape)
    self.assertAllClose(normalized_input, np.array([[[0.0, 0.0], [0.0, 0.0]]]))

  @parameterized.named_parameters(
      ('input1', 2, 5, (10, 10, 10), (10, 10, 5, 2)),
      ('input2', 1, 2, (10, 10, 10), (10, 2, 5, 10)),
      ('input3', 1, 10, (10, 10, 10), (10, 10, 10)),
      ('input4', 1, 1, (10, 10, 10), (10, 1, 10, 10)),
  )
  def test_reshape(self, axis, group, input_shape, expected_shape):
    group_layer = group_norm.GroupNormalization(groups=group, axis=axis)
    group_layer.build(input_shape)

    inputs = np.ones(input_shape)
    tensor_input_shape = tf.convert_to_tensor(input_shape)
    _, group_shape = group_layer._reshape_into_groups(inputs, (10, 10, 10),
                                                      tensor_input_shape)
    self.assertAllEqual(group_shape, expected_shape)

  @parameterized.named_parameters(('groups_1', 1), ('groups_10', 10),
                                  ('groups_20', 20))
  def test_model_with_groupnorm_layer_trains(self, groups):
    # Check if Axis is working for CONV nets
    np.random.seed(0)
    model = tf.keras.models.Sequential()
    model.add(
        group_norm.GroupNormalization(
            axis=1, groups=groups, input_shape=(20, 20, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss='mse')
    x = np.random.randint(1000, size=(10, 20, 20, 3))
    y = np.random.randint(1000, size=(10, 1))
    model.fit(x=x, y=y, epochs=1)

  @parameterized.named_parameters(
      ('input1', 10, 1),
      ('input2', 10, 2),
      ('input3', 10, 5),
      ('input4', 10, 10),
  )
  def test_groups_have_mean_0_var_leq_1_on_1d_data(self, tensor_dim, groups):
    # After group normalization, component groups should have mean 0 and
    # variance at most 1 (note that while generally it should be variance 1
    # exactly, certain tensors will have smaller variance, potentially even 0.
    # This test ensures that the output of GroupNorm on 1d tensors has groups
    # with mean 0 and variance at most 1.
    model = tf.keras.models.Sequential()
    norm = group_norm.GroupNormalization(
        input_shape=(tensor_dim,), groups=groups)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')
    group_size = tensor_dim // groups
    expected_group_means = tf.zeros((1, groups), dtype=tf.float32)
    # We seed our random input for reproducibility, but this test should
    # succeed independently of the input to the GroupNorm layer.
    for random_seed in range(100):
      np.random.seed(random_seed)
      x = np.random.normal(loc=2.0, scale=3.0, size=(1, 10))
      out = model.predict(x)
      reshaped_out = tf.reshape(out, (1, groups, group_size))
      group_means = tf.math.reduce_mean(reshaped_out, axis=2)
      group_variances = tf.math.reduce_variance(reshaped_out, axis=2)
      self.assertAllClose(group_means, expected_group_means, atol=1e-2)
      self.assertAllLessEqual(group_variances, 1.0)

  @parameterized.named_parameters(
      ('input1', (10, 3), 1),
      ('input2', (10, 5), 2),
      ('input3', (10, 3), 5),
      ('input4', (10, 1), 10),
  )
  def test_groups_have_mean_0_var_leq_1_on_2d_data(self, input_shape, groups):
    # After group normalization, component groups should have mean 0 and
    # variance at most 1 (note that while generally it should be variance 1
    # exactly, certain tensors will have smaller variance, potentially even 0.
    # This test ensures that the output of GroupNorm on 2d tensors has groups
    # with mean 0 and variance at most 1.
    model = tf.keras.models.Sequential()
    norm = group_norm.GroupNormalization(
        axis=1, groups=groups, input_shape=input_shape)
    batched_input_shape = (1,) + input_shape
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')
    group_size = (input_shape[0] // groups) * input_shape[1]
    expected_group_means = tf.zeros((1, groups), dtype=tf.float32)
    # We seed our random input for reproducibility, but this test should
    # succeed independently of the input to the GroupNorm layer.
    for random_seed in range(100):
      np.random.seed(random_seed)
      x = np.random.normal(loc=2.0, scale=3.0, size=batched_input_shape)
      out = model.predict(x)
      reshaped_out = tf.reshape(out, (1, groups, group_size))
      group_means = tf.math.reduce_mean(reshaped_out, axis=2)
      group_variances = tf.math.reduce_variance(reshaped_out, axis=2)
      self.assertAllClose(group_means, expected_group_means, atol=1e-2)
      self.assertAllLessEqual(group_variances, 1.0)


if __name__ == '__main__':
  tf.test.main()
