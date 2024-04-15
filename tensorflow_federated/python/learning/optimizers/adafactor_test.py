# Copyright 2024, The TensorFlow Federated Authors.
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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.learning.optimizers import adafactor

_TEST_PARAMETERS = (
    dict(
        testcase_name='defaults',
        learning_rate=0.1,
        beta_2_decay=-0.8,
        epsilon_1=1e-30,
        epsilon_2=1e-3,
        clip_threshold=1.0,
    ),
    dict(
        testcase_name='non_default',
        learning_rate=0.001,
        beta_2_decay=-0.6,
        epsilon_1=1e-10,
        epsilon_2=1e-2,
        clip_threshold=0.5,
    ),
    dict(
        testcase_name='no_lr_decay',
        learning_rate=0.001,
        beta_2_decay=-0.6,
        epsilon_1=1e-10,
        epsilon_2=1e-2,
        clip_threshold=0.5,
        relative_step=False,
    ),
)


class AdafactorTest(parameterized.TestCase, tf.test.TestCase):

  def test_initialization_and_step_in_eager_mode(self):
    optimizer = adafactor.build_adafactor(learning_rate=0.003)
    weights = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    state = optimizer.initialize(tf.TensorSpec.from_tensor(weights))
    # Assert the shapes of the factorized moments are zero for
    # weights with less than rank two.
    self.assertLen(state['moments'], 1)
    self.assertSequenceEqual(state['moments'][0].r.shape, [1])
    self.assertSequenceEqual(state['moments'][0].c.shape, [3])
    self.assertSequenceEqual(state['moments'][0].v.shape, [1, 3])

    @tf.function
    def f(x):
      return tf.reduce_sum(x**2.0)

    expected_values = [13.92, 13.84, 13.76, 13.70, 13.61]
    for expected_value in expected_values:
      with tf.GradientTape() as tape:
        tape.watch(weights)
        y = f(weights)
      gradients = tape.gradient(y, weights)
      state, weights = optimizer.next(state, weights, gradients)
      self.assertNear(f(weights), expected_value, err=0.01)

  def test_initialization_and_step_in_traced_computation(self):
    optimizer = adafactor.build_adafactor(learning_rate=0.003)
    weights = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)

    @tensorflow_computation.tf_computation
    def init():
      return optimizer.initialize(tf.TensorSpec.from_tensor(weights))

    state = init()
    # Assert the shapes of the factorized moments are zero for
    # weights with less than rank two.
    self.assertLen(state['moments'], 1)
    self.assertSequenceEqual(state['moments'][0].r.shape, [1])
    self.assertSequenceEqual(state['moments'][0].c.shape, [3])
    self.assertSequenceEqual(state['moments'][0].v.shape, [1, 3])

    @tf.function
    def f(x):
      return tf.reduce_sum(x**2.0)

    @tensorflow_computation.tf_computation
    @tf.function
    def step(state, weights):
      with tf.GradientTape() as tape:
        tape.watch(weights)
        y = f(weights)
      gradients = tape.gradient(y, weights)
      return optimizer.next(state, weights, gradients)

    expected_values = [13.92, 13.84, 13.76, 13.70, 13.61]
    for expected_value in expected_values:
      state, weights = step(state, weights)
      self.assertNear(f(weights), expected_value, err=0.01)

  @parameterized.named_parameters(*_TEST_PARAMETERS)
  def test_single_rank_compare_to_keras(self, **optimizer_kwargs):
    test_shape = [1]
    tff_optimizer = adafactor.build_adafactor(**optimizer_kwargs)
    tff_state = tff_optimizer.initialize(
        tf.TensorSpec(shape=test_shape, dtype=tf.float32)
    )
    # Assert the shapes of the factorized moments are zero for
    # weights with less than rank two.
    self.assertLen(tff_state['moments'], 1)
    self.assertSequenceEqual(tff_state['moments'][0].r.shape, [0])
    self.assertSequenceEqual(tff_state['moments'][0].c.shape, [0])
    self.assertSequenceEqual(tff_state['moments'][0].v.shape, test_shape)

    tff_weights = tf.zeros(shape=test_shape)
    keras_optimizer = tf.keras.optimizers.Adafactor(**optimizer_kwargs)
    keras_variable = tf.Variable(initial_value=tff_weights, dtype=tf.float32)
    gradient = tf.ones_like(tff_weights) * -0.1
    for _ in range(10):
      tff_state, tff_weights = tff_optimizer.next(
          state=tff_state, weights=tff_weights, gradients=gradient
      )
      keras_optimizer.apply_gradients([(gradient, keras_variable)])
      self.assertAllClose(tff_weights, keras_variable)

  @parameterized.named_parameters(*_TEST_PARAMETERS)
  def test_double_rank_compare_to_keras(self, **optimizer_kwargs):
    test_shape = [10, 20]
    tff_optimizer = adafactor.build_adafactor(**optimizer_kwargs)
    tff_state = tff_optimizer.initialize(
        tf.TensorSpec(shape=test_shape, dtype=tf.float32)
    )
    # Assert the shapes of the factorized moments are as expected for weights
    # with rank 2 or more.
    self.assertSequenceEqual(tff_state['moments'][0].r.shape, test_shape[:1])
    self.assertSequenceEqual(tff_state['moments'][0].c.shape, test_shape[1:])
    self.assertSequenceEqual(tff_state['moments'][0].v.shape, test_shape)

    tff_weights = tf.zeros(shape=test_shape)
    keras_optimizer = tf.keras.optimizers.Adafactor(**optimizer_kwargs)
    keras_variable = tf.Variable(initial_value=tff_weights, dtype=tf.float32)
    gradient = tf.ones_like(tff_weights) * -0.1
    for _ in range(10):
      tff_state, tff_weights = tff_optimizer.next(
          state=tff_state, weights=tff_weights, gradients=gradient
      )
      keras_optimizer.apply_gradients([(gradient, keras_variable)])
      self.assertAllClose(tff_weights, keras_variable)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  absltest.main()
