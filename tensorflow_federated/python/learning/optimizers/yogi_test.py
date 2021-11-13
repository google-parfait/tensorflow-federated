# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import optimizer_test_utils
from tensorflow_federated.python.learning.optimizers import yogi

_SCALAR_SPEC = tf.TensorSpec([1], tf.float32)
_STRUCT_SPEC = [tf.TensorSpec([2], tf.float32), tf.TensorSpec([3], tf.float32)]
_NESTED_SPEC = [
    tf.TensorSpec([10], tf.float32),
    [tf.TensorSpec([20], tf.float32), [tf.TensorSpec([30], tf.float32)]]
]


class YogiTest(optimizer_test_utils.TestCase, parameterized.TestCase):

  def test_state_structure(self):
    optimizer = yogi.build_yogi(0.01)
    state = optimizer.initialize(_SCALAR_SPEC)
    self.assertLen(state, 7)
    self.assertIn(yogi._LEARNING_RATE_KEY, state)
    self.assertIn(yogi._BETA_1_KEY, state)
    self.assertIn(yogi._BETA_2_KEY, state)
    self.assertIn(yogi._EPSILON_KEY, state)
    self.assertIn(yogi._STEP_KEY, state)
    self.assertIn(yogi._PRECONDITIONER_KEY, state)
    self.assertIn(yogi._ACCUMULATOR_KEY, state)

  def test_math(self):
    weights = tf.constant([1.0], tf.float32)
    gradients = tf.constant([2.0], tf.float32)
    optimizer = yogi.build_yogi(
        0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=0.0,
        initial_preconditioner_value=0.0)
    history = [weights]

    state = optimizer.initialize(_SCALAR_SPEC)

    for _ in range(4):
      state, weights = optimizer.next(state, weights, gradients)
      history.append(weights)
    self.assertAllClose(
        [[1.0], [0.9000007], [0.8000267], [0.700077], [0.600153]], history)

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC),
      ('struct_spec', _STRUCT_SPEC),
      ('nested_spec', _NESTED_SPEC),
  )
  def test_executes_with(self, spec):
    weights = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    gradients = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    optimizer = yogi.build_yogi(0.01)

    state = optimizer.initialize(spec)
    for _ in range(10):
      state, weights = optimizer.next(state, weights, gradients)

    tf.nest.map_structure(lambda w: self.assertTrue(all(tf.math.is_finite(w))),
                          weights)

  def test_executes_with_indexed_slices(self):
    # TF can represent gradients as tf.IndexedSlices. This test makes sure this
    # case is supported by the optimizer.
    weights = tf.ones([4, 2])
    gradients = tf.IndexedSlices(
        values=tf.constant([[1.0, 1.0], [1.0, 1.0]]),
        indices=tf.constant([0, 2]),
        dense_shape=tf.constant([4, 2]))
    # Always-zero preconditioner and accumulator, for simplicity of this test.
    optimizer = yogi.build_yogi(
        0.5,
        beta_1=0.0,
        beta_2=0.0,
        epsilon=1e-7,
        initial_preconditioner_value=0.0)

    state = optimizer.initialize(tf.TensorSpec([4, 2]))
    _, weights = optimizer.next(state, weights, gradients)
    self.assertAllClose([[0.5, 0.5], [1.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
                        weights)

  def test_convergence(self):
    init_w, fn, grad_fn = optimizer_test_utils.test_quadratic_problem()
    weights = init_w()
    self.assertGreater(fn(weights), 5.0)

    optimizer = yogi.build_yogi(0.5)
    state = optimizer.initialize(tf.TensorSpec(weights.shape, weights.dtype))

    for _ in range(100):
      gradients = grad_fn(weights)
      state, weights = optimizer.next(state, weights, gradients)
    self.assertLess(fn(weights), 0.005)

  def test_build_yogi(self):
    optimizer = yogi.build_yogi(0.01)
    self.assertIsInstance(optimizer, optimizer_base.Optimizer)

  # pylint: disable=pointless-string-statement
  # Following test works but commented out as depending on tensorflow_addons
  # is lot of extra work for OSS.
  """Uncomment to run.

  def test_match_keras(self):

    # f#r#o#m tensorflow_addons i#m#p#o#r#t optimizers as tfa_optimizers
    weight_spec = [
        tf.TensorSpec([10, 2], tf.float32),
        tf.TensorSpec([2], tf.float32)
    ]
    steps = 10
    genarator = tf.random.Generator.from_seed(2021)

    def random_vector():
      return [
          genarator.normal(shape=s.shape, dtype=s.dtype) for s in weight_spec
      ]

    intial_weight = random_vector()
    model_variables_fn = lambda: [tf.Variable(v) for v in intial_weight]
    gradients = [random_vector() for _ in range(steps)]
    tff_optimizer_fn = lambda: yogi.build_yogi(0.01, 0.9, 0.999)
    keras_optimizer_fn = lambda: tfa_optimizers.Yogi(0.01, 0.9, 0.999)

    self.assert_optimizers_numerically_close(model_variables_fn, gradients,
                                             tff_optimizer_fn,
                                             keras_optimizer_fn)
  """

  @parameterized.named_parameters(
      ('negative_lr', -1.0, 0.9, 0.999, 1e-7, 'learning rate'),
      ('negative_beta_1', 1.0, -0.9, 0.999, 1e-7, 'Beta'),
      ('beta_1_one', 1.0, 1.0, 0.999, 1e-7, 'Beta'),
      ('negative_beta_2', 1.0, 0.9, -0.999, 1e-7, 'Beta'),
      ('beta_2_one', 1.0, 0.9, 1.0, 1e-7, 'Beta'),
      ('negative_epsilon', 1.0, 0.9, 0.999, -1e-7, 'epsilon'),
  )
  def test_invalid_args_raises(self, lr, beta_1, beta_2, epsilon, regex):
    with self.assertRaisesRegex(ValueError, regex):
      yogi.build_yogi(lr, beta_1, beta_2, epsilon)

  def test_weights_gradients_mismatch_raises(self):
    optimizer = yogi.build_yogi(0.1)
    state = optimizer.initialize(_SCALAR_SPEC)
    with self.assertRaises(ValueError):
      optimizer.next(state, tf.zeros([1]), tf.zeros([2]))

  def test_initialize_next_weights_mismatch_raises(self):
    optimizer = yogi.build_yogi(0.1)
    state = optimizer.initialize(_SCALAR_SPEC)
    with self.assertRaises(ValueError):
      optimizer.next(state, tf.zeros([2]), tf.zeros([2]))


if __name__ == '__main__':
  tf.test.main()
