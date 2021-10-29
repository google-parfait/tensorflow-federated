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
from tensorflow_federated.python.learning.optimizers import sgdm

_SCALAR_SPEC = tf.TensorSpec([1], tf.float32)
_STRUCT_SPEC = [tf.TensorSpec([2], tf.float32), tf.TensorSpec([3], tf.float32)]
_NESTED_SPEC = [
    tf.TensorSpec([10], tf.float32),
    [tf.TensorSpec([20], tf.float32), [tf.TensorSpec([30], tf.float32)]]
]


class SGDTest(optimizer_test_utils.TestCase, parameterized.TestCase):

  def test_state_structure(self):
    optimizer = sgdm.build_sgdm(0.01)
    state = optimizer.initialize(_SCALAR_SPEC)
    self.assertLen(state, 1)
    self.assertIn(sgdm.LEARNING_RATE_KEY, state)

  def test_state_structure_momentum(self):
    optimizer = sgdm.build_sgdm(0.01, momentum=0.9)
    state = optimizer.initialize(_SCALAR_SPEC)
    self.assertLen(state, 3)
    self.assertIn(sgdm.LEARNING_RATE_KEY, state)
    self.assertIn(sgdm.MOMENTUM_KEY, state)
    self.assertIn(sgdm.ACCUMULATOR_KEY, state)

  def test_math_no_momentum(self):
    weights = tf.constant([1.0], tf.float32)
    gradients = tf.constant([2.0], tf.float32)
    optimizer = sgdm.build_sgdm(0.01)
    history = [weights]

    state = optimizer.initialize(_SCALAR_SPEC)

    for _ in range(4):
      state, weights = optimizer.next(state, weights, gradients)
      history.append(weights)
    self.assertAllClose([[1.0], [0.98], [0.96], [0.94], [0.92]], history)

  def test_math_momentum_0_5(self):
    weights = tf.constant([1.0], tf.float32)
    gradients = tf.constant([2.0], tf.float32)
    optimizer = sgdm.build_sgdm(0.01, momentum=0.5)
    history = [weights]

    state = optimizer.initialize(_SCALAR_SPEC)

    for _ in range(4):
      state, weights = optimizer.next(state, weights, gradients)
      history.append(weights)
    self.assertAllClose([[1.0], [0.98], [0.95], [0.915], [0.8775]], history)

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC, None),
      ('struct_spec', _STRUCT_SPEC, None),
      ('nested_spec', _NESTED_SPEC, None),
      ('scalar_spec_and_momentum', _SCALAR_SPEC, 0.5),
      ('struct_spec_and_momentum', _STRUCT_SPEC, 0.5),
      ('nested_spec_and_momentum', _NESTED_SPEC, 0.5),
  )
  def test_executes_with(self, spec, momentum):
    weights = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    gradients = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    optimizer = sgdm.build_sgdm(0.01, momentum=momentum)

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
    optimizer = sgdm.build_sgdm(0.5)

    state = optimizer.initialize(tf.TensorSpec([4, 2]))
    _, weights = optimizer.next(state, weights, gradients)
    self.assertAllClose([[0.5, 0.5], [1.0, 1.0], [0.5, 0.5], [1.0, 1.0]],
                        weights)

  @parameterized.named_parameters(('no_momentum', None), ('momentum_0_5', 0.5))
  def test_convergence(self, momentum):
    init_w, fn, grad_fn = optimizer_test_utils.test_quadratic_problem()
    weights = init_w()
    self.assertGreater(fn(weights), 5.0)

    optimizer = sgdm.build_sgdm(0.1, momentum=momentum)
    state = optimizer.initialize(tf.TensorSpec(weights.shape, weights.dtype))

    for _ in range(100):
      gradients = grad_fn(weights)
      state, weights = optimizer.next(state, weights, gradients)
    self.assertLess(fn(weights), 0.005)

  @parameterized.named_parameters(('lr_0_1_m_none', 0.1, None),
                                  ('lr_0_01_m_0_9', 0.01, 0.9))
  def test_build_sgdm(self, learning_rate, momentum):
    optimizer = sgdm.build_sgdm(learning_rate, momentum)
    self.assertIsInstance(optimizer, optimizer_base.Optimizer)
    self.assertEqual(learning_rate, optimizer._lr)
    self.assertEqual(momentum, optimizer._momentum)

  @parameterized.named_parameters(('lr_0_1_m_0', 0.1, 0.),
                                  ('lr_0_01_m_0_9', 0.01, 0.9))
  def test_match_keras(self, learning_rate, momentum):
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
    tff_optimizer_fn = lambda: sgdm.build_sgdm(learning_rate, momentum)

    def keras_optimizer_fn():
      return tf.keras.optimizers.SGD(learning_rate, momentum)

    self.assert_optimizers_numerically_close(model_variables_fn, gradients,
                                             tff_optimizer_fn,
                                             keras_optimizer_fn)


if __name__ == '__main__':
  tf.test.main()
