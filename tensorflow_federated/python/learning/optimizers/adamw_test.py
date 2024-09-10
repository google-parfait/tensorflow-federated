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

import collections
import copy

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.learning.optimizers import adamw
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import optimizer_test_utils

_SCALAR_SPEC = tf.TensorSpec([1], tf.float32)
_STRUCT_SPEC = [tf.TensorSpec([2], tf.float32), tf.TensorSpec([3], tf.float32)]
_NESTED_SPEC = [
    tf.TensorSpec([10], tf.float32),
    [tf.TensorSpec([20], tf.float32), [tf.TensorSpec([30], tf.float32)]],
]


class AdamTest(optimizer_test_utils.TestCase, parameterized.TestCase):

  def test_state_structure(self):
    optimizer = adamw.build_adamw(0.01)
    state = optimizer.initialize(_SCALAR_SPEC)
    self.assertLen(state, 8)
    self.assertIn(optimizer_base.LEARNING_RATE_KEY, state)
    self.assertIn(adamw._BETA_1_KEY, state)
    self.assertIn(adamw._BETA_2_KEY, state)
    self.assertIn(adamw._EPSILON_KEY, state)
    self.assertIn(adamw._STEP_KEY, state)
    self.assertIn(adamw._PRECONDITIONER_KEY, state)
    self.assertIn(adamw._ACCUMULATOR_KEY, state)
    self.assertIn(adamw._WEIGHT_DECAY_KEY, state)

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC),
      ('struct_spec', _STRUCT_SPEC),
      ('nested_spec', _NESTED_SPEC),
  )
  def test_executes_with(self, spec):
    weights = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    gradients = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    optimizer = adamw.build_adamw(0.01)

    state = optimizer.initialize(spec)
    for _ in range(10):
      state, weights = optimizer.next(state, weights, gradients)

    tf.nest.map_structure(
        lambda w: self.assertTrue(all(tf.math.is_finite(w))), weights
    )

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC),
      ('struct_spec', _STRUCT_SPEC),
      ('nested_spec', _NESTED_SPEC),
  )
  def test_skips_none_gradients(self, spec):
    weights = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
    gradients = tf.nest.map_structure(lambda s: None, spec)
    optimizer = adamw.build_adamw(0.01)

    state = optimizer.initialize(spec)
    updated_state, updated_weights = optimizer.next(state, weights, gradients)
    state[adamw._STEP_KEY] += 1

    tf.nest.map_structure(self.assertAllEqual, weights, updated_weights)
    tf.nest.map_structure(self.assertAllEqual, state, updated_state)

  @parameterized.named_parameters(
      ('empty_list', []),
      ('empty_dict', {}),
      ('empty_nested_structure', [([], []), {}]),
  )
  def test_behavior_on_empty_tree(self, structure):
    weights = gradients = structure
    optimizer = adamw.build_adamw(0.01)

    state = optimizer.initialize(weights)
    updated_state, updated_weights = optimizer.next(state, weights, gradients)
    state[adamw._STEP_KEY] += 1

    self.assertEqual(updated_state, state)
    self.assertEqual(updated_weights, weights)

  def test_executes_with_indexed_slices(self):
    # TF can represent gradients as tf.IndexedSlices. This test makes sure this
    # case is supported by the optimizer.
    weights = tf.ones([4, 2])
    gradients = tf.IndexedSlices(
        values=tf.constant([[1.0, 1.0], [1.0, 1.0]]),
        indices=tf.constant([0, 2]),
        dense_shape=tf.constant([4, 2]),
    )
    # Always-zero preconditioner and accumulator, for simplicity of this test.
    optimizer = adamw.build_adamw(0.5, beta_1=0.0, beta_2=0.0)

    state = optimizer.initialize(tf.TensorSpec([4, 2]))
    _, weights = optimizer.next(state, weights, gradients)
    self.assertAllClose(
        [[0.498, 0.498], [0.998, 0.998], [0.498, 0.498], [0.998, 0.998]],
        weights,
    )

  def test_convergence(self):
    init_w, fn, grad_fn = optimizer_test_utils.test_quadratic_problem()
    weights = init_w()
    self.assertGreater(fn(weights), 5.0)

    optimizer = adamw.build_adamw(0.5)
    state = optimizer.initialize(tf.TensorSpec(weights.shape, weights.dtype))

    for _ in range(100):
      gradients = grad_fn(weights)
      state, weights = optimizer.next(state, weights, gradients)
    self.assertLess(fn(weights), 0.005)

  def test_build_adamw(self):
    optimizer = adamw.build_adamw(0.01)
    self.assertIsInstance(optimizer, optimizer_base.Optimizer)

  def test_match_keras(self):
    weight_spec = [
        tf.TensorSpec([10, 2], tf.float32),
        tf.TensorSpec([2], tf.float32),
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
    tff_optimizer_fn = lambda: adamw.build_adamw(
        learning_rate=0.01, beta_1=0.9, beta_2=0.999, weight_decay=0.004
    )
    keras_optimizer_fn = lambda: tf.keras.optimizers.AdamW(
        learning_rate=0.01, beta_1=0.9, beta_2=0.999, weight_decay=0.004
    )

    self.assert_optimizers_numerically_close(
        model_variables_fn, gradients, tff_optimizer_fn, keras_optimizer_fn
    )

  @parameterized.named_parameters(
      ('negative_lr', -1.0, 0.9, 0.999, 1e-7, 0.001, 'learning_rate'),
      ('negative_beta_1', 1.0, -0.9, 0.999, 1e-7, 0.001, 'beta_1'),
      ('beta_1_greater_than_1', 1.0, 1.1, 0.999, 1e-7, 0.001, 'beta_1'),
      ('negative_beta_2', 1.0, 0.9, -0.999, 1e-7, 0.001, 'beta_2'),
      ('beta_2_greater_than_1', 1.0, 0.9, 1.1, 1e-7, 0.001, 'beta_2'),
      ('negative_epsilon', 1.0, 0.9, 0.999, -1e-7, 0.001, 'epsilon'),
      ('negative_weight_decay', 1.0, 0.9, 0.999, 1e-7, -0.004, 'weight_decay'),
  )
  def test_invalid_args_raises(
      self, lr, beta_1, beta_2, epsilon, weight_decay, regex
  ):
    with self.assertRaisesRegex(ValueError, regex):
      adamw.build_adamw(lr, beta_1, beta_2, epsilon, weight_decay)

  def test_weights_gradients_mismatch_raises(self):
    optimizer = adamw.build_adamw(0.1)
    state = optimizer.initialize(_SCALAR_SPEC)
    with self.assertRaises(ValueError):
      optimizer.next(state, tf.zeros([1]), tf.zeros([2]))

  def test_initialize_next_weights_mismatch_raises(self):
    optimizer = adamw.build_adamw(0.1)
    state = optimizer.initialize(_SCALAR_SPEC)
    with self.assertRaises(ValueError):
      optimizer.next(state, tf.zeros([2]), tf.zeros([2]))

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC),
      ('struct_spec', _STRUCT_SPEC),
      ('nested_spec', _NESTED_SPEC),
  )
  def test_get_hparams_returns_expected_result(self, spec):
    optimizer = adamw.build_adamw(
        learning_rate=0.1,
        beta_1=0.92,
        beta_2=0.97,
        epsilon=0.01,
        weight_decay=0.002,
    )
    state = optimizer.initialize(spec)
    expected_hparams = collections.OrderedDict(
        learning_rate=0.1,
        beta_1=0.92,
        beta_2=0.97,
        epsilon=0.01,
        weight_decay=0.002,
    )
    actual_hparams = optimizer.get_hparams(state)
    self.assertIsInstance(actual_hparams, collections.OrderedDict)
    self.assertEqual(actual_hparams, expected_hparams)

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC),
      ('struct_spec', _STRUCT_SPEC),
      ('nested_spec', _NESTED_SPEC),
  )
  def test_set_hparams_returns_expected_result(self, spec):
    optimizer = adamw.build_adamw(
        learning_rate=0.1,
        beta_1=0.92,
        beta_2=0.97,
        epsilon=0.01,
        weight_decay=0.002,
    )
    state = optimizer.initialize(spec)
    hparams = collections.OrderedDict(
        learning_rate=0.5,
        beta_1=0.12,
        beta_2=0.56,
        epsilon=2.0,
        weight_decay=0.005,
    )
    expected_state = copy.deepcopy(state)
    for k, v in hparams.items():
      expected_state[k] = v
    updated_state = optimizer.set_hparams(state, hparams)
    self.assertIsInstance(updated_state, collections.OrderedDict)
    self.assertEqual(updated_state, expected_state)

  @parameterized.named_parameters(
      ('scalar_spec', _SCALAR_SPEC),
      ('struct_spec', _STRUCT_SPEC),
      ('nested_spec', _NESTED_SPEC),
  )
  def test_set_get_hparams_is_no_op(self, spec):
    optimizer = adamw.build_adamw(0.1)
    state = optimizer.initialize(spec)
    hparams = optimizer.get_hparams(state)
    updated_state = optimizer.set_hparams(state, hparams)
    self.assertEqual(state, updated_state)

  def test_lr_with_different_weight_dtypes(self):
    weights = (
        tf.constant([0.1], dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float64),
        tf.constant([10.0, 10.0], dtype=tf.bfloat16),
    )
    adamw_optimizer = adamw.build_adamw(
        learning_rate=tf.constant(0.1, dtype=tf.float32),
        beta_1=tf.constant(0.1, dtype=tf.float32),
        beta_2=tf.constant(0.1, dtype=tf.float32),
        epsilon=tf.constant(0.1, dtype=tf.float64),
    )
    state = adamw_optimizer.initialize(weights)
    adamw_optimizer.next(
        state, weights, tf.nest.map_structure(tf.zeros_like, weights)
    )


if __name__ == '__main__':
  tf.test.main()
