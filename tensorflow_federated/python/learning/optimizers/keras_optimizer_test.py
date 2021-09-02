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

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning.optimizers import keras_optimizer
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import optimizer_test_utils
from tensorflow_federated.python.learning.optimizers import sgdm

_SCALAR_SPEC = tf.TensorSpec([1], tf.float32)
_STRUCT_SPEC = [tf.TensorSpec([2], tf.float32), tf.TensorSpec([3], tf.float32)]
_NESTED_SPEC = [
    tf.TensorSpec([10], tf.float32),
    [tf.TensorSpec([20], tf.float32), [tf.TensorSpec([30], tf.float32)]]
]


class KerasOptimizerTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('no_momentum', 0.0), ('momentum_0_5', 0.5))
  def test_disjoint_init_and_next_true(self, momentum):
    """Tests behavior expected as 'TFF server optimizer'.

    This test creates two `tff.tf_computation`s, which would correspond to parts
    of the two arguments for creation of a `tff.templates.IterativeProcess`.

    The `KerasOptimizers` is instantiated in both of these computations, and
    only one of its `initialize` and `next` methods is invoked in each of them.
    The state which optimizers need is exposed by the `KerasOptimizer` and needs
    to be carried between the invocations of the created `tff.Computation`s.

    Note that even though it is expected that `variables` passed to the
    `single_step` method are expected to be `tf.Variable` instances, the code
    can still be written in a functional manner.

    Args:
      momentum: Momentum parameter to be used in tf.keras.optimizers.SGD.
    """
    init_w, fn, grad_fn = optimizer_test_utils.test_quadratic_problem()
    weights = init_w()
    self.assertGreater(fn(weights), 5.0)
    optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1, momentum=momentum)

    @computations.tf_computation()
    def initialize_fn():
      variables = tf.Variable(tf.zeros([5, 1]))
      optimizer = keras_optimizer.KerasOptimizer(
          optimizer_fn, variables, disjoint_init_and_next=True)
      return optimizer.initialize(
          tf.TensorSpec(variables.shape, variables.dtype))

    @tf.function
    def single_step(optimizer, state, variables):
      gradients = grad_fn(variables)
      new_state, updated_weights = optimizer.next(state, variables, gradients)
      return new_state, updated_weights

    @computations.tf_computation()
    def next_fn(state, initial_weights):
      variables = tf.Variable(initial_weights)
      optimizer = keras_optimizer.KerasOptimizer(
          optimizer_fn, variables, disjoint_init_and_next=True)
      return single_step(optimizer, state, variables)

    state = initialize_fn()
    for _ in range(100):
      state, weights = next_fn(state, weights)

    self.assertLess(fn(weights), 0.005)
    # The optimizer variables are exposed by the KerasOptimizer. First variable
    # of a keras optimzier is the number of steps taken.
    self.assertEqual(100, state[0])

  @parameterized.named_parameters(('no_momentum', 0.0), ('momentum_0_5', 0.5))
  def test_disjoint_init_and_next_false(self, momentum):
    """Tests behavior expected as 'TFF client optimizer'.

    The main part of this test is `training_loop` method, which works with
    already instantiated `KerasOptimizer` and uses both of its `initialize` and
    `next` methods to perform a number of training steps. This is the behavior
    expected to happen during local training at clients.

    Note that even though it is expected that `variables` passed to the
    `training_loop` method are expected to be `tf.Variable` instances, the code
    can still be written in a functional manner.

    Args:
      momentum: Momentum parameter to be used in tf.keras.optimizers.SGD.
    """
    init_w, fn, grad_fn = optimizer_test_utils.test_quadratic_problem()
    weights = init_w()
    self.assertGreater(fn(weights), 5.0)
    optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1, momentum=momentum)

    @tf.function
    def training_loop(optimizer, variables):
      state = optimizer.initialize(
          tf.TensorSpec(variables.shape, variables.dtype))
      for _ in range(100):
        gradients = grad_fn(variables)
        state, variables = optimizer.next(state, variables, gradients)
      return state, variables

    @computations.tf_computation()
    def local_training(initial_weights):
      variables = tf.Variable(initial_weights)
      optimizer = keras_optimizer.KerasOptimizer(
          optimizer_fn, variables, disjoint_init_and_next=False)
      return training_loop(optimizer, variables)

    state, optimized_weights = local_training(weights)
    self.assertLess(fn(optimized_weights), 0.005)
    # The optimizer variables are handled internally in the KerasOptimizer.
    self.assertEmpty(state)

  def test_disjoint_init_and_next_false_keras_state_updated(self):
    """Tests optimizer state is updated inside of KerasOptimizer."""
    optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1)

    @tf.function
    def training_loop(optimizer, variables):
      state = optimizer.initialize(
          tf.TensorSpec(variables.shape, variables.dtype))
      for _ in range(3):
        gradients = tf.constant(1.0)
        state, variables = optimizer.next(state, variables, gradients)
      # Return also the private variables of the optimizer.
      return state, variables, optimizer._optimizer.variables()

    @computations.tf_computation()
    def test_computation(initial_weights):
      variables = tf.Variable(initial_weights)
      optimizer = keras_optimizer.KerasOptimizer(
          optimizer_fn, variables, disjoint_init_and_next=False)
      return training_loop(optimizer, variables)

    state, weights, optimizer_variables = test_computation(1.0)
    self.assertAllClose(0.7, weights)  # 3 steps with learning rate 0.1
    # The optimizer variables are handled internally in the KerasOptimizer.
    self.assertEmpty(state)
    # The optimizer has a single variable counring number of steps.
    self.assertAllEqual([3], optimizer_variables)

  @parameterized.named_parameters(
      ('scalar_server', _SCALAR_SPEC, True),
      ('struct_server', _STRUCT_SPEC, True),
      ('nested_server', _NESTED_SPEC, True),
      ('scalar_client', _SCALAR_SPEC, False),
      ('struct_client', _STRUCT_SPEC, False),
      ('nested_client', _NESTED_SPEC, False),
  )
  def test_spec(self, specs, disjoint_init_and_next):
    """Test compatibility with different structures of variables."""
    optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1)
    variables = tf.nest.map_structure(
        lambda s: tf.Variable(tf.ones(s.shape, s.dtype)), specs)
    gradients = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype),
                                      specs)

    optimizer = keras_optimizer.KerasOptimizer(
        optimizer_fn, variables, disjoint_init_and_next=disjoint_init_and_next)
    state = optimizer.initialize(specs)
    for _ in range(3):
      state, variables = optimizer.next(state, variables, gradients)

    expected_variables = tf.nest.map_structure(
        lambda s: 0.7 * tf.ones(s.shape, s.dtype), specs)
    self.assertAllClose(expected_variables, variables)

  @parameterized.named_parameters(
      ('scalar_server', _SCALAR_SPEC, True),
      ('struct_server', _STRUCT_SPEC, True),
      ('nested_server', _NESTED_SPEC, True),
      ('scalar_client', _SCALAR_SPEC, False),
      ('struct_client', _STRUCT_SPEC, False),
      ('nested_client', _NESTED_SPEC, False),
  )
  def test_build_tff_optimizer_keras(self, specs, disjoint_init_and_next):
    optimizer_fn = lambda: tf.keras.optimizers.SGD(0.1)
    variables = tf.nest.map_structure(
        lambda s: tf.Variable(tf.ones(s.shape, s.dtype)), specs)
    optimizer = keras_optimizer.build_or_verify_tff_optimizer(
        optimizer_fn, variables, disjoint_init_and_next)
    self.assertIsInstance(optimizer, optimizer_base.Optimizer)

  def test_build_tff_optimizer_tff(self):
    optimizer = sgdm.build_sgdm()
    optimizer2 = keras_optimizer.build_or_verify_tff_optimizer(optimizer)
    self.assertIs(optimizer, optimizer2)

  @parameterized.named_parameters(
      ('server', True),
      ('client', False),
  )
  def test_build_tff_optimizer_raise(self, disjoint_init_and_next):
    with self.assertRaisesRegex(TypeError,
                                '`optimizer_fn` must be a callable or '):
      keras_optimizer.build_or_verify_tff_optimizer(None, None,
                                                    disjoint_init_and_next)

  @parameterized.named_parameters(
      ('server', True),
      ('client', False),
  )
  def test_build_tff_optimizer_arg_callable(self, disjoint_init_and_next):
    with self.assertRaises(TypeError):
      keras_optimizer.build_or_verify_tff_optimizer(
          optimizer_fn=lambda x: x,
          trainable_weights=None,
          disjoint_init_and_next=disjoint_init_and_next)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
