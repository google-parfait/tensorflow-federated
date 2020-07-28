# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import values
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.templates import iterative_process


# Create two tff.Computations that perform sum on a sequence: initializes the
# state to 0 and add each item in a sequence to the state.
@computations.tf_computation
def initialize():
  return tf.constant(0)


@computations.tf_computation(tf.int32, tf.int32)
def add_int32(current, val):
  return current + val


@computations.tf_computation(tf.int32, tf.int32)
def add_mul_int32(current, val):
  return current + val, current * val


@computations.tf_computation(tf.int32)
def count_int32(current):
  return current + 1


@computations.tf_computation
def initialize_empty_tuple():
  return []


@computations.tf_computation([])
def next_empty_tuple(x):
  return x


class IterativeProcessTest(test.TestCase):

  def test_constructor_with_state_only(self):
    ip = iterative_process.IterativeProcess(initialize, count_int32)

    state = ip.initialize()
    iterations = 10
    for _ in range(iterations):
      # TODO(b/122321354): remove the .item() call on `state` once numpy.int32
      # type is supported.
      state = ip.next(state.item())
    self.assertEqual(state, iterations)

  def test_constructor_with_tensors_unknown_dimensions(self):

    @computations.tf_computation
    def init():
      return tf.constant([], dtype=tf.string)

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def next_fn(strings):
      return tf.concat([strings, tf.constant(['abc'])], axis=0)

    try:
      iterative_process.IterativeProcess(init, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an IterativeProcess with parameter types '
                'including unknown dimension tennsors.')

  def test_constructor_with_state_tuple_arg(self):
    ip = iterative_process.IterativeProcess(initialize, add_int32)

    state = ip.initialize()
    iterations = 10
    for val in range(iterations):
      state = ip.next(state, val)
    self.assertEqual(state, sum(range(iterations)))

  def test_constructor_with_state_multiple_return_values(self):
    ip = iterative_process.IterativeProcess(initialize, add_mul_int32)

    state = ip.initialize()
    iterations = 10
    for val in range(iterations):
      state, product = ip.next(state, val)
    self.assertEqual(state, sum(range(iterations)))
    self.assertEqual(product, sum(range(iterations - 1)) * (iterations - 1))

  def test_constructor_with_empty_tuple(self):
    ip = iterative_process.IterativeProcess(initialize_empty_tuple,
                                            next_empty_tuple)

    state = ip.initialize()
    iterations = 2
    for _ in range(iterations):
      state = ip.next(state)
    self.assertEqual(state, [])

  def test_constructor_with_initialize_bad_type(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      iterative_process.IterativeProcess(initialize_fn=None, next_fn=add_int32)

    with self.assertRaisesRegex(
        TypeError, r'initialize_fn must be a no-arg tff.Computation'):

      @computations.federated_computation(tf.int32)
      def one_arg_initialize(one_arg):
        del one_arg  # Unused.
        return values.to_value(0)

      iterative_process.IterativeProcess(
          initialize_fn=one_arg_initialize, next_fn=add_int32)

  def test_constructor_with_next_bad_type(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      iterative_process.IterativeProcess(initialize_fn=initialize, next_fn=None)

  def test_constructor_with_type_mismatch(self):
    with self.assertRaisesRegex(
        TypeError, r'The return type of initialize_fn must be assignable.*'):

      @computations.federated_computation(tf.float32, tf.float32)
      def add_float32(current, val):
        return current + val

      iterative_process.IterativeProcess(
          initialize_fn=initialize, next_fn=add_float32)

    with self.assertRaisesRegex(
        TypeError,
        'The return type of next_fn must be assignable to the first parameter'):

      @computations.federated_computation(tf.int32)
      def add_bad_result(_):
        return 0.0

      iterative_process.IterativeProcess(
          initialize_fn=initialize, next_fn=add_bad_result)

    with self.assertRaisesRegex(
        TypeError,
        'The return type of next_fn must be assignable to the first parameter'):

      @computations.federated_computation(tf.int32)
      def add_bad_multi_result(_):
        return 0.0, 0

      iterative_process.IterativeProcess(
          initialize_fn=initialize, next_fn=add_bad_multi_result)


if __name__ == '__main__':
  # Note: num_clients must be explicit here to correctly test the broadcast
  # behavior. Otherwise TFF will infer there are zero clients, which is an
  # error.
  factory = executor_stacks.local_executor_factory(num_clients=3)
  context = execution_context.ExecutionContext(factory)
  context_stack_impl.context_stack.set_default_context(context)
  test.main()
