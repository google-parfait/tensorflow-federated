# Copyright 2020, The TensorFlow Federated Authors.
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

import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import values
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.templates import measured_process


def _build_initialize_comp(constant):

  @computations.tf_computation
  def initialize():
    return tf.constant(constant)

  return initialize


@computations.tf_computation(tf.int32, tf.int32)
def add_int32(current, val):
  """Performs arbitrary Tensor math for testing."""
  return measured_process.MeasuredProcessOutput(
      state=current + val, result=val, measurements=[current / (val + 1)])


@computations.tf_computation(tf.float32, tf.float32)
def add_mul_int32(current, val):
  """Performs arbitrary Tensor math for testing."""
  return measured_process.MeasuredProcessOutput(
      state=current + val,
      result=current * val,
      measurements=[current / (val + 1.0)])


@computations.tf_computation(tf.int32)
def count_int32(current):
  """Performs arbitrary Tensor math for testing."""
  # NOTE: the empty tuple () is the NoneType of TFF.
  return measured_process.MeasuredProcessOutput(
      state=current + 1, result=(), measurements=())


class MeasuredProcessTest(test.TestCase):

  def test_constructor_with_state_only(self):
    ip = measured_process.MeasuredProcess(
        _build_initialize_comp(0), count_int32)

    state = ip.initialize()
    iterations = 10
    for _ in range(iterations):
      state, result, measurements = attr.astuple(ip.next(state))
      self.assertLen(result, 0)
      self.assertLen(measurements, 0)
    self.assertEqual(state, iterations)

  def test_constructor_with_tensors_unknown_dimensions_succeeds(self):

    @computations.tf_computation
    def init():
      return tf.constant([], dtype=tf.string)

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def next_fn(strings):
      return measured_process.MeasuredProcessOutput(
          state=tf.concat([strings, tf.constant(['abc'])], axis=0),
          result=(),
          measurements=())

    try:
      measured_process.MeasuredProcess(init, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an MeasuredProcess with parameter types '
                'including unknown dimension tennsors.')

  def test_constructor_with_state_tuple_arg(self):
    ip = measured_process.MeasuredProcess(_build_initialize_comp(0), add_int32)

    state = ip.initialize()
    iterations = 10
    for val in range(iterations):
      output = ip.next(state, val)
      state = output.state
    self.assertEqual(output.state, sum(range(iterations)))
    self.assertEqual(output.result, val)
    expected_measurment = sum(range(iterations - 1)) / iterations
    self.assertAllClose(output.measurements, [expected_measurment])

  def test_constructor_with_state_multiple_return_values(self):
    ip = measured_process.MeasuredProcess(
        _build_initialize_comp(0.0), add_mul_int32)

    state = ip.initialize()
    iterations = 10
    for val in range(iterations):
      output = ip.next(state, float(val))
      state = output.state
    self.assertEqual(output.state, sum(range(iterations)))
    self.assertEqual(output.result,
                     sum(range(iterations - 1)) * (iterations - 1))
    expected_measurment = sum(range(iterations - 1)) / iterations
    self.assertAllClose(output.measurements, [expected_measurment])

  def test_constructor_with_next_federated_same_placed_struct_result(self):

    @computations.federated_computation
    def initialize_comp():
      return intrinsics.federated_value(0, placements.SERVER)

    # A `next` function that returns all the same placement and is zipped so
    # the FederatedType is at the top of the type hierarchy.
    @computations.federated_computation(initialize_comp.type_signature.result)
    def next_comp(state):
      return intrinsics.federated_zip(
          measured_process.MeasuredProcessOutput(
              state=state,
              result=intrinsics.federated_value(0, placements.SERVER),
              measurements=intrinsics.federated_value((), placements.SERVER))),

    with self.assertRaisesRegex(
        TypeError,
        'The return type of next_fn must be assignable to the first parameter'):
      measured_process.MeasuredProcess(
          initialize_fn=initialize_comp, next_fn=next_comp)

  def test_constructor_with_next_struct_of_different_placedresult(self):

    @computations.federated_computation
    def initialize_comp():
      return intrinsics.federated_value(0, placements.SERVER)

    # A `next` function that returns different placements for the components.
    @computations.federated_computation(initialize_comp.type_signature.result)
    def next_comp(state):
      return measured_process.MeasuredProcessOutput(
          state=state,
          result=intrinsics.federated_value(0, placements.CLIENTS),
          measurements=intrinsics.federated_value((), placements.SERVER))

    try:
      measured_process.MeasuredProcess(
          initialize_fn=initialize_comp, next_fn=next_comp)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Failed to construct MeasuredProcess: {e}')

  def test_constructor_with_initialize_bad_type(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      measured_process.MeasuredProcess(initialize_fn=None, next_fn=add_int32)

    with self.assertRaisesRegex(
        TypeError, r'initialize_fn must be a no-arg tff.Computation'):

      @computations.federated_computation(tf.int32)
      def one_arg_initialize(one_arg):
        del one_arg  # Unused.
        return values.to_value(0)

      measured_process.MeasuredProcess(
          initialize_fn=one_arg_initialize, next_fn=add_int32)

  def test_constructor_with_next_bad_type(self):
    initialize = _build_initialize_comp(0)
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      measured_process.MeasuredProcess(initialize_fn=initialize, next_fn=None)

  def test_constructor_with_init_next_type_mismatch(self):
    initialize = _build_initialize_comp(0)
    with self.assertRaisesRegex(
        TypeError, r'The return type of initialize_fn must be assignable.*'):

      @computations.federated_computation(tf.float32, tf.float32)
      def add_float32(current, val):
        return current + val

      measured_process.MeasuredProcess(
          initialize_fn=initialize, next_fn=add_float32)

  def test_constructor_with_next_result_param_type_mismatch(self):
    initialize = _build_initialize_comp(0)
    with self.assertRaisesRegex(
        TypeError,
        'The return type of next_fn must be assignable to the first parameter'):

      @computations.federated_computation(tf.int32)
      def add_bad_result(_):
        return 0.0

      measured_process.MeasuredProcess(
          initialize_fn=initialize, next_fn=add_bad_result)

  def test_constructor_with_next_result_not_measuredprocessoutput(self):
    initialize = _build_initialize_comp(0)
    with self.assertRaisesRegex(
        TypeError, 'MeasuredProcess must return a MeasuredProcessOutput'):

      @computations.federated_computation(tf.int32)
      def add_not_tuple_result(_):
        return 0

      measured_process.MeasuredProcess(
          initialize_fn=initialize, next_fn=add_not_tuple_result)

    with self.assertRaisesRegex(
        TypeError, 'MeasuredProcess must return a MeasuredProcessOutput'):

      @computations.federated_computation(tf.int32)
      def add_not_named_tuple_result(_):
        return 0, 0, 0

      measured_process.MeasuredProcess(
          initialize_fn=initialize, next_fn=add_not_named_tuple_result)


if __name__ == '__main__':
  factory = executor_stacks.local_executor_factory(num_clients=3)
  context = execution_context.ExecutionContext(factory)
  context_stack_impl.context_stack.set_default_context(context)
  test.main()
