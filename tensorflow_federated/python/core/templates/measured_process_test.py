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

import collections

import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process

MeasuredProcessOutput = measured_process.MeasuredProcessOutput


@computations.tf_computation()
def test_initialize_fn():
  return tf.constant(0, tf.int32)


@computations.tf_computation(tf.int32, tf.float32)
def test_next_fn(state, value):
  return MeasuredProcessOutput(state, value, ())


class MeasuredProcessTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      measured_process.MeasuredProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid MeasuredProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.tf_computation()(lambda: ())
    next_fn = computations.tf_computation(
        ())(lambda x: MeasuredProcessOutput(x, (), ()))
    try:
      measured_process.MeasuredProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an MeasuredProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):
    initialize_fn = computations.tf_computation()(
        lambda: tf.constant([], dtype=tf.string))

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string))
    def next_fn(strings):
      return MeasuredProcessOutput(
          tf.concat([strings, tf.constant(['abc'])], axis=0), (), ())

    try:
      measured_process.MeasuredProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an MeasuredProcess with parameter types '
                'with statically unknown shape.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      measured_process.MeasuredProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      measured_process.MeasuredProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state: MeasuredProcessOutput(state, (), ()))

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.tf_computation(tf.int32)(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      measured_process.MeasuredProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.tf_computation()(lambda: 0.0)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(float_initialize_fn, test_next_fn)

  def test_federated_init_state_not_assignable(self):
    zero = lambda: intrinsics.federated_value(0, placements.SERVER)
    initialize_fn = computations.federated_computation()(zero)
    next_fn = computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))(
            lambda state: MeasuredProcessOutput(state, zero(), zero()))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  def test_next_state_not_assignable(self):
    float_next_fn = computations.tf_computation(tf.float32)(
        lambda state: MeasuredProcessOutput(tf.cast(state, tf.float32), (), ()))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(test_initialize_fn, float_next_fn)

  def test_federated_next_state_not_assignable(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))

    @computations.federated_computation(initialize_fn.type_signature.result)
    def next_fn(state):
      return MeasuredProcessOutput(
          intrinsics.federated_broadcast(state), (), ())

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  # Tests specific only for the MeasuredProcess contract below.

  def test_measured_process_output_as_state_raises(self):
    empty_output = lambda: MeasuredProcessOutput((), (), ())
    initialize_fn = computations.tf_computation(empty_output)
    next_fn = computations.tf_computation(
        initialize_fn.type_signature.result)(lambda state: empty_output())
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  def test_next_return_tensor_type_raises(self):
    next_fn = computations.tf_computation(tf.int32)(lambda state: state)
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, next_fn)

  def test_next_return_tuple_raises(self):
    tuple_next_fn = computations.tf_computation(
        tf.int32)(lambda state: (state, (), ()))
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements'])
    namedtuple_next_fn = computations.tf_computation(
        tf.int32)(lambda state: measured_process_output(state, (), ()))
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):
    odict_next_fn = computations.tf_computation(tf.int32)(
        lambda s: collections.OrderedDict(state=s, result=(), measurements=()))
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, odict_next_fn)

  def test_federated_measured_process_output_raises(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    empty = lambda: intrinsics.federated_value((), placements.SERVER)
    state_type = initialize_fn.type_signature.result

    # Using federated_zip to place FederatedType at the top of the hierarchy.
    @computations.federated_computation(state_type)
    def next_fn(state):
      return intrinsics.federated_zip(
          MeasuredProcessOutput(state, empty(), empty()))

    # A MeasuredProcessOutput containing three `FederatedType`s is different
    # than a `FederatedType` containing a MeasuredProcessOutput. Corrently, only
    # the former is considered valid.
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)


def test_measured_process_double(state_type, state_init, values_type):

  @computations.tf_computation
  def double(x):
    return x * 2

  @computations.federated_computation(
      computation_types.at_server(state_type),
      computation_types.at_clients(values_type))
  def map_double(state, values):
    return MeasuredProcessOutput(
        state=intrinsics.federated_map(double, state),
        result=intrinsics.federated_map(double, values),
        measurements=intrinsics.federated_value({'a': 1}, placements.SERVER))

  return measured_process.MeasuredProcess(
      initialize_fn=computations.federated_computation(
          lambda: intrinsics.federated_value(state_init, placements.SERVER)),
      next_fn=map_double)


def test_measured_process_sum(state_type, state_init, values_type):

  @computations.tf_computation
  def add_one(x):
    return x + 1

  @computations.federated_computation(
      computation_types.at_server(state_type),
      computation_types.at_clients(values_type))
  def map_sum(state, values):
    return MeasuredProcessOutput(
        state=intrinsics.federated_map(add_one, state),
        result=intrinsics.federated_sum(values),
        measurements=intrinsics.federated_value({'b': 2}, placements.SERVER))

  return measured_process.MeasuredProcess(
      initialize_fn=computations.federated_computation(
          lambda: intrinsics.federated_value(state_init, placements.SERVER)),
      next_fn=map_sum)


class MeasuredProcessCompositionComputationTest(test_case.TestCase):

  def test_composite_process_type_properties(self):
    state_type = tf.float32
    values_type = tf.int32
    composite_process = measured_process.chain_measured_processes(
        collections.OrderedDict(
            double=test_measured_process_double(state_type, 1.0, values_type),
            sum=test_measured_process_sum(state_type, 0.0, values_type)))
    self.assertIsInstance(composite_process, measured_process.MeasuredProcess)

    expected_state_type = computation_types.at_server(
        collections.OrderedDict(double=state_type, sum=state_type))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        composite_process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    param_value_type = computation_types.at_clients(values_type)
    result_value_type = computation_types.at_server(values_type)
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            double=collections.OrderedDict(a=tf.int32),
            sum=collections.OrderedDict(b=tf.int32)))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, values=param_value_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        composite_process.next.type_signature.is_equivalent_to(
            expected_next_type))

  def test_values_type_mismatching_raises(self):
    measured_processes = collections.OrderedDict(
        double=test_measured_process_double(tf.int32, 1, tf.int32),
        sum=test_measured_process_sum(tf.int32, 0, tf.float32))

    first_process_result_type = measured_processes[
        'double'].next.type_signature.result.result
    second_process_values_type = measured_processes[
        'sum'].next.type_signature.parameter.values
    self.assertFalse(
        second_process_values_type.is_equivalent_to(first_process_result_type))

    with self.assertRaisesRegex(TypeError, 'Cannot call function'):
      measured_process.chain_measured_processes(measured_processes)


class MeasuredProcessCompositionExecutionTest(test_case.TestCase):

  def test_composite_process_gets_expected_output(self):

    measured_processes = collections.OrderedDict(
        double=test_measured_process_double(tf.int32, 1, tf.int32),
        sum=test_measured_process_sum(tf.int32, 0, tf.int32))
    composite_process = measured_process.chain_measured_processes(
        measured_processes)
    output = composite_process.next(composite_process.initialize(),
                                    [1, 2, 3, 4])
    self.assertEqual(output.state, collections.OrderedDict(double=2, sum=1))
    self.assertEqual(output.result, 20)
    self.assertEqual(output.measurements,
                     collections.OrderedDict(double={'a': 1}, sum={'b': 2}))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
