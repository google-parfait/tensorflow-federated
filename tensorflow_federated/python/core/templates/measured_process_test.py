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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process

MeasuredProcessOutput = measured_process.MeasuredProcessOutput


@tensorflow_computation.tf_computation()
def test_initialize_fn():
  return tf.constant(0, tf.int32)


@tensorflow_computation.tf_computation(tf.int32, tf.float32)
def test_next_fn(state, value):
  return MeasuredProcessOutput(state, value, ())


class MeasuredProcessTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      measured_process.MeasuredProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid MeasuredProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = tensorflow_computation.tf_computation()(lambda: ())
    next_fn = tensorflow_computation.tf_computation(
        ())(lambda x: MeasuredProcessOutput(x, (), ()))
    try:
      measured_process.MeasuredProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an MeasuredProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):
    initialize_fn = tensorflow_computation.tf_computation()(
        lambda: tf.constant([], dtype=tf.string))

    @tensorflow_computation.tf_computation(
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
    one_arg_initialize_fn = tensorflow_computation.tf_computation(
        tf.int32)(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      measured_process.MeasuredProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = tensorflow_computation.tf_computation()(lambda: 0.0)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(float_initialize_fn, test_next_fn)

  def test_federated_init_state_not_assignable(self):
    zero = lambda: intrinsics.federated_value(0, placements.SERVER)
    initialize_fn = federated_computation.federated_computation()(zero)
    next_fn = federated_computation.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))(
            lambda state: MeasuredProcessOutput(state, zero(), zero()))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  def test_next_state_not_assignable(self):
    float_next_fn = tensorflow_computation.tf_computation(tf.float32)(
        lambda state: MeasuredProcessOutput(tf.cast(state, tf.float32), (), ()))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(test_initialize_fn, float_next_fn)

  def test_federated_next_state_not_assignable(self):
    initialize_fn = federated_computation.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))

    @federated_computation.federated_computation(
        initialize_fn.type_signature.result)
    def next_fn(state):
      return MeasuredProcessOutput(
          intrinsics.federated_broadcast(state), (), ())

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  # Tests specific only for the MeasuredProcess contract below.

  def test_measured_process_output_as_state_raises(self):
    empty_output = lambda: MeasuredProcessOutput((), (), ())
    initialize_fn = tensorflow_computation.tf_computation(empty_output)
    next_fn = tensorflow_computation.tf_computation(
        initialize_fn.type_signature.result)(lambda state: empty_output())
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  def test_next_return_tensor_type_raises(self):
    next_fn = tensorflow_computation.tf_computation(
        tf.int32)(lambda state: state)
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, next_fn)

  def test_next_return_tuple_raises(self):
    tuple_next_fn = tensorflow_computation.tf_computation(
        tf.int32)(lambda state: (state, (), ()))
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements'])
    namedtuple_next_fn = tensorflow_computation.tf_computation(
        tf.int32)(lambda state: measured_process_output(state, (), ()))
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):
    odict_next_fn = tensorflow_computation.tf_computation(tf.int32)(
        lambda s: collections.OrderedDict(state=s, result=(), measurements=()))
    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(test_initialize_fn, odict_next_fn)

  def test_federated_measured_process_output_raises(self):
    initialize_fn = federated_computation.federated_computation()(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    empty = lambda: intrinsics.federated_value((), placements.SERVER)
    state_type = initialize_fn.type_signature.result

    # Using federated_zip to place FederatedType at the top of the hierarchy.
    @federated_computation.federated_computation(state_type)
    def next_fn(state):
      return intrinsics.federated_zip(
          MeasuredProcessOutput(state, empty(), empty()))

    # A MeasuredProcessOutput containing three `FederatedType`s is different
    # than a `FederatedType` containing a MeasuredProcessOutput. Corrently, only
    # the former is considered valid.
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)


def _create_test_measured_process_double(state_type, state_init, values_type):

  @tensorflow_computation.tf_computation
  def double(x):
    return x * 2

  @federated_computation.federated_computation(
      computation_types.at_server(state_type),
      computation_types.at_clients(values_type))
  def map_double(state, values):
    return MeasuredProcessOutput(
        state=intrinsics.federated_map(double, state),
        result=intrinsics.federated_map(double, values),
        measurements=intrinsics.federated_value(
            collections.OrderedDict(a=1), placements.SERVER))

  return measured_process.MeasuredProcess(
      initialize_fn=federated_computation.federated_computation(
          lambda: intrinsics.federated_value(state_init, placements.SERVER)),
      next_fn=map_double)


def _create_test_measured_process_sum(state_type, state_init, values_type):

  @tensorflow_computation.tf_computation
  def add_one(x):
    return x + 1

  @federated_computation.federated_computation(
      computation_types.at_server(state_type),
      computation_types.at_clients(values_type))
  def map_sum(state, values):
    return MeasuredProcessOutput(
        state=intrinsics.federated_map(add_one, state),
        result=intrinsics.federated_sum(values),
        measurements=intrinsics.federated_value(
            collections.OrderedDict(b=2), placements.SERVER))

  return measured_process.MeasuredProcess(
      initialize_fn=federated_computation.federated_computation(
          lambda: intrinsics.federated_value(state_init, placements.SERVER)),
      next_fn=map_sum)


def _create_test_measured_process_state_at_clients():

  @federated_computation.federated_computation(
      computation_types.at_clients(tf.int32),
      computation_types.at_clients(tf.int32))
  def next_fn(state, values):
    return measured_process.MeasuredProcessOutput(
        state, intrinsics.federated_sum(values),
        intrinsics.federated_value(1, placements.SERVER))

  return measured_process.MeasuredProcess(
      initialize_fn=federated_computation.federated_computation(
          lambda: intrinsics.federated_value(0, placements.CLIENTS)),
      next_fn=next_fn)


def _create_test_measured_process_state_missing_placement():
  return measured_process.MeasuredProcess(
      initialize_fn=test_initialize_fn, next_fn=test_next_fn)


def _create_test_aggregation_process(state_type, state_init, values_type):

  @federated_computation.federated_computation(
      computation_types.at_server(state_type),
      computation_types.at_clients(values_type))
  def next_fn(state, values):
    return measured_process.MeasuredProcessOutput(
        state, intrinsics.federated_sum(values),
        intrinsics.federated_value(1, placements.SERVER))

  return aggregation_process.AggregationProcess(
      initialize_fn=federated_computation.federated_computation(
          lambda: intrinsics.federated_value(state_init, placements.SERVER)),
      next_fn=next_fn)


def _create_test_iterative_process(state_type, state_init):

  @tensorflow_computation.tf_computation(state_type)
  def next_fn(state):
    return state

  return iterative_process.IterativeProcess(
      initialize_fn=tensorflow_computation.tf_computation(
          lambda: tf.constant(state_init)),
      next_fn=next_fn)


# Name the compiled computations to avoid the issue that the TF graphs being
# generated are different at HEAD vs in OSS, resulting in different hash values
# for the computation name which fail to compare.
# TODO(b/234016763): Extend `golden` to be able to take Computation arguments
# directly, and call this method on them.
def _name_compiled_computations(
    tree: building_blocks.ComputationBuildingBlock
) -> building_blocks.ComputationBuildingBlock:
  counter = 1

  def _transform(building_block):
    nonlocal counter
    if building_block.is_compiled_computation():
      new_name = str(counter)
      counter += 1
      return building_blocks.CompiledComputation(
          proto=building_block.proto, name=new_name), True
    return building_block, False

  return transformation_utils.transform_postorder(tree, _transform)[0]


class MeasuredProcessCompositionComputationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('all_measured_processes', _create_test_measured_process_sum),
      ('with_aggregation_process', _create_test_aggregation_process),
  ])
  def test_composition_type_properties(self, last_process):
    state_type = tf.float32
    values_type = tf.int32
    last_process = last_process(state_type, 0.0, values_type)
    composite_process = measured_process.chain_measured_processes(
        collections.OrderedDict(
            double=_create_test_measured_process_double(state_type, 1.0,
                                                        values_type),
            last_process=last_process))
    self.assertIsInstance(composite_process, measured_process.MeasuredProcess)

    expected_state_type = computation_types.at_server(
        collections.OrderedDict(double=state_type, last_process=state_type))
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
            last_process=last_process.next.type_signature.result.measurements
            .member))
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
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        sum=_create_test_measured_process_sum(tf.int32, 0, tf.float32))

    first_process_result_type = measured_processes[
        'double'].next.type_signature.result.result
    second_process_values_type = measured_processes[
        'sum'].next.type_signature.parameter.values
    self.assertFalse(
        second_process_values_type.is_equivalent_to(first_process_result_type))

    with self.assertRaisesRegex(TypeError, 'Cannot call function'):
      measured_process.chain_measured_processes(measured_processes)

  def test_composition_with_iterative_process_raises(self):
    processes = collections.OrderedDict(
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        iterative=_create_test_iterative_process(tf.int32, 0))

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'):
      measured_process.chain_measured_processes(processes)

  @parameterized.named_parameters([
      ('state_at_server_and_clients',
       _create_test_measured_process_double(tf.int32, 1, tf.int32),
       _create_test_measured_process_state_at_clients()),
      ('state_at_server_and_missing_placement',
       _create_test_measured_process_double(tf.int32, 1, tf.int32),
       _create_test_measured_process_state_missing_placement()),
      ('state_at_clients_and_missing_placement',
       _create_test_measured_process_state_at_clients(),
       _create_test_measured_process_state_missing_placement()),
  ])
  def test_composition_with_mixed_state_placement_raises(
      self, first_process, second_process):
    measured_processes = collections.OrderedDict(
        first_process=first_process, second_process=second_process)

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'):
      measured_process.chain_measured_processes(measured_processes)


# We verify the AST (Abstract Syntax Trees) of the `initialize` and `next`
# of the composite process. So the tests don't need to actually invoke these
# computations and depend on the execution context.
class MeasuredProcessCompositionASTTest(absltest.TestCase):

  def test_composition_with_measured_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        sum=_create_test_measured_process_sum(tf.int32, 0, tf.int32))
    composite_process = measured_process.chain_measured_processes(
        measured_processes)

    actual_initialize_ast = _name_compiled_computations(
        composite_process.initialize.to_building_block())
    actual_next_ast = _name_compiled_computations(
        composite_process.next.to_building_block())
    golden.check_string(
        'composition_with_measured_processes.expected',
        f'initialize:\n\n{actual_initialize_ast.formatted_representation()}\n\n'
        f'next:\n\n{actual_next_ast.formatted_representation()}')

  def test_composition_with_aggregation_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        aggregate=_create_test_aggregation_process(tf.int32, 0, tf.int32))
    composite_process = measured_process.chain_measured_processes(
        measured_processes)

    actual_initialize_ast = _name_compiled_computations(
        composite_process.initialize.to_building_block())
    actual_next_ast = _name_compiled_computations(
        composite_process.next.to_building_block())
    golden.check_string(
        'composition_with_aggregation_processes.expected',
        f'initialize:\n\n{actual_initialize_ast.formatted_representation()}\n\n'
        f'next:\n\n{actual_next_ast.formatted_representation()}')


class MeasuredProcessConcatenationComputationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('all_measured_processes', _create_test_measured_process_sum),
      ('with_aggregation_process', _create_test_aggregation_process),
  ])
  def test_concatenation_type_properties(self, last_process):
    state_type = tf.int32
    values_type = tf.int32
    last_process = last_process(state_type, 0, values_type)
    concatenated_process = measured_process.concatenate_measured_processes(
        collections.OrderedDict(
            double=_create_test_measured_process_double(state_type, 1,
                                                        values_type),
            last_process=last_process))
    self.assertIsInstance(concatenated_process,
                          measured_process.MeasuredProcess)

    expected_state_type = computation_types.at_server(
        collections.OrderedDict(double=state_type, last_process=state_type))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        concatenated_process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    param_value_type = collections.OrderedDict(
        double=computation_types.at_clients(values_type),
        last_process=computation_types.at_clients(values_type))
    result_value_type = collections.OrderedDict(
        double=computation_types.at_clients(values_type),
        last_process=computation_types.at_server(values_type))
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(
            double=collections.OrderedDict(a=tf.int32),
            last_process=last_process.next.type_signature.result.measurements
            .member))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, values=param_value_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        concatenated_process.next.type_signature.is_equivalent_to(
            expected_next_type))

  def test_concatenation_with_iterative_process_raises(self):
    processes = collections.OrderedDict(
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        iterative=_create_test_iterative_process(tf.int32, 0))

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'):
      measured_process.concatenate_measured_processes(processes)

  @parameterized.named_parameters([
      ('state_at_server_and_clients',
       _create_test_measured_process_double(tf.int32, 1, tf.int32),
       _create_test_measured_process_state_at_clients()),
      ('state_at_server_and_missing_placement',
       _create_test_measured_process_double(tf.int32, 1, tf.int32),
       _create_test_measured_process_state_missing_placement()),
      ('state_at_clients_and_missing_placement',
       _create_test_measured_process_state_at_clients(),
       _create_test_measured_process_state_missing_placement()),
  ])
  def test_concatenation_with_mixed_state_placement_raises(
      self, first_process, second_process):
    measured_processes = collections.OrderedDict(
        first_process=first_process, second_process=second_process)

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'):
      measured_process.concatenate_measured_processes(measured_processes)


# We verify the AST (Abstract Syntax Trees) of the `initialize` and `next`
# of the concatednated process. So the tests don't need to actually invoke these
# computations and depend on the execution context.
class MeasuredProcessConcatenationASTTest(absltest.TestCase):

  def test_concatenation_with_measured_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        sum=_create_test_measured_process_sum(tf.int32, 0, tf.int32))
    concatenated_process = measured_process.concatenate_measured_processes(
        measured_processes)

    actual_initialize_ast = _name_compiled_computations(
        concatenated_process.initialize.to_building_block())
    actual_next_ast = _name_compiled_computations(
        concatenated_process.next.to_building_block())
    golden.check_string(
        'concatenation_with_measured_processes.expected',
        f'initialize:\n\n{actual_initialize_ast.formatted_representation()}\n\n'
        f'next:\n\n{actual_next_ast.formatted_representation()}')

  def test_concatenation_with_aggregation_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(tf.int32, 1, tf.int32),
        aggregate=_create_test_aggregation_process(tf.int32, 0, tf.int32))
    concatenated_process = measured_process.concatenate_measured_processes(
        measured_processes)

    actual_initialize_ast = _name_compiled_computations(
        concatenated_process.initialize.to_building_block())
    actual_next_ast = _name_compiled_computations(
        concatenated_process.next.to_building_block())
    golden.check_string(
        'concatenation_with_aggregation_processes.expected',
        f'initialize:\n\n{actual_initialize_ast.formatted_representation()}\n\n'
        f'next:\n\n{actual_next_ast.formatted_representation()}')


if __name__ == '__main__':
  absltest.main()
