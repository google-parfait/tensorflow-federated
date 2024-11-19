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
import federated_language
import numpy as np

from tensorflow_federated.python.core.impl.compiler import compiler_test_utils
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process
from tensorflow_federated.python.core.templates import measured_process

_MeasuredProcessConstructionError = (
    TypeError,
    errors.TemplateInitFnParamNotEmptyError,
    errors.TemplateStateNotAssignableError,
    errors.TemplateNotMeasuredProcessOutputError,
)


@federated_language.federated_computation()
def _initialize():
  return federated_language.federated_value(0, federated_language.SERVER)


@federated_language.federated_computation(
    federated_language.StructType([
        federated_language.FederatedType(np.int32, federated_language.SERVER),
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
    ])
)
def _next(state, value):
  return measured_process.MeasuredProcessOutput(state, value, ())


class MeasuredProcessTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      measured_process.MeasuredProcess(_initialize, _next)
    except _MeasuredProcessConstructionError:
      self.fail('Could not construct a valid MeasuredProcess.')

  def test_construction_with_empty_state_does_not_raise(self):

    @federated_language.federated_computation()
    def _initialize_empty():
      return federated_language.federated_value((), federated_language.SERVER)

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType((), federated_language.SERVER),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ])
    )
    def _next_empty(state, value):
      return measured_process.MeasuredProcessOutput(state, value, ())

    try:
      measured_process.MeasuredProcess(_initialize_empty, _next_empty)
    except _MeasuredProcessConstructionError:
      self.fail('Could not construct an MeasuredProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):

    @federated_language.federated_computation()
    def _initialize_unknown():
      return federated_language.federated_value(
          np.array([], np.str_), federated_language.SERVER
      )

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType(
                federated_language.TensorType(np.str_, [None]),
                federated_language.SERVER,
            ),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ])
    )
    def _next_unknown(state, value):
      return measured_process.MeasuredProcessOutput(state, value, ())

    try:
      measured_process.MeasuredProcess(_initialize_unknown, _next_unknown)
    except _MeasuredProcessConstructionError:
      self.fail(
          'Could not construct an MeasuredProcess with parameter types '
          'with statically unknown shape.'
      )

  def test_init_not_tff_computation_raises(self):

    def _initialize_py():
      return 0

    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      measured_process.MeasuredProcess(_initialize_py, _next)

  def test_next_not_tff_computation_raises(self):

    def _next_py(state, value):
      return measured_process.MeasuredProcessOutput(state, value, ())

    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      measured_process.MeasuredProcess(_initialize, _next_py)

  def test_init_param_not_empty_raises(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.SERVER)
    )
    def _initialize_arg(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      measured_process.MeasuredProcess(_initialize_arg, _next)

  def test_init_state_not_assignable(self):

    @federated_language.federated_computation()
    def _initialize_float():
      return federated_language.federated_value(0.0, federated_language.SERVER)

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(_initialize_float, _next)

  def test_next_state_not_assignable(self):

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ])
    )
    def _next_float(state, value):
      return measured_process.MeasuredProcessOutput(state, value, ())

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(_initialize, _next_float)

  # Tests specific only for the MeasuredProcess contract below.

  def test_measured_process_output_as_state_raises(self):

    @federated_language.federated_computation()
    def _initialize_process():
      value = measured_process.MeasuredProcessOutput((), (), ())
      return federated_language.federated_value(
          value, federated_language.SERVER
      )

    @federated_language.federated_computation(
        federated_language.StructWithPythonType(
            elements=[[
                federated_language.StructType([]),
                federated_language.StructType([]),
                federated_language.StructType([]),
            ]],
            container_type=measured_process.MeasuredProcessOutput,
        )
    )
    def _next_process(state):
      return state

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(_initialize_process, _next_process)

  def test_next_return_tensor_type_raises(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.SERVER),
    )
    def _next_tensor(state):
      return state

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(_initialize, _next_tensor)

  def test_next_return_tuple_raises(self):

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType(
                np.int32, federated_language.SERVER
            ),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ])
    )
    def _next_tuple(state, value):
      return (state, value, ())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(_initialize, _next_tuple)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements']
    )

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType(
                np.int32, federated_language.SERVER
            ),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ])
    )
    def _next_named_tuple(state, value):
      return measured_process_output(state, value, ())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(_initialize, _next_named_tuple)

  def test_next_return_odict_raises(self):

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType(
                np.int32, federated_language.SERVER
            ),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ])
    )
    def _next_odict(state, value):
      return collections.OrderedDict([
          ('state', state),
          ('result', value),
          ('measurements', ()),
      ])

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      measured_process.MeasuredProcess(_initialize, _next_odict)

  def test_federated_measured_process_output_raises(self):
    initialize_fn = federated_language.federated_computation()(
        lambda: federated_language.federated_value(0, federated_language.SERVER)
    )
    empty = lambda: federated_language.federated_value(
        (), federated_language.SERVER
    )
    state_type = initialize_fn.type_signature.result

    # Using federated_zip to place FederatedType at the top of the hierarchy.
    @federated_language.federated_computation(state_type)
    def next_fn(state):
      return federated_language.federated_zip(
          measured_process.MeasuredProcessOutput(state, empty(), empty())
      )

    # A MeasuredProcessOutput containing three `FederatedType`s is different
    # than a `FederatedType` containing a MeasuredProcessOutput. Corrently, only
    # the former is considered valid.
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)


def _create_test_measured_process_double(state_type, state_init, values_type):

  @federated_language.federated_computation()
  def _initialize_double():
    return federated_language.federated_value(
        state_init, federated_language.SERVER
    )

  @federated_language.federated_computation()
  def _double(x):
    return x

  @federated_language.federated_computation(
      federated_language.FederatedType(state_type, federated_language.SERVER),
      federated_language.FederatedType(values_type, federated_language.CLIENTS),
  )
  def _next_double(state, values):
    return measured_process.MeasuredProcessOutput(
        state=federated_language.federated_map(_double, state),
        result=federated_language.federated_map(_double, values),
        measurements=federated_language.federated_value(
            collections.OrderedDict(a=1), federated_language.SERVER
        ),
    )

  return measured_process.MeasuredProcess(_initialize_double, _next_double)


def _create_test_measured_process_sum(state_type, state_init, values_type):

  @federated_language.federated_computation()
  def _initialize_sum():
    return federated_language.federated_value(
        state_init, federated_language.SERVER
    )

  @federated_language.federated_computation()
  def _sum(x):
    return x

  @federated_language.federated_computation(
      federated_language.FederatedType(state_type, federated_language.SERVER),
      federated_language.FederatedType(values_type, federated_language.CLIENTS),
  )
  def _next_sum(state, values):
    return measured_process.MeasuredProcessOutput(
        state=federated_language.federated_map(_sum, state),
        result=federated_language.federated_sum(values),
        measurements=federated_language.federated_value(
            collections.OrderedDict(b=2), federated_language.SERVER
        ),
    )

  return measured_process.MeasuredProcess(_initialize_sum, _next_sum)


def _create_test_measured_process_state_at_clients():

  @federated_language.federated_computation(
      federated_language.FederatedType(np.int32, federated_language.CLIENTS),
      federated_language.FederatedType(np.int32, federated_language.CLIENTS),
  )
  def next_fn(state, values):
    return measured_process.MeasuredProcessOutput(
        state,
        federated_language.federated_sum(values),
        federated_language.federated_value(1, federated_language.SERVER),
    )

  return measured_process.MeasuredProcess(
      initialize_fn=federated_language.federated_computation(
          lambda: federated_language.federated_value(
              0, federated_language.CLIENTS
          )
      ),
      next_fn=next_fn,
  )


def _create_test_measured_process_state_missing_placement():

  @federated_language.federated_computation()
  def _initialize_unplaced():
    return 0

  @federated_language.federated_computation(
      federated_language.StructType([np.int32, np.int32])
  )
  def _next_unplaced(state, value):
    return measured_process.MeasuredProcessOutput(state, value, ())

  return measured_process.MeasuredProcess(_initialize_unplaced, _next_unplaced)


def _create_test_aggregation_process(state_type, state_init, values_type):

  @federated_language.federated_computation(
      federated_language.FederatedType(state_type, federated_language.SERVER),
      federated_language.FederatedType(values_type, federated_language.CLIENTS),
  )
  def next_fn(state, values):
    return measured_process.MeasuredProcessOutput(
        state,
        federated_language.federated_sum(values),
        federated_language.federated_value(1, federated_language.SERVER),
    )

  return aggregation_process.AggregationProcess(
      initialize_fn=federated_language.federated_computation(
          lambda: federated_language.federated_value(
              state_init, federated_language.SERVER
          )
      ),
      next_fn=next_fn,
  )


def _create_test_iterative_process(state_type, state_init):

  @federated_language.federated_computation()
  def _initialize_ip():
    return state_init

  @federated_language.federated_computation(state_type)
  def _next_ip(state):
    return state

  return iterative_process.IterativeProcess(_initialize_ip, _next_ip)


class MeasuredProcessCompositionComputationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('all_measured_processes', _create_test_measured_process_sum),
      ('with_aggregation_process', _create_test_aggregation_process),
  ])
  def test_composition_type_properties(self, last_process):
    state_type = np.float32
    values_type = np.int32
    last_process = last_process(state_type, 0.0, values_type)
    composite_process = measured_process.chain_measured_processes(
        collections.OrderedDict(
            double=_create_test_measured_process_double(
                state_type, 1.0, values_type
            ),
            last_process=last_process,
        )
    )
    self.assertIsInstance(composite_process, measured_process.MeasuredProcess)

    expected_state_type = federated_language.FederatedType(
        collections.OrderedDict(double=state_type, last_process=state_type),
        federated_language.SERVER,
    )
    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        composite_process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    param_value_type = federated_language.FederatedType(
        values_type, federated_language.CLIENTS
    )
    result_value_type = federated_language.FederatedType(
        values_type, federated_language.SERVER
    )
    expected_measurements_type = federated_language.FederatedType(
        collections.OrderedDict(
            double=collections.OrderedDict(a=np.int32),
            last_process=last_process.next.type_signature.result.measurements.member,
        ),
        federated_language.SERVER,
    )
    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, values=param_value_type
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type
        ),
    )
    self.assertTrue(
        composite_process.next.type_signature.is_equivalent_to(
            expected_next_type
        )
    )

  def test_values_type_mismatching_raises(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        sum=_create_test_measured_process_sum(np.int32, 0, np.float32),
    )

    first_process_result_type = measured_processes[
        'double'
    ].next.type_signature.result.result
    second_process_values_type = measured_processes[
        'sum'
    ].next.type_signature.parameter.values
    self.assertFalse(
        second_process_values_type.is_equivalent_to(first_process_result_type)
    )

    with self.assertRaisesRegex(TypeError, 'Cannot call function'):
      measured_process.chain_measured_processes(measured_processes)

  def test_composition_with_iterative_process_raises(self):
    processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        iterative=_create_test_iterative_process(np.int32, 0),
    )

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'
    ):
      measured_process.chain_measured_processes(processes)

  @parameterized.named_parameters([
      (
          'state_at_server_and_clients',
          _create_test_measured_process_double(np.int32, 1, np.int32),
          _create_test_measured_process_state_at_clients(),
      ),
      (
          'state_at_server_and_missing_placement',
          _create_test_measured_process_double(np.int32, 1, np.int32),
          _create_test_measured_process_state_missing_placement(),
      ),
      (
          'state_at_clients_and_missing_placement',
          _create_test_measured_process_state_at_clients(),
          _create_test_measured_process_state_missing_placement(),
      ),
  ])
  def test_composition_with_mixed_state_placement_raises(
      self, first_process, second_process
  ):
    measured_processes = collections.OrderedDict(
        first_process=first_process, second_process=second_process
    )

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'
    ):
      measured_process.chain_measured_processes(measured_processes)


# We verify the AST (Abstract Syntax Trees) of the `initialize` and `next`
# of the composite process. So the tests don't need to actually invoke these
# computations and depend on the execution context.
class MeasuredProcessCompositionASTTest(absltest.TestCase):

  def test_composition_with_measured_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        sum=_create_test_measured_process_sum(np.int32, 0, np.int32),
    )
    composite_process = measured_process.chain_measured_processes(
        measured_processes
    )
    computations = collections.OrderedDict(
        initialize=composite_process.initialize.to_building_block(),
        next=composite_process.next.to_building_block(),
    )
    compiler_test_utils.check_computations(
        'composition_with_measured_processes.expected', computations
    )

  def test_composition_with_aggregation_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        aggregate=_create_test_aggregation_process(np.int32, 0, np.int32),
    )
    composite_process = measured_process.chain_measured_processes(
        measured_processes
    )
    computations = collections.OrderedDict(
        initialize=composite_process.initialize.to_building_block(),
        next=composite_process.next.to_building_block(),
    )
    compiler_test_utils.check_computations(
        'composition_with_aggregation_processes.expected', computations
    )


class MeasuredProcessConcatenationComputationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('all_measured_processes', _create_test_measured_process_sum),
      ('with_aggregation_process', _create_test_aggregation_process),
  ])
  def test_concatenation_type_properties(self, last_process):
    state_type = np.int32
    values_type = np.int32
    last_process = last_process(state_type, 0, values_type)
    concatenated_process = measured_process.concatenate_measured_processes(
        collections.OrderedDict(
            double=_create_test_measured_process_double(
                state_type, 1, values_type
            ),
            last_process=last_process,
        )
    )
    self.assertIsInstance(
        concatenated_process, measured_process.MeasuredProcess
    )

    expected_state_type = federated_language.FederatedType(
        collections.OrderedDict(double=state_type, last_process=state_type),
        federated_language.SERVER,
    )
    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        concatenated_process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    param_value_type = collections.OrderedDict(
        double=federated_language.FederatedType(
            values_type, federated_language.CLIENTS
        ),
        last_process=federated_language.FederatedType(
            values_type, federated_language.CLIENTS
        ),
    )
    result_value_type = collections.OrderedDict(
        double=federated_language.FederatedType(
            values_type, federated_language.CLIENTS
        ),
        last_process=federated_language.FederatedType(
            values_type, federated_language.SERVER
        ),
    )
    expected_measurements_type = federated_language.FederatedType(
        collections.OrderedDict(
            double=collections.OrderedDict(a=np.int32),
            last_process=last_process.next.type_signature.result.measurements.member,
        ),
        federated_language.SERVER,
    )
    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, values=param_value_type
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type
        ),
    )
    self.assertTrue(
        concatenated_process.next.type_signature.is_equivalent_to(
            expected_next_type
        )
    )

  def test_concatenation_with_iterative_process_raises(self):
    processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        iterative=_create_test_iterative_process(np.int32, 0),
    )

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'
    ):
      measured_process.concatenate_measured_processes(processes)

  @parameterized.named_parameters([
      (
          'state_at_server_and_clients',
          _create_test_measured_process_double(np.int32, 1, np.int32),
          _create_test_measured_process_state_at_clients(),
      ),
      (
          'state_at_server_and_missing_placement',
          _create_test_measured_process_double(np.int32, 1, np.int32),
          _create_test_measured_process_state_missing_placement(),
      ),
      (
          'state_at_clients_and_missing_placement',
          _create_test_measured_process_state_at_clients(),
          _create_test_measured_process_state_missing_placement(),
      ),
  ])
  def test_concatenation_with_mixed_state_placement_raises(
      self, first_process, second_process
  ):
    measured_processes = collections.OrderedDict(
        first_process=first_process, second_process=second_process
    )

    with self.assertRaisesRegex(
        TypeError, 'Cannot concatenate the initialization functions'
    ):
      measured_process.concatenate_measured_processes(measured_processes)


# We verify the AST (Abstract Syntax Trees) of the `initialize` and `next`
# of the concatednated process. So the tests don't need to actually invoke these
# computations and depend on the execution context.
class MeasuredProcessConcatenationASTTest(absltest.TestCase):

  def test_concatenation_with_measured_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        sum=_create_test_measured_process_sum(np.int32, 0, np.int32),
    )
    concatenated_process = measured_process.concatenate_measured_processes(
        measured_processes
    )
    computations = collections.OrderedDict(
        initialize=concatenated_process.initialize.to_building_block(),
        next=concatenated_process.next.to_building_block(),
    )
    compiler_test_utils.check_computations(
        'concatenation_with_measured_processes.expected', computations
    )

  def test_concatenation_with_aggregation_processes(self):
    measured_processes = collections.OrderedDict(
        double=_create_test_measured_process_double(np.int32, 1, np.int32),
        aggregate=_create_test_aggregation_process(np.int32, 0, np.int32),
    )
    concatenated_process = measured_process.concatenate_measured_processes(
        measured_processes
    )
    computations = collections.OrderedDict(
        initialize=concatenated_process.initialize.to_building_block(),
        next=concatenated_process.next.to_building_block(),
    )
    compiler_test_utils.check_computations(
        'concatenation_with_aggregation_processes.expected', computations
    )


if __name__ == '__main__':
  absltest.main()
