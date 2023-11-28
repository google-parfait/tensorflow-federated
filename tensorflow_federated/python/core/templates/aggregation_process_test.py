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
import numpy as np

from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process

_AggregationProcessConstructionError = (
    TypeError,
    errors.TemplateInitFnParamNotEmptyError,
    errors.TemplateStateNotAssignableError,
    errors.TemplateNotMeasuredProcessOutputError,
    errors.TemplateNextFnNumArgsError,
    aggregation_process.AggregationNotFederatedError,
    aggregation_process.AggregationPlacementError,
)


def _server_zero():
  """Returns zero integer placed at SERVER."""
  return intrinsics.federated_value(0, placements.SERVER)


@federated_computation.federated_computation()
def _initialize():
  return _server_zero()


@federated_computation.federated_computation(
    computation_types.FederatedType(np.int32, placements.SERVER),
    computation_types.FederatedType(np.float32, placements.CLIENTS),
)
def _next(state, value):
  return measured_process.MeasuredProcessOutput(
      state,
      intrinsics.federated_sum(value),
      intrinsics.federated_value(1, placements.SERVER),
  )


class AggregationProcessTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      aggregation_process.AggregationProcess(_initialize, _next)
    except _AggregationProcessConstructionError:
      self.fail('Could not construct a valid AggregationProcess.')

  def test_construction_with_empty_state_does_not_raise(self):

    @federated_computation.federated_computation()
    def _initialize_empty():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.FederatedType((), placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def _next_empty(state, value):
      return measured_process.MeasuredProcessOutput(
          state,
          intrinsics.federated_sum(value),
          intrinsics.federated_value(1, placements.SERVER),
      )

    try:
      aggregation_process.AggregationProcess(_initialize_empty, _next_empty)
    except _AggregationProcessConstructionError:
      self.fail('Could not construct an AggregationProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):

    @federated_computation.federated_computation()
    def _initialize_unknown():
      return intrinsics.federated_value(
          np.array([], np.str_), placements.SERVER
      )

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            computation_types.TensorType(np.str_, [None]),
            placements.SERVER,
        ),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def next_fn(strings, value):
      return measured_process.MeasuredProcessOutput(
          strings,
          intrinsics.federated_sum(value),
          intrinsics.federated_value(1, placements.SERVER),
      )

    try:
      aggregation_process.AggregationProcess(_initialize_unknown, next_fn)
    except _AggregationProcessConstructionError:
      self.fail(
          'Could not construct an AggregationProcess with parameter '
          'types with statically unknown shape.'
      )

  def test_construction_with_value_type_mismatch_does_not_raise(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def next_fn(state, value):
      del value  # Unused.
      return measured_process.MeasuredProcessOutput(
          state, state, _server_zero()
      )

    try:
      aggregation_process.AggregationProcess(_initialize, next_fn)
    except _AggregationProcessConstructionError:
      self.fail(
          'Could not construct an AggregationProcess with different '
          'client and server placed types.'
      )

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      aggregation_process.AggregationProcess(
          initialize_fn=lambda: 0, next_fn=_next
      )

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      aggregation_process.AggregationProcess(
          initialize_fn=_initialize,
          next_fn=lambda state: measured_process.MeasuredProcessOutput(
              state, (), ()
          ),
      )

  def test_init_param_not_empty_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER)
    )
    def _initialize_arg(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      aggregation_process.AggregationProcess(_initialize_arg, _next)

  def test_init_state_not_assignable(self):

    @federated_computation.federated_computation()
    def _initialize_float():
      return intrinsics.federated_value(0.0, placements.SERVER)

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      aggregation_process.AggregationProcess(_initialize_float, _next)

  def test_next_state_not_assignable(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def _next_float(state, value):
      del state  # Unused.
      return measured_process.MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          intrinsics.federated_sum(value),
          _server_zero(),
      )

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      aggregation_process.AggregationProcess(_initialize, _next_float)

  def test_measured_process_output_as_state_raises(self):
    no_value = lambda: intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation()
    def initialize_fn():
      return intrinsics.federated_zip(
          measured_process.MeasuredProcessOutput(
              no_value(), no_value(), no_value()
          )
      )

    @federated_computation.federated_computation(
        initialize_fn.type_signature.result,
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def next_fn(state, value):
      del state, value
      return measured_process.MeasuredProcessOutput(
          no_value(), no_value(), no_value()
      )

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  def test_next_return_tuple_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def tuple_next_fn(state, value):
      return state, intrinsics.federated_sum(value), _server_zero()

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      aggregation_process.AggregationProcess(_initialize, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements']
    )

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def namedtuple_next_fn(state, value):
      return measured_process_output(
          state, intrinsics.federated_sum(value), _server_zero()
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      aggregation_process.AggregationProcess(_initialize, namedtuple_next_fn)

  def test_next_return_odict_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def odict_next_fn(state, value):
      return collections.OrderedDict(
          state=state,
          result=intrinsics.federated_sum(value),
          measurements=_server_zero(),
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      aggregation_process.AggregationProcess(_initialize, odict_next_fn)

  def test_federated_measured_process_output_raises(self):
    # Using federated_zip to place FederatedType at the top of the hierarchy.
    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def next_fn(state, value):
      return intrinsics.federated_zip(
          measured_process.MeasuredProcessOutput(
              state, intrinsics.federated_sum(value), _server_zero()
          )
      )

    # A MeasuredProcessOutput containing three `FederatedType`s is different
    # than a `FederatedType` containing a MeasuredProcessOutput. Corrently, only
    # the former is considered valid.
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      aggregation_process.AggregationProcess(_initialize, next_fn)

  # Tests specific only for the AggregationProcess contract below.

  def test_non_federated_init_next_raises(self):

    @federated_computation.federated_computation()
    def _initialize_unplaced():
      return 0

    @federated_computation.federated_computation(np.int32, np.float32)
    def _next_unplaced(state, value):
      return measured_process.MeasuredProcessOutput(state, value, ())

    with self.assertRaises(aggregation_process.AggregationNotFederatedError):
      aggregation_process.AggregationProcess(
          _initialize_unplaced, _next_unplaced
      )

  def test_init_tuple_of_federated_types_raises(self):

    @federated_computation.federated_computation()
    def _initialize_tuple():
      return (_server_zero(), _server_zero())

    @federated_computation.federated_computation(
        computation_types.StructType([
            computation_types.FederatedType(np.int32, placements.SERVER),
            computation_types.FederatedType(np.int32, placements.SERVER),
        ]),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def _next_tuple(state, value):
      return measured_process.MeasuredProcessOutput(
          state,
          intrinsics.federated_sum(value),
          intrinsics.federated_value(1, placements.SERVER),
      )

    with self.assertRaises(aggregation_process.AggregationNotFederatedError):
      aggregation_process.AggregationProcess(_initialize_tuple, _next_tuple)

  def test_non_server_placed_init_state_raises(self):

    @federated_computation.federated_computation()
    def _initialize_clients():
      return intrinsics.federated_value(0, placements.CLIENTS)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def _next_non_server(state, value):
      return measured_process.MeasuredProcessOutput(
          state,
          intrinsics.federated_sum(value),
          intrinsics.federated_value(1, placements.SERVER),
      )

    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(
          _initialize_clients, _next_non_server
      )

  def test_single_param_next_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS),
    )
    def _next_single_parameter(state):
      return measured_process.MeasuredProcessOutput(
          state,
          intrinsics.federated_value(1.0, placements.SERVER),
          intrinsics.federated_value(1, placements.SERVER),
      )

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      aggregation_process.AggregationProcess(
          _initialize, _next_single_parameter
      )

  def test_non_clients_placed_next_value_param_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.SERVER),
    )
    def _next_non_clients(state, value):
      return measured_process.MeasuredProcessOutput(
          state,
          value,
          intrinsics.federated_value(1, placements.SERVER),
      )

    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(_initialize, _next_non_clients)

  def test_non_server_placed_next_result_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.SERVER),
    )
    def _next_non_server_result(state, value):
      return measured_process.MeasuredProcessOutput(
          state,
          value,
          intrinsics.federated_value(1, placements.SERVER),
      )

    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(
          _initialize, _next_non_server_result
      )

  def test_non_server_placed_next_measurements_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.int32, placements.CLIENTS),
    )
    def next_fn(state, value):
      return measured_process.MeasuredProcessOutput(
          state, intrinsics.federated_sum(value), value
      )

    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(_initialize, next_fn)

  def test_is_weighted_property(self):
    process = aggregation_process.AggregationProcess(_initialize, _next)
    self.assertFalse(process.is_weighted)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def weighted_next_fn(state, value, weight):
      del weight
      return measured_process.MeasuredProcessOutput(
          state,
          intrinsics.federated_sum(value),
          intrinsics.federated_value(1, placements.SERVER),
      )

    process = aggregation_process.AggregationProcess(
        _initialize, weighted_next_fn
    )
    self.assertTrue(process.is_weighted)


if __name__ == '__main__':
  absltest.main()
