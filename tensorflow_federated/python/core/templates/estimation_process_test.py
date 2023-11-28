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

from absl.testing import absltest
import numpy as np

from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import estimation_process

_EstimationProcessConstructionError = (
    TypeError,
    errors.TemplateInitFnParamNotEmptyError,
    errors.TemplateStateNotAssignableError,
)


@federated_computation.federated_computation()
def _initialize():
  return intrinsics.federated_value(0, placements.SERVER)


@federated_computation.federated_computation(
    computation_types.FederatedType(np.int32, placements.SERVER)
)
def _next(state):
  return state


@federated_computation.federated_computation(
    computation_types.FederatedType(np.int32, placements.SERVER)
)
def _report(state):
  del state  # Unused.
  return intrinsics.federated_value(1.0, placements.SERVER)


@federated_computation.federated_computation(
    computation_types.FederatedType(np.float32, placements.SERVER)
)
def _map(state):
  return (state, state)


class EstimationProcessTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      estimation_process.EstimationProcess(_initialize, _next, _report)
    except _EstimationProcessConstructionError:
      self.fail('Could not construct a valid EstimationProcess.')

  def test_construction_with_empty_state_does_not_raise(self):

    @federated_computation.federated_computation()
    def _initialize_empty():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.FederatedType((), placements.SERVER)
    )
    def _next_empty(state):
      return (state, 1.0)

    @federated_computation.federated_computation(
        computation_types.FederatedType((), placements.SERVER)
    )
    def _report_empty(state):
      return state

    try:
      estimation_process.EstimationProcess(
          _initialize_empty, _next_empty, _report_empty
      )
    except _EstimationProcessConstructionError:
      self.fail('Could not construct an EstimationProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):

    @federated_computation.federated_computation()
    def _initialize_unknown():
      return intrinsics.federated_value(
          np.array([], np.str_), placements.SERVER
      )

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            computation_types.TensorType(np.str_, [None]), placements.SERVER
        )
    )
    def _next_unknown(state):
      return state

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            computation_types.TensorType(np.str_, [None]), placements.SERVER
        )
    )
    def _report_unknown(state):
      return state

    try:
      estimation_process.EstimationProcess(
          _initialize_unknown, _next_unknown, _report_unknown
      )
    except _EstimationProcessConstructionError:
      self.fail(
          'Could not construct an EstimationProcess with parameter types '
          'with statically unknown shape.'
      )

  def test_init_not_tff_computation_raises(self):

    def _initialize_py():
      return 0

    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(_initialize_py, _next, _report)

  def test_next_not_tff_computation_raises(self):

    def _next_py(x):
      return x

    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(_initialize, _next_py, _report)

  def test_report_not_tff_computation_raises(self):

    def _report_py(x):
      return x

    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      estimation_process.EstimationProcess(_initialize, _next, _report_py)

  def test_init_param_not_empty_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER)
    )
    def _initialize_arg(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      estimation_process.EstimationProcess(_initialize_arg, _next, _report)

  def test_init_state_not_assignable(self):

    @federated_computation.federated_computation()
    def _initialize_float():
      return intrinsics.federated_value(0.0, placements.SERVER)

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(_initialize_float, _next, _report)

  def test_next_state_not_assignable(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.float32, placements.SERVER)
    )
    def _next_float(state):
      return state

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(_initialize, _next_float, _report)

  def test_next_state_not_assignable_tuple_result(self):
    @federated_computation.federated_computation(
        computation_types.StructType([
            computation_types.FederatedType(np.float32, placements.SERVER),
            computation_types.FederatedType(np.float32, placements.SERVER),
        ]),
    )
    def _next_float(state, value):
      return state, value

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(_initialize, _next_float, _report)

  # Tests specific only for the EstimationProcess contract below.

  def test_report_state_not_assignable(self):

    @federated_computation.federated_computation(np.float32)
    def _report_float(state):
      return state

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      estimation_process.EstimationProcess(_initialize, _next, _report_float)

  def test_mapped_process_as_expected(self):
    process = estimation_process.EstimationProcess(_initialize, _next, _report)
    mapped_process = process.map(_map)

    self.assertIsInstance(mapped_process, estimation_process.EstimationProcess)
    self.assertEqual(process.initialize, mapped_process.initialize)
    self.assertEqual(process.next, mapped_process.next)
    self.assertEqual(
        process.report.type_signature.parameter,
        mapped_process.report.type_signature.parameter,
    )
    self.assertEqual(
        _map.type_signature.result,
        mapped_process.report.type_signature.result,
    )

  def test_map_estimate_not_assignable(self):

    @federated_computation.federated_computation(np.int32)
    def _map_int(x):
      return x

    process = estimation_process.EstimationProcess(_initialize, _next, _report)
    with self.assertRaises(estimation_process.EstimateNotAssignableError):
      process.map(_map_int)


if __name__ == '__main__':
  absltest.main()
