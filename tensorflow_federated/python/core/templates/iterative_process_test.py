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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


_IterativeProcessConstructionError = (
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


class IterativeProcessTest(absltest.TestCase):

  def test_construction_does_not_raise(self):
    try:
      iterative_process.IterativeProcess(_initialize, _next)
    except _IterativeProcessConstructionError:
      self.fail('Could not construct a valid IterativeProcess.')

  def test_construction_with_empty_state_does_not_raise(self):

    @federated_computation.federated_computation()
    def _initialize_empty():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.FederatedType((), placements.SERVER)
    )
    def _next_empty(state):
      return state

    try:
      iterative_process.IterativeProcess(_initialize_empty, _next_empty)
    except _IterativeProcessConstructionError:
      self.fail('Could not construct an IterativeProcess with empty state.')

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

    try:
      iterative_process.IterativeProcess(_initialize_unknown, _next_unknown)
    except _IterativeProcessConstructionError:
      self.fail(
          'Could not construct an IterativeProcess with parameter types '
          'with statically unknown shape.'
      )

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      iterative_process.IterativeProcess(initialize_fn=lambda: 0, next_fn=_next)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      iterative_process.IterativeProcess(
          initialize_fn=_initialize, next_fn=lambda state: state
      )

  def test_init_param_not_empty_raises(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.SERVER)
    )
    def _initialize_arg(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      iterative_process.IterativeProcess(_initialize_arg, _next)

  def test_init_state_not_assignable(self):
    @federated_computation.federated_computation()
    def _initialize_float():
      return intrinsics.federated_value(0.0, placements.SERVER)

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(_initialize_float, _next)

  def test_next_state_not_assignable(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.float32, placements.SERVER)
    )
    def _next_float(state):
      return state

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(_initialize, _next_float)

  def test_next_state_not_assignable_tuple_result(self):
    @federated_computation.federated_computation(
        computation_types.StructType([
            computation_types.FederatedType(np.float32, placements.SERVER),
            computation_types.FederatedType(np.int32, placements.CLIENTS),
        ]),
    )
    def _next_float(state, value):
      return state, value

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(_initialize, _next_float)


def _create_test_process(
    state_type: computation_types.Type, state: object
) -> iterative_process.IterativeProcess:
  @federated_computation.federated_computation
  def _init_process():
    if isinstance(state_type, computation_types.FederatedType):
      return intrinsics.federated_value(state, state_type.placement)
    else:
      return state

  @federated_computation.federated_computation(state_type, np.int32)
  def _next_process(state, value):
    return state, value

  return iterative_process.IterativeProcess(_init_process, _next_process)


class HasEmptyStateTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      ('struct_tuple_empty', computation_types.StructType([]), ()),
      (
          'struct_list_empty',
          computation_types.StructWithPythonType([], list),
          [],
      ),
      (
          'struct_nested_empty',
          computation_types.StructType([[], [[]]]),
          ((), ((),)),
      ),
      (
          'federated_struct_empty',
          computation_types.FederatedType([], placements.SERVER),
          (),
      ),
      (
          'federated_struct_nested_empty',
          computation_types.FederatedType([[], [[]]], placements.SERVER),
          ((), ((),)),
      ),
  )
  def test_is_stateful_returns_false(self, state_type, state):
    process = _create_test_process(state_type, state)
    self.assertFalse(iterative_process.is_stateful(process))

  @parameterized.named_parameters(
      ('tensor', computation_types.TensorType(np.int32), 1),
      (
          'struct_tuple_tensor',
          computation_types.StructType([np.int32]),
          (1,),
      ),
      (
          'struct_list_tensor',
          computation_types.StructWithPythonType([np.int32], list),
          [1],
      ),
      (
          'struct_nested_tensor',
          computation_types.StructType([[], [[np.int32]]]),
          ((), ((1,),)),
      ),
      (
          'federated_tensor',
          computation_types.FederatedType(np.int32, placements.SERVER),
          1,
      ),
      (
          'federated_struct_nested_tensor',
          computation_types.FederatedType(
              [[], [[np.int32]]], placements.SERVER
          ),
          ((), ((1,),)),
      ),
  )
  def test_is_stateful_returns_true(self, state_type, state):
    process = _create_test_process(state_type, state)
    self.assertTrue(iterative_process.is_stateful(process))


if __name__ == '__main__':
  absltest.main()
