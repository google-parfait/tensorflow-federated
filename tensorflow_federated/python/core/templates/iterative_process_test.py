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
import federated_language
import numpy as np
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import iterative_process


_IterativeProcessConstructionError = (
    TypeError,
    errors.TemplateInitFnParamNotEmptyError,
    errors.TemplateStateNotAssignableError,
)


@federated_language.federated_computation()
def _initialize():
  return federated_language.federated_value(0, federated_language.SERVER)


@federated_language.federated_computation(
    federated_language.FederatedType(np.int32, federated_language.SERVER)
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

    @federated_language.federated_computation()
    def _initialize_empty():
      return federated_language.federated_value((), federated_language.SERVER)

    @federated_language.federated_computation(
        federated_language.FederatedType((), federated_language.SERVER)
    )
    def _next_empty(state):
      return state

    try:
      iterative_process.IterativeProcess(_initialize_empty, _next_empty)
    except _IterativeProcessConstructionError:
      self.fail('Could not construct an IterativeProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):

    @federated_language.federated_computation()
    def _initialize_unknown():
      return federated_language.federated_value(
          np.array([], np.str_), federated_language.SERVER
      )

    @federated_language.federated_computation(
        federated_language.FederatedType(
            federated_language.TensorType(np.str_, [None]),
            federated_language.SERVER,
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

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.SERVER)
    )
    def _initialize_arg(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      iterative_process.IterativeProcess(_initialize_arg, _next)

  def test_init_state_not_assignable(self):

    @federated_language.federated_computation()
    def _initialize_float():
      return federated_language.federated_value(0.0, federated_language.SERVER)

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(_initialize_float, _next)

  def test_next_state_not_assignable(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.float32, federated_language.SERVER)
    )
    def _next_float(state):
      return state

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(_initialize, _next_float)

  def test_next_state_not_assignable_tuple_result(self):

    @federated_language.federated_computation(
        federated_language.StructType([
            federated_language.FederatedType(
                np.float32, federated_language.SERVER
            ),
            federated_language.FederatedType(
                np.int32, federated_language.CLIENTS
            ),
        ]),
    )
    def _next_float(state, value):
      return state, value

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      iterative_process.IterativeProcess(_initialize, _next_float)


def _create_test_process(
    state_type: federated_language.Type, state: object
) -> iterative_process.IterativeProcess:
  @federated_language.federated_computation
  def _init_process():
    if isinstance(state_type, federated_language.FederatedType):
      return federated_language.federated_value(state, state_type.placement)
    else:
      return state

  @federated_language.federated_computation(state_type, np.int32)
  def _next_process(state, value):
    return state, value

  return iterative_process.IterativeProcess(_init_process, _next_process)


class HasEmptyStateTest(parameterized.TestCase, absltest.TestCase):

  @parameterized.named_parameters(
      ('struct_tuple_empty', federated_language.StructType([]), ()),
      (
          'struct_list_empty',
          federated_language.StructWithPythonType([], list),
          [],
      ),
      (
          'struct_nested_empty',
          federated_language.StructType([[], [[]]]),
          ((), ((),)),
      ),
      (
          'federated_struct_empty',
          federated_language.FederatedType([], federated_language.SERVER),
          (),
      ),
      (
          'federated_struct_nested_empty',
          federated_language.FederatedType(
              [[], [[]]], federated_language.SERVER
          ),
          ((), ((),)),
      ),
  )
  def test_is_stateful_returns_false(self, state_type, state):
    process = _create_test_process(state_type, state)
    self.assertFalse(iterative_process.is_stateful(process))

  @parameterized.named_parameters(
      ('tensor', federated_language.TensorType(np.int32), 1),
      (
          'struct_tuple_tensor',
          federated_language.StructType([np.int32]),
          (1,),
      ),
      (
          'struct_list_tensor',
          federated_language.StructWithPythonType([np.int32], list),
          [1],
      ),
      (
          'struct_nested_tensor',
          federated_language.StructType([[], [[np.int32]]]),
          ((), ((1,),)),
      ),
      (
          'federated_tensor',
          federated_language.FederatedType(np.int32, federated_language.SERVER),
          1,
      ),
      (
          'federated_struct_nested_tensor',
          federated_language.FederatedType(
              [[], [[np.int32]]], federated_language.SERVER
          ),
          ((), ((1,),)),
      ),
  )
  def test_is_stateful_returns_true(self, state_type, state):
    process = _create_test_process(state_type, state)
    self.assertTrue(iterative_process.is_stateful(process))


if __name__ == '__main__':
  absltest.main()
