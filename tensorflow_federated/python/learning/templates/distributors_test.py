# Copyright 2021, The TensorFlow Federated Authors.
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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.templates import distributors

SERVER_INT = computation_types.FederatedType(np.int32, placements.SERVER)
SERVER_FLOAT = computation_types.FederatedType(np.float32, placements.SERVER)
CLIENTS_INT = computation_types.FederatedType(np.int32, placements.CLIENTS)
MeasuredProcessOutput = measured_process.MeasuredProcessOutput

_DistributionProcessConstructionError = (
    errors.TemplateNotFederatedError,
    errors.TemplatePlacementError,
    errors.TemplateNextFnNumArgsError,
)


def server_zero():
  return intrinsics.federated_value(0, placements.SERVER)


@federated_computation.federated_computation()
def test_initialize_fn():
  return server_zero()


@federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
def test_next_fn(state, val):
  return MeasuredProcessOutput(
      state,
      intrinsics.federated_broadcast(val),
      intrinsics.federated_value(1, placements.SERVER),
  )


class DistributionProcessTest(tf.test.TestCase):

  def test_construction_does_not_raise(self):
    try:
      distributors.DistributionProcess(test_initialize_fn, test_next_fn)
    except _DistributionProcessConstructionError:
      self.fail('Could not construct a valid DistributionProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = federated_computation.federated_computation()(
        lambda: intrinsics.federated_value((), placements.SERVER)
    )

    @federated_computation.federated_computation(
        initialize_fn.type_signature.result, SERVER_FLOAT
    )
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_broadcast(val),
          intrinsics.federated_value(1, placements.SERVER),
      )

    try:
      distributors.DistributionProcess(initialize_fn, next_fn)
    except _DistributionProcessConstructionError:
      self.fail('Could not construct an DistributionProcess with empty state.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      distributors.DistributionProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn
      )

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      distributors.DistributionProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state, val: MeasuredProcessOutput(state, (), ()),
      )

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = federated_computation.federated_computation(
        SERVER_INT
    )(lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      distributors.DistributionProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = federated_computation.federated_computation()(
        lambda: intrinsics.federated_value(0.0, placements.SERVER)
    )
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      distributors.DistributionProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):
    @federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
    def float_next_fn(state, val):
      del state
      return MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          intrinsics.federated_broadcast(val),
          server_zero(),
      )

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      distributors.DistributionProcess(test_initialize_fn, float_next_fn)

  def test_next_return_tuple_raises(self):
    @federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
    def tuple_next_fn(state, val):
      return state, intrinsics.federated_broadcast(val), server_zero()

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      distributors.DistributionProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements']
    )

    @federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
    def namedtuple_next_fn(state, val):
      return measured_process_output(
          state, intrinsics.federated_broadcast(val), server_zero()
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      distributors.DistributionProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):
    @federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
    def odict_next_fn(state, val):
      return collections.OrderedDict(
          state=state,
          result=intrinsics.federated_broadcast(val),
          measurements=server_zero(),
      )

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      distributors.DistributionProcess(test_initialize_fn, odict_next_fn)

  # Tests specific only for the DistributionProcess contract below.

  def test_construction_with_value_type_mismatch_does_not_raise(self):
    bad_cast_fn = tensorflow_computation.tf_computation(
        lambda x: tf.cast(x, tf.float64)
    )

    @federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
    def next_fn(state, val):
      result = intrinsics.federated_map(
          bad_cast_fn, intrinsics.federated_broadcast(val)
      )
      return MeasuredProcessOutput(
          state, result, intrinsics.federated_value(1, placements.SERVER)
      )

    try:
      distributors.DistributionProcess(test_initialize_fn, next_fn)
    except _DistributionProcessConstructionError:
      self.fail(
          'Could not construct an DistributionProcess with different '
          'client and server placed types.'
      )

  def test_non_federated_init_next_raises(self):
    initialize_fn = tensorflow_computation.tf_computation(lambda: 0)

    @tensorflow_computation.tf_computation(np.int32, np.float32)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, val, ())

    with self.assertRaises(errors.TemplateNotFederatedError):
      distributors.DistributionProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = federated_computation.federated_computation()(
        lambda: (server_zero(), server_zero())
    )

    @federated_computation.federated_computation(
        initialize_fn.type_signature.result, SERVER_FLOAT
    )
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state, intrinsics.federated_broadcast(val), server_zero()
      )

    with self.assertRaises(errors.TemplateNotFederatedError):
      distributors.DistributionProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = federated_computation.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS)
    )

    @federated_computation.federated_computation(CLIENTS_INT, SERVER_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state, intrinsics.federated_broadcast(val), server_zero()
      )

    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(initialize_fn, next_fn)

  def test_single_param_next_raises(self):
    @federated_computation.federated_computation(SERVER_INT)
    def next_fn(state):
      return MeasuredProcessOutput(
          state, intrinsics.federated_broadcast(state), server_zero()
      )

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_three_params_next_raises(self):
    @federated_computation.federated_computation(
        SERVER_INT, SERVER_FLOAT, SERVER_FLOAT
    )
    def next_fn(state, value, extra_value):
      return MeasuredProcessOutput(
          state, intrinsics.federated_broadcast(value), extra_value
      )

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_value_param_raises(self):
    next_fn = federated_computation.federated_computation(
        SERVER_INT, CLIENTS_INT
    )(lambda state, val: MeasuredProcessOutput(state, val, server_zero()))
    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_result_raises(self):
    next_fn = federated_computation.federated_computation(
        SERVER_INT, SERVER_INT
    )(lambda state, val: MeasuredProcessOutput(state, val, server_zero()))
    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):
    @federated_computation.federated_computation(SERVER_INT, SERVER_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_broadcast(val),
          intrinsics.federated_value(1.0, placements.CLIENTS),
      )

    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)


class BroadcastProcessComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float', computation_types.TensorType(np.float32)),
      ('struct', computation_types.to_type([(np.float32, (2,)), np.int32])),
  )
  def test_type_properties(self, value_type):
    broadcast_process = distributors.build_broadcast_process(value_type)
    self.assertIsInstance(broadcast_process, distributors.DistributionProcess)

    expected_param_value_type = computation_types.FederatedType(
        value_type, placements.SERVER
    )
    expected_result_type = computation_types.FederatedType(
        value_type, placements.CLIENTS, all_equal=True
    )
    expected_state_type = computation_types.FederatedType((), placements.SERVER)
    expected_measurements_type = computation_types.FederatedType(
        (), placements.SERVER
    )

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    expected_initialize_type.check_equivalent_to(
        broadcast_process.initialize.type_signature
    )

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, value=expected_param_value_type
        ),
        result=MeasuredProcessOutput(
            expected_state_type,
            expected_result_type,
            expected_measurements_type,
        ),
    )
    expected_next_type.check_equivalent_to(
        broadcast_process.next.type_signature
    )

  @parameterized.named_parameters(
      ('federated_type', SERVER_FLOAT),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(np.float32)),
  )
  def test_incorrect_value_type_raises(self, bad_value_type):
    with self.assertRaises(TypeError):
      distributors.build_broadcast_process(bad_value_type)

  def test_inner_federated_type_raises(self):
    with self.assertRaisesRegex(TypeError, 'FederatedType'):
      distributors.build_broadcast_process(
          computation_types.to_type([SERVER_FLOAT, SERVER_FLOAT])
      )


class BroadcastProcessExecutionTest(tf.test.TestCase):

  def test_broadcast_scalar(self):
    broadcast_process = distributors.build_broadcast_process(
        SERVER_FLOAT.member
    )
    output = broadcast_process.next(broadcast_process.initialize(), 2.5)
    self.assertEqual((), output.state)
    self.assertAllClose(2.5, output.result)
    self.assertEqual((), output.measurements)

  def test_broadcast_struct(self):
    struct_type = computation_types.to_type([(np.float32, (2,)), np.int32])
    broadcast_process = distributors.build_broadcast_process(struct_type)
    output = broadcast_process.next(
        broadcast_process.initialize(), ((1.0, 2.5), 3)
    )
    self.assertEqual((), output.state)
    self.assertAllClose(((1.0, 2.5), 3), output.result)
    self.assertEqual((), output.measurements)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context(
      default_num_clients=1
  )
  tf.test.main()
