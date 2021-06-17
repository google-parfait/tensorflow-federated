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

import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.framework import distributors

SERVER_INT = computation_types.FederatedType(tf.int32, placements.SERVER)
SERVER_FLOAT = computation_types.FederatedType(tf.float32, placements.SERVER)
CLIENTS_INT = computation_types.FederatedType(tf.int32, placements.CLIENTS)
MeasuredProcessOutput = measured_process.MeasuredProcessOutput


def server_zero():
  return intrinsics.federated_value(0, placements.SERVER)


@computations.federated_computation()
def test_initialize_fn():
  return server_zero()


@computations.federated_computation(SERVER_INT, SERVER_FLOAT)
def test_next_fn(state, val):
  return MeasuredProcessOutput(state, intrinsics.federated_broadcast(val),
                               intrinsics.federated_value(1, placements.SERVER))


class DistributionProcessTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      distributors.DistributionProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid DistributionProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value((), placements.SERVER))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        SERVER_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state, intrinsics.federated_broadcast(val),
          intrinsics.federated_value(1, placements.SERVER))

    try:
      distributors.DistributionProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an DistributionProcess with empty state.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      distributors.DistributionProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      distributors.DistributionProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state, val: MeasuredProcessOutput(state, (), ()))

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.federated_computation(SERVER_INT)(
        lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      distributors.DistributionProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0.0, placements.SERVER))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      distributors.DistributionProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT)
    def float_next_fn(state, val):
      del state
      return MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          intrinsics.federated_broadcast(val), server_zero())

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      distributors.DistributionProcess(test_initialize_fn, float_next_fn)

  def test_next_return_tuple_raises(self):

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT)
    def tuple_next_fn(state, val):
      return state, intrinsics.federated_broadcast(val), server_zero()

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      distributors.DistributionProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements'])

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT)
    def namedtuple_next_fn(state, val):
      return measured_process_output(state, intrinsics.federated_broadcast(val),
                                     server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      distributors.DistributionProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT)
    def odict_next_fn(state, val):
      return collections.OrderedDict(
          state=state,
          result=intrinsics.federated_broadcast(val),
          measurements=server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      distributors.DistributionProcess(test_initialize_fn, odict_next_fn)

  # Tests specific only for the DistributionProcess contract below.

  def test_construction_with_value_type_mismatch_does_not_raise(self):
    bad_cast_fn = computations.tf_computation(lambda x: tf.cast(x, tf.float64))

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT)
    def next_fn(state, val):
      result = intrinsics.federated_map(bad_cast_fn,
                                        intrinsics.federated_broadcast(val))
      return MeasuredProcessOutput(
          state, result, intrinsics.federated_value(1, placements.SERVER))

    try:
      distributors.DistributionProcess(test_initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an DistributionProcess with different '
                'client and server placed types.')

  def test_non_federated_init_next_raises(self):
    initialize_fn = computations.tf_computation(lambda: 0)

    @computations.tf_computation(tf.int32, tf.float32)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, val, ())

    with self.assertRaises(errors.TemplateNotFederatedError):
      distributors.DistributionProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = computations.federated_computation()(
        lambda: (server_zero(), server_zero()))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        SERVER_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, intrinsics.federated_broadcast(val),
                                   server_zero())

    with self.assertRaises(errors.TemplateNotFederatedError):
      distributors.DistributionProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS))

    @computations.federated_computation(CLIENTS_INT, SERVER_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, intrinsics.federated_broadcast(val),
                                   server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(initialize_fn, next_fn)

  def test_single_param_next_raises(self):

    @computations.federated_computation(SERVER_INT)
    def next_fn(state):
      return MeasuredProcessOutput(state, intrinsics.federated_broadcast(state),
                                   server_zero())

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_three_params_next_raises(self):

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT, SERVER_FLOAT)
    def next_fn(state, value, extra_value):
      return MeasuredProcessOutput(state, intrinsics.federated_broadcast(value),
                                   extra_value)

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_value_param_raises(self):
    next_fn = computations.federated_computation(SERVER_INT, CLIENTS_INT)(
        lambda state, val: MeasuredProcessOutput(state, val, server_zero()))
    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_result_raises(self):
    next_fn = computations.federated_computation(SERVER_INT, SERVER_INT)(
        lambda state, val: MeasuredProcessOutput(state, val, server_zero()))
    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):

    @computations.federated_computation(SERVER_INT, SERVER_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state, intrinsics.federated_broadcast(val),
          intrinsics.federated_value(1.0, placements.CLIENTS))

    with self.assertRaises(errors.TemplatePlacementError):
      distributors.DistributionProcess(test_initialize_fn, next_fn)


if __name__ == '__main__':
  test_case.main()
