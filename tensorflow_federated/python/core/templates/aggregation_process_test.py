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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process

SERVER_INT = computation_types.FederatedType(tf.int32, placements.SERVER)
SERVER_FLOAT = computation_types.FederatedType(tf.float32, placements.SERVER)
CLIENTS_INT = computation_types.FederatedType(tf.int32, placements.CLIENTS)
CLIENTS_FLOAT = computation_types.FederatedType(tf.float32, placements.CLIENTS)
MeasuredProcessOutput = measured_process.MeasuredProcessOutput


def server_zero():
  """Returns zero integer placed at SERVER."""
  return intrinsics.federated_value(0, placements.SERVER)


@computations.federated_computation()
def test_initialize_fn():
  return server_zero()


@computations.federated_computation(test_initialize_fn.type_signature.result,
                                    computation_types.FederatedType(
                                        tf.float32, placements.CLIENTS))
def test_next_fn(state, val):
  return MeasuredProcessOutput(state, intrinsics.federated_sum(val),
                               intrinsics.federated_value(1, placements.SERVER))


class AggregationProcessTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      aggregation_process.AggregationProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid AggregationProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.federated_computation()(server_zero)

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(
          state, intrinsics.federated_sum(val),
          intrinsics.federated_value(1, placements.SERVER))

    try:
      aggregation_process.AggregationProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an AggregationProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):

    @computations.federated_computation()
    def initialize_fn():
      return intrinsics.federated_eval(
          computations.tf_computation()(
              lambda: tf.constant([], dtype=tf.string)), placements.SERVER)

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.TensorType(shape=[None], dtype=tf.string),
            placements.SERVER), CLIENTS_FLOAT)
    def next_fn(strings, val):
      new_state_fn = computations.tf_computation()(
          lambda s: tf.concat([s, tf.constant(['abc'])], axis=0))
      return MeasuredProcessOutput(
          intrinsics.federated_map(new_state_fn, strings),
          intrinsics.federated_sum(val),
          intrinsics.federated_value(1, placements.SERVER))

    try:
      aggregation_process.AggregationProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an AggregationProcess with parameter '
                'types with statically unknown shape.')

  def test_construction_with_value_type_mismatch_does_not_raise(self):

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def next_fn(state, val):
      del val
      return MeasuredProcessOutput(state, state, server_zero())

    try:
      aggregation_process.AggregationProcess(test_initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an AggregationProcess with different '
                'client and server placed types.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      aggregation_process.AggregationProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      aggregation_process.AggregationProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state: MeasuredProcessOutput(state, (), ()))

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.federated_computation(SERVER_INT)(
        lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      aggregation_process.AggregationProcess(one_arg_initialize_fn,
                                             test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0.0, placements.SERVER))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      aggregation_process.AggregationProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def float_next_fn(state, val):
      del state
      return MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          intrinsics.federated_sum(val), server_zero())

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      aggregation_process.AggregationProcess(test_initialize_fn, float_next_fn)

  def test_measured_process_output_as_state_raises(self):
    no_value = lambda: intrinsics.federated_value((), placements.SERVER)

    @computations.federated_computation()
    def initialize_fn():
      return intrinsics.federated_zip(
          MeasuredProcessOutput(no_value(), no_value(), no_value()))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        CLIENTS_FLOAT)
    def next_fn(state, value):
      del state, value
      return MeasuredProcessOutput(no_value(), no_value(), no_value())

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      measured_process.MeasuredProcess(initialize_fn, next_fn)

  def test_next_return_tuple_raises(self):

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def tuple_next_fn(state, val):
      return state, intrinsics.federated_sum(val), server_zero()

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      aggregation_process.AggregationProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements'])

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def namedtuple_next_fn(state, val):
      return measured_process_output(state, intrinsics.federated_sum(val),
                                     server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      aggregation_process.AggregationProcess(test_initialize_fn,
                                             namedtuple_next_fn)

  def test_next_return_odict_raises(self):

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def odict_next_fn(state, val):
      return collections.OrderedDict(
          state=state,
          result=intrinsics.federated_sum(val),
          measurements=server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      aggregation_process.AggregationProcess(test_initialize_fn, odict_next_fn)

  def test_federated_measured_process_output_raises(self):

    # Using federated_zip to place FederatedType at the top of the hierarchy.
    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)
    def next_fn(state, val):
      return intrinsics.federated_zip(
          MeasuredProcessOutput(state, intrinsics.federated_sum(val),
                                server_zero()))

    # A MeasuredProcessOutput containing three `FederatedType`s is different
    # than a `FederatedType` containing a MeasuredProcessOutput. Corrently, only
    # the former is considered valid.
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      aggregation_process.AggregationProcess(test_initialize_fn, next_fn)

  # Tests specific only for the AggregationProcess contract below.

  def test_non_federated_init_next_raises(self):
    initialize_fn = computations.tf_computation(lambda: 0)

    @computations.tf_computation(tf.int32, tf.float32)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, val, ())

    with self.assertRaises(aggregation_process.AggregationNotFederatedError):
      aggregation_process.AggregationProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = computations.federated_computation()(
        lambda: (server_zero(), server_zero()))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        CLIENTS_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, intrinsics.federated_sum(val), ())

    with self.assertRaises(aggregation_process.AggregationNotFederatedError):
      aggregation_process.AggregationProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS))

    @computations.federated_computation(CLIENTS_INT, CLIENTS_FLOAT)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, intrinsics.federated_sum(val),
                                   server_zero())

    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(initialize_fn, next_fn)

  def test_single_param_next_raises(self):
    next_fn = computations.federated_computation(SERVER_INT)(
        lambda state: MeasuredProcessOutput(state, state, state))
    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      aggregation_process.AggregationProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_value_param_raises(self):
    next_fn = computations.federated_computation(SERVER_INT, SERVER_INT)(
        lambda state, val: MeasuredProcessOutput(state, val, server_zero()))
    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_result_raises(self):
    next_fn = computations.federated_computation(SERVER_INT, CLIENTS_INT)(
        lambda state, val: MeasuredProcessOutput(state, val, server_zero()))
    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):

    @computations.federated_computation(SERVER_INT, CLIENTS_INT)
    def next_fn(state, val):
      return MeasuredProcessOutput(state, intrinsics.federated_sum(val), val)

    with self.assertRaises(aggregation_process.AggregationPlacementError):
      aggregation_process.AggregationProcess(test_initialize_fn, next_fn)

  def test_is_weighted_property(self):
    process = aggregation_process.AggregationProcess(test_initialize_fn,
                                                     test_next_fn)
    self.assertFalse(process.is_weighted)

    @computations.federated_computation(SERVER_INT, CLIENTS_FLOAT,
                                        CLIENTS_FLOAT)
    def weighted_next_fn(state, value, weight):
      del weight
      return MeasuredProcessOutput(
          state, intrinsics.federated_sum(value),
          intrinsics.federated_value(1, placements.SERVER))

    process = aggregation_process.AggregationProcess(test_initialize_fn,
                                                     weighted_next_fn)
    self.assertTrue(process.is_weighted)


if __name__ == '__main__':
  test_case.main()
