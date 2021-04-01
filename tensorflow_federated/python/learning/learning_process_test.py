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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.learning import learning_process

LearningProcessOutput = learning_process.LearningProcessOutput


@computations.federated_computation()
def test_init_fn():
  return intrinsics.federated_value(0, placements.SERVER)


test_state_type = test_init_fn.type_signature.result


@computations.tf_computation
def sum_sequence(s):
  spec = s.element_spec
  return s.reduce(
      tf.zeros(spec.shape, spec.dtype),
      lambda s, t: tf.nest.map_structure(tf.add, s, t))


ClientIntSequenceType = computation_types.at_clients(
    computation_types.SequenceType(tf.int32))


def build_next_fn(server_init_fn):

  @computations.federated_computation(server_init_fn.type_signature.result,
                                      ClientIntSequenceType)
  def next_fn(state, client_values):
    metrics = intrinsics.federated_map(sum_sequence, client_values)
    metrics = intrinsics.federated_sum(metrics)
    return LearningProcessOutput(state, metrics)

  return next_fn


def build_report_fn(server_init_fn):

  @computations.tf_computation(server_init_fn.type_signature.result.member)
  def report_fn(state):
    return state

  return report_fn


test_next_fn = build_next_fn(test_init_fn)
test_report_fn = build_report_fn(test_init_fn)


class LearningProcessTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      learning_process.LearningProcess(test_init_fn, test_next_fn,
                                       test_report_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid LearningProcess.')

  def test_construction_with_empty_state_does_not_raise(self):

    @computations.federated_computation()
    def empty_initialize_fn():
      return intrinsics.federated_value((), placements.SERVER)

    next_fn = build_next_fn(empty_initialize_fn)
    report_fn = build_report_fn(empty_initialize_fn)

    try:
      learning_process.LearningProcess(empty_initialize_fn, next_fn, report_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a LearningProcess with empty state.')

  def test_construction_with_unknown_dimension_does_not_raise(self):
    create_empty_string = computations.tf_computation()(
        lambda: tf.constant([], dtype=tf.string))
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(create_empty_string(), placements.
                                           SERVER))
    next_fn = build_next_fn(initialize_fn)
    report_fn = build_report_fn(initialize_fn)

    try:
      learning_process.LearningProcess(initialize_fn, next_fn, report_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a LearningProcess with state type having '
                'statically unknown shape.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      init_fn = lambda: 0
      learning_process.LearningProcess(init_fn, test_next_fn, test_report_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      learning_process.LearningProcess(
          initialize_fn=test_init_fn,
          next_fn=lambda state, client_data: LearningProcessOutput(state, ()),
          report_fn=test_report_fn)

  def test_init_param_not_empty_raises(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER))
    def one_arg_initialize_fn(x):
      return x

    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      learning_process.LearningProcess(one_arg_initialize_fn, test_next_fn,
                                       test_report_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.federated_computation()(lambda: 0.0)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      learning_process.LearningProcess(float_initialize_fn, test_next_fn,
                                       test_report_fn)

  def test_next_state_not_assignable(self):
    float_initialize_fn = computations.federated_computation()(lambda: 0.0)
    float_next_fn = build_next_fn(float_initialize_fn)
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      learning_process.LearningProcess(test_init_fn, float_next_fn,
                                       test_report_fn)

  def test_init_fn_with_client_placed_state_raises(self):

    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS))

    @computations.federated_computation(init_fn.type_signature.result,
                                        ClientIntSequenceType)
    def next_fn(state, client_values):
      return LearningProcessOutput(state, client_values)

    with self.assertRaises(learning_process.LearningProcessPlacementError):
      learning_process.LearningProcess(init_fn, next_fn, test_report_fn)

  def test_next_return_tuple_raises(self):

    @computations.federated_computation(test_state_type, ClientIntSequenceType)
    def tuple_next_fn(state, client_values):
      metrics = intrinsics.federated_map(sum_sequence, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return (state, metrics)

    with self.assertRaises(learning_process.LearningProcessOutputError):
      learning_process.LearningProcess(test_init_fn, tuple_next_fn,
                                       test_report_fn)

  def test_next_return_namedtuple_raises(self):

    learning_process_output = collections.namedtuple('LearningProcessOutput',
                                                     ['state', 'metrics'])

    @computations.federated_computation(test_state_type, ClientIntSequenceType)
    def namedtuple_next_fn(state, client_values):
      metrics = intrinsics.federated_map(sum_sequence, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return learning_process_output(state, metrics)

    with self.assertRaises(learning_process.LearningProcessOutputError):
      learning_process.LearningProcess(test_init_fn, namedtuple_next_fn,
                                       test_report_fn)

  def test_next_return_odict_raises(self):

    @computations.federated_computation(test_state_type, ClientIntSequenceType)
    def odict_next_fn(state, client_values):
      metrics = intrinsics.federated_map(sum_sequence, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return collections.OrderedDict(state=state, metrics=metrics)

    with self.assertRaises(learning_process.LearningProcessOutputError):
      learning_process.LearningProcess(test_init_fn, odict_next_fn,
                                       test_report_fn)

  def test_next_fn_with_one_parameter_raises(self):

    @computations.federated_computation(test_state_type)
    def next_fn(state):
      return LearningProcessOutput(state, 0)

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      learning_process.LearningProcess(test_init_fn, next_fn, test_report_fn)

  def test_next_fn_with_three_parameters_raises(self):

    @computations.federated_computation(test_state_type, ClientIntSequenceType,
                                        test_state_type)
    def next_fn(state, client_values, second_state):  # pylint: disable=unused-argument
      metrics = intrinsics.federated_map(sum_sequence, client_values)
      metrics = intrinsics.federated_sum(metrics)
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      learning_process.LearningProcess(test_init_fn, next_fn, test_report_fn)

  def test_next_fn_with_non_client_placed_second_arg_raises(self):

    int_sequence_at_server = computation_types.FederatedType(
        computation_types.SequenceType(tf.int32), placements.SERVER)

    @computations.federated_computation(test_state_type, int_sequence_at_server)
    def next_fn(state, server_values):
      metrics = intrinsics.federated_map(sum_sequence, server_values)
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(learning_process.LearningProcessPlacementError):
      learning_process.LearningProcess(test_init_fn, next_fn, test_report_fn)

  def test_next_fn_with_non_sequence_second_arg_raises(self):
    ints_at_clients = computation_types.FederatedType(tf.int32,
                                                      placements.CLIENTS)

    @computations.federated_computation(test_state_type, ints_at_clients)
    def next_fn(state, client_values):
      metrics = intrinsics.federated_sum(client_values)
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(learning_process.LearningProcessSequenceTypeError):
      learning_process.LearningProcess(test_init_fn, next_fn, test_report_fn)

  def test_next_fn_with_client_placed_metrics_result_raises(self):

    @computations.federated_computation(test_state_type, ClientIntSequenceType)
    def next_fn(state, metrics):
      return LearningProcessOutput(state, metrics)

    with self.assertRaises(learning_process.LearningProcessPlacementError):
      learning_process.LearningProcess(test_init_fn, next_fn, test_report_fn)

  def test_non_tff_computation_report_fn_raises(self):
    report_fn = lambda x: x
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      learning_process.LearningProcess(test_init_fn, test_next_fn, report_fn)

  def test_federated_report_fn_raises(self):
    report_fn = computations.federated_computation(test_state_type)(lambda x: x)
    with self.assertRaises(learning_process.ReportFnTypeSignatureError):
      learning_process.LearningProcess(test_init_fn, test_next_fn, report_fn)

  def test_report_param_not_assignable(self):
    report_fn = computations.tf_computation(tf.float32)(lambda x: x)
    with self.assertRaises(learning_process.ReportFnTypeSignatureError):
      learning_process.LearningProcess(test_init_fn, test_next_fn, report_fn)


if __name__ == '__main__':
  test_case.main()
