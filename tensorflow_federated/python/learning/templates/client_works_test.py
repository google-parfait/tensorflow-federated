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

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works

SERVER_INT = computation_types.FederatedType(tf.int32, placements.SERVER)
SERVER_FLOAT = computation_types.FederatedType(tf.float32, placements.SERVER)
CLIENTS_FLOAT_SEQUENCE = computation_types.FederatedType(
    computation_types.SequenceType(tf.float32), placements.CLIENTS)
CLIENTS_FLOAT = computation_types.FederatedType(tf.float32, placements.CLIENTS)
CLIENTS_INT = computation_types.FederatedType(tf.int32, placements.CLIENTS)
MODEL_WEIGHTS_TYPE = computation_types.at_clients(
    computation_types.to_type(model_utils.ModelWeights(tf.float32, ())))
MeasuredProcessOutput = measured_process.MeasuredProcessOutput


def server_zero():
  return intrinsics.federated_value(0, placements.SERVER)


def client_one():
  return intrinsics.federated_value(1.0, placements.CLIENTS)


def federated_add(a, b):
  return intrinsics.federated_map(
      computations.tf_computation(lambda x, y: x + y), (a, b))


@computations.tf_computation()
def tf_data_sum(data):
  return data.reduce(0.0, lambda x, y: x + y)


@computations.federated_computation()
def test_initialize_fn():
  return server_zero()


def test_client_result(weights, data):
  reduced_data = intrinsics.federated_map(tf_data_sum, data)
  return intrinsics.federated_zip(
      client_works.ClientResult(
          update=federated_add(weights.trainable, reduced_data),
          update_weight=client_one()))


@computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                    CLIENTS_FLOAT_SEQUENCE)
def test_next_fn(state, weights, data):
  return MeasuredProcessOutput(state, test_client_result(weights, data),
                               intrinsics.federated_value(1, placements.SERVER))


class ClientWorkTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      client_works.ClientWorkProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid ClientWorkProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value((), placements.SERVER))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state, test_client_result(weights, data),
          intrinsics.federated_value(1, placements.SERVER))

    try:
      client_works.ClientWorkProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an ClientWorkProcess with empty state.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      client_works.ClientWorkProcess(
          initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      client_works.ClientWorkProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state, w, d: MeasuredProcessOutput(state, w + d, ()))

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.federated_computation(SERVER_INT)(
        lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      client_works.ClientWorkProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0.0, placements.SERVER))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      client_works.ClientWorkProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def float_next_fn(state, weights, data):
      del state
      return MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          test_client_result(weights, data),
          intrinsics.federated_value(1, placements.SERVER))

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      client_works.ClientWorkProcess(test_initialize_fn, float_next_fn)

  def test_next_return_tuple_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def tuple_next_fn(state, weights, data):
      return (state, test_client_result(weights, data), server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      client_works.ClientWorkProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements'])

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def namedtuple_next_fn(state, weights, data):
      return measured_process_output(state, test_client_result(weights, data),
                                     server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      client_works.ClientWorkProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def odict_next_fn(state, weights, data):
      return collections.OrderedDict(
          state=state,
          result=test_client_result(weights, data),
          measurements=server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      client_works.ClientWorkProcess(test_initialize_fn, odict_next_fn)

  # Tests specific only for the ClientWorkProcess contract below.

  def test_non_federated_init_next_raises(self):
    initialize_fn = computations.tf_computation(lambda: 0)

    @computations.tf_computation(tf.int32, MODEL_WEIGHTS_TYPE.member,
                                 computation_types.SequenceType(tf.float32))
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          client_works.ClientResult(weights.trainable + tf_data_sum(data), ()),
          ())

    with self.assertRaises(errors.TemplateNotFederatedError):
      client_works.ClientWorkProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = computations.federated_computation()(
        lambda: (server_zero(), server_zero()))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(state, test_client_result(weights, data),
                                   server_zero())

    with self.assertRaises(errors.TemplateNotFederatedError):
      client_works.ClientWorkProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(state, test_client_result(weights, data),
                                   server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(initialize_fn, next_fn)

  def test_two_param_next_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE)
    def next_fn(state, weights):
      return MeasuredProcessOutput(state, weights.trainable, server_zero())

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_weights_param_raises(self):

    @computations.federated_computation(SERVER_INT,
                                        computation_types.at_server(
                                            MODEL_WEIGHTS_TYPE.member),
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          test_client_result(intrinsics.federated_broadcast(weights), data),
          server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_bad_next_weights_param_type_raises(self):
    bad_model_weights_type = computation_types.at_clients(
        computation_types.to_type(
            collections.OrderedDict(trainable=tf.float32, non_trainable=())))

    @computations.federated_computation(SERVER_INT, bad_model_weights_type,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(state, test_client_result(weights, data),
                                   server_zero())

    with self.assertRaises(client_works.ModelWeightsTypeError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_data_param_raises(self):
    server_sequence_float_type = computation_types.at_server(
        computation_types.SequenceType(tf.float32))

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        server_sequence_float_type)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          test_client_result(weights, intrinsics.federated_broadcast(data)),
          server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_sequence_next_data_param_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_zip(
              client_works.ClientResult(
                  federated_add(weights.trainable, data), client_one())),
          server_zero())

    with self.assertRaises(client_works.ClientDataTypeError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_clients_placed_next_result_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state, intrinsics.federated_sum(test_client_result(weights, data)),
          server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_zipped_next_result_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      reduced_data = intrinsics.federated_map(tf_data_sum, data)
      return MeasuredProcessOutput(
          state,
          client_works.ClientResult(
              federated_add(weights.trainable, reduced_data), client_one()),
          server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_incorrect_client_result_container_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      reduced_data = intrinsics.federated_map(tf_data_sum, data)
      bad_client_result = intrinsics.federated_zip(
          collections.OrderedDict(
              update=federated_add(weights.trainable, reduced_data),
              update_weight=client_one()))
      return MeasuredProcessOutput(state, bad_client_result, server_zero())

    with self.assertRaises(client_works.ClientResultTypeError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_trainable_weights_not_assignable_from_update_raises(self):
    bad_cast_fn = computations.tf_computation(lambda x: tf.cast(x, tf.float64))

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      reduced_data = intrinsics.federated_map(tf_data_sum, data)
      not_assignable_update = intrinsics.federated_map(
          bad_cast_fn, federated_add(weights.trainable, reduced_data))
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_zip(
              client_works.ClientResult(not_assignable_update, client_one())),
          server_zero())

    with self.assertRaises(client_works.ClientResultTypeError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT_SEQUENCE)
    def next_fn(state, weights, data):
      return MeasuredProcessOutput(
          state, test_client_result(weights, data),
          intrinsics.federated_value(1.0, placements.CLIENTS))

    with self.assertRaises(errors.TemplatePlacementError):
      client_works.ClientWorkProcess(test_initialize_fn, next_fn)


class ModelDeltaClientWorkComputationTest(test_case.TestCase):

  def test_type_properties(self):
    model_fn = model_examples.LinearRegression
    client_work_process = client_works.build_model_delta_client_work(
        model_fn, sgdm.build_sgdm(1.0))
    self.assertIsInstance(client_work_process, client_works.ClientWorkProcess)

    mw_type = model_utils.ModelWeights(
        trainable=computation_types.to_type([(tf.float32, (2, 1)), tf.float32]),
        non_trainable=computation_types.to_type([tf.float32]))
    expected_param_model_weights_type = computation_types.at_clients(mw_type)
    expected_param_data_type = computation_types.at_clients(
        computation_types.SequenceType(
            computation_types.to_type(model_fn().input_spec)))
    expected_result_type = computation_types.at_clients(
        client_works.ClientResult(
            update=mw_type.trainable,
            update_weight=computation_types.TensorType(tf.float32)))
    expected_state_type = computation_types.at_server(())
    expected_measurements_type = computation_types.at_server(())

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    expected_initialize_type.check_equivalent_to(
        client_work_process.initialize.type_signature)

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_model_weights_type,
            client_data=expected_param_data_type),
        result=MeasuredProcessOutput(expected_state_type, expected_result_type,
                                     expected_measurements_type))
    expected_next_type.check_equivalent_to(
        client_work_process.next.type_signature)

  def test_keras_optimizer_raises(self):
    with self.assertRaises(TypeError):
      client_works.build_model_delta_client_work(
          model_examples.LinearRegression, tf.keras.optimizers.SGD(1.0))

  def test_created_model_raises(self):
    with self.assertRaises(TypeError):
      client_works.build_model_delta_client_work(
          model_examples.LinearRegression(), sgdm.build_sgdm(1.0))


class ModelDeltaClientWorkExecutionTest(test_case.TestCase):

  def test_execution_stateless_optimizer(self):
    client_work_process = client_works.build_model_delta_client_work(
        model_examples.LinearRegression, sgdm.build_sgdm(0.1))
    data = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)
    data = [data, data.repeat(2)]  # 1st client has 2 examples, 2nd has 4.
    model_weights = model_utils.ModelWeights(
        trainable=[[[0.0], [0.0]], 0.0], non_trainable=[0.0])
    client_model_weights = [model_weights] * 2

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, data)

    expected_result = (
        client_works.ClientResult([[[-1.15], [-1.7]], -0.55], 2.0),
        client_works.ClientResult([[[-0.425], [-0.73]], -0.305], 4.0),
    )

    self.assertEqual((), output.state)
    for i in range(len(expected_result)):
      self.assertAllClose(expected_result[i].update, output.result[i].update)
      self.assertAllClose(expected_result[i].update_weight,
                          output.result[i].update_weight)
    self.assertEqual((), output.measurements)

  def test_execution_stateful_optimizer(self):
    client_work_process = client_works.build_model_delta_client_work(
        model_examples.LinearRegression, sgdm.build_sgdm(0.1, momentum=0.9))
    data = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0]],
            y=[[5.0], [6.0]],
        )).batch(2)
    data = [data, data.repeat(2)]  # 1st client has 2 examples, 2nd has 4.
    model_weights = model_utils.ModelWeights(
        trainable=[[[0.0], [0.0]], 0.0], non_trainable=[0.0])
    client_model_weights = [model_weights] * 2

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, data)

    expected_result = (
        client_works.ClientResult([[[-1.15], [-1.7]], -0.55], 2.0),
        client_works.ClientResult([[[-1.46], [-2.26]], -0.8], 4.0),
    )

    self.assertEqual((), output.state)
    for i in range(len(expected_result)):
      self.assertAllClose(expected_result[i].update, output.result[i].update)
      self.assertAllClose(expected_result[i].update_weight,
                          output.result[i].update_weight)
    self.assertEqual((), output.measurements)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
