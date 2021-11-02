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
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import finalizers
from tensorflow_federated.python.learning.optimizers import sgdm

SERVER_INT = computation_types.FederatedType(tf.int32, placements.SERVER)
SERVER_FLOAT = computation_types.FederatedType(tf.float32, placements.SERVER)
CLIENTS_INT = computation_types.FederatedType(tf.int32, placements.CLIENTS)
CLIENTS_FLOAT = computation_types.FederatedType(tf.float32, placements.CLIENTS)
MODEL_WEIGHTS_TYPE = computation_types.at_server(
    computation_types.to_type(model_utils.ModelWeights(tf.float32, ())))
MeasuredProcessOutput = measured_process.MeasuredProcessOutput


def server_zero():
  """Returns zero integer placed at SERVER."""
  return intrinsics.federated_value(0, placements.SERVER)


def federated_add(a, b):
  return intrinsics.federated_map(
      computations.tf_computation(lambda x, y: x + y), (a, b))


@computations.federated_computation()
def test_initialize_fn():
  return server_zero()


def test_finalizer_result(weights, update):
  return intrinsics.federated_zip(
      model_utils.ModelWeights(federated_add(weights.trainable, update), ()))


@computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                    SERVER_FLOAT)
def test_next_fn(state, weights, update):
  return MeasuredProcessOutput(state, test_finalizer_result(weights, update),
                               intrinsics.federated_value(1, placements.SERVER))


class FinalizerTest(test_case.TestCase):

  def test_construction_does_not_raise(self):
    try:
      finalizers.FinalizerProcess(test_initialize_fn, test_next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct a valid FinalizerProcess.')

  def test_construction_with_empty_state_does_not_raise(self):
    initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value((), placements.SERVER))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        MODEL_WEIGHTS_TYPE, SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state, test_finalizer_result(weights, update),
          intrinsics.federated_value(1, placements.SERVER))

    try:
      finalizers.FinalizerProcess(initialize_fn, next_fn)
    except:  # pylint: disable=bare-except
      self.fail('Could not construct an FinalizerProcess with empty state.')

  def test_init_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      finalizers.FinalizerProcess(initialize_fn=lambda: 0, next_fn=test_next_fn)

  def test_next_not_tff_computation_raises(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*\.Computation, .*'):
      finalizers.FinalizerProcess(
          initialize_fn=test_initialize_fn,
          next_fn=lambda state, w, u: MeasuredProcessOutput(state, w + u, ()))

  def test_init_param_not_empty_raises(self):
    one_arg_initialize_fn = computations.federated_computation(SERVER_INT)(
        lambda x: x)
    with self.assertRaises(errors.TemplateInitFnParamNotEmptyError):
      finalizers.FinalizerProcess(one_arg_initialize_fn, test_next_fn)

  def test_init_state_not_assignable(self):
    float_initialize_fn = computations.federated_computation()(
        lambda: intrinsics.federated_value(0.0, placements.SERVER))
    with self.assertRaises(errors.TemplateStateNotAssignableError):
      finalizers.FinalizerProcess(float_initialize_fn, test_next_fn)

  def test_next_state_not_assignable(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def float_next_fn(state, weights, update):
      del state
      return MeasuredProcessOutput(
          intrinsics.federated_value(0.0, placements.SERVER),
          test_finalizer_result(weights, update),
          intrinsics.federated_value(1, placements.SERVER))

    with self.assertRaises(errors.TemplateStateNotAssignableError):
      finalizers.FinalizerProcess(test_initialize_fn, float_next_fn)

  def test_next_return_tuple_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def tuple_next_fn(state, weights, update):
      return state, test_finalizer_result(weights, update), server_zero()

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      finalizers.FinalizerProcess(test_initialize_fn, tuple_next_fn)

  def test_next_return_namedtuple_raises(self):
    measured_process_output = collections.namedtuple(
        'MeasuredProcessOutput', ['state', 'result', 'measurements'])

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def namedtuple_next_fn(state, weights, update):
      return measured_process_output(state,
                                     test_finalizer_result(weights, update),
                                     server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      finalizers.FinalizerProcess(test_initialize_fn, namedtuple_next_fn)

  def test_next_return_odict_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def odict_next_fn(state, weights, update):
      return collections.OrderedDict(
          state=state,
          result=test_finalizer_result(weights, update),
          measurements=server_zero())

    with self.assertRaises(errors.TemplateNotMeasuredProcessOutputError):
      finalizers.FinalizerProcess(test_initialize_fn, odict_next_fn)

  # Tests specific only for the FinalizerProcess contract below.

  def test_non_federated_init_next_raises(self):
    initialize_fn = computations.tf_computation(lambda: 0)

    @computations.tf_computation(tf.int32,
                                 computation_types.to_type(
                                     model_utils.ModelWeights(tf.float32, ())),
                                 tf.float32)
    def next_fn(state, weights, update):
      new_weigths = model_utils.ModelWeights(weights.trainable + update, ())
      return MeasuredProcessOutput(state, new_weigths, 0)

    with self.assertRaises(errors.TemplateNotFederatedError):
      finalizers.FinalizerProcess(initialize_fn, next_fn)

  def test_init_tuple_of_federated_types_raises(self):
    initialize_fn = computations.federated_computation()(
        lambda: (server_zero(), server_zero()))

    @computations.federated_computation(initialize_fn.type_signature.result,
                                        MODEL_WEIGHTS_TYPE, SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(state,
                                   test_finalizer_result(weights, update),
                                   server_zero())

    with self.assertRaises(errors.TemplateNotFederatedError):
      finalizers.FinalizerProcess(initialize_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    initialize_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS))

    @computations.federated_computation(CLIENTS_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(state,
                                   test_finalizer_result(weights, update),
                                   server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(initialize_fn, next_fn)

  def test_two_param_next_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE)
    def next_fn(state, weights):
      return MeasuredProcessOutput(state, weights, server_zero())

    with self.assertRaises(errors.TemplateNextFnNumArgsError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_weight_param_raises(self):

    @computations.federated_computation(SERVER_INT,
                                        computation_types.at_clients(
                                            MODEL_WEIGHTS_TYPE.member),
                                        SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          test_finalizer_result(intrinsics.federated_sum(weights), update),
          server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_bad_next_weights_param_type_raises(self):
    bad_model_weights_type = computation_types.at_server(
        computation_types.to_type(
            collections.OrderedDict(trainable=tf.float32, non_trainable=())))

    @computations.federated_computation(SERVER_INT, bad_model_weights_type,
                                        SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_zip(
              model_utils.ModelWeights(
                  federated_add(weights['trainable'], update), ())),
          server_zero())

    with self.assertRaises(finalizers.ModelWeightsTypeError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_update_param_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        CLIENTS_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state, test_finalizer_result(weights,
                                       intrinsics.federated_sum(update)),
          server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_result_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_broadcast(
              test_finalizer_result(weights, update)), server_zero())

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_result_not_assignable_to_weight_raises(self):
    bad_cast_fn = computations.tf_computation(
        lambda x: tf.nest.map_structure(lambda y: tf.cast(y, tf.float64), x))

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state,
          intrinsics.federated_map(bad_cast_fn,
                                   test_finalizer_result(weights, update)),
          server_zero())

    with self.assertRaises(finalizers.FinalizerResultTypeError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):

    @computations.federated_computation(SERVER_INT, MODEL_WEIGHTS_TYPE,
                                        SERVER_FLOAT)
    def next_fn(state, weights, update):
      return MeasuredProcessOutput(
          state, test_finalizer_result(weights, update),
          intrinsics.federated_value(1.0, placements.CLIENTS))

    with self.assertRaises(errors.TemplatePlacementError):
      finalizers.FinalizerProcess(test_initialize_fn, next_fn)


class ApplyOptimizerFinalizerComputationTest(test_case.TestCase,
                                             parameterized.TestCase):

  def test_type_properties(self):
    mw_type = computation_types.to_type(
        model_utils.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))

    finalizer = finalizers.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), mw_type)
    self.assertIsInstance(finalizer, finalizers.FinalizerProcess)

    expected_param_weights_type = computation_types.at_server(mw_type)
    expected_param_update_type = computation_types.at_server(mw_type.trainable)
    expected_result_type = computation_types.at_server(mw_type)
    expected_state_type = computation_types.at_server(
        computation_types.to_type(
            collections.OrderedDict([(sgdm.LEARNING_RATE_KEY, tf.float32)])))
    expected_measurements_type = computation_types.at_server(())

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    expected_initialize_type.check_equivalent_to(
        finalizer.initialize.type_signature)

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_weights_type,
            update=expected_param_update_type),
        result=MeasuredProcessOutput(expected_state_type, expected_result_type,
                                     expected_measurements_type))
    expected_next_type.check_equivalent_to(finalizer.next.type_signature)

  @parameterized.named_parameters(
      ('not_struct', computation_types.TensorType(tf.float32)),
      ('federated_type', MODEL_WEIGHTS_TYPE),
      ('model_weights_of_federated_types',
       computation_types.to_type(
           model_utils.ModelWeights(SERVER_FLOAT, SERVER_FLOAT))),
      ('not_model_weights', computation_types.to_type(
          (tf.float32, tf.float32))),
      ('function_type', computation_types.FunctionType(None,
                                                       MODEL_WEIGHTS_TYPE)),
      ('sequence_type', computation_types.SequenceType(
          MODEL_WEIGHTS_TYPE.member)))
  def test_incorrect_value_type_raises(self, bad_type):
    with self.assertRaises(TypeError):
      finalizers.build_apply_optimizer_finalizer(sgdm.build_sgdm(1.0), bad_type)

  def test_unexpected_optimizer_fn_raises(self):
    optimizer = tf.keras.optimizers.SGD(1.0)
    with self.assertRaises(TypeError):
      finalizers.build_apply_optimizer_finalizer(optimizer,
                                                 MODEL_WEIGHTS_TYPE.member)


class ApplyOptimizerFinalizerExecutionTest(test_case.TestCase):

  def test_execution_with_stateless_tff_optimizer(self):
    finalizer = finalizers.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), MODEL_WEIGHTS_TYPE.member)

    weights = model_utils.ModelWeights(1.0, ())
    update = 0.1
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      weights = output.result
      self.assertEqual(1.0, optimizer_state[sgdm.LEARNING_RATE_KEY])
      self.assertAllClose(1.0 - 0.1 * (i + 1), weights.trainable)
      self.assertEqual((), output.measurements)

  def test_execution_with_nearly_stateless_keras_optimizer(self):
    server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    # Note that SGD only maintains a counter of how many times it has been
    # called. No other state is used.
    finalizer = finalizers.build_apply_optimizer_finalizer(
        server_optimizer_fn, MODEL_WEIGHTS_TYPE.member)

    weights = model_utils.ModelWeights(1.0, ())
    update = 0.1
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      weights = output.result
      # We check that the optimizer state is the number of calls.
      self.assertEqual([i + 1], optimizer_state)
      self.assertAllClose(1.0 - 0.1 * (i + 1), weights.trainable)
      self.assertEqual((), output.measurements)

  def test_execution_with_stateful_tff_optimizer(self):
    momentum = 0.5
    finalizer = finalizers.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0, momentum=momentum), MODEL_WEIGHTS_TYPE.member)

    weights = model_utils.ModelWeights(1.0, ())
    update = 0.1
    expected_velocity = 0.0
    optimizer_state = finalizer.initialize()
    for _ in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      expected_velocity = expected_velocity * momentum + update
      self.assertNear(expected_velocity, optimizer_state['accumulator'], 1e-6)
      self.assertAllClose(weights.trainable - expected_velocity,
                          output.result.trainable)
      self.assertEqual((), output.measurements)
    weights = output.result

  def test_execution_with_stateful_keras_optimizer(self):
    momentum = 0.5

    def server_optimizer_fn():
      return tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.5)

    finalizer = finalizers.build_apply_optimizer_finalizer(
        server_optimizer_fn, MODEL_WEIGHTS_TYPE.member)

    weights = model_utils.ModelWeights(1.0, ())
    update = 0.1
    expected_velocity = 0.0
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      expected_velocity = expected_velocity * momentum + update
      # Keras stores the negative of the velocity term used by
      # tff.learning.optimizers.SGDM
      self.assertAllClose([i + 1, -expected_velocity], optimizer_state)
      self.assertAllClose(weights.trainable - expected_velocity,
                          output.result.trainable)
      self.assertEqual((), output.measurements)
      weights = output.result


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
