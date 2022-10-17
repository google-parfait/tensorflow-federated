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
import copy

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import apply_optimizer_finalizer

SERVER_FLOAT = computation_types.FederatedType(tf.float32, placements.SERVER)
MODEL_WEIGHTS_TYPE = computation_types.at_server(
    computation_types.to_type(model_weights.ModelWeights(tf.float32, ())))
MeasuredProcessOutput = measured_process.MeasuredProcessOutput


class ApplyOptimizerFinalizerComputationTest(tf.test.TestCase,
                                             parameterized.TestCase):

  def test_initialize_has_expected_type_with_keras_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, mw_type)
    expected_state_type = computation_types.at_server([tf.int64])
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    type_test_utils.assert_types_equivalent(finalizer.initialize.type_signature,
                                            expected_initialize_type)

  def test_next_has_expected_type_with_keras_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, mw_type)

    expected_param_weights_type = computation_types.at_server(mw_type)
    expected_param_update_type = computation_types.at_server(mw_type.trainable)
    expected_result_type = computation_types.at_server(mw_type)
    expected_state_type = computation_types.at_server([tf.int64])
    expected_measurements_type = computation_types.at_server(())
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_weights_type,
            update=expected_param_update_type),
        result=MeasuredProcessOutput(expected_state_type, expected_result_type,
                                     expected_measurements_type))
    type_test_utils.assert_types_equivalent(finalizer.next.type_signature,
                                            expected_next_type)

  def test_get_hparams_has_expected_type_with_keras_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, mw_type)

    expected_state_type = [tf.int64]
    expected_hparams_type = collections.OrderedDict()
    expected_get_hparams_type = computation_types.FunctionType(
        parameter=expected_state_type, result=expected_hparams_type)
    type_test_utils.assert_types_equivalent(
        finalizer.get_hparams.type_signature, expected_get_hparams_type)

  def test_set_hparams_has_expected_type_with_keras_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, mw_type)

    expected_state_type = [tf.int64]
    expected_hparams_type = collections.OrderedDict()
    expected_set_hparams_type = computation_types.FunctionType(
        parameter=computation_types.StructType([('state', expected_state_type),
                                                ('hparams',
                                                 expected_hparams_type)]),
        result=expected_state_type)
    type_test_utils.assert_types_equivalent(
        finalizer.set_hparams.type_signature, expected_set_hparams_type)

  def test_initialize_has_expected_type_with_tff_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), mw_type)

    expected_state_type = computation_types.at_server(
        computation_types.to_type(
            collections.OrderedDict([(optimizer_base.LEARNING_RATE_KEY,
                                      tf.float32)])))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    type_test_utils.assert_types_equivalent(finalizer.initialize.type_signature,
                                            expected_initialize_type)

  def test_next_has_expected_type_with_tff_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), mw_type)

    expected_param_weights_type = computation_types.at_server(mw_type)
    expected_param_update_type = computation_types.at_server(mw_type.trainable)
    expected_result_type = computation_types.at_server(mw_type)
    expected_state_type = computation_types.at_server(
        computation_types.to_type(
            collections.OrderedDict([(optimizer_base.LEARNING_RATE_KEY,
                                      tf.float32)])))
    expected_measurements_type = computation_types.at_server(())
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_weights_type,
            update=expected_param_update_type),
        result=MeasuredProcessOutput(expected_state_type, expected_result_type,
                                     expected_measurements_type))
    type_test_utils.assert_types_equivalent(finalizer.next.type_signature,
                                            expected_next_type)

  def test_get_hparams_has_expected_type_with_tff_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), mw_type)

    expected_state_type = collections.OrderedDict([
        (optimizer_base.LEARNING_RATE_KEY, tf.float32)
    ])
    expected_hparams_type = expected_state_type
    expected_get_hparams_type = computation_types.FunctionType(
        parameter=expected_state_type, result=expected_hparams_type)
    type_test_utils.assert_types_equivalent(
        finalizer.get_hparams.type_signature, expected_get_hparams_type)

  def test_set_hparams_has_expected_type_with_tff_optimizer(self):
    mw_type = computation_types.to_type(
        model_weights.ModelWeights(
            trainable=(tf.float32, tf.float32), non_trainable=tf.float32))

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), mw_type)

    expected_state_type = collections.OrderedDict([
        (optimizer_base.LEARNING_RATE_KEY, tf.float32)
    ])
    expected_hparams_type = expected_state_type
    expected_set_hparams_type = computation_types.FunctionType(
        parameter=computation_types.StructType([('state', expected_state_type),
                                                ('hparams',
                                                 expected_hparams_type)]),
        result=expected_state_type)
    type_test_utils.assert_types_equivalent(
        finalizer.set_hparams.type_signature, expected_set_hparams_type)

  @parameterized.named_parameters(
      ('not_struct', computation_types.TensorType(tf.float32)),
      ('federated_type', MODEL_WEIGHTS_TYPE),
      ('model_weights_of_federated_types',
       computation_types.to_type(
           model_weights.ModelWeights(SERVER_FLOAT, SERVER_FLOAT))),
      ('not_model_weights', computation_types.to_type(
          (tf.float32, tf.float32))),
      ('function_type', computation_types.FunctionType(None,
                                                       MODEL_WEIGHTS_TYPE)),
      ('sequence_type', computation_types.SequenceType(
          MODEL_WEIGHTS_TYPE.member)))
  def test_incorrect_value_type_raises(self, bad_type):
    with self.assertRaises(TypeError):
      apply_optimizer_finalizer.build_apply_optimizer_finalizer(
          sgdm.build_sgdm(1.0), bad_type)

  def test_unexpected_optimizer_fn_raises(self):
    optimizer = tf.keras.optimizers.SGD(1.0)
    with self.assertRaises(TypeError):
      apply_optimizer_finalizer.build_apply_optimizer_finalizer(
          optimizer, MODEL_WEIGHTS_TYPE.member)


class ApplyOptimizerFinalizerExecutionTest(tf.test.TestCase):

  def test_execution_with_stateless_tff_optimizer(self):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), MODEL_WEIGHTS_TYPE.member)

    weights = model_weights.ModelWeights(1.0, ())
    update = 0.1
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      weights = output.result
      self.assertEqual(1.0, optimizer_state[optimizer_base.LEARNING_RATE_KEY])
      self.assertAllClose(1.0 - 0.1 * (i + 1), weights.trainable)
      self.assertEqual((), output.measurements)

  def test_execution_with_keras_sgd_optimizer(self):
    server_optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(1.0)
    # Note that SGD only maintains a counter of how many times it has been
    # called. No other state is used.
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        server_optimizer_fn, MODEL_WEIGHTS_TYPE.member)

    weights = model_weights.ModelWeights(1.0, ())
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
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0, momentum=momentum), MODEL_WEIGHTS_TYPE.member)

    weights = model_weights.ModelWeights(1.0, ())
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

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        server_optimizer_fn, MODEL_WEIGHTS_TYPE.member)

    weights = model_weights.ModelWeights(1.0, ())
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

  def test_get_hparams_with_keras_optimizer(self):
    optimizer = lambda: tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, MODEL_WEIGHTS_TYPE.member)
    state = finalizer.initialize()

    hparams = finalizer.get_hparams(state)

    expected_hparams = collections.OrderedDict()
    self.assertIsInstance(hparams, collections.OrderedDict)
    self.assertDictEqual(hparams, expected_hparams)

  def test_set_hparams_with_keras_optimizer(self):
    optimizer = lambda: tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, MODEL_WEIGHTS_TYPE.member)
    state = finalizer.initialize()
    hparams = collections.OrderedDict()

    updated_state = finalizer.set_hparams(state, hparams)

    self.assertEqual(updated_state, state)

  def test_get_hparams_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, MODEL_WEIGHTS_TYPE.member)
    state = finalizer.initialize()

    hparams = finalizer.get_hparams(state)

    expected_hparams = collections.OrderedDict(learning_rate=1.0, momentum=0.9)
    self.assertIsInstance(hparams, collections.OrderedDict)
    self.assertDictEqual(hparams, expected_hparams)

  def test_set_hparams_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, MODEL_WEIGHTS_TYPE.member)
    state = finalizer.initialize()
    hparams = collections.OrderedDict(learning_rate=0.5, momentum=0.4)

    updated_state = finalizer.set_hparams(state, hparams)

    self.assertIsInstance(updated_state, collections.OrderedDict)
    expected_state = copy.deepcopy(state)
    for k, v in hparams.items():
      expected_state[k] = v
    self.assertDictEqual(updated_state, expected_state)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
