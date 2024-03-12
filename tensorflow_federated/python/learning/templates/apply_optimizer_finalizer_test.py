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
import numpy as np
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
from tensorflow_federated.python.learning.templates import finalizers

SERVER_FLOAT = computation_types.FederatedType(np.float32, placements.SERVER)
_MODEL_WEIGHTS_SPEC = model_weights.ModelWeights(
    trainable=(np.float32,), non_trainable=(np.float32,)
)
_MODEL_WEIGHTS_TYPE = computation_types.to_type(_MODEL_WEIGHTS_SPEC)
_SERVER_MODEL_WEIGHTS_TYPE = computation_types.FederatedType(
    _MODEL_WEIGHTS_TYPE, placements.SERVER
)
MeasuredProcessOutput = measured_process.MeasuredProcessOutput


def _get_should_reject_update_fn(state_threshold: int):
  """Returns a test `should_reject_update` function with state threshold."""

  def should_reject_update(state, update):
    reject_update, measurements = (
        apply_optimizer_finalizer.reject_non_finite_update(state, update)
    )
    measurements['state_large_values'] = 0
    if reject_update:
      return reject_update, measurements
    else:
      has_large_values = tf.math.reduce_any([
          tf.math.reduce_any(
              tf.math.greater(
                  tf.math.abs(t),
                  tf.cast(
                      state_threshold,
                      dtype=t.dtype,
                  ),
              )
          )
          for t in tf.nest.flatten(state)
      ])
      if has_large_values:
        measurements['state_large_values'] = 1
        return True, measurements
      else:
        return False, measurements

  return should_reject_update


class ApplyOptimizerFinalizerComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_initialize_has_expected_type_with_keras_optimizer(self):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, _MODEL_WEIGHTS_TYPE
    )
    expected_state_type = computation_types.FederatedType(
        [np.int64], placements.SERVER
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    type_test_utils.assert_types_equivalent(
        finalizer.initialize.type_signature, expected_initialize_type
    )

  def test_next_has_expected_type_with_keras_optimizer(self):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, _MODEL_WEIGHTS_TYPE
    )

    expected_param_weights_type = _SERVER_MODEL_WEIGHTS_TYPE
    expected_param_update_type = computation_types.FederatedType(
        _MODEL_WEIGHTS_TYPE.trainable, placements.SERVER
    )
    expected_result_type = computation_types.FederatedType(
        _MODEL_WEIGHTS_TYPE, placements.SERVER
    )
    expected_state_type = computation_types.FederatedType(
        [np.int64], placements.SERVER
    )
    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(update_non_finite=np.int32), placements.SERVER
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_weights_type,
            update=expected_param_update_type,
        ),
        result=MeasuredProcessOutput(
            expected_state_type,
            expected_result_type,
            expected_measurements_type,
        ),
    )
    type_test_utils.assert_types_equivalent(
        finalizer.next.type_signature, expected_next_type
    )

  def test_get_hparams_has_expected_type_with_keras_optimizer(self):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, _MODEL_WEIGHTS_TYPE
    )

    expected_state_type = [np.int64]
    expected_hparams_type = collections.OrderedDict()
    expected_get_hparams_type = computation_types.FunctionType(
        parameter=expected_state_type, result=expected_hparams_type
    )
    type_test_utils.assert_types_equivalent(
        finalizer.get_hparams.type_signature, expected_get_hparams_type
    )

  def test_set_hparams_has_expected_type_with_keras_optimizer(self):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn, _MODEL_WEIGHTS_TYPE
    )

    expected_state_type = [np.int64]
    expected_hparams_type = collections.OrderedDict()
    expected_set_hparams_type = computation_types.FunctionType(
        parameter=computation_types.StructType(
            [('state', expected_state_type), ('hparams', expected_hparams_type)]
        ),
        result=expected_state_type,
    )
    type_test_utils.assert_types_equivalent(
        finalizer.set_hparams.type_signature, expected_set_hparams_type
    )

  def test_initialize_has_expected_type_with_tff_optimizer(self):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), _MODEL_WEIGHTS_TYPE
    )

    expected_state_type = computation_types.FederatedType(
        computation_types.to_type(
            collections.OrderedDict(
                [(optimizer_base.LEARNING_RATE_KEY, np.float32)]
            )
        ),
        placements.SERVER,
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    type_test_utils.assert_types_equivalent(
        finalizer.initialize.type_signature, expected_initialize_type
    )

  def test_next_has_expected_type_with_tff_optimizer(self):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), _MODEL_WEIGHTS_TYPE
    )

    expected_param_weights_type = _SERVER_MODEL_WEIGHTS_TYPE
    expected_param_update_type = computation_types.FederatedType(
        _MODEL_WEIGHTS_TYPE.trainable, placements.SERVER
    )
    expected_result_type = _SERVER_MODEL_WEIGHTS_TYPE
    expected_state_type = computation_types.FederatedType(
        computation_types.to_type(
            collections.OrderedDict(
                [(optimizer_base.LEARNING_RATE_KEY, np.float32)]
            )
        ),
        placements.SERVER,
    )
    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(update_non_finite=np.int32), placements.SERVER
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_weights_type,
            update=expected_param_update_type,
        ),
        result=MeasuredProcessOutput(
            expected_state_type,
            expected_result_type,
            expected_measurements_type,
        ),
    )
    type_test_utils.assert_types_equivalent(
        finalizer.next.type_signature, expected_next_type
    )

  def test_get_hparams_has_expected_type_with_tff_optimizer(self):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), _MODEL_WEIGHTS_TYPE
    )

    expected_state_type = collections.OrderedDict(
        [(optimizer_base.LEARNING_RATE_KEY, np.float32)]
    )
    expected_hparams_type = expected_state_type
    expected_get_hparams_type = computation_types.FunctionType(
        parameter=expected_state_type, result=expected_hparams_type
    )
    type_test_utils.assert_types_equivalent(
        finalizer.get_hparams.type_signature, expected_get_hparams_type
    )

  def test_set_hparams_has_expected_type_with_tff_optimizer(self):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), _MODEL_WEIGHTS_TYPE
    )

    expected_state_type = collections.OrderedDict(
        [(optimizer_base.LEARNING_RATE_KEY, np.float32)]
    )
    expected_hparams_type = expected_state_type
    expected_set_hparams_type = computation_types.FunctionType(
        parameter=computation_types.StructType(
            [('state', expected_state_type), ('hparams', expected_hparams_type)]
        ),
        result=expected_state_type,
    )
    type_test_utils.assert_types_equivalent(
        finalizer.set_hparams.type_signature, expected_set_hparams_type
    )

  @parameterized.named_parameters(
      ('not_struct', computation_types.TensorType(np.float32)),
      ('federated_type', _SERVER_MODEL_WEIGHTS_TYPE),
      (
          'model_weights_of_federated_types',
          computation_types.to_type(
              model_weights.ModelWeights(SERVER_FLOAT, SERVER_FLOAT)
          ),
      ),
      (
          'not_model_weights',
          computation_types.to_type((np.float32, np.float32)),
      ),
      (
          'function_type',
          computation_types.FunctionType(None, _SERVER_MODEL_WEIGHTS_TYPE),
      ),
      (
          'sequence_type',
          computation_types.SequenceType(_SERVER_MODEL_WEIGHTS_TYPE.member),
      ),
  )
  def test_incorrect_value_type_raises(self, bad_type):
    with self.assertRaises(TypeError):
      apply_optimizer_finalizer.build_apply_optimizer_finalizer(
          sgdm.build_sgdm(1.0), bad_type
      )

  def test_unexpected_optimizer_fn_raises(self):
    optimizer = tf.keras.optimizers.SGD(1.0)
    with self.assertRaises(TypeError):
      apply_optimizer_finalizer.build_apply_optimizer_finalizer(
          optimizer, _SERVER_MODEL_WEIGHTS_TYPE.member
      )

  @parameterized.named_parameters(
      ('tff_optimizer', sgdm.build_sgdm(1.0)),
      (
          'keras_optimizer',
          lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0),
      ),
  )
  def test_custom_should_reject_update_builds(self, optimizer_fn):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer_fn,
        _SERVER_MODEL_WEIGHTS_TYPE.member,
        _get_should_reject_update_fn(100),
    )
    self.assertIsInstance(finalizer, finalizers.FinalizerProcess)


class ApplyOptimizerFinalizerExecutionTest(tf.test.TestCase):

  def test_execution_with_stateless_tff_optimizer(self):
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0), _SERVER_MODEL_WEIGHTS_TYPE.member
    )

    weights = model_weights.ModelWeights(trainable=(1.0,), non_trainable=(2.0,))
    update = (0.1,)
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      weights = output.result
      self.assertEqual(1.0, optimizer_state[optimizer_base.LEARNING_RATE_KEY])
      self.assertAllClose((1.0 - 0.1 * (i + 1),), weights.trainable)
      self.assertEqual(
          collections.OrderedDict(update_non_finite=0), output.measurements
      )

  def test_execution_with_keras_sgd_optimizer(self):
    server_optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(1.0)
    # Note that SGD only maintains a counter of how many times it has been
    # called. No other state is used.
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        server_optimizer_fn, _SERVER_MODEL_WEIGHTS_TYPE.member
    )

    weights = model_weights.ModelWeights(trainable=(1.0,), non_trainable=(2.0,))
    update = (0.1,)
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      weights = output.result
      # We check that the optimizer state is the number of calls.
      self.assertEqual([i + 1], optimizer_state)
      self.assertAllClose((1.0 - 0.1 * (i + 1),), weights.trainable)
      self.assertEqual(
          collections.OrderedDict(update_non_finite=0), output.measurements
      )

  def test_keras_finalizer_execution_with_non_finite_update(self):
    init_fn, next_fn = (
        apply_optimizer_finalizer._build_keras_optimizer_initialize_and_next(
            _MODEL_WEIGHTS_TYPE,
            optimizer_fn=tf.keras.optimizers.SGD,
            should_reject_update=apply_optimizer_finalizer.reject_non_finite_update,
        )
    )

    initial_state = init_fn()
    test_trainable_weights = (0.0,)

    with self.subTest('inf'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('inf'),)
      )
      self.assertAllClose(state, initial_state)
      self.assertAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements, collections.OrderedDict(update_non_finite=1)
      )
    with self.subTest('nan'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('nan'),)
      )
      self.assertAllClose(state, initial_state)
      self.assertAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements, collections.OrderedDict(update_non_finite=1)
      )

  def test_tff_finalizer_execution_with_non_finite_update(self):
    init_fn, next_fn = (
        apply_optimizer_finalizer._build_tff_optimizer_initialize_and_next(
            _MODEL_WEIGHTS_TYPE,
            optimizer=sgdm.build_sgdm(1.0),
            should_reject_update=apply_optimizer_finalizer.reject_non_finite_update,
        )
    )

    initial_state = init_fn()
    test_trainable_weights = (0.0,)

    with self.subTest('inf'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('inf'),)
      )
      self.assertAllClose(state, initial_state)
      self.assertAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements, collections.OrderedDict(update_non_finite=1)
      )
    with self.subTest('nan'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('nan'),)
      )
      self.assertAllClose(state, initial_state)
      self.assertAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements, collections.OrderedDict(update_non_finite=1)
      )

  def test_keras_finalizer_execution_with_custom_should_reject_update(self):
    init_fn, next_fn = (
        apply_optimizer_finalizer._build_keras_optimizer_initialize_and_next(
            _MODEL_WEIGHTS_TYPE,
            optimizer_fn=tf.keras.optimizers.Adam,
            should_reject_update=_get_should_reject_update_fn(10),
        )
    )

    initial_state = init_fn()
    test_trainable_weights = (0.0,)

    with self.subTest('larger_than_threshold_rejects_update'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('101'),)
      )
      self.assertAllClose(state, initial_state)
      self.assertAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements,
          collections.OrderedDict(update_non_finite=0, state_large_values=1),
      )
    with self.subTest('equal_to_threshold_accepts_update'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('100'),)
      )
      self.assertNotAllClose(state, initial_state)
      self.assertNotAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements,
          collections.OrderedDict(update_non_finite=0, state_large_values=0),
      )
    with self.subTest('less_than_threshold_accepts_update'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('50'),)
      )
      self.assertNotAllClose(state, initial_state)
      self.assertNotAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements,
          collections.OrderedDict(update_non_finite=0, state_large_values=0),
      )

  def test_tff_finalizer_execution_with_custom_should_reject_update(self):
    init_fn, next_fn = (
        apply_optimizer_finalizer._build_tff_optimizer_initialize_and_next(
            _MODEL_WEIGHTS_TYPE,
            optimizer=sgdm.build_sgdm(learning_rate=0.1, momentum=0.1),
            should_reject_update=_get_should_reject_update_fn(10),
        )
    )

    initial_state = init_fn()
    test_trainable_weights = (0.0,)

    with self.subTest('larger_than_threshold_rejects_update'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('11'),)
      )
      self.assertAllClose(state, initial_state)
      self.assertAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements,
          collections.OrderedDict(update_non_finite=0, state_large_values=1),
      )
    with self.subTest('equal_to_threshold_accepts_update'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('10'),)
      )
      self.assertNotAllClose(state, initial_state)
      self.assertNotAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements,
          collections.OrderedDict(update_non_finite=0, state_large_values=0),
      )
    with self.subTest('less_than_threshold_accepts_update'):
      state, weights, measurements = next_fn(
          initial_state, test_trainable_weights, (float('5'),)
      )
      self.assertNotAllClose(state, initial_state)
      self.assertNotAllClose(weights, test_trainable_weights)
      self.assertAllEqual(
          measurements,
          collections.OrderedDict(update_non_finite=0, state_large_values=0),
      )

  def test_execution_with_stateful_tff_optimizer(self):
    momentum = 0.5
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        sgdm.build_sgdm(1.0, momentum=momentum),
        _SERVER_MODEL_WEIGHTS_TYPE.member,
    )

    weights = model_weights.ModelWeights(trainable=(1.0,), non_trainable=(2.0,))
    update = (0.1,)
    expected_velocity = 0.0
    optimizer_state = finalizer.initialize()
    for _ in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      expected_velocity = expected_velocity * momentum + update[0]
      self.assertAllClose((expected_velocity,), optimizer_state['accumulator'])
      self.assertAllClose(
          (weights.trainable[0] - expected_velocity,), output.result.trainable
      )
      self.assertEqual(
          collections.OrderedDict(update_non_finite=0), output.measurements
      )

  def test_execution_with_stateful_keras_optimizer(self):
    momentum = 0.5

    def server_optimizer_fn():
      return tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.5)

    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        server_optimizer_fn, _SERVER_MODEL_WEIGHTS_TYPE.member
    )

    weights = model_weights.ModelWeights(trainable=(1.0,), non_trainable=(2.0,))
    update = (0.1,)
    expected_velocity = 0.0
    optimizer_state = finalizer.initialize()
    for i in range(5):
      output = finalizer.next(optimizer_state, weights, update)
      optimizer_state = output.state
      expected_velocity = expected_velocity * momentum + update[0]
      # Keras stores the negative of the velocity term used by
      # tff.learning.optimizers.SGDM
      self.assertAllClose([i + 1, -expected_velocity], optimizer_state)
      self.assertAllClose(
          (weights.trainable[0] - expected_velocity,), output.result.trainable
      )
      self.assertEqual(
          collections.OrderedDict(update_non_finite=0), output.measurements
      )
      weights = output.result

  def test_get_hparams_with_keras_optimizer(self):
    optimizer = lambda: tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, _SERVER_MODEL_WEIGHTS_TYPE.member
    )
    state = finalizer.initialize()

    hparams = finalizer.get_hparams(state)

    expected_hparams = collections.OrderedDict()
    self.assertIsInstance(hparams, collections.OrderedDict)
    self.assertDictEqual(hparams, expected_hparams)

  def test_set_hparams_with_keras_optimizer(self):
    optimizer = lambda: tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, _SERVER_MODEL_WEIGHTS_TYPE.member
    )
    state = finalizer.initialize()
    hparams = collections.OrderedDict()

    updated_state = finalizer.set_hparams(state, hparams)

    self.assertEqual(updated_state, state)

  def test_get_hparams_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, _SERVER_MODEL_WEIGHTS_TYPE.member
    )
    state = finalizer.initialize()

    hparams = finalizer.get_hparams(state)

    expected_hparams = collections.OrderedDict(learning_rate=1.0, momentum=0.9)
    self.assertIsInstance(hparams, collections.OrderedDict)
    self.assertDictEqual(hparams, expected_hparams)

  def test_set_hparams_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    finalizer = apply_optimizer_finalizer.build_apply_optimizer_finalizer(
        optimizer, _SERVER_MODEL_WEIGHTS_TYPE.member
    )
    state = finalizer.initialize()
    hparams = collections.OrderedDict(learning_rate=0.5, momentum=0.4)

    updated_state = finalizer.set_hparams(state, hparams)

    self.assertIsInstance(updated_state, collections.OrderedDict)
    expected_state = copy.deepcopy(state)
    for k, v in hparams.items():
      expected_state[k] = v
    self.assertDictEqual(updated_state, expected_state)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
