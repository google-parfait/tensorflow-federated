# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.learning.optimizers import adagrad
from tensorflow_federated.python.learning.optimizers import adam
from tensorflow_federated.python.learning.optimizers import rmsprop
from tensorflow_federated.python.learning.optimizers import scheduling
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.optimizers import yogi

_SCALAR_SPEC = tf.TensorSpec([1], tf.float32)
_STRUCT_SPEC = [tf.TensorSpec([2], tf.float32), tf.TensorSpec([3], tf.float32)]
_NESTED_SPEC = [
    tf.TensorSpec([10], tf.float32),
    [tf.TensorSpec([20], tf.float32), [tf.TensorSpec([30], tf.float32)]],
]


@tf.function
def _example_schedule_fn(round_num):
  if round_num < 2:
    return 0.1
  return 0.01


def _scheduled_sgd():
  return scheduling.schedule_learning_rate(
      sgdm.build_sgdm(0.1), _example_schedule_fn
  )


def _run_in_eager_mode(optimizer, spec):
  weights = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
  gradients = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)

  state = optimizer.initialize(spec)
  state_history = [state]
  weights_history = [weights]
  for _ in range(3):
    state, weights = optimizer.next(state, weights, gradients)
    state_history.append(state)
    weights_history.append(weights)

  return state_history, weights_history


def _run_in_tf_computation(optimizer, spec):
  weights = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
  gradients = tf.nest.map_structure(lambda s: tf.ones(s.shape, s.dtype), spec)
  init_fn = tensorflow_computation.tf_computation(
      lambda: optimizer.initialize(spec)
  )
  next_fn = tensorflow_computation.tf_computation(optimizer.next)

  state = init_fn()
  state_history = [state]
  weights_history = [weights]
  for _ in range(3):
    state, weights = next_fn(state, weights, gradients)
    state_history.append(state)
    weights_history.append(weights)

  return state_history, weights_history


def _run_in_federated_computation(optimizer, spec):
  weights = tf.nest.map_structure(
      lambda s: np.ones(s.shape, s.dtype.as_numpy_dtype()), spec
  )
  gradients = tf.nest.map_structure(
      lambda s: np.ones(s.shape, s.dtype.as_numpy_dtype()), spec
  )

  @federated_computation.federated_computation()
  def init_fn():
    return intrinsics.federated_eval(
        tensorflow_computation.tf_computation(
            lambda: optimizer.initialize(spec)
        ),
        placements.SERVER,
    )

  @federated_computation.federated_computation(
      init_fn.type_signature.result,
      computation_types.FederatedType(
          computation_types.tensorflow_to_type(spec), placements.SERVER
      ),
      computation_types.FederatedType(
          computation_types.tensorflow_to_type(spec), placements.SERVER
      ),
  )
  def next_fn(state, weights, gradients):
    return intrinsics.federated_map(
        tensorflow_computation.tf_computation(optimizer.next),
        (state, weights, gradients),
    )

  state = init_fn()
  state_history = [state]
  weights_history = [weights]
  for _ in range(3):
    state, weights = next_fn(state, weights, gradients)
    state_history.append(state)
    weights_history.append(weights)

  return state_history, weights_history


class IntegrationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('adagrad_scalar', adagrad.build_adagrad(0.1), _SCALAR_SPEC),
      ('adagrad_struct', adagrad.build_adagrad(0.1), _STRUCT_SPEC),
      ('adagrad_nested', adagrad.build_adagrad(0.1), _NESTED_SPEC),
      ('adam_scalar', adam.build_adam(0.1), _SCALAR_SPEC),
      ('adam_struct', adam.build_adam(0.1), _STRUCT_SPEC),
      ('adam_nested', adam.build_adam(0.1), _NESTED_SPEC),
      ('rmsprop_scalar', rmsprop.build_rmsprop(0.1), _SCALAR_SPEC),
      ('rmsprop_struct', rmsprop.build_rmsprop(0.1), _STRUCT_SPEC),
      ('rmsprop_nested', rmsprop.build_rmsprop(0.1), _NESTED_SPEC),
      ('scheduled_sgd_scalar', _scheduled_sgd(), _SCALAR_SPEC),
      ('scheduled_sgd_struct', _scheduled_sgd(), _STRUCT_SPEC),
      ('scheduled_sgd_nested', _scheduled_sgd(), _NESTED_SPEC),
      ('sgd_scalar', sgdm.build_sgdm(0.1), _SCALAR_SPEC),
      ('sgd_struct', sgdm.build_sgdm(0.1), _STRUCT_SPEC),
      ('sgd_nested', sgdm.build_sgdm(0.1), _NESTED_SPEC),
      ('sgdm_scalar', sgdm.build_sgdm(0.1, 0.9), _SCALAR_SPEC),
      ('sgdm_struct', sgdm.build_sgdm(0.1, 0.9), _STRUCT_SPEC),
      ('sgdm_nested', sgdm.build_sgdm(0.1, 0.9), _NESTED_SPEC),
      ('yogi_scalar', yogi.build_yogi(0.1), _SCALAR_SPEC),
      ('yogi_struct', yogi.build_yogi(0.1), _STRUCT_SPEC),
      ('yogi_nested', yogi.build_yogi(0.1), _NESTED_SPEC),
  )
  def test_integration_produces_identical_results(self, optimizer, spec):
    eager_history = _run_in_eager_mode(optimizer, spec)
    tf_comp_history = _run_in_tf_computation(optimizer, spec)
    federated_comp_history = _run_in_federated_computation(optimizer, spec)

    self.assertAllClose(eager_history, tf_comp_history, rtol=1e-5, atol=1e-5)
    self.assertAllClose(
        eager_history, federated_comp_history, rtol=1e-5, atol=1e-5
    )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
