# Copyright 2018, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python.learning.models import model_weights


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_state_with_model_weights_success(self):
    trainable = [np.array([1.0, 2.0]), np.array([[1.0]]), np.int64(3)]
    non_trainable = [np.array(1), b'bytes type', 5, 2.0]

    new_trainable = [np.array([3.0, 3.0]), np.array([[3.0]]), np.int64(4)]
    new_non_trainable = [np.array(3), b'bytes check', 6, 3.0]

    state = optimizer_utils.ServerState(
        model=model_weights.ModelWeights(
            trainable=trainable, non_trainable=non_trainable
        ),
        optimizer_state=[],
        delta_aggregate_state=tf.constant(0),
        model_broadcast_state=tf.constant(0),
    )

    new_state = optimizer_utils.state_with_new_model_weights(
        state,
        trainable_weights=new_trainable,
        non_trainable_weights=new_non_trainable,
    )
    self.assertAllClose(new_state.model.trainable, new_trainable)
    self.assertEqual(new_state.model.non_trainable, new_non_trainable)

  @parameterized.named_parameters(
      (
          'int_to_float',
          None,
          [np.array(1.5), b'bytes type', 5, 2.0],
          'tensor type',
      ),
      (
          '2d_to_1d',
          [np.array([3.0, 3.0]), np.array([3.0]), np.int64(4)],
          None,
          'tensor type',
      ),
      ('different_lengths', [np.array([3.0, 3.0])], None, 'different lengths'),
      ('bytes_to_int', None, [np.array(1), 5, 5, 2.0], 'not the same type'),
      ('wrong_struct', {'a': np.array([3.0, 3.0])}, None, 'cannot be handled'),
  )
  def test_state_with_new_model_weights_failure(
      self, new_trainable, new_non_trainable, expected_err_msg
  ):
    trainable = [np.array([1.0, 2.0]), np.array([[1.0]]), np.int64(3)]
    non_trainable = [np.array(1), b'bytes type', 5, 2.0]
    state = optimizer_utils.ServerState(
        model=model_weights.ModelWeights(
            trainable=trainable, non_trainable=non_trainable
        ),
        optimizer_state=[],
        delta_aggregate_state=tf.constant(0),
        model_broadcast_state=tf.constant(0),
    )

    new_trainable = trainable if new_trainable is None else new_trainable
    non_trainable = (
        non_trainable if new_non_trainable is None else non_trainable
    )

    with self.assertRaisesRegex(TypeError, expected_err_msg):
      optimizer_utils.state_with_new_model_weights(
          state,
          trainable_weights=new_trainable,
          non_trainable_weights=new_non_trainable,
      )

  def test_is_valid_broadcast_process_true(self):
    @federated_computation.federated_computation()
    def stateless_init():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.at_server(()), computation_types.at_server(())
    )
    def stateless_broadcast(state, value):
      empty_metrics = intrinsics.federated_value(1.0, placements.SERVER)
      return measured_process.MeasuredProcessOutput(
          state=state,
          result=intrinsics.federated_broadcast(value),
          measurements=empty_metrics,
      )

    stateless_process = measured_process.MeasuredProcess(
        initialize_fn=stateless_init, next_fn=stateless_broadcast
    )

    self.assertTrue(
        optimizer_utils.is_valid_broadcast_process(stateless_process)
    )

  def test_is_valid_broadcast_process_bad_placement(self):
    @federated_computation.federated_computation()
    def stateless_init():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.at_server(()), computation_types.at_server(())
    )
    def fake_broadcast(state, value):
      empty_metrics = intrinsics.federated_value(1.0, placements.SERVER)
      return measured_process.MeasuredProcessOutput(
          state=state, result=value, measurements=empty_metrics
      )

    stateless_process = measured_process.MeasuredProcess(
        initialize_fn=stateless_init, next_fn=fake_broadcast
    )

    # Expect to be false because `result` of `next` is on the server.
    self.assertFalse(
        optimizer_utils.is_valid_broadcast_process(stateless_process)
    )

    @federated_computation.federated_computation()
    def stateless_init2():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        computation_types.at_server(()), computation_types.at_clients(())
    )
    def stateless_broadcast(state, value):
      empty_metrics = intrinsics.federated_value(1.0, placements.SERVER)
      return measured_process.MeasuredProcessOutput(
          state=state, result=value, measurements=empty_metrics
      )

    stateless_process = measured_process.MeasuredProcess(
        initialize_fn=stateless_init2, next_fn=stateless_broadcast
    )

    # Expect to be false because second param of `next` is on the clients.
    self.assertFalse(
        optimizer_utils.is_valid_broadcast_process(stateless_process)
    )


if __name__ == '__main__':
  tf.test.main()
