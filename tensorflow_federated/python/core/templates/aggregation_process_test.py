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

import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


SERVER_INT = computation_types.FederatedType(tf.int32, placements.SERVER)
CLIENTS_INT = computation_types.FederatedType(tf.int32, placements.CLIENTS)
CLIENTS_FLOAT = computation_types.FederatedType(tf.float32, placements.CLIENTS)


class AggregationProcessTest(test.TestCase):

  def test_construction_does_not_raise(self):
    # This test makes sure an AggregationProcess with *three* input arguments
    # can be constructed.

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_value(0, placements.SERVER)

    @computations.federated_computation(SERVER_INT, CLIENTS_INT, CLIENTS_FLOAT)
    def next_fn(x, y, z):
      return measured_process.MeasuredProcessOutput(x,
                                                    intrinsics.federated_sum(y),
                                                    intrinsics.federated_sum(z))

    try:
      aggregation_process.AggregationProcess(init_fn, next_fn)
    except TypeError:
      self.fail('TypeError raised unexpectedly.')

  def test_non_federated_init_state_raises(self):
    init_fn = computations.tf_computation(lambda: 0)
    next_fn = computations.tf_computation(init_fn.type_signature.result)(
        lambda x: measured_process.MeasuredProcessOutput(x, x, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_non_server_placed_init_state_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.CLIENTS))
    next_fn = computations.federated_computation(CLIENTS_INT)(
        lambda x: measured_process.MeasuredProcessOutput(x, x, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_single_param_next_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(SERVER_INT)(
        lambda x: measured_process.MeasuredProcessOutput(x, x, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_non_server_placed_next_state_param_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(CLIENTS_INT, CLIENTS_INT)(
        lambda x, y: measured_process.MeasuredProcessOutput(x, y, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_non_clients_placed_next_value_param_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(SERVER_INT, SERVER_INT)(
        lambda x, y: measured_process.MeasuredProcessOutput(x, y, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_non_server_placed_next_state_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(SERVER_INT, CLIENTS_INT)(
        lambda x, y: measured_process.MeasuredProcessOutput(y, x, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_non_server_placed_next_result_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(SERVER_INT, CLIENTS_INT)(
        lambda x, y: measured_process.MeasuredProcessOutput(x, y, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_non_server_placed_next_measurements_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(SERVER_INT, CLIENTS_INT)(
        lambda x, y: measured_process.MeasuredProcessOutput(x, x, y))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)

  def test_next_value_type_mismatch_raises(self):
    init_fn = computations.federated_computation(
        lambda: intrinsics.federated_value(0, placements.SERVER))
    next_fn = computations.federated_computation(SERVER_INT, CLIENTS_FLOAT)(
        lambda x, y: measured_process.MeasuredProcessOutput(x, x, x))
    with self.assertRaises(TypeError):
      aggregation_process.AggregationProcess(init_fn, next_fn)


if __name__ == '__main__':
  test.main()
