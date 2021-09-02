# Copyright 2020, The TensorFlow Federated Authors.
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

import collections

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import test_utils as aggregators_test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class SumPlusOneFactoryComputationTest(test_case.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(('float', tf.float32),
                                  ('struct', ((tf.float32, (2,)), tf.int32)))
  def test_type_properties(self, value_type):
    sum_f = aggregators_test_utils.SumPlusOneFactory()
    self.assertIsInstance(sum_f, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.FederatedType(value_type,
                                                       placements.CLIENTS)
    result_value_type = computation_types.FederatedType(value_type,
                                                        placements.SERVER)
    expected_state_type = computation_types.FederatedType(
        tf.int32, placements.SERVER)
    expected_measurements_type = computation_types.FederatedType(
        tf.int32, placements.SERVER)

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, value=param_value_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('federated_type',
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)))
  def test_incorrect_value_type_raises(self, bad_value_type):
    sum_f = aggregators_test_utils.SumPlusOneFactory()
    with self.assertRaises(TypeError):
      sum_f.create(bad_value_type)


class SumPlusOneFactoryExecutionTest(test_case.TestCase):

  def test_sum_scalar(self):
    sum_f = aggregators_test_utils.SumPlusOneFactory()
    value_type = computation_types.to_type(tf.float32)
    process = sum_f.create(value_type)

    state = process.initialize()
    self.assertEqual(0, state)

    client_data = [1.0, 2.0, 3.0]
    output = process.next(state, client_data)
    self.assertEqual(1, output.state)
    self.assertAllClose(7.0, output.result)
    self.assertEqual(42, output.measurements)

  def test_sum_structure(self):
    sum_f = aggregators_test_utils.SumPlusOneFactory()
    value_type = computation_types.to_type(((tf.float32, (2,)), tf.int32))
    process = sum_f.create(value_type)

    state = process.initialize()
    self.assertEqual(0, state)

    client_data = [((1.0, 2.0), 3), ((2.0, 5.0), 4), ((3.0, 0.0), 5)]
    output = process.next(state, client_data)
    self.assertEqual(1, output.state)
    self.assertAllClose(((7.0, 8.0), 13), output.result)
    self.assertEqual(42, output.measurements)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
