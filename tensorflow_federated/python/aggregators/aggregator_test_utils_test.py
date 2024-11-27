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
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import aggregator_test_utils
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class SumPlusOneFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('float', np.float32), ('struct', ((np.float32, (2,)), np.int32))
  )
  def test_type_properties(self, value_type):
    sum_f = aggregator_test_utils.SumPlusOneFactory()
    self.assertIsInstance(sum_f, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type(value_type)
    process = sum_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = federated_language.FederatedType(
        value_type, federated_language.CLIENTS
    )
    result_value_type = federated_language.FederatedType(
        value_type, federated_language.SERVER
    )
    expected_state_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    expected_measurements_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, value=param_value_type
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      (
          'federated_type',
          federated_language.FederatedType(
              np.float32, federated_language.SERVER
          ),
      ),
      ('function_type', federated_language.FunctionType(None, ())),
      ('sequence_type', federated_language.SequenceType(np.float32)),
  )
  def test_incorrect_value_type_raises(self, bad_value_type):
    sum_f = aggregator_test_utils.SumPlusOneFactory()
    with self.assertRaises(TypeError):
      sum_f.create(bad_value_type)


class SumPlusOneFactoryExecutionTest(tf.test.TestCase):

  def test_sum_scalar(self):
    sum_f = aggregator_test_utils.SumPlusOneFactory()
    value_type = federated_language.to_type(np.float32)
    process = sum_f.create(value_type)

    state = process.initialize()
    self.assertEqual(0, state)

    client_data = [1.0, 2.0, 3.0]
    output = process.next(state, client_data)
    self.assertEqual(1, output.state)
    self.assertAllClose(7.0, output.result)
    self.assertEqual(42, output.measurements)

  def test_sum_structure(self):
    sum_f = aggregator_test_utils.SumPlusOneFactory()
    value_type = federated_language.to_type(((np.float32, (2,)), np.int32))
    process = sum_f.create(value_type)

    state = process.initialize()
    self.assertEqual(0, state)

    client_data = [((1.0, 2.0), 3), ((2.0, 5.0), 4), ((3.0, 0.0), 5)]
    output = process.next(state, client_data)
    self.assertEqual(1, output.state)
    self.assertAllClose(((7.0, 8.0), 13), output.result)
    self.assertEqual(42, output.measurements)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
