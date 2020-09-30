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
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import test_utils as aggregators_test_utils
from tensorflow_federated.python.common_libs import test_utils as common_libs_test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

M_CONST = aggregators_test_utils.MEASUREMENT_CONSTANT


class MeanFactoryComputationTest(common_libs_test_utils.TestCase,
                                 parameterized.TestCase):

  @parameterized.named_parameters(('float', tf.float32),
                                  ('struct', ((tf.float32, (2,)), tf.float64)))
  def test_type_properties(self, value_type):
    mean_f = mean_factory.MeanFactory()
    self.assertIsInstance(mean_f, factory.AggregationProcessFactory)
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.FederatedType(tf.float32,
                                                  placements.CLIENTS)
    process = mean_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.FederatedType(value_type,
                                                       placements.CLIENTS)
    result_value_type = computation_types.FederatedType(value_type,
                                                        placements.SERVER)
    expected_state_type = computation_types.FederatedType(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        placements.SERVER)
    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        placements.SERVER)

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=param_value_type,
            weight=weight_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(('float', tf.float32),
                                  ('struct', ((tf.float32, (2,)), tf.float64)))
  def test_type_properties_with_inner_factory(self, value_type):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    mean_f = mean_factory.MeanFactory(
        value_sum_factory=sum_factory, weight_sum_factory=sum_factory)
    self.assertIsInstance(mean_f, factory.AggregationProcessFactory)
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.FederatedType(tf.float32,
                                                  placements.CLIENTS)
    process = mean_f.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.FederatedType(value_type,
                                                       placements.CLIENTS)
    result_value_type = computation_types.FederatedType(value_type,
                                                        placements.SERVER)
    expected_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            value_sum_process=tf.int32, weight_sum_process=tf.int32),
        placements.SERVER)
    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(
            value_sum_process=tf.int32, weight_sum_process=tf.int32),
        placements.SERVER)

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=param_value_type,
            weight=weight_type),
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
    mean_f = mean_factory.MeanFactory()
    with self.assertRaises(TypeError):
      mean_f.create(bad_value_type)


class MeanFactoryExecutionTest(common_libs_test_utils.TestCase):

  def test_scalar_value(self):
    mean_f = mean_factory.MeanFactory()
    value_type = computation_types.to_type(tf.float32)
    process = mean_f.create(value_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        state)

    client_data = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        output.state)
    self.assertAllClose(2.0, output.result)
    self.assertEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        output.measurements)

  def test_structure_value(self):
    mean_f = mean_factory.MeanFactory()
    value_type = computation_types.to_type(((tf.float32, (2,)), tf.float64))
    process = mean_f.create(value_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        state)

    client_data = [((1.0, 2.0), 3.0), ((2.0, 5.0), 4.0), ((3.0, 0.0), 5.0)]
    weights = [1.0, 1.0, 1.0]
    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        output.state)
    self.assertAllClose(((2.0, 7 / 3), 4.0), output.result)
    self.assertEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()),
        output.measurements)

  def test_weight_arg(self):
    mean_f = mean_factory.MeanFactory()
    value_type = computation_types.to_type(tf.float32)
    process = mean_f.create(value_type)

    state = process.initialize()
    client_data = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    self.assertEqual(2.0, process.next(state, client_data, weights).result)
    weights = [0.1, 0.1, 0.1]
    self.assertEqual(2.0, process.next(state, client_data, weights).result)
    weights = [6.0, 3.0, 1.0]
    self.assertEqual(1.5, process.next(state, client_data, weights).result)

  def test_inner_value_sum_factory(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    mean_f = mean_factory.MeanFactory(value_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    process = mean_f.create(value_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=0, weight_sum_process=()),
        state)

    client_data = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    # Weighted values will be summed to 7.0.
    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=1, weight_sum_process=()),
        output.state)
    self.assertAllClose(7 / 3, output.result)
    self.assertEqual(
        collections.OrderedDict(
            value_sum_process=M_CONST, weight_sum_process=()),
        output.measurements)

  def test_inner_weight_sum_factory(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    mean_f = mean_factory.MeanFactory(weight_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    process = mean_f.create(value_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=0),
        state)

    client_data = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    # Weights will be summed to 4.0.
    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=1),
        output.state)
    self.assertAllClose(1.5, output.result)
    self.assertEqual(
        collections.OrderedDict(
            value_sum_process=(), weight_sum_process=M_CONST),
        output.measurements)

  def test_inner_value_and_weight_sum_factory(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    mean_f = mean_factory.MeanFactory(
        value_sum_factory=sum_factory, weight_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    process = mean_f.create(value_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=0, weight_sum_process=0),
        state)

    client_data = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    # Weighted values will be summed to 7.0 and weights will be summed to 4.0.
    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=1, weight_sum_process=1),
        output.state)
    self.assertAllClose(7 / 4, output.result)
    self.assertEqual(
        collections.OrderedDict(
            value_sum_process=M_CONST, weight_sum_process=M_CONST),
        output.measurements)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  common_libs_test_utils.main()
