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
import math
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import test_utils as aggregators_test_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

M_CONST = aggregators_test_utils.MEASUREMENT_CONSTANT

_test_struct_type = ((tf.float32, (2,)), tf.float64)


class MeanFactoryComputationTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('float_value_float32_weight', tf.float32, tf.float32),
      ('struct_value_float32_weight', _test_struct_type, tf.float32),
      ('float_value_float64_weight', tf.float32, tf.float64),
      ('struct_value_float64_weight', _test_struct_type, tf.float64),
      ('float_value_int32_weight', tf.float32, tf.int32),
      ('struct_value_int32_weight', _test_struct_type, tf.int32),
      ('float_value_int64_weight', tf.float32, tf.int64),
      ('struct_value_int64_weight', _test_struct_type, tf.int64),
  )
  def test_type_properties(self, value_type, weight_type):
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)

    factory_ = mean.MeanFactory()
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(value_type, weight_type)

    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.at_clients(value_type)
    result_value_type = computation_types.at_server(value_type)

    expected_state_type = computation_types.at_server(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=()))
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(mean_value=(), mean_weight=()))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_parameter = collections.OrderedDict(
        state=expected_state_type,
        value=param_value_type,
        weight=computation_types.at_clients(weight_type))

    expected_next_type = computation_types.FunctionType(
        parameter=expected_parameter,
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('float_value', tf.float32),
      ('struct_value', _test_struct_type),
  )
  def test_type_properties_unweighted(self, value_type):
    value_type = computation_types.to_type(value_type)

    factory_ = mean.UnweightedMeanFactory()
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(value_type)

    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.at_clients(value_type)
    result_value_type = computation_types.at_server(value_type)

    expected_state_type = computation_types.at_server(())
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(mean_value=()))

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
      ('float_value_float32_weight', tf.float32, tf.float32),
      ('struct_value_float32_weight', _test_struct_type, tf.float32),
      ('float_value_float64_weight', tf.float32, tf.float64),
      ('struct_value_float64_weight', _test_struct_type, tf.float64),
      ('float_value_int32_weight', tf.float32, tf.int32),
      ('struct_value_int32_weight', _test_struct_type, tf.int32),
      ('float_value_int64_weight', tf.float32, tf.int64),
      ('struct_value_int64_weight', _test_struct_type, tf.int64),
  )
  def test_type_properties_with_inner_factory(self, value_type, weight_type):
    value_type = computation_types.to_type(value_type)
    weight_type = computation_types.to_type(weight_type)
    sum_factory = aggregators_test_utils.SumPlusOneFactory()

    factory_ = mean.MeanFactory(
        value_sum_factory=sum_factory, weight_sum_factory=sum_factory)
    self.assertIsInstance(factory_, factory.WeightedAggregationFactory)
    process = factory_.create(value_type, weight_type)

    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.at_clients(value_type)
    result_value_type = computation_types.at_server(value_type)

    expected_state_type = computation_types.at_server(
        collections.OrderedDict(
            value_sum_process=tf.int32, weight_sum_process=tf.int32))
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(mean_value=tf.int32, mean_weight=tf.int32))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_parameter = collections.OrderedDict(
        state=expected_state_type,
        value=param_value_type,
        weight=computation_types.at_clients(weight_type))

    expected_next_type = computation_types.FunctionType(
        parameter=expected_parameter,
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('float_value', tf.float32),
      ('struct_value', _test_struct_type),
  )
  def test_type_properties_with_inner_factory_unweighted(self, value_type):
    value_type = computation_types.to_type(value_type)
    sum_factory = aggregators_test_utils.SumPlusOneFactory()

    factory_ = mean.UnweightedMeanFactory(value_sum_factory=sum_factory)
    self.assertIsInstance(factory_, factory.UnweightedAggregationFactory)
    process = factory_.create(value_type)

    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.at_clients(value_type)
    result_value_type = computation_types.at_server(value_type)

    expected_state_type = computation_types.at_server(tf.int32)
    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(mean_value=tf.int32))

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
      ('federated_type', computation_types.at_server(tf.float32)),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)))
  def test_incorrect_create_type_raises(self, wrong_type):
    factory_ = mean.MeanFactory()
    correct_type = computation_types.to_type(tf.float32)
    with self.assertRaises(TypeError):
      factory_.create(wrong_type, correct_type)
    with self.assertRaises(TypeError):
      factory_.create(correct_type, wrong_type)

    factory_ = mean.UnweightedMeanFactory()
    with self.assertRaises(TypeError):
      factory_.create(wrong_type)


class MeanFactoryExecutionTest(test_case.TestCase):

  def test_scalar_value(self):
    factory_ = mean.MeanFactory()
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)

    process = factory_.create(value_type, weight_type)
    expected_state = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=())
    expected_measurements = collections.OrderedDict(
        mean_value=(), mean_weight=())

    state = process.initialize()
    self.assertAllEqual(expected_state, state)

    client_data = [1.0, 2.0, 3.0]
    weights = [3.0, 2.0, 1.0]
    output = process.next(state, client_data, weights)
    self.assertAllClose(10. / 6., output.result)

    self.assertAllEqual(expected_state, output.state)
    self.assertEqual(expected_measurements, output.measurements)

  def test_scalar_value_unweighted(self):
    factory_ = mean.UnweightedMeanFactory()
    value_type = computation_types.to_type(tf.float32)

    process = factory_.create(value_type)
    expected_state = ()
    expected_measurements = collections.OrderedDict(mean_value=())

    state = process.initialize()
    self.assertAllEqual(expected_state, state)

    client_data = [1.0, 2.0, 3.0]
    output = process.next(state, client_data)
    self.assertAllClose(2.0, output.result)

    self.assertAllEqual(expected_state, output.state)
    self.assertEqual(expected_measurements, output.measurements)

  def test_structure_value(self):
    factory_ = mean.MeanFactory()
    value_type = computation_types.to_type(_test_struct_type)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)
    expected_state = collections.OrderedDict(
        value_sum_process=(), weight_sum_process=())
    expected_measurements = collections.OrderedDict(
        mean_value=(), mean_weight=())

    state = process.initialize()
    self.assertAllEqual(expected_state, state)

    client_data = [((1.0, 2.0), 3.0), ((2.0, 5.0), 4.0), ((3.0, 0.0), 5.0)]
    weights = [3.0, 2.0, 1.0]
    output = process.next(state, client_data, weights)
    self.assertAllEqual(expected_state, output.state)
    self.assertAllClose(((10. / 6., 16. / 6.), 22. / 6.), output.result)
    self.assertEqual(expected_measurements, output.measurements)

  def test_structure_value_unweighted(self):
    factory_ = mean.UnweightedMeanFactory()
    value_type = computation_types.to_type(_test_struct_type)
    process = factory_.create(value_type)
    expected_state = ()
    expected_measurements = collections.OrderedDict(mean_value=())

    state = process.initialize()
    self.assertAllEqual(expected_state, state)

    client_data = [((1.0, 2.0), 3.0), ((2.0, 5.0), 4.0), ((3.0, 0.0), 5.0)]
    output = process.next(state, client_data)

    self.assertAllEqual(expected_state, output.state)
    self.assertAllClose(((2.0, 7 / 3), 4.0), output.result)
    self.assertEqual(expected_measurements, output.measurements)

  def test_weight_arg(self):
    factory_ = mean.MeanFactory()
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)

    state = process.initialize()
    client_data = [1.0, 2.0, 3.0]
    weights = [1.0, 1.0, 1.0]
    self.assertEqual(2.0, process.next(state, client_data, weights).result)
    weights = [0.1, 0.1, 0.1]
    self.assertEqual(2.0, process.next(state, client_data, weights).result)
    weights = [6.0, 3.0, 1.0]
    self.assertEqual(1.5, process.next(state, client_data, weights).result)

  def test_weight_arg_all_zeros_nan_division(self):
    factory_ = mean.MeanFactory(no_nan_division=False)
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)

    state = process.initialize()
    client_data = [1.0, 2.0, 3.0]
    weights = [0.0, 0.0, 0.0]

    # Division by zero resulting in NaN/Inf *should* occur.
    self.assertFalse(
        math.isfinite(process.next(state, client_data, weights).result))

  def test_weight_arg_all_zeros_no_nan_division(self):
    factory_ = mean.MeanFactory(no_nan_division=True)
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)

    state = process.initialize()
    client_data = [1.0, 2.0, 3.0]
    weights = [0.0, 0.0, 0.0]

    # Division by zero resulting in NaN/Inf *should not* occur.
    self.assertEqual(0.0, process.next(state, client_data, weights).result)

  def test_inner_value_sum_factory(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    factory_ = mean.MeanFactory(value_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=0, weight_sum_process=()),
        state)

    # Weighted values will be summed to 11.0.
    client_data = [1.0, 2.0, 3.0]
    weights = [3.0, 2.0, 1.0]

    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=1, weight_sum_process=()),
        output.state)
    self.assertAllClose(11 / 6, output.result)
    self.assertEqual(
        collections.OrderedDict(mean_value=M_CONST, mean_weight=()),
        output.measurements)

  def test_inner_value_sum_factory_unweighted(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    factory_ = mean.UnweightedMeanFactory(value_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type)

    state = process.initialize()
    self.assertAllEqual(0, state)

    # Values will be summed to 7.0.
    client_data = [1.0, 2.0, 3.0]

    output = process.next(state, client_data)
    self.assertAllEqual(1, output.state)
    self.assertAllClose(7 / 3, output.result)
    self.assertEqual(
        collections.OrderedDict(mean_value=M_CONST), output.measurements)

  def test_inner_weight_sum_factory(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    factory_ = mean.MeanFactory(weight_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=0),
        state)

    # Weights will be summed to 7.0.
    client_data = [1.0, 2.0, 3.0]
    weights = [3.0, 2.0, 1.0]

    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=(), weight_sum_process=1),
        output.state)
    self.assertAllClose(10 / 7, output.result)
    self.assertEqual(
        collections.OrderedDict(mean_value=(), mean_weight=M_CONST),
        output.measurements)

  def test_inner_value_and_weight_sum_factory(self):
    sum_factory = aggregators_test_utils.SumPlusOneFactory()
    factory_ = mean.MeanFactory(
        value_sum_factory=sum_factory, weight_sum_factory=sum_factory)
    value_type = computation_types.to_type(tf.float32)
    weight_type = computation_types.to_type(tf.float32)
    process = factory_.create(value_type, weight_type)

    state = process.initialize()
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=0, weight_sum_process=0),
        state)

    # Weighted values will be summed to 11.0 and weights will be summed to 7.0.
    client_data = [1.0, 2.0, 3.0]
    weights = [3.0, 2.0, 1.0]

    output = process.next(state, client_data, weights)
    self.assertAllEqual(
        collections.OrderedDict(value_sum_process=1, weight_sum_process=1),
        output.state)
    self.assertAllClose(11 / 7, output.result)
    self.assertEqual(
        collections.OrderedDict(mean_value=M_CONST, mean_weight=M_CONST),
        output.measurements)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
