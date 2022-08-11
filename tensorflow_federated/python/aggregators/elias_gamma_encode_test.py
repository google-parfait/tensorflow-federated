# Copyright 2022, Google LLC.
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
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

from tensorflow_federated.python.aggregators import elias_gamma_encode
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

_test_value_type_int32_tensor_rank_1 = (tf.int32, (4,))
_test_value_type_int32_tensor_rank_2 = (tf.int32, (2, 4))
_test_value_type_float32_tensor = (tf.float32, (3,))
_test_value_type_int64_tensor = (tf.int64, (4,))
_test_value_type_struct_int32_tensors = collections.OrderedDict({
    'layer1': (tf.int32, (4,)),
    'layer2': (tf.int32, (4,)),
})
_test_federated_value_type_int32_tensor = computation_types.at_clients(
    _test_value_type_int32_tensor_rank_1)
_test_federated_value_type_struct_int32_tensors = collections.OrderedDict({
    'layer1': computation_types.at_server((tf.int32, (4,))),
    'layer2': (tf.int32, (4,))
})

_test_client_values_int32_tensor_rank_1 = [[-5, 3, 0, 0], [-3, 1, 0, 0]]
_test_expected_result_int32_tensor_rank_1 = [-8, 4, 0, 0]

_test_client_values_int32_tensor_rank_2 = [[[-5, 3, 0, 0], [-3, 1, 0, 0]],
                                           [[-5, 3, 0, 0], [-3, 1, 0, 0]]]
_test_expected_result_int32_tensor_rank_2 = [[-10, 6, 0, 0], [-6, 2, 0, 0]]
_test_client_values_struct_int32_tensors = [
    collections.OrderedDict({
        'layer1': [-5, 3, 0, 0],
        'layer2': [-3, 1, 0, 0],
    }),
    collections.OrderedDict({
        'layer1': [-5, 3, 0, 0],
        'layer2': [-3, 1, 0, 0],
    })
]
_test_expected_result_struct_int32_tensors = collections.OrderedDict({
    'layer1': [-10, 6, 0, 0],
    'layer2': [-6, 2, 0, 0],
})

_test_avg_bitrate_int32_tensor_rank_1 = 16. / 4.
_test_avg_bitrate_int32_tensor_rank_2 = 32. / 8.
_test_avg_bitrate_struct_int32_tensors = 32. / 8.


class EncodeUtilTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32_tensor_rank_1', _test_value_type_int32_tensor_rank_1),
      ('int32_tensor_rank_2', _test_value_type_int32_tensor_rank_2),
      ('struct_int32_tensors', _test_value_type_struct_int32_tensors))
  def test_check_is_non_federated_structure_of_int32_passes(self, value_type):
    value_type = computation_types.to_type(value_type)
    self.assertTrue(
        elias_gamma_encode._is_int32_or_structure_of_int32s(value_type))

  @parameterized.named_parameters(
      ('float32_tensor', _test_value_type_float32_tensor),
      ('int64_tensor', _test_value_type_int64_tensor),
      ('federated_int32_tensor', _test_federated_value_type_int32_tensor),
      ('federated_struct_int32_tensors',
       _test_federated_value_type_struct_int32_tensors))
  def test_check_is_non_federated_structure_of_int32_fails(self, value_type):
    value_type = computation_types.to_type(value_type)
    self.assertFalse(
        elias_gamma_encode._is_int32_or_structure_of_int32s(value_type))


class EncodeComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32_tensor_rank_1', _test_value_type_int32_tensor_rank_1),
      ('int32_tensor_rank_2', _test_value_type_int32_tensor_rank_2),
      ('struct_int32_tensors', _test_value_type_struct_int32_tensors))
  def test_encode_properties(self, value_type):
    factory = elias_gamma_encode.EliasGammaEncodedSumFactory(
        bitrate_mean_factory=mean.UnweightedMeanFactory())
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.at_server(())
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    type_test_utils.assert_types_equivalent(process.initialize.type_signature,
                                            expected_initialize_type)

    expected_measurements_type = computation_types.StructType([
        ('elias_gamma_code_avg_bitrate', tf.float64)
    ])
    expected_measurements_type = computation_types.at_server(
        expected_measurements_type)
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    type_test_utils.assert_types_equivalent(process.next.type_signature,
                                            expected_next_type)

  @parameterized.named_parameters(
      ('float32_tensor', _test_value_type_float32_tensor),
      ('int64_tensor', _test_value_type_int64_tensor))
  def test_encode_create_raises(self, value_type):
    factory = elias_gamma_encode.EliasGammaEncodedSumFactory()
    value_type = computation_types.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class EncodeExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32_tensor_rank_1', _test_value_type_int32_tensor_rank_1,
       _test_client_values_int32_tensor_rank_1,
       _test_expected_result_int32_tensor_rank_1,
       _test_avg_bitrate_int32_tensor_rank_1),
      ('int32_tensor_rank_2', _test_value_type_int32_tensor_rank_2,
       _test_client_values_int32_tensor_rank_2,
       _test_expected_result_int32_tensor_rank_2,
       _test_avg_bitrate_int32_tensor_rank_2),
      ('struct_int32_tensors', _test_value_type_struct_int32_tensors,
       _test_client_values_struct_int32_tensors,
       _test_expected_result_struct_int32_tensors,
       _test_avg_bitrate_struct_int32_tensors))
  def test_encode_impl(self, value_type, client_values, expected_result,
                       avg_bitrate):
    factory = elias_gamma_encode.EliasGammaEncodedSumFactory(
        bitrate_mean_factory=mean.UnweightedMeanFactory())
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    expected_measurements = collections.OrderedDict(
        elias_gamma_code_avg_bitrate=avg_bitrate)

    measurements = process.next(state, client_values).measurements
    self.assertAllClose(measurements, expected_measurements)
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

  @parameterized.named_parameters(
      ('int32_tensor_rank_1', _test_client_values_int32_tensor_rank_1,
       _test_avg_bitrate_int32_tensor_rank_1),
      ('int32_tensor_rank_2', _test_client_values_int32_tensor_rank_2,
       _test_avg_bitrate_int32_tensor_rank_2))
  def test_bitstring_impl(self, client_values, avg_bitrate):
    num_elements = tf.size(client_values[0], out_type=tf.float64)
    bitstrings = [tfc.run_length_gamma_encode(x) for x in client_values]
    bitrates = [
        elias_gamma_encode._get_bitrate(x, num_elements) for x in bitstrings
    ]
    self.assertEqual(np.mean(bitrates), avg_bitrate)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
