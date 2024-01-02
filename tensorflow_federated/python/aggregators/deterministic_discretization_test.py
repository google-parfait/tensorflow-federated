# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for DeterministicDiscretizationFactory."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import deterministic_discretization
from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import measurements
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

_test_struct_type_int = [np.int32, (np.int32, (2,)), (np.int32, (3, 3))]
_test_struct_type_float = [np.float32, (np.float32, (2,)), (np.float32, (3, 3))]

_test_nested_struct_type_float = collections.OrderedDict(
    a=[np.float32, [(np.float32, (2, 2, 1))]], b=(np.float32, (3, 3))
)


def _make_test_nested_struct_value(value):
  return collections.OrderedDict(
      a=[
          tf.constant(value, dtype=np.float32),
          [tf.constant(value, dtype=np.float32, shape=[2, 2, 1])],
      ],
      b=tf.constant(value, dtype=np.float32, shape=(3, 3)),
  )


def _named_test_cases_product(*args):
  """Utility for creating parameterized named test cases."""
  named_cases = []
  dict1, dict2 = args
  for k1, v1 in dict1.items():
    for k2, v2 in dict2.items():
      named_cases.append(('_'.join([k1, k2]), v1, v2))
  return named_cases


_measurement_aggregator = measurements.add_measurements(
    sum_factory.SumFactory(), client_measurement_fn=intrinsics.federated_sum
)


class DeterministicDiscretizationComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('float', np.float32),
      ('struct_list_float_scalars', [np.float16, np.float32, np.float64]),
      ('struct_list_float_mixed', _test_struct_type_float),
      ('struct_nested', _test_nested_struct_type_float),
  )
  def test_type_properties(self, value_type):
    factory = deterministic_discretization.DeterministicDiscretizationFactory(
        step_size=0.1,
        inner_agg_factory=_measurement_aggregator,
        distortion_aggregation_factory=mean.UnweightedMeanFactory(),
    )
    value_type = computation_types.to_type(value_type)
    quantize_type = type_conversions.structure_from_tensor_type_tree(
        lambda x: (np.int32, x.shape), value_type
    )
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.StructType(
        [('step_size', np.float32), ('inner_agg_process', ())]
    )
    server_state_type = computation_types.FederatedType(
        server_state_type, placements.SERVER
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    type_test_utils.assert_types_equivalent(
        process.initialize.type_signature, expected_initialize_type
    )

    expected_measurements_type = computation_types.StructType([
        ('deterministic_discretization', quantize_type),
        ('distortion', np.float32),
    ])
    expected_measurements_type = computation_types.FederatedType(
        expected_measurements_type, placements.SERVER
    )
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.FederatedType(
                value_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.FederatedType(
                value_type, placements.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    type_test_utils.assert_types_equivalent(
        process.next.type_signature, expected_next_type
    )

  @parameterized.named_parameters(
      ('bool', np.bool_),
      ('string', np.str_),
      ('int32', np.int32),
      ('int64', np.int64),
      ('int_nested', [np.int32, [np.int32]]),
  )
  def test_raises_on_bad_component_tensor_dtypes(self, value_type):
    factory = deterministic_discretization.DeterministicDiscretizationFactory(
        inner_agg_factory=_measurement_aggregator, step_size=0.1
    )
    value_type = computation_types.to_type(value_type)
    self.assertRaises(TypeError, factory.create, value_type)

  @parameterized.named_parameters(
      ('plain_struct', [('a', np.int32)]),
      ('sequence', computation_types.SequenceType(np.int32)),
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('nested_sequence', [[[computation_types.SequenceType(np.int32)]]]),
  )
  def test_raises_on_bad_tff_value_types(self, value_type):
    factory = deterministic_discretization.DeterministicDiscretizationFactory(
        inner_agg_factory=_measurement_aggregator, step_size=0.1
    )
    value_type = computation_types.to_type(value_type)
    self.assertRaises(TypeError, factory.create, value_type)


class DeterministicDiscretizationExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('scalar', np.float32, [1, 2, 3], 6),
      (
          'rank_1_tensor',
          (np.float32, [7]),
          [np.arange(7.0), np.arange(7.0) * 2],
          np.arange(7.0) * 3,
      ),
      (
          'rank_2_tensor',
          (np.float32, [1, 2]),
          [((1, 1),), ((2, 2),)],
          ((3, 3),),
      ),
      (
          'nested',
          _test_nested_struct_type_float,
          [
              _make_test_nested_struct_value(123),
              _make_test_nested_struct_value(456),
          ],
          _make_test_nested_struct_value(579),
      ),
  )
  def test_discretize_impl(self, value_type, client_values, expected_sum):
    factory = deterministic_discretization.DeterministicDiscretizationFactory(
        inner_agg_factory=_measurement_aggregator,
        step_size=0.1,
        distortion_aggregation_factory=mean.UnweightedMeanFactory(),
    )
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    expected_result = expected_sum
    expected_quantized_result = tf.nest.map_structure(
        lambda x: x * 10, expected_sum
    )
    expected_measurements = collections.OrderedDict(
        deterministic_discretization=expected_quantized_result, distortion=0.0
    )

    for _ in range(3):
      output = process.next(state, client_values)
      output_measurements = output.measurements
      self.assertAllClose(output_measurements, expected_measurements)
      result = output.result
      self.assertAllClose(result, expected_result)

  @parameterized.named_parameters(
      ('int32', np.int32), ('int64', np.int64), ('float64', np.float64)
  )
  def test_output_dtype(self, dtype):
    """Checks the tensor type gets casted during preprocessing."""
    x = tf.range(8, dtype=dtype)
    encoded_x = deterministic_discretization._discretize_struct(
        x, step_size=0.1
    )
    self.assertEqual(
        encoded_x.dtype, deterministic_discretization.OUTPUT_TF_TYPE
    )

  @parameterized.named_parameters(
      ('int32', np.int32), ('int64', np.int64), ('float64', np.float64)
  )
  def test_revert_to_input_dtype(self, dtype):
    """Checks that postprocessing restores the original dtype."""
    x = tf.range(8, dtype=dtype)
    encoded_x = deterministic_discretization._discretize_struct(x, step_size=1)
    decoded_x = deterministic_discretization._undiscretize_struct(
        encoded_x, step_size=1, tf_dtype_struct=dtype
    )
    self.assertEqual(dtype, decoded_x.dtype)


class QuantizationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'step_size_1': 2**-10,
              'step_size_2': 0.1,
              'step_size_3': 1,
              'step_size_4': 2**5,
          },
          {'shape_1': (10,), 'shape_2': (10, 10), 'shape_3': (10, 5, 2)},
      )
  )
  def test_error_from_rounding(self, step_size, shape):
    x = tf.random.uniform(shape=shape, minval=-10, maxval=10, dtype=tf.float32)
    encoded_x = deterministic_discretization._discretize_struct(x, step_size)
    decoded_x = deterministic_discretization._undiscretize_struct(
        encoded_x, step_size, tf_dtype_struct=np.float32
    )
    x, decoded_x = self.evaluate([x, decoded_x])

    self.assertAllEqual(x.shape, decoded_x.shape)
    # For deterministic rounding, errors are bounded by half of the bin width.
    quantization_atol = 0.5 * step_size
    self.assertAllClose(x, decoded_x, rtol=0.0, atol=quantization_atol)


class ScalingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('step_size_1', 1), ('step_size_2', 0.01), ('step_size_3', 10**-5)
  )
  def test_scaling(self, step_size):
    # Integers to prevent rounding.
    x = tf.random.stateless_uniform([100], (1, 1), -100, 100, dtype=tf.int32)
    discretized_x = deterministic_discretization._discretize_struct(
        x, tf.cast(step_size, np.float32)
    )
    reverted_x = deterministic_discretization._undiscretize_struct(
        discretized_x, step_size, tf_dtype_struct=np.int32
    )
    x, discretized_x, reverted_x = self.evaluate([x, discretized_x, reverted_x])
    self.assertAllEqual(
        tf.round(tf.divide(tf.cast(x, np.float32), step_size)), discretized_x
    )  # Scaling up.
    self.assertAllEqual(x, reverted_x)  # Scaling down.


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
