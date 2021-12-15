# Copyright 2021, Google LLC.
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
"""Tests for `rotation.py`."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import mean
from tensorflow_federated.python.aggregators import measurements
from tensorflow_federated.python.aggregators import rotation
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

SEED_PAIR = (42, 42)

_test_struct_type_int_mixed = [tf.int32, (tf.int32, (3,)), (tf.int32, (2, 2))]
_test_struct_type_float_mixed = [
    tf.float32, (tf.float32, (3,)), (tf.float32, (2, 2))
]
_test_struct_type_nested = collections.OrderedDict(
    a=[tf.float32, [(tf.float32, (2, 1))]], b=(tf.float32, (1, 1, 2)))


def _make_test_struct_value_nested(value):
  return collections.OrderedDict(
      a=[tf.cast(value, tf.float32), [tf.ones([2, 1]) * value]],
      b=tf.ones((1, 1, 2)) * value)


def _measured_test_sum_factory():
  # SumFactory which also returns the sum as measurements. This is useful for
  # monitoring what values are passed through an inner aggregator.
  return measurements.add_measurements(
      sum_factory.SumFactory(),
      server_measurement_fn=lambda x: collections.OrderedDict(sum=x))


def _hadamard_mean():
  return rotation.HadamardTransformFactory(mean.UnweightedMeanFactory())


def _hadamard_sum():
  return rotation.HadamardTransformFactory(sum_factory.SumFactory())


def _measured_hadamard_sum():
  return rotation.HadamardTransformFactory(_measured_test_sum_factory())


def _dft_mean():
  return rotation.DiscreteFourierTransformFactory(mean.UnweightedMeanFactory())


def _dft_sum():
  return rotation.DiscreteFourierTransformFactory(sum_factory.SumFactory())


def _measured_dft_sum():
  return rotation.DiscreteFourierTransformFactory(_measured_test_sum_factory())


def _named_test_cases_product(dict1, dict2):
  """Utility for creating parameterized named test cases."""
  named_cases = []
  for k1, v1 in dict1.items():
    for k2, v2 in dict2.items():
      named_cases.append(('_'.join([k1, k2]), v1, v2))
  return named_cases


class RotationsComputationTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      _named_test_cases_product({
          'hd': 'hd',
          'dft': 'dft',
      }, {
          'float': tf.float32,
          'ints': [tf.int32, tf.int32, tf.int32],
          'struct': _test_struct_type_float_mixed,
          'nested_struct': _test_struct_type_nested,
      }))
  def test_type_properties(self, name, value_type):
    factory = _hadamard_sum() if name == 'hd' else _dft_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.at_server(
        ((), rotation.SEED_TFF_TYPE))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assert_types_equivalent(process.initialize.type_signature,
                                 expected_initialize_type)

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict([(name, ())]))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=expected_measurements_type))
    self.assert_types_equivalent(process.next.type_signature,
                                 expected_next_type)

  @parameterized.named_parameters(
      _named_test_cases_product({
          'hd': _hadamard_sum,
          'dft': _dft_sum,
      }, {
          'bool': tf.bool,
          'string': tf.string,
          'nested_string': [tf.string, [tf.string]],
      }))
  def test_raises_on_non_numeric_component_tensor_dtypes(
      self, factory_fn, value_type):
    factory = factory_fn()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'all integers or all floats'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('plain_struct_hadamard', _hadamard_sum, [('a', tf.int32)]),
      ('plain_struct_dft', _dft_sum, [('a', tf.int32)]),
      ('sequence_hadamard', _hadamard_sum,
       computation_types.SequenceType(tf.int32)),
      ('sequence_dft', _dft_sum, computation_types.SequenceType(tf.int32)),
      ('func_hadamard', _hadamard_sum,
       computation_types.FunctionType(tf.int32, tf.int32)),
      ('func_dft', _dft_sum, computation_types.FunctionType(tf.int32,
                                                            tf.int32)),
      ('nested_sequence', _dft_sum,
       [[[computation_types.SequenceType(tf.int32)]]]))
  def test_raises_on_bad_tff_value_types(self, factory_fn, value_type):
    factory = _hadamard_sum()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'Expected `value_type` to be'):
      factory.create(value_type)


class RotationsExecutionTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_hd', tf.int32, [1, 2, 3], 6, _hadamard_sum),
      ('rank_1_tensor_hd', (tf.int32, [4]),
       [np.arange(4), np.arange(4) * 2], np.arange(4) * 3, _hadamard_sum),
      ('rank_2_tensor_hd', (tf.int32, [1, 2]), [((1, 1),), ((2, 2),)],
       ((3, 3),), _hadamard_sum),
      ('nested_hd', _test_struct_type_nested, [
          _make_test_struct_value_nested(123),
          _make_test_struct_value_nested(456)
      ], _make_test_struct_value_nested(579), _hadamard_sum),
      ('scalar_dft', tf.int32, [1, 2, 3], 6, _dft_sum),
      ('rank_1_tensor_dft', (tf.int32, [4]),
       [np.arange(4), np.arange(4) * 2], np.arange(4) * 3, _dft_sum),
      ('rank_2_tensor_dft', (tf.int32, [1, 2]), [((1, 1),), ((2, 2),)],
       ((3, 3),), _dft_sum),
      ('nested_dft', _test_struct_type_nested, [
          _make_test_struct_value_nested(123),
          _make_test_struct_value_nested(456)
      ], _make_test_struct_value_nested(579), _dft_sum),
  )
  def test_sum(self, value_type, client_data, expected_sum, factory_fn):
    """Integration test with sum for the all implementations."""
    factory = factory_fn()
    process = factory.create(computation_types.to_type(value_type))
    state = process.initialize()

    for _ in range(3):
      output = process.next(state, client_data)
      self.assertAllClose(output.result, expected_sum, atol=0)
      state = output.state

  @parameterized.named_parameters(
      ('scalar_hd', tf.float32, [1, 2, 3, 4], 2.5, _hadamard_mean),
      ('rank_1_tensor_hd', (tf.float32, [2]), [(1, 1), (6, 6)],
       (3.5, 3.5), _hadamard_mean),
      ('rank_2_tensor_hd', (tf.float32, [1, 2]), [((-1, -1),), ((5, 5),)],
       ((2, 2),), _hadamard_mean),
      ('nested_hd', _test_struct_type_nested, [
          _make_test_struct_value_nested(123),
          _make_test_struct_value_nested(-321)
      ], _make_test_struct_value_nested(-99), _hadamard_mean),
      ('scalar_dft', tf.float32, [1, 2, 3, 4], 2.5, _dft_mean),
      ('rank_1_tensor_dft', (tf.float32, [2]), [(1, 1),
                                                (6, 6)], (3.5, 3.5), _dft_mean),
      ('rank_2_tensor_dft', (tf.float32, [1, 2]), [((-1, -1),), ((5, 5),)],
       ((2, 2),), _dft_mean),
      ('nested_dft', _test_struct_type_nested, [
          _make_test_struct_value_nested(123),
          _make_test_struct_value_nested(-321)
      ], _make_test_struct_value_nested(-99), _dft_mean),
  )
  def test_mean(self, value_type, client_data, expected_mean, factory_fn):
    """Integration test for the factory with mean."""
    factory = factory_fn()
    process = factory.create(computation_types.to_type(value_type))
    state = process.initialize()

    for _ in range(3):
      output = process.next(state, client_data)
      self.assertAllClose(output.result, expected_mean)
      state = output.state

  @parameterized.named_parameters(
      ('hd-5', 'hd', [5], [8]),
      ('hd-8', 'hd', [8], [8]),
      ('hd-2x3x4', 'hd', [2, 3, 4], [32]),
      ('dft-5', 'dft', [5], [6]),
      ('dft-8', 'dft', [8], [8]),
      ('dft-3x5x3', 'dft', [3, 5, 3], [46]),
  )
  def test_inner_aggregation_acts_on_padded_space(self, name, input_shape,
                                                  expected_inner_shape):
    factory = _measured_hadamard_sum() if name == 'hd' else _measured_dft_sum()
    process = factory.create(
        computation_types.to_type((tf.float32, input_shape)))

    client_input = tf.ones(input_shape)
    output = process.next(process.initialize(), [client_input])
    inner_shape = output.measurements[name]['sum'].shape
    self.assertAllEqual(expected_inner_shape, inner_shape)

  @parameterized.named_parameters(('hd', 'hd'), ('dft', 'dft'))
  def test_inner_aggregation_acts_on_rotated_space(self, name):
    factory = _measured_hadamard_sum() if name == 'hd' else _measured_dft_sum()
    process = factory.create(computation_types.TensorType(tf.float32, [8]))

    client_input = np.array([1.0, -1.0, 2.5, -1.5, -0.5, 1.9, 2.2, -2.0])
    state = process.initialize()
    output = process.next(state, [client_input])
    inner_aggregand_1 = output.measurements[name]['sum']
    # The value passed to the inner aggregation after projection should be
    # different than the input to the outer aggregation.
    self.assertNotAllClose(np.zeros([8]), inner_aggregand_1 - client_input)
    # Rotation preserves L2 norm.
    self.assertAllClose(
        np.linalg.norm(inner_aggregand_1), np.linalg.norm(client_input))

    output = process.next(output.state, [client_input])
    inner_aggregand_2 = output.measurements[name]['sum']
    # The projections are randomized, independent in each round.
    self.assertNotAllClose(np.zeros([8]), inner_aggregand_2 - client_input)
    self.assertNotAllClose(np.zeros([8]), inner_aggregand_2 - inner_aggregand_1)
    self.assertAllClose(
        np.linalg.norm(inner_aggregand_1), np.linalg.norm(inner_aggregand_2))

  def test_hd_spreads_information(self):
    factory = _measured_hadamard_sum()
    process = factory.create(computation_types.TensorType(tf.float32, [256]))

    client_input = 256 * tf.one_hot(indices=17, depth=256, dtype=tf.float32)
    output = process.next(process.initialize(), [client_input])
    inner_aggregand = output.measurements['hd']['sum']

    # For Hadamard we expect values equal to +/- sqrt(256) = 16.
    self.assertAllEqual(
        np.logical_or(
            np.isclose(inner_aggregand, 16), np.isclose(inner_aggregand, -16)),
        [True] * 256)
    self.assertBetween(np.var(inner_aggregand), 255, 257)

  def test_dft_spreads_information(self):
    factory = _measured_dft_sum()
    process = factory.create(computation_types.TensorType(tf.float32, [256]))

    client_input = 256 * tf.one_hot(indices=17, depth=256, dtype=tf.float32)
    output = process.next(process.initialize(), [client_input])
    inner_aggregand = output.measurements['dft']['sum']

    # We expect the values to be non-zero. Check that numerically the values are
    # bounded away from zero, with some slack (240 out of 256 scalars).
    self.assertGreaterEqual(
        np.sum(np.greater(np.abs(inner_aggregand), 1e-6)), 240)
    self.assertBetween(np.var(inner_aggregand), 255, 257)


class SeedUtilsTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('stride-1', 1), ('stride-3', 3))
  def test_init_and_next(self, stride):
    # First element of the seed of shape [2] is used as a timestamp kept fixed
    # across rounds, the second as a counter starting at 0, making sure fresh
    # seeds are always used.
    seed = rotation._init_global_seed()
    self.assertEqual((2,), seed.shape)
    self.assertEqual(0, seed[1])

    next_seed_fn = rotation._build_next_global_seed_fn(stride=stride)
    for _ in range(3):
      new_seed = next_seed_fn(seed)
      self.assertEqual(seed[0], new_seed[0])
      self.assertEqual(seed[1] + stride, new_seed[1])
      seed = new_seed

  def test_unique_seeds_for_struct(self):
    value = (tf.constant(1.0), (tf.constant(2.0), tf.constant([3.0, 4.0])))
    seed = (1, 101)
    unique_seeds = rotation._unique_seeds_for_struct(value, seed, stride=1)
    tf.nest.map_structure(self.assertAllEqual,
                          (np.array([1, 101]),
                           (np.array([1, 102]), np.array([1, 103]))),
                          unique_seeds)

    unique_seeds = rotation._unique_seeds_for_struct(value, seed, stride=3)
    tf.nest.map_structure(self.assertAllEqual,
                          (np.array([1, 101]),
                           (np.array([1, 104]), np.array([1, 107]))),
                          unique_seeds)


class PaddingUtilsTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('1', [1], [1]),
      ('2', [1, 1], [1, 1]),
      ('3', [1, 1, 1], [1, 1, 1, 0]),
      ('5', [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]),
  )
  def test_pad_pow2(self, value, expected_padded):
    padded = rotation._pad_zeros_pow2(tf.convert_to_tensor(value))
    self.assertAllEqual(expected_padded, padded)

  @parameterized.named_parameters(
      ('1', [1], [1, 0]),
      ('2', [1, 1], [1, 1]),
      ('3', [1, 1, 1], [1, 1, 1, 0]),
      ('5', [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]),
  )
  def test_pad_even(self, value, expected_padded):
    padded = rotation._pad_zeros_even(tf.convert_to_tensor(value))
    self.assertAllEqual(expected_padded, padded)

  def test_pad_pow_2_struct(self):
    struct = (tf.ones([3]), tf.ones([2, 3]), tf.ones([1, 2, 2]))
    flat_padded_struct = rotation._flatten_and_pad_zeros_pow2(struct)
    tf.nest.map_structure(
        self.assertAllEqual,
        (tf.constant([1, 1, 1, 0]), tf.constant([1, 1, 1, 1, 1, 1, 0, 0]),
         tf.constant([1, 1, 1, 1])), flat_padded_struct)

  def test_pad_even_struct(self):
    struct = (tf.ones([3]), tf.ones([2, 2]), tf.ones([1, 3, 3]))
    flat_padded_struct = rotation._flatten_and_pad_zeros_even(struct)
    tf.nest.map_structure(self.assertAllEqual,
                          (tf.constant([1, 1, 1, 0]), tf.constant([1, 1, 1, 1]),
                           tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])),
                          flat_padded_struct)

  def test_revert_padding(self):
    value = tf.range(8)
    spec = tf.TensorSpec([1, 2, 3], tf.float32)
    self.assertAllEqual([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]],
                        rotation._slice_and_reshape_to_template_spec(
                            value, spec))


class SampleRademacherTest(test_case.TestCase, parameterized.TestCase):

  def _assert_signs(self, x):
    """Helper function that checks every element of a tensor is +1/-1."""
    flat_x = x.reshape([-1])
    result = np.logical_or(np.isclose(1.0, flat_x), np.isclose(-1.0, flat_x))
    self.assertAllEqual(result, [True] * x.size)

  @parameterized.named_parameters(
      ('num_elements_1', 1),
      ('num_elements_10', 10),
      ('num_elements_100', 100),
  )
  def test_output_values(self, num_elements):
    shape = (1, num_elements)
    signs = rotation.sample_rademacher(shape, tf.int32, SEED_PAIR)
    signs = self.evaluate(signs)
    self._assert_signs(signs)

  def test_both_values_present(self):
    signs = rotation.sample_rademacher((100, 10), tf.int32, SEED_PAIR)
    signs = self.evaluate(signs)
    self._assert_signs(signs)
    self.assertGreater(np.sum(np.isclose(1.0, signs)), 450)
    self.assertGreater(np.sum(np.isclose(-1.0, signs)), 450)

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def test_expected_dtype(self, dtype):
    shape = (1000,)
    signs = rotation.sample_rademacher(shape, dtype, SEED_PAIR)
    self.assertEqual(dtype, signs.dtype)
    signs = self.evaluate(signs)
    self._assert_signs(signs)

  @parameterized.named_parameters(
      ('shape-10', [10]),
      ('shape-10x10', [10, 10]),
      ('shape-10x10x10', [10, 10, 10]),
      ('shape-10x1x1x1', [10, 1, 1, 1]),
  )
  def test_expected_shape(self, shape):
    signs = rotation.sample_rademacher(shape, tf.int32, SEED_PAIR)
    signs = self.evaluate(signs)
    self._assert_signs(signs)
    self.assertAllEqual(signs.shape, shape)

  def test_different_sampels_with_different_seeds(self):
    shape = (100,)
    signs_1 = rotation.sample_rademacher(shape, tf.int32, seed=(42, 42))
    signs_2 = rotation.sample_rademacher(shape, tf.int32, seed=(420, 420))
    signs_1, signs_2 = self.evaluate([signs_1, signs_2])
    self.assertFalse(np.array_equal(signs_1, signs_2))


class SampleCisTest(test_case.TestCase, parameterized.TestCase):

  def test_uniform_angles(self):
    # Checks that the average is close to zero.
    trials = 1000
    angles = rotation.sample_cis((trials,), SEED_PAIR)
    value = self.evaluate(tf.reduce_mean(angles))
    self.assertAllClose(value, 0 + 0j, atol=0.1)

  @parameterized.named_parameters(('true', True), ('false', False))
  def test_unit_length(self, inverse):
    # Checks that each complex number has absolute value |r| = 1.
    shape = (100,)
    angles = rotation.sample_cis(shape, SEED_PAIR, inverse=inverse)
    lengths = self.evaluate(tf.math.abs(angles))
    self.assertAllClose(lengths, np.ones(shape))

  def test_inverse(self):
    shape = (100,)
    forward_angles = rotation.sample_cis(shape, SEED_PAIR, inverse=False)
    backward_angles = rotation.sample_cis(shape, SEED_PAIR, inverse=True)
    # Angles should revert to identity.
    actual = forward_angles * backward_angles
    expected = tf.complex(real=tf.ones(shape), imag=tf.zeros(shape))
    actual, expected = self.evaluate([actual, expected])
    self.assertAllClose(actual, expected)

  @parameterized.named_parameters(('true', True), ('false', False))
  def test_output_dtype(self, inverse):
    shape = (100,)
    angles = rotation.sample_cis(shape, SEED_PAIR, inverse=inverse)
    self.assertEqual(angles.dtype, tf.complex64)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'shape-10': [10],
              'shape-10x10': [10, 10],
              'shape-10x10x10': [10, 10, 10],
              'shape-10x1x1x1': [10, 1, 1, 1],
          }, {
              'true': True,
              'false': False,
          }))
  def test_expected_shape(self, shape, inverse):
    angles = rotation.sample_cis(shape, SEED_PAIR, inverse=inverse)
    self.assertAllEqual(angles.shape, shape)

  def test_different_samples_with_different_seeds(self):
    shape = (100,)
    angles_1 = rotation.sample_cis(shape, seed=(42, 42))
    angles_2 = rotation.sample_cis(shape, seed=(4200, 4200))
    angles_1, angles_2 = self.evaluate([angles_1, angles_2])
    self.assertFalse(np.array_equal(angles_1, angles_2))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
