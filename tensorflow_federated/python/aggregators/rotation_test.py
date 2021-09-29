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


def _hadamard_mean():
  return rotation.HadamardTransformFactory(mean.UnweightedMeanFactory())


def _hadamard_sum():
  return rotation.HadamardTransformFactory(sum_factory.SumFactory())


def _dft_mean():
  return rotation.DiscreteFourierTransformFactory(mean.UnweightedMeanFactory())


def _dft_sum():
  return rotation.DiscreteFourierTransformFactory(sum_factory.SumFactory())


def _named_test_cases_product(*args):
  """Utility for creating parameterized named test cases."""
  named_cases = []
  if len(args) == 2:
    dict1, dict2 = args
    for k1, v1 in dict1.items():
      for k2, v2 in dict2.items():
        named_cases.append(('_'.join([k1, k2]), v1, v2))
  elif len(args) == 3:
    dict1, dict2, dict3 = args
    for k1, v1 in dict1.items():
      for k2, v2 in dict2.items():
        for k3, v3 in dict3.items():
          named_cases.append(('_'.join([k1, k2, k3]), v1, v2, v3))
  return named_cases


class FlatteningFactoryComputationTest(test_case.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(
      _named_test_cases_product({
          'hd': _hadamard_sum,
          'dft': _dft_sum,
      }, {
          'float': tf.float32,
          'ints': [tf.int32, tf.int32, tf.int32],
          'struct': _test_struct_type_float_mixed,
          'nested_struct': _test_struct_type_nested,
      }))
  def test_type_properties(self, factory_fn, value_type):
    factory = factory_fn()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.at_server(
        collections.OrderedDict(
            round_seed=(rotation.SEED_TF_TYPE, (2,)), inner_agg_process=()))

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assert_types_equivalent(process.initialize.type_signature,
                                 expected_initialize_type)

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(rotation=()))
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
    with self.assertRaisesRegex(TypeError, 'must all be integers or floats'):
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


class FlatteningFactoryExecutionTest(test_case.TestCase,
                                     parameterized.TestCase):
  """Shared tests for the parent class FlatteningFactory."""

  def setUp(self):
    super().setUp()
    self.seed_pair = tf.cast(SEED_PAIR, rotation.SEED_TF_TYPE)

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
      self.assertEqual(output.state['round_seed'].dtype,
                       rotation.SEED_TF_TYPE.as_numpy_dtype)
      self.assertEqual(output.state['round_seed'].shape, (2,))
      self.assertEqual(output.state['inner_agg_process'], ())
      self.assertEqual(output.measurements,
                       collections.OrderedDict(rotation=()))
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
    expected_measurements = collections.OrderedDict(
        rotation=collections.OrderedDict(mean_value=()))
    state = process.initialize()

    for _ in range(3):
      output = process.next(state, client_data)
      self.assertEqual(output.state['round_seed'].dtype,
                       rotation.SEED_TF_TYPE.as_numpy_dtype)
      self.assertEqual(output.state['round_seed'].shape, (2,))
      self.assertEqual(output.state['inner_agg_process'], ())
      self.assertEqual(output.measurements, expected_measurements)
      self.assertAllClose(output.result, expected_mean)
      state = output.state

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'shape-1': [1],
              'shape-2': [2],
              'shape-3': [3],
              'shape-2x2': [2, 2],
              'shape-3x3x1': [3, 3, 1],
              'shape-5x3x4': [5, 3, 4],
          }, {
              'hd': _hadamard_sum,
              'dft': _dft_sum,
          }))
  def test_postprocess_tensor_restores_shape(self, shape, factory_fn):
    """Checks that postprocessing restores the tensor (with padding)."""
    factory = factory_fn()
    input_dim = tf.TensorShape(shape).num_elements()
    x = tf.reshape(tf.range(input_dim), shape)
    original_spec = tf.TensorSpec(x.shape, x.dtype)
    prep_x = factory._preprocess_tensor(x)
    post_x = factory._postprocess_tensor(prep_x, original_spec)
    self.assertAllEqual(x, post_x)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'int32': tf.int32,
              'float32': tf.float32,
              'float64': tf.float64,
          }, {
              'hd': _hadamard_sum,
              'dft': _dft_sum,
          }))
  def test_postprocess_tensor_restores_dtype(self, dtype, factory_fn):
    """Checks that postprocessing restores the original dtype."""
    factory = factory_fn()
    x = tf.range(8, dtype=dtype)
    prep_x = factory._preprocess_tensor(x)
    post_x = factory._postprocess_tensor(prep_x, tf.TensorSpec([8], dtype))
    self.assertEqual(dtype, post_x.dtype)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'dim-2': 2,
              'dim-8': 8,
              'dim-32': 32,
              'dim-128': 128,
          },
          {
              'repeat-1': 1,
              'repeat-2': 3,
          },
          {
              'hd': _hadamard_sum,
              'dft': _dft_sum,
          },
      ))
  def test_forward_transform_vector_same_l2_norm(self, dim, repeat, factory_fn):
    """Checks that the L2 norm doesn't change much after rotation."""
    factory = factory_fn()
    x = tf.random.uniform((dim,))
    rotated_x = factory._forward_transform_vector(x, self.seed_pair, repeat)
    self.assertAllClose(np.linalg.norm(x), np.linalg.norm(rotated_x), atol=1e-5)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'dim-4': 4,
              'dim-16': 16,
              'dim-64': 64,
              'dim-256': 256,
          }, {
              'hd': _hadamard_sum,
              'dft': _dft_sum,
          }))
  def test_forward_transform_vector_has_rotation(self, dim, factory_fn):
    """Checks the vector difference norm after rotation is reasonably big."""
    factory = factory_fn()
    x = tf.random.uniform((dim,), minval=0, maxval=10)
    rotated_x = factory._forward_transform_vector(
        x, self.seed_pair, num_repeats=1)
    self.assertGreater(np.linalg.norm(rotated_x - x), 5)

  @parameterized.named_parameters(
      ('hd_dim-16', 16, 3, _hadamard_sum),
      ('hd_dim-64', 64, 6, _hadamard_sum),
      ('dft_dim-16', 16, 3, _dft_sum),
      ('dft_dim-64', 64, 3, _dft_sum),
  )
  def test_forward_transform_vector_use_different_repeat_seeds(
      self, dim, diff_norm, factory_fn):
    """Checks that forward transform repeats with different seed."""
    factory = factory_fn()
    x = tf.random.uniform((dim,))
    # Repeat transformations by making a single call.
    repeat_rotated_x = factory._forward_transform_vector(
        x, self.seed_pair, num_repeats=2)
    # Repeat by making separate calls.
    separate_rotated_x = factory._forward_transform_vector(
        x, self.seed_pair, num_repeats=1)
    separate_rotated_x = factory._forward_transform_vector(
        separate_rotated_x, self.seed_pair, num_repeats=1)

    difference = separate_rotated_x - repeat_rotated_x
    self.assertGreater(np.linalg.norm(difference), diff_norm)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'repeat-1': 1,
              'repeat-2': 2,
          },
          {
              'dim-2': 2,
              'dim-4': 4,
              'dim-128': 128,
          },
          {
              'hd': _hadamard_sum,
              'dft': _dft_sum,
          },
      ))
  def test_backward_transform_vector(self, repeat, dim, factory_fn):
    """Verifies that the inverse transform reverses the ops with same seeds."""
    factory = factory_fn()
    seed_pair = tf.constant((11, 22), dtype=rotation.SEED_TF_TYPE)
    x = tf.random.uniform((dim,))
    forward_x = factory._forward_transform_vector(
        x, seed_pair, num_repeats=repeat)
    reverted_x = factory._backward_transform_vector(
        forward_x, seed_pair, num_repeats=repeat)
    self.assertAllClose(x, reverted_x, atol=1e-5)

  @parameterized.named_parameters(
      _named_test_cases_product({
          'repeat-1': 1,
          'repeat-2': 2,
      }, {
          'hd': _hadamard_sum,
          'dft': _dft_sum,
      }))
  def test_transform_structure_integration(self, repeat, factory_fn):
    """Integration test for TF structure transforms."""
    factory = factory_fn()
    rand = tf.random.uniform
    input_struct = [
        rand([1]),
        rand([3, 3]),
        rand([4, 4, 4]), [rand([3]), rand([1, 1, 3])],
        collections.OrderedDict(a=rand([2]), b=rand([10]))
    ]
    seed_pair = tf.constant((11, 22), dtype=rotation.SEED_TF_TYPE)
    input_struct_type = tf.nest.map_structure(
        lambda x: tf.TensorSpec(x.shape, x.dtype), input_struct)
    forward_struct = factory._forward_transform_struct(
        input_struct, seed_pair, num_repeats=repeat)
    backward_struct = factory._backward_transform_struct(
        forward_struct, input_struct_type, seed_pair, num_repeats=repeat)
    self.assertAllClose(input_struct, backward_struct, atol=1e-5)


class HadamardTransformFactoryExecutionTest(test_case.TestCase,
                                            parameterized.TestCase):

  @parameterized.named_parameters(
      ('shape-1', [1]),
      ('shape-2', [2]),
      ('shape-4', [4]),
      ('shape-2x2', [2, 2]),
      ('shape-16x16', [16]),
      ('shape-64x64', [64]),
      ('shape-4x4x4', [4, 4, 4]),
  )
  def test_preprocess_tensor_no_padding(self, shape):
    """Checks the same input is returned if no padding is requried."""
    factory = _hadamard_sum()
    dim = tf.TensorShape(shape).num_elements()
    x = tf.reshape(tf.range(dim), shape)
    prep_x = factory._preprocess_tensor(x)
    self.assertAllEqual(tf.reshape(x, [-1]), prep_x)

  @parameterized.named_parameters(
      ('shape-3', [3], 4),
      ('shape-9', [9], 16),
      ('shape-31', [31], 32),
      ('shape-65', [65], 128),
      ('shape-3x3', [3, 3], 16),
      ('shape-5x3x17', [5, 3, 17], 256),
      ('shape-1x1x1x3', [1, 1, 1, 3], 4),
  )
  def test_preprocess_tensor_padding(self, input_shape, padded_dim):
    """Checks size, dtype, and content of the padded vector."""
    factory = _hadamard_sum()
    input_dim = tf.TensorShape(input_shape).num_elements()
    num_zeros = padded_dim - input_dim
    flat_x = tf.range(input_dim)
    x = tf.reshape(flat_x, input_shape)

    prep_x = factory._preprocess_tensor(x)
    self.assertEqual(prep_x.shape, (padded_dim,))
    self.assertAllEqual(prep_x[:input_dim], flat_x)
    self.assertAllEqual(prep_x[input_dim:], np.zeros((num_zeros,)))

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
      ('float64', tf.float64),
  )
  def test_preprocess_tensor_dtype(self, dtype):
    """Checks the tensor type gets casted during preprocessing."""
    factory = _hadamard_sum()
    x = tf.range(8, dtype=dtype)
    prep_x = factory._preprocess_tensor(x)
    expected_dtype = tf.float32
    self.assertEqual(prep_x.dtype, expected_dtype)

  def test_flattening(self):
    """A basic test for the flattening behaviour of this transform."""
    dim = 128
    factory = _hadamard_sum()
    scaled_onehot_x = dim * tf.one_hot(indices=17, depth=dim)

    seed_pair = tf.constant([1234, 5678], dtype=tf.int64)
    flat_x = factory._forward_transform_vector(
        scaled_onehot_x, seed_pair, num_repeats=1)

    # For Hadamard we expect +/- sqrt(dim) values.
    flat_val = np.sqrt(dim)
    is_flat = np.logical_or(
        np.isclose(flat_val, flat_x), np.isclose(-flat_val, flat_x))
    self.assertAllEqual(is_flat, [True] * dim)
    # Variance should be ~ norm^2 / dim = dim.
    self.assertBetween(np.var(flat_x), dim - 1, dim + 1)


class DiscreteFourierTransformFactoryExecutionTest(test_case.TestCase,
                                                   parameterized.TestCase):

  @parameterized.named_parameters(
      ('shape-2', [2]),
      ('shape-4', [4]),
      ('shape-2x3', [2, 3]),
      ('shape-14', [14]),
      ('shape-60', [60]),
      ('shape-6x2x4', [6, 2, 4]),
  )
  def test_preprocess_tensor_no_padding(self, shape):
    """Checks the same input is returned if no padding is requried."""
    factory = _dft_sum()
    dim = tf.TensorShape(shape).num_elements()
    x = tf.reshape(tf.range(dim), shape)
    prep_x = factory._preprocess_tensor(x)
    self.assertAllEqual(tf.reshape(x, [-1]), prep_x)

  @parameterized.named_parameters(
      ('shape-1', [1], 2),
      ('shape-9', [9], 10),
      ('shape-31', [31], 32),
      ('shape-65', [65], 66),
      ('shape-3x3', [3, 3], 10),
      ('shape-3x17x1', [3, 17, 1], 52),
      ('shape-1x1x1x3', [1, 1, 1, 5], 6),
  )
  def test_preprocess_tensor_padding(self, input_shape, padded_dim):
    """Checks size, dtype, and content of the padded vector."""
    factory = _dft_sum()
    input_dim = tf.TensorShape(input_shape).num_elements()
    num_zeros = padded_dim - input_dim
    flat_x = tf.range(input_dim)
    x = tf.reshape(flat_x, input_shape)

    prep_x = factory._preprocess_tensor(x)
    self.assertEqual(prep_x.shape, (padded_dim,))
    self.assertAllEqual(prep_x[:input_dim], flat_x)
    self.assertAllEqual(prep_x[input_dim:], np.zeros((num_zeros,)))

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('int64', tf.int64),
      ('float64', tf.float64),
  )
  def test_preprocess_tensor_dtype(self, dtype):
    """Checks the tensor type gets casted during preprocessing."""
    factory = _dft_sum()
    x = tf.range(8, dtype=dtype)
    prep_x = factory._preprocess_tensor(x)
    expected_dtype = tf.float32
    self.assertEqual(prep_x.dtype, expected_dtype)

  def test_flattening(self):
    """A basic test for the flattening behaviour of this transform."""
    dim = 128
    factory = _dft_sum()
    scaled_onehot_x = dim * tf.one_hot(indices=17, depth=dim)

    seed_pair = tf.constant([1234, 5678], dtype=tf.int64)
    flat_x = factory._forward_transform_vector(
        scaled_onehot_x, seed_pair, num_repeats=1)
    flat_x = self.evaluate(flat_x)

    # Columns of DFT matrix have roots of unity, so a prime index column should
    # give around `dim` unique real/imag components that are negatives of each
    # other. We can simply test that we have at least half as much unique
    # components to account for precision issues, zero components, etc.
    num_uniq = len(np.unique(np.around(flat_x, 2)))
    num_abs_uniq = len(np.unique(np.abs(np.around(flat_x, 2))))
    self.assertGreaterEqual(num_uniq, dim // 2)
    self.assertGreaterEqual(num_abs_uniq, dim // 4)
    # Variance should be ~ norm^2 / dim = dim.
    self.assertBetween(np.var(flat_x), dim - 1, dim + 1)


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
    signs_1 = rotation.sample_rademacher(shape, tf.int32, seed_pair=(42, 42))
    signs_2 = rotation.sample_rademacher(shape, tf.int32, seed_pair=(420, 420))
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
    angles_1 = rotation.sample_cis(shape, seed_pair=(42, 42))
    angles_2 = rotation.sample_cis(shape, seed_pair=(4200, 4200))
    angles_1, angles_2 = self.evaluate([angles_1, angles_2])
    self.assertFalse(np.array_equal(angles_1, angles_2))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
