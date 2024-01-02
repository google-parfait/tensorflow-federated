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

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import discretization
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
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
          tf.constant(value, dtype=tf.float32),
          [tf.constant(value, dtype=tf.float32, shape=[2, 2, 1])],
      ],
      b=tf.constant(value, dtype=tf.float32, shape=(3, 3)),
  )


def _discretization_sum(
    scale_factor=2, stochastic=False, beta=0, prior_norm_bound=None
):
  return discretization.DiscretizationFactory(
      sum_factory.SumFactory(), scale_factor, stochastic, beta, prior_norm_bound
  )


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


class DiscretizationFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('float', np.float32),
      ('struct_list_float_scalars', [np.float16, np.float32, np.float64]),
      ('struct_list_float_mixed', _test_struct_type_float),
      ('struct_nested', _test_nested_struct_type_float),
  )
  def test_type_properties(self, value_type):
    factory = _discretization_sum()
    value_type = computation_types.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    server_state_type = computation_types.FederatedType(
        collections.OrderedDict(
            scale_factor=np.float32,
            prior_norm_bound=np.float32,
            inner_agg_process=(),
        ),
        placements.SERVER,
    )

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    type_test_utils.assert_types_equivalent(
        process.initialize.type_signature, expected_initialize_type
    )

    expected_measurements_type = computation_types.FederatedType(
        collections.OrderedDict(discretize=()), placements.SERVER
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
    factory = _discretization_sum()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'must all be floats'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('plain_struct', [('a', np.int32)]),
      ('sequence', computation_types.SequenceType(np.int32)),
      ('function', computation_types.FunctionType(np.int32, np.int32)),
      ('nested_sequence', [[[computation_types.SequenceType(np.int32)]]]),
  )
  def test_raises_on_bad_tff_value_types(self, value_type):
    factory = _discretization_sum()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'Expected `value_type` to be'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('negative', -1),
      ('zero', 0),
      ('string', 'lol'),
      ('tensor', tf.constant(3)),
  )
  def test_raises_on_bad_scale_factor(self, scale_factor):
    with self.assertRaisesRegex(ValueError, '`scale_factor` should be a'):
      _discretization_sum(scale_factor=scale_factor)

  @parameterized.named_parameters(
      ('number', 3.14), ('string', 'lol'), ('tensor', tf.constant(True))
  )
  def test_raises_on_bad_stochastic(self, stochastic):
    with self.assertRaisesRegex(ValueError, '`stochastic` should be a'):
      _discretization_sum(stochastic=stochastic)

  @parameterized.named_parameters(
      ('negative', -1),
      ('too_large', 1),
      ('string', 'lol'),
      ('tensor', tf.constant(0.5)),
  )
  def test_raises_on_bad_beta(self, beta):
    with self.assertRaisesRegex(ValueError, '`beta` should be a'):
      _discretization_sum(beta=beta)

  @parameterized.named_parameters(
      ('negative', -0.5),
      ('zero', 0),
      ('string', 'lol'),
      ('tensor', tf.constant(1)),
  )
  def test_raises_on_bad_prior_norm_bound(self, prior_norm_bound):
    with self.assertRaisesRegex(ValueError, '`prior_norm_bound` should be a'):
      _discretization_sum(prior_norm_bound=prior_norm_bound)


class DiscretizationFactoryExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('scalar', np.float32, [1, 2, 3], 6, False),
      (
          'rank_1_tensor',
          (np.float32, [7]),
          [np.arange(7.0), np.arange(7.0) * 2],
          np.arange(7.0) * 3,
          False,
      ),
      (
          'rank_2_tensor',
          (np.float32, [1, 2]),
          [((1, 1),), ((2, 2),)],
          ((3, 3),),
          False,
      ),
      (
          'nested',
          _test_nested_struct_type_float,
          [
              _make_test_nested_struct_value(123),
              _make_test_nested_struct_value(456),
          ],
          _make_test_nested_struct_value(579),
          False,
      ),
      ('stochastic', np.float32, [1, 2, 3], 6, True),
  )
  def test_sum(self, value_type, client_data, expected_sum, stochastic):
    """Integration test with sum."""
    scale_factor = 3
    factory = _discretization_sum(scale_factor, stochastic=stochastic)
    process = factory.create(computation_types.to_type(value_type))
    state = process.initialize()

    for _ in range(3):
      output = process.next(state, client_data)
      self.assertEqual(output.state['scale_factor'], scale_factor)
      self.assertEqual(output.state['prior_norm_bound'], 0)
      self.assertEqual(output.state['inner_agg_process'], ())
      self.assertEqual(
          output.measurements, collections.OrderedDict(discretize=())
      )
      # Use `assertAllClose` to compare structures.
      self.assertAllClose(output.result, expected_sum, atol=0)
      state = output.state

  @parameterized.named_parameters(
      ('int32', tf.int32), ('int64', tf.int64), ('float64', tf.float64)
  )
  def test_output_dtype(self, dtype):
    """Checks the tensor type gets casted during preprocessing."""
    x = tf.range(8, dtype=dtype)
    encoded_x = discretization._discretize_struct(
        x, scale_factor=10, stochastic=False, beta=0, prior_norm_bound=0
    )
    self.assertEqual(encoded_x.dtype, discretization.OUTPUT_TF_TYPE)

  @parameterized.named_parameters(
      ('int32', tf.int32), ('int64', tf.int64), ('float64', tf.float64)
  )
  def test_revert_to_input_dtype(self, dtype):
    """Checks that postprocessing restores the original dtype."""
    x = tf.range(8, dtype=dtype)
    encoded_x = discretization._discretize_struct(
        x, scale_factor=1, stochastic=True, beta=0, prior_norm_bound=0
    )
    decoded_x = discretization._undiscretize_struct(
        encoded_x, scale_factor=1, tf_dtype_struct=dtype
    )
    self.assertEqual(dtype, decoded_x.dtype)


class QuantizationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'scale_factor_1': 0.1,
              'scale_factor_2': 1,
              'scale_factor_3': 314,
              'scale_factor_4': 2**24,
          },
          {'stochastic_true': True, 'stochastic_false': False},
          {'shape_1': (10,), 'shape_2': (10, 10), 'shape_3': (10, 5, 2)},
      )
  )
  def test_error_from_rounding(self, scale_factor, stochastic, shape):
    dtype = tf.float32
    x = tf.random.uniform(shape=shape, minval=-10, maxval=10, dtype=dtype)
    encoded_x = discretization._discretize_struct(
        x, scale_factor, stochastic=stochastic, beta=0, prior_norm_bound=0
    )
    decoded_x = discretization._undiscretize_struct(
        encoded_x, scale_factor, tf_dtype_struct=dtype
    )
    x, decoded_x = self.evaluate([x, decoded_x])

    self.assertAllEqual(x.shape, decoded_x.shape)
    # For stochastic rounding, errors are bounded by the effective bin width;
    # for deterministic rounding, they are bounded by the half of the bin width.
    quantization_atol = (1 if stochastic else 0.5) / scale_factor
    self.assertAllClose(x, decoded_x, rtol=0.0, atol=quantization_atol)


class ScalingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scale_factor_1', 1), ('scale_factor_2', 97), ('scale_factor_3', 10**6)
  )
  def test_scaling(self, scale_factor):
    # Integers to prevent rounding.
    x = tf.random.stateless_uniform([100], (1, 1), -100, 100, dtype=tf.int32)
    discretized_x = discretization._discretize_struct(
        x, scale_factor, stochastic=True, beta=0, prior_norm_bound=0
    )
    reverted_x = discretization._undiscretize_struct(
        discretized_x, scale_factor, tf_dtype_struct=tf.int32
    )
    x, discretized_x, reverted_x = self.evaluate([x, discretized_x, reverted_x])
    self.assertAllEqual(x * scale_factor, discretized_x)  # Scaling up.
    self.assertAllEqual(x, reverted_x)  # Scaling down.


class StochasticRoundingTest(tf.test.TestCase, parameterized.TestCase):

  def test_conditional_rounding_bounds_norm(self):
    """Compare avg rounded norm across different values of beta."""
    num_trials = 500
    x = tf.random.uniform([100], -100, 100, dtype=tf.float32)
    rounded_norms = []
    for beta in [0, 0.9]:
      avg_rounded_norm_beta = tf.reduce_mean([
          tf.norm(discretization._stochastic_rounding(x, beta=beta))
          for _ in range(num_trials)
      ])
      rounded_norms.append(avg_rounded_norm_beta)

    rounded_norms = self.evaluate(rounded_norms)
    # Larger beta should give smaller average norms.
    self.assertAllEqual(rounded_norms, sorted(rounded_norms, reverse=True))

  @parameterized.named_parameters(('beta_1', 0.0), ('beta_2', 0.6))
  def test_noop_on_integers(self, beta):
    x = tf.range(100, dtype=tf.float32)
    rounded_x = discretization._stochastic_rounding(x, beta=beta)
    x, rounded_x = self.evaluate([x, rounded_x])
    self.assertAllEqual(x, rounded_x)
    self.assertEqual(rounded_x.dtype, np.float32)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {'beta_1': 0.0, 'beta_2': 0.6},
          {'value_1': 0.2, 'value_2': 42.6, 'value_3': -3.3},
      )
  )
  def test_biased_inputs(self, beta, value):
    num_trials = 5000
    x = tf.constant(value, shape=[num_trials])
    rounded_x = discretization._stochastic_rounding(x, beta=beta)
    err_x = rounded_x - x
    x, rounded_x, err_x = self.evaluate([x, rounded_x, err_x])
    # Check errors match.
    self.assertAllClose(np.mean(x), np.mean(rounded_x) - np.mean(err_x))
    # Check expected value.
    self.assertTrue(np.floor(value) < np.mean(rounded_x) < np.ceil(value))
    # The rounding events are binomially distributed and we can compute the
    # stddev of the error given the `num_trials` and allow for 4 stddevs as
    # the tolerance to give ~0.006% probability of test failure.
    decimal = np.modf(value)[0]
    stddev = np.sqrt(num_trials * abs(decimal) * (1 - abs(decimal)))
    self.assertAllClose(np.mean(rounded_x), value, atol=4 * stddev / num_trials)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {'float32': tf.float32, 'float64': tf.float64},
          {'beta_1': 0.0, 'beta_2': 0.6},
      )
  )
  def test_output_dtype(self, dtype, beta):
    x = tf.random.uniform([100], minval=-10, maxval=10, dtype=dtype)
    rounded_x = discretization._stochastic_rounding(x, beta=beta)
    self.assertEqual(rounded_x.dtype, dtype)

  def test_fails_with_too_many_tries(self):
    # Seeds chosen to induce greater than two tries.
    x = tf.random.stateless_uniform(
        [100], [0xBAD5EED, 21], -100, 100, dtype=tf.float32
    )
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'Stochastic rounding failed for 2 tries.',
    ):
      discretization._stochastic_rounding(x, beta=0.99, seed=21, max_tries=2)


class InflatedNormTest(tf.test.TestCase, parameterized.TestCase):

  # The expected inflated norm bounds for these test are computed independently
  # in Python using Equations (18) and (19) in Section 4.1 of
  # https://arxiv.org/pdf/2102.06387.pdf.
  @parameterized.named_parameters(
      ('zero_beta_small_gamma', 1.0, 1e-9, 0, 1e9, 1.0000316227766017),
      ('zero_beta_large_gamma', 1.0, 1e-2, 0, 1e9, 317.2277660168379),
      ('small_beta_small_gamma', 1.0, 1e-9, 0.01, 1e9, 1.000000001642451),
      ('small_beta_large_gamma', 1.0, 1e-2, 0.01, 1e9, 158.13231445360802),
      ('large_beta_small_gamma', 1.0, 1e-9, 0.5, 1e9, 1.0000000007137142),
      ('large_beta_large_gamma', 1.0, 1e-2, 0.5, 1e9, 158.12296930808552),
      ('one_beta_small_gamma', 1.0, 1e-9, 1, 1e9, 1.000000000125),
      ('one_beta_large_gamma', 1.0, 1e-2, 1, 1e9, 158.11704525445697),
  )
  def test_inflated_l2_norm_bound(
      self, l2_norm_bound, gamma, beta, dim, expected_inflated_norm_bound
  ):
    inflated_norm_bound = discretization.inflated_l2_norm_bound(
        l2_norm_bound, gamma, beta, dim
    )
    self.assertAllClose(inflated_norm_bound, expected_inflated_norm_bound)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
