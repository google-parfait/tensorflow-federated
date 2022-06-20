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
"""Tests for secure_dp."""

import collections
import types

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import concat
from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import discretization
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import rotation
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.aggregators import secure_dp
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert

_test_struct_type = [(tf.float32, (2, 2)), tf.float32]
_test_nested_struct_type = collections.OrderedDict(
    a=[tf.float32, [(tf.float32, (2, 2, 1))]], b=(tf.float32, (3,)))


class AttributeHolder:

  def __init__(self, **kwargs):
    for name, value in kwargs.items():
      setattr(self, name, value)


def _make_test_nested_struct(value):
  return collections.OrderedDict(
      a=[
          tf.constant(value, dtype=tf.float32),
          [tf.constant(value, dtype=tf.float32, shape=[2, 2, 1])]
      ],
      b=tf.constant(value, dtype=tf.float32, shape=(3,)))


def _make_onehot(value, dim=100):
  return value * tf.one_hot(indices=0, depth=dim)


def _run_query(query, records, global_state=None, weights=None):
  """Executes query on the given set of records as a single sample.

  Args:
    query: A PrivateQuery to run.
    records: An iterable containing records to pass to the query.
    global_state: The current global state. If None, an initial global state is
      generated.
    weights: An optional iterable containing the weights of the records.

  Returns:
    A tuple (result, new_global_state) where "result" is the result of the
      query and "new_global_state" is the updated global state.
  """
  if not global_state:
    global_state = query.initial_global_state()
  params = query.derive_sample_params(global_state)
  sample_state = query.initial_sample_state(next(iter(records)))
  if weights is None:
    for record in records:
      sample_state = query.accumulate_record(params, sample_state, record)
  else:
    for weight, record in zip(weights, records):
      sample_state = query.accumulate_record(params, sample_state, record,
                                             weight)
  result, global_state, _ = query.get_noised_result(sample_state, global_state)
  return result, global_state


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


class ParameterRestrictionTests(parameterized.TestCase):

  @parameterized.named_parameters([('left_bound', -1.0), ('right_bound', 5.0)])
  def test_check_in_range_bounds_exclusive(self, attribute_value):
    attribute_name = 'test_attribute'
    test_attribute = AttributeHolder(name=attribute_name)
    range_validator = secure_dp._numeric_range_validator(-1.0, 5.0)

    with self.assertRaisesRegex(
        ValueError,
        f'{attribute_name}.*between.*\(-1.0, 5.0\).*{attribute_value}'):  # pylint: disable=anomalous-backslash-in-string
      range_validator(None, test_attribute, attribute_value)

  def test_check_in_range_bounds_left_inclusive(self):
    attribute_name = 'test_attribute'
    test_attribute = AttributeHolder(name=attribute_name)
    left_value, right_value = -1.0, 5.0
    range_validator = secure_dp._numeric_range_validator(
        left_value, right_value, left_inclusive=True)

    range_validator(None, test_attribute, left_value)  # Should pass

    with self.assertRaisesRegex(
        ValueError, f'{attribute_name}.*between.*\[-1.0, 5.0\).*{right_value}'):  # pylint: disable=anomalous-backslash-in-string
      range_validator(None, test_attribute, right_value)

  def test_check_in_range_bounds_right_inclusive(self):
    attribute_name = 'test_attribute'
    test_attribute = AttributeHolder(name=attribute_name)
    left_value, right_value = -1.0, 5.0
    range_validator = secure_dp._numeric_range_validator(
        left_value, right_value, right_inclusive=True)

    range_validator(None, test_attribute, right_value)  # Should pass

    with self.assertRaisesRegex(
        ValueError, f'{attribute_name}.*between.*\(-1.0, 5.0\].*{left_value}'):  # pylint: disable=anomalous-backslash-in-string
      range_validator(None, test_attribute, left_value)


class DistributedDpComputationTest(tf.test.TestCase, parameterized.TestCase):

  @property
  def default_ddp_kwargs(self):
    # Use a large bit to make quantization errors negligible.
    ddp_kwargs = {'noise_multiplier': 0.5, 'bits': 20}
    return ddp_kwargs.copy()

  @property
  def default_static_clip_kwargs(self):
    static_clip_kwargs = {'l2_clip': 10.0}
    return static_clip_kwargs.copy()

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'value_type_1': tf.int64,
              'value_type_2': tf.float32,
              'value_type_3': _test_struct_type,
              'value_type_4': _test_nested_struct_type
          }, {
              'mechanism_1': secure_dp.DDPMechanism.GAUSSIAN,
              'mechanism_2': secure_dp.DDPMechanism.SKELLAM,
          }))
  def test_type_properties(self, value_type, mechanism):
    ddp_options = self.default_ddp_kwargs
    ddp_options['mechanism'] = mechanism
    ddp_options = secure_dp.DistributedDPArgs(**ddp_options)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)
    ddp_factory = secure_dp.SecureDPFactory(100, ddp_options, clip_options)
    self.assertIsInstance(ddp_factory, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type(value_type)
    process = ddp_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    # The state is a nested object with component factory states. Construct
    # test factories directly and compare the signatures.
    modsum_f = secure.SecureModularSumFactory(2**15, True)

    if mechanism == secure_dp.DDPMechanism.GAUSSIAN:
      dp_query = tfp.DistributedDiscreteGaussianSumQuery(
          l2_norm_bound=10.0, local_stddev=10.0)
    else:
      dp_query = tfp.DistributedSkellamSumQuery(
          l1_norm_bound=10.0, l2_norm_bound=10.0, local_stddev=10.0)

    dp_f = differential_privacy.DifferentiallyPrivateFactory(dp_query, modsum_f)
    discrete_f = discretization.DiscretizationFactory(dp_f)
    l2clip_f = robust.clipping_factory(
        clipping_norm=10.0, inner_agg_factory=discrete_f)
    rot_f = rotation.HadamardTransformFactory(inner_agg_factory=l2clip_f)
    expected_process = concat.concat_factory(rot_f).create(value_type)

    # Check init_fn/state.
    expected_init_type = expected_process.initialize.type_signature
    expected_state_type = expected_init_type.result
    actual_init_type = process.initialize.type_signature
    type_test_utils.assert_types_equivalent(actual_init_type,
                                            expected_init_type)

    # Check next_fn/measurements.
    tensor2type = type_conversions.type_from_tensors
    discrete_state = discrete_f.create(computation_types.to_type(
        tf.float32)).initialize()
    dp_query_state = dp_query.initial_global_state()
    dp_query_metrics_type = tensor2type(dp_query.derive_metrics(dp_query_state))
    expected_measurements_type = collections.OrderedDict(
        l2_clip=robust.NORM_TF_TYPE,
        scale_factor=tensor2type(discrete_state['scale_factor']),
        scaled_inflated_l2=tensor2type(dp_query_state.l2_norm_bound),
        scaled_local_stddev=tensor2type(dp_query_state.local_stddev),
        actual_num_clients=tf.int32,
        padded_dim=tf.int32,
        dp_query_metrics=dp_query_metrics_type)
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=expected_state_type,
            result=computation_types.at_server(value_type),
            measurements=computation_types.at_server(
                expected_measurements_type)))
    actual_next_type = process.next.type_signature
    print('printing here', actual_next_type, expected_next_type)
    type_test_utils.assert_types_equivalent(actual_next_type,
                                            expected_next_type)
    try:
      static_assert.assert_not_contains_unsecure_aggregation(process.next)
    except:  # pylint: disable=bare-except
      self.fail('Factory returned an AggregationProcess containing '
                'non-secure aggregation.')

  def test_expected_num_clients_positive(self):
    ddp_options = secure_dp.DistributedDPArgs(**self.default_ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)

    with self.assertRaisesRegex(
        ValueError,
        '.*must be positive.*Found.*[0-9]*',
        msg='Error not raised when `expected_num_clients` was not positive.'):
      secure_dp.SecureDPFactory(0, ddp_options, clip_options)

  def test_expected_num_clients_integer(self):
    ddp_options = secure_dp.DistributedDPArgs(**self.default_ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)

    with self.assertRaisesRegex(
        TypeError,
        '.*Expected.*int.*found.*',
        msg='Error not raised when `expected_num_clients` was not an integer.'):
      secure_dp.SecureDPFactory(1.0, ddp_options, clip_options)

  def test_expected_num_clients_scalar(self):
    ddp_options = secure_dp.DistributedDPArgs(**self.default_ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)

    with self.assertRaisesRegex(
        TypeError,
        '.*must be a scalar.*Got.*[0-9]*',
        msg='Error not raised when `expected_num_clients` was not a scalar.'):
      secure_dp.SecureDPFactory([1, 1], ddp_options, clip_options)

  @parameterized.named_parameters(
      ('plain_struct', [('a', tf.int32)]),
      ('sequence', computation_types.SequenceType(tf.int32)),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('nested_sequence', [[[computation_types.SequenceType(tf.int32)]]]))
  def test_tff_value_types_raise_on(self, value_type):
    ddp_options = secure_dp.DistributedDPArgs(**self.default_ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)
    ddp_factory = secure_dp.SecureDPFactory(100, ddp_options, clip_options)
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'Expected `value_type` to be'):
      ddp_factory.create(value_type)

  @parameterized.named_parameters(('bool', tf.bool), ('string', tf.string),
                                  ('string_nested', [tf.string, [tf.string]]))
  def test_component_tensor_dtypes_raise_on(self, value_type):
    ddp_options = secure_dp.DistributedDPArgs(**self.default_ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)
    ddp_factory = secure_dp.SecureDPFactory(100, ddp_options, clip_options)
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'must all be integers or floats'):
      ddp_factory.create(value_type)


class DistributedDpExecutionTest(tf.test.TestCase, parameterized.TestCase):
  """Basic integration tests for the nesting of aggregators.

  More precise tests are deferred to the component aggregators (secure, modular
  clipping, DP noising, discretization, norm clipping, rotation, and concat).
  """

  @property
  def default_ddp_kwargs(self):
    # Use a large bit to make quantization errors negligible.
    ddp_kwargs = {'noise_multiplier': 0.5, 'bits': 20}
    return ddp_kwargs.copy()

  @property
  def default_static_clip_kwargs(self):
    static_clip_kwargs = {'l2_clip': 10.0}
    return static_clip_kwargs.copy()

  def test_auto_l2_builds_estimation_process(self):
    ddp_options = secure_dp.DistributedDPArgs(**self.default_ddp_kwargs)
    clip_options = secure_dp.AdaptiveClipArgs()
    ddp_factory = secure_dp.SecureDPFactory(100, ddp_options, clip_options)

    self.assertIsInstance(ddp_factory._l2_clip,
                          estimation_process.EstimationProcess)

  @parameterized.named_parameters(
      ('hd_scalar', tf.float32, [-1., 2., 3.], 4., 'hd'),
      ('dft_scalar', tf.float32, [-1., 2., 3.], 4., 'dft'),
      ('rank_2_hd',
       (tf.float32, [2, 2]), [[[1., 2.], [1., 2.]], [[2., 1.], [2., 1.]]
                             ], [[3., 3.], [3., 3.]], 'hd'),
      ('rank_2_dft',
       (tf.float32, [2, 2]), [[[1., 2.], [1., 2.]], [[2., 1.], [2., 1.]]
                             ], [[3., 3.], [3., 3.]], 'dft'),
      ('struct_hd', _test_nested_struct_type,
       [_make_test_nested_struct(-1),
        _make_test_nested_struct(-2)], _make_test_nested_struct(-3), 'hd'),
      ('struct_dft', _test_nested_struct_type,
       [_make_test_nested_struct(-1),
        _make_test_nested_struct(-2)], _make_test_nested_struct(-3), 'dft'))
  def test_sum_no_noise(self, value_type, client_values, expected_sum,
                        rotation_type):
    ddp_kwargs = self.default_ddp_kwargs
    ddp_kwargs['rotation_type'] = rotation_type
    ddp_kwargs['noise_multiplier'] = 0.0
    ddp_options = secure_dp.DistributedDPArgs(**ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(**self.default_static_clip_kwargs)
    ddp_factory = secure_dp.SecureDPFactory(
        len(client_values), ddp_options, clip_options)
    process = ddp_factory.create(computation_types.to_type(value_type))
    state = process.initialize()
    for _ in range(2):
      output = process.next(state, client_values)
      state = output.state
      # Larger atol to account for side effects (e.g. quantization).
      self.assertAllClose(output.result, expected_sum, atol=1e-3)

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'mechanism_dgauss': 'distributed_dgauss',
              'mechanism_skellam': 'skellam'
          }, {
              'rotation_type_hd': 'hd',
              'rotation_type_dft': 'dft'
          }))
  def test_auto_tuning(self, mechanism, rotation_type):
    noise_mult = 0.0
    initial_clip = 1.0
    ddp_kwargs = self.default_ddp_kwargs
    ddp_kwargs['noise_multiplier'] = noise_mult
    ddp_kwargs['mechanism'] = mechanism
    ddp_kwargs['rotation_type'] = rotation_type
    ddp_options = secure_dp.DistributedDPArgs(**ddp_kwargs)
    clip_options = secure_dp.AdaptiveClipArgs(
        l2_clip=initial_clip,
        target_unclipped_quantile=1.0,
        learning_rate=1.0,
        l2_clip_count_stddev=0.0)

    ddp_factory = secure_dp.SecureDPFactory(3, ddp_options, clip_options)

    dim = 99
    padded_dim = 100.0 if rotation_type == 'dft' else 128.0
    value_type = (tf.float32, _make_onehot(0.0, dim).shape.as_list())
    process = ddp_factory.create(computation_types.to_type(value_type))
    state = process.initialize()
    _, discrete_state, dp_state = ddp_factory._unpack_state(state)
    cur_scale = discrete_state['scale_factor']

    # Two clipped on first round.
    client_data = [_make_onehot(val, dim) for val in [0.5, 1.5, 2.0]]
    expected_sum = _make_onehot(0.5 + 1.0 + 1.0, dim)
    output = process.next(state, client_data)
    state = output.state
    # Slightly larger tolerance to account for side effects.
    self.assertAllClose(expected_sum, output.result, atol=1e-4)

    # Clip is increased by exp(2/3) with geometric update as 2/3 was clipped.
    new_clip = initial_clip * np.exp(2 / 3)
    _, discrete_state, dp_state = ddp_factory._unpack_state(state)
    new_scale = discrete_state['scale_factor']
    actual_new_clip = discrete_state['prior_norm_bound']
    self.assertAllClose(new_clip, actual_new_clip)

    # Check the new scale is updated along with the L2 clip. With only l2_clip
    # changing, the scaling factor should be updated by the same factor.
    self.assertAllClose(new_scale * np.exp(2 / 3), cur_scale, atol=1e-4)

    # Check the norm used for DP operations. The norm inflation from rounding
    # should be small as `bits` is large.
    new_scaled_inflated_l2 = dp_state[0].l2_norm_bound
    self.assertAllClose(new_scaled_inflated_l2 / new_scale, new_clip, atol=1e-4)
    if mechanism == 'distributed_skellam':
      new_scaled_inflated_l1 = dp_state[0].l1_norm_bound
      expected_l1 = tf.math.ceil(
          new_scaled_inflated_l2 *
          tf.minimum(new_scaled_inflated_l2, tf.sqrt(padded_dim)))
      self.assertAllClose(expected_l1, new_scaled_inflated_l1, atol=0)

    # Check sum with new clipping.
    expected_sum = min(new_clip, 0.5) + min(new_clip, 1.5) + min(new_clip, 2.0)
    expected_sum = _make_onehot(expected_sum, dim)
    output = process.next(output.state, client_data)
    self.assertAllClose(expected_sum, output.result, atol=1e-4)

  def test_auto_tuning_fails_on_non_shifted_binary_flags(self):
    """A test that fails when `tfp.QuantileEstimatorQuery` changes value shift.

    Currently, the implementation of the private quantile estimation algorithm
    (via `tfp.QuantileEstimatorQuery`) fixes/hard-codes a value shifting of 0.5
    to reduce sensitivity. Since this value is not exposed, this aggregator
    must also hard-code this info for the SecAgg bounds. This test ensures that
    this aggregator fails if the underlying value shift is changed externally.

    See https://arxiv.org/pdf/1905.03871.pdf and `tfp.QuantileEstimatorQuery`
    for more details on the implementation of the quantile estimation algorithm.
    """
    qe_query_args = dict(
        initial_estimate=1.0,
        target_quantile=1.0,
        learning_rate=1.0,
        below_estimate_stddev=0.0,
        expected_num_records=2,
        geometric_update=True)

    expected_qe_query = tfp.QuantileEstimatorQuery(**qe_query_args)
    mocked_qe_query = tfp.QuantileEstimatorQuery(**qe_query_args)

    def mocked_prep_record(self, params, record):
      del self  # Unused.
      output = expected_qe_query.preprocess_record(params, record)
      # Assertion would fail if the value range shift is changed.
      is_shifted_by_half = tf.logical_or(
          tf.equal(output, 0.5), tf.equal(output, -0.5))
      with tf.control_dependencies(
          [tf.debugging.assert_equal(is_shifted_by_half, True)]):
        return tf.identity(output)

    # Replace the instance method with one that wrapped with assertions.
    mocked_qe_query.preprocess_record = types.MethodType(
        mocked_prep_record, expected_qe_query)

    global_state = mocked_qe_query.initial_global_state()

    # One clipped and one unclipped record to give both binary flags.
    record1 = tf.constant(qe_query_args['initial_estimate'] - 0.1)
    record2 = tf.constant(qe_query_args['initial_estimate'] + 0.1)

    try:
      _run_query(mocked_qe_query, [record1, record2], global_state)
    except tf.errors.InvalidArgumentError:
      self.fail('Output record of `tfp.QuantileEstimatorQuery` is not '
                '+0.5/-0.5.')

  @parameterized.named_parameters(('ddgauss', 'distributed_dgauss'),
                                  ('skellam', 'skellam'))
  def test_noisy_sum(self, mechanism):
    # We only do a rough test to show that noise is indeed added as repeated
    # runs of the nested aggregator is expensive.
    client_values = [1.0, 1.0]
    num_iterations = 12

    noise_mult = 5
    l2_clip = 1.0
    ddp_kwargs = self.default_ddp_kwargs
    ddp_kwargs['noise_multiplier'] = noise_mult
    ddp_kwargs['mechanism'] = mechanism
    ddp_options = secure_dp.DistributedDPArgs(**ddp_kwargs)
    clip_options = secure_dp.StaticClipArgs(l2_clip=l2_clip)

    ddp_factory = secure_dp.SecureDPFactory(
        len(client_values), ddp_options, clip_options)
    process = ddp_factory.create(computation_types.to_type(tf.float32))

    state = process.initialize()
    outputs = []
    for _ in range(num_iterations):
      output = process.next(state, client_values)
      outputs.append(output.result)
      state = output.state

    stddev = np.std(outputs, ddof=1)
    # The standard error of the stddev should be roughly sigma / sqrt(2N - 2),
    # (https://stats.stackexchange.com/questions/156518) so set a rtol to give
    # < 0.01% of failure (within ~4 standard errors).
    rtol = 4 / np.sqrt(2 * num_iterations - 2)
    self.assertAllClose(stddev, noise_mult * l2_clip, rtol=rtol, atol=0)


if __name__ == '__main__':
  execution_contexts.set_test_python_execution_context()
  tf.test.main()
