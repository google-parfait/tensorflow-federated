# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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

from tensorflow_federated.python.aggregators import modular_clipping
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

_test_struct_type = [(tf.int32, (3,)), tf.int32]

_int_at_server = computation_types.at_server(tf.int32)
_int_at_clients = computation_types.at_clients(tf.int32)


def _make_test_struct_value(x):
  return [tf.constant(x, dtype=tf.int32, shape=(3,)), x]


def _test_factory(clip_lower=-2, clip_upper=2, estimate_stddev=False):
  return modular_clipping.ModularClippingSumFactory(
      clip_lower, clip_upper, sum_factory.SumFactory(), estimate_stddev
  )


def _named_test_cases_product(*args):
  """Utility for creating parameterized named test cases."""
  named_cases = []
  if len(args) == 2:
    dict1, dict2 = args
    for k1, v1 in dict1.items():
      for k2, v2 in dict2.items():
        named_cases.append(('_'.join([k1, k2]), v1, v2))
  return named_cases


class ModularClippingSumFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      _named_test_cases_product(
          {
              'value_type_1': (tf.int32, [10]),
              'value_type_2': _test_struct_type,
              'value_type_3': computation_types.StructType(
                  [('a', tf.int32), ('b', tf.int32)]
              ),
          },
          {'true_stddev': True, 'false_stddev': False},
      )
  )
  def test_type_properties_simple(self, value_type, estimate_stddev):
    factory = _test_factory(estimate_stddev=estimate_stddev)
    process = factory.create(computation_types.to_type(value_type))
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    # Inner SumFactory has no state.
    server_state_type = computation_types.at_server(())

    expected_init_type = computation_types.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(expected_init_type)
    )

    expected_measurements_type = collections.OrderedDict(modclip=())
    if estimate_stddev:
      expected_measurements_type['estimated_stddev'] = tf.float32

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(value_type),
            measurements=computation_types.at_server(
                expected_measurements_type
            ),
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('lower_is_larger', 1, -5),
      ('lower_is_larger_negative', -1, -2),
      ('lower_is_larger_positive', 3, 2),
      ('overflow', -(2**30), 2**30 + 5),
      ('overflow_positive', 0, 2**31),
      ('overflow_negative', -(2**31) - 1, 0),
  )
  def test_raise_on_clip_range(self, lower, upper):
    with self.assertRaises(ValueError):
      _ = _test_factory(lower, upper)

  @parameterized.named_parameters(
      ('string', 'lol'), ('float', 10.0), ('tensor', tf.constant(10))
  )
  def test_raise_on_invalid_clip_type(self, value):
    with self.assertRaises(TypeError):
      _ = _test_factory(clip_lower=value)
    with self.assertRaises(TypeError):
      _ = _test_factory(clip_upper=value)

  @parameterized.named_parameters(
      ('string', 'lol'),
      ('float', 10.0),
      ('int', 10),
      ('tensor', tf.constant(True)),
  )
  def test_raise_on_invalid_estimate_stddev_type(self, value):
    with self.assertRaises(TypeError):
      _ = _test_factory(estimate_stddev=value)

  @parameterized.named_parameters(
      ('scalar', tf.int32), ('rank-2', (tf.int32, [1, 1]))
  )
  def test_raise_on_estimate_stddev_for_single_element(self, value_type):
    factory = _test_factory(estimate_stddev=True)
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(ValueError, 'more than 1 element'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('sequence', computation_types.SequenceType(tf.int32)),
      ('function', computation_types.FunctionType(tf.int32, tf.int32)),
      ('nested_sequence', [[[computation_types.SequenceType(tf.int32)]]]),
  )
  def test_tff_value_types_raise_on(self, value_type):
    factory = _test_factory()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'Expected `value_type` to be'):
      factory.create(value_type)

  @parameterized.named_parameters(
      ('bool', tf.bool),
      ('string', tf.string),
      ('string_nested', [tf.string, [tf.string]]),
  )
  def test_component_tensor_dtypes_raise_on(self, value_type):
    factory = _test_factory()
    value_type = computation_types.to_type(value_type)
    with self.assertRaisesRegex(TypeError, 'must all be integers'):
      factory.create(value_type)


class ModularClippingSumFactoryExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  def _check_result(self, expected, result):
    for exp, res in zip(_make_test_struct_value(expected), result):
      self.assertAllClose(exp, res, atol=0)

  @parameterized.named_parameters([
      ('in_range', -5, 10, [5], [5]),
      ('out_range_left', -5, 10, [-15], [0]),
      ('out_range_right', -5, 10, [20], [5]),
      ('boundary_left', -5, 10, [-5], [-5]),
      ('boundary_right', -5, 10, [10], [-5]),
      ('negative_in_range', -20, -10, [-15], [-15]),
      ('negative_out_range_left', -20, -10, [-25], [-15]),
      ('negative_out_range_right', -20, -10, [-5], [-15]),
      ('positive_in_range', 20, 40, [30], [30]),
      ('positive_out_range_left', 20, 40, [10], [30]),
      ('positive_out_range_right', 20, 40, [50], [30]),
      (
          'large_range_symmetric',
          -(2**30),
          2**30 - 1,
          [2**30 + 5],
          [-(2**30) + 6],
      ),
      ('large_range_left', -(2**31) + 1, 0, [5], [-(2**31) + 6]),
      ('large_range_right', 0, 2**31 - 1, [-5], [2**31 - 6]),
  ])
  def test_clip_individual_values(
      self, clip_range_lower, clip_range_upper, client_data, expected_sum
  ):
    factory = _test_factory(clip_range_lower, clip_range_upper)
    value_type = computation_types.to_type(tf.int32)
    process = factory.create(value_type)
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(output.result, expected_sum)

  @parameterized.named_parameters([
      ('in_range_clip', -3, 3, [1, -2, 1, -2], -2),
      ('boundary_clip', -3, 3, [-3, 3, 3, 3], 0),
      ('out_range_clip', -2, 2, [-3, 3, 5], 1),
      ('mixed_clip', -2, 2, [-4, -2, 1, 2, 7], 0),
  ])
  def test_clip_sum(
      self, clip_range_lower, clip_range_upper, client_data, expected_sum
  ):
    factory = _test_factory(clip_range_lower, clip_range_upper)
    value_type = computation_types.to_type(tf.int32)
    process = factory.create(value_type)
    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(output.result, expected_sum)

  @parameterized.named_parameters([
      ('in_range_clip', -3, 3, [1, -2, 1, -2], -2),
      ('boundary_clip', -3, 3, [-3, 3, 3, 3], 0),
      ('out_range_clip', -2, 2, [-3, 3, 5], 1),
      ('mixed_clip', -2, 2, [-4, -2, 1, 2, 7], 0),
  ])
  def test_clip_sum_struct(
      self, clip_range_lower, clip_range_upper, client_data, expected_sum
  ):
    factory = _test_factory(clip_range_lower, clip_range_upper)
    value_type = computation_types.to_type(_test_struct_type)
    process = factory.create(value_type)
    state = process.initialize()
    client_struct_data = [_make_test_struct_value(v) for v in client_data]
    output = process.next(state, client_struct_data)
    self._check_result(expected_sum, output.result)


class StddevEstimationTest(tf.test.TestCase, parameterized.TestCase):

  def _modclip_by_value(self, x, clip_lo, clip_hi):
    width = clip_hi - clip_lo
    period = np.floor(x / width - clip_lo / width)
    return x - period * width

  def _sample_wrapped_gaussian(self, mean, stddev, size, clip_lo, clip_hi):
    gaussian = np.random.normal(loc=mean, scale=stddev, size=size)
    return self._modclip_by_value(gaussian, clip_lo, clip_hi)

  @parameterized.named_parameters(
      ('range_1', (-15, 15)),
      ('range_2', (-20, 20)),
      ('range_3', (0, 60)),
      ('range_4', (-100, -20)),
  )
  def test_estimation(self, clip_range):
    stddev = 10
    size = 1000
    clip_lo, clip_hi = clip_range
    mean = clip_lo + (clip_hi - clip_lo) / 2
    values = self._sample_wrapped_gaussian(mean, stddev, size, clip_lo, clip_hi)
    values = tf.convert_to_tensor(values)
    est_stddev = modular_clipping.estimate_wrapped_gaussian_stddev(
        values, clip_lo, clip_hi
    )
    est_stddev = self.evaluate(est_stddev)
    # The standard error of the standard deviation should be roughly
    # `sigma / sqrt(2N - 2)` if the data are normally distributed
    # (https://stats.stackexchange.com/questions/156518). While in this case the
    # data are not Gaussian, we use this as a rough heuristic for setting rtol.
    # We expect this to be close as the mod clip range widens (modular wrap-
    # around becomes rare). Usually ~4 standard errors give < 0.01% of test
    # failure, and we use a slightly larger rtol here.
    rtol = 4.5 / np.sqrt(2 * size - 2)
    self.assertAllClose(stddev, est_stddev, rtol=rtol, atol=0)


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
