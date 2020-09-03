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
"""Tests for adaptive_zeroing."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_privacy

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.utils import adaptive_zeroing

_Odict = collections.OrderedDict


def _make_value(x, shapes):
  return [tf.constant(x, dtype=tf.float32, shape=shape) for shape in shapes]


test_value_types = [('scalar', [()]), ('vector', [(2,)]), ('matrix', [(3, 4)]),
                    ('complex', [(), (2,), (3, 4)])]


class AdaptiveZeroingTest(test.TestCase, parameterized.TestCase):

  def _check_result(self, output, expected, shapes):
    for res, exp in zip(output.result, _make_value(expected, shapes)):
      self.assertAllEqual(res, exp)

  @parameterized.named_parameters(
      ('float', 0.0),
      ('list', [0.0, 0.0]),
      ('odict', _Odict(a=0.0, b=0.0)),
      ('nested', _Odict(a=_Odict(b=[0.0]), c=[0.0, (0.0,)])),
      ('tensors', [0.0, tf.zeros([1]), tf.zeros([2, 2])]),
  )
  def test_process_type_signature(self, value_template):
    value_type = type_conversions.type_from_tensors(value_template)
    mean_process = adaptive_zeroing.build_adaptive_zeroing_mean_process(
        value_type, 100.0, 0.99, 2.0, 1.0, np.inf)

    dummy_quantile_query = tensorflow_privacy.NoPrivacyQuantileEstimatorQuery(
        50.0, 0.99, 1.0, True)
    quantile_query_state = dummy_quantile_query.initial_global_state()
    server_state_type = computation_types.FederatedType(
        type_conversions.type_from_tensors(quantile_query_state),
        placements.SERVER)

    self.assertEqual(
        mean_process.initialize.type_signature,
        computation_types.FunctionType(
            parameter=None, result=server_state_type))

    client_value_type = computation_types.FederatedType(value_type,
                                                        placements.CLIENTS)
    client_value_weight_type = computation_types.FederatedType(
        tf.float32, placements.CLIENTS)
    server_result_type = computation_types.FederatedType(
        value_type, placements.SERVER)
    server_metrics_type = computation_types.FederatedType(
        adaptive_zeroing.AdaptiveZeroingMetrics(
            current_threshold=tf.float32, num_zeroed=tf.int32),
        placements.SERVER)
    self.assertTrue(
        mean_process.next.type_signature.is_equivalent_to(
            computation_types.FunctionType(
                parameter=collections.OrderedDict(
                    global_state=server_state_type,
                    value=client_value_type,
                    weight=client_value_weight_type),
                result=collections.OrderedDict(
                    state=server_state_type,
                    result=server_result_type,
                    measurements=server_metrics_type,
                ))))

  def test_raises_with_empty_value(self):
    value_type = type_conversions.type_from_tensors(())
    with self.assertRaises(ValueError):
      adaptive_zeroing.build_adaptive_zeroing_mean_process(
          value_type, 100.0, 0.99, 2.0, 1.0, np.inf)

  @parameterized.named_parameters(test_value_types)
  def test_simple_average(self, shapes):

    value_type = type_conversions.type_from_tensors(_make_value(0, shapes))
    mean_process = adaptive_zeroing.build_adaptive_zeroing_mean_process(
        value_type, 100.0, 1.0, 1.0, 0.0, np.inf)

    # Basic average.
    global_state = mean_process.initialize()
    values = [_make_value(x, shapes) for x in [0, 1, 2]]
    output = mean_process.next(global_state, values, [1, 1, 1])
    self._check_result(output, 1, shapes)
    metrics = output.measurements
    self.assertEqual(metrics.num_zeroed, 0)

    # Weighted average.
    global_state = output.state
    values = [_make_value(x, shapes) for x in [3, 5, 1, 0]]
    output = mean_process.next(global_state, values, [1, 2, 3, 2])
    self._check_result(output, 2, shapes)
    metrics = output.measurements
    self.assertEqual(metrics.num_zeroed, 0)

    # One value zeroed.
    global_state = output.state
    values = [_make_value(x, shapes) for x in [50, 150]]
    output = mean_process.next(global_state, values, [1, 1])
    self._check_result(output, 50, shapes)
    metrics = output.measurements
    self.assertEqual(metrics.num_zeroed, 1)

    # Both values zeroed.
    global_state = mean_process.initialize()
    values = [_make_value(x, shapes) for x in [250, 150]]
    output = mean_process.next(global_state, values, [1, 1])
    self._check_result(output, 0, shapes)
    metrics = output.measurements
    self.assertEqual(metrics.num_zeroed, 2)

  @parameterized.named_parameters(test_value_types)
  def test_adaptation_down(self, shapes):
    value_type = type_conversions.type_from_tensors(_make_value(0, shapes))
    mean_process = adaptive_zeroing.build_adaptive_zeroing_mean_process(
        value_type, 100.0, 0.0, 1.0, np.log(2.0), np.inf)

    global_state = mean_process.initialize()

    values = [_make_value(x, shapes) for x in [0, 1, 2]]
    output = mean_process.next(global_state, values, [1, 1, 1])
    self._check_result(output, 1, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 50.0)
    self.assertEqual(metrics.num_zeroed, 0)

    output = mean_process.next(global_state, values, [1, 1, 1])
    self._check_result(output, 1, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 25.0)
    self.assertEqual(metrics.num_zeroed, 0)

  @parameterized.named_parameters(test_value_types)
  def test_adaptation_up(self, shapes):
    value_type = type_conversions.type_from_tensors(_make_value(0, shapes))
    mean_process = adaptive_zeroing.build_adaptive_zeroing_mean_process(
        value_type, 1.0, 1.0, 1.0, np.log(2.0), np.inf)

    global_state = mean_process.initialize()

    values = [_make_value(x, shapes) for x in [90, 91, 92]]
    output = mean_process.next(global_state, values, [1, 1, 1])
    self._check_result(output, 0, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 2.0)
    self.assertEqual(metrics.num_zeroed, 3)

    output = mean_process.next(global_state, values, [1, 1, 1])
    self._check_result(output, 0, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 4.0)
    self.assertEqual(metrics.num_zeroed, 3)

  @parameterized.named_parameters(test_value_types)
  def test_adaptation_achieved(self, shapes):
    value_type = type_conversions.type_from_tensors(_make_value(0, shapes))
    mean_process = adaptive_zeroing.build_adaptive_zeroing_mean_process(
        value_type, 100.0, 0.5, 1.0, np.log(4.0), np.inf)

    global_state = mean_process.initialize()

    values = [_make_value(x, shapes) for x in [30, 60]]

    # With target 0.5, learning rate λ=ln(4), estimate should be cut in
    # half in first round: exp(ln(4)(-0.5)) = 1/sqrt(4) = 0.5.
    output = mean_process.next(global_state, values, [1, 1])
    self._check_result(output, 45, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 50.0)
    self.assertEqual(metrics.num_zeroed, 0)

    # In second round, target is achieved, no adaptation occurs, but one update
    # is zeroed.
    output = mean_process.next(global_state, values, [1, 1])
    self._check_result(output, 30, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 50.0)
    self.assertEqual(metrics.num_zeroed, 1)

  @parameterized.named_parameters(test_value_types)
  def test_adaptation_achieved_with_multiplier(self, shapes):
    value_type = type_conversions.type_from_tensors(_make_value(0, shapes))
    mean_process = adaptive_zeroing.build_adaptive_zeroing_mean_process(
        value_type, 200.0, 0.5, 2.0, np.log(4.0), np.inf)

    global_state = mean_process.initialize()

    values = [_make_value(x, shapes) for x in [30, 60]]

    # With target 0.5, learning rate λ=ln(4), estimate should be cut in
    # half in first round: exp(ln(4)(-0.5)) = 1/sqrt(4) = 0.5.
    output = mean_process.next(global_state, values, [1, 1])
    self._check_result(output, 45, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 100.0)
    self.assertEqual(metrics.num_zeroed, 0)

    # In second round, target is achieved, no adaptation occurs, and no updates
    # are zeroed becaues of multiplier.
    output = mean_process.next(global_state, values, [1, 1])
    self._check_result(output, 45, shapes)
    global_state = output.state
    metrics = output.measurements
    self.assertAllClose(metrics.current_threshold, 100.0)
    self.assertEqual(metrics.num_zeroed, 0)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
