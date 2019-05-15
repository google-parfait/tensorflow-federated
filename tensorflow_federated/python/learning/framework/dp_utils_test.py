# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for learning.framework.dp_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_privacy import privacy

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning.framework import dp_utils


def wrap_aggregate_fn(dp_aggregate_fn):
  @tff.federated_computation
  def run_initialize():
    return tff.federated_value(dp_aggregate_fn.initialize(), tff.SERVER)

  @tff.federated_computation(
      run_initialize.type_signature.result,
      tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=False))
  def run_aggregate(global_state, client_values):
    return dp_aggregate_fn(global_state, client_values)

  return run_initialize, run_aggregate


class DpUtilsTest(test.TestCase):

  def test_dp_sum(self):
    query = privacy.GaussianSumQuery(4.0, 0.0)

    dp_aggregate_fn = dp_utils.build_dp_aggregate(query)

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn)
    global_state = initialize()

    global_state, result = aggregate(global_state, [1.0, 3.0, 5.0])

    self.assertEqual(getattr(global_state, 'l2_norm_clip'), 4.0)
    self.assertEqual(getattr(global_state, 'stddev'), 0.0)
    self.assertEqual(result, 8.0)

  def test_dp_stateful_mean(self):
    class ShrinkingSumQuery(privacy.GaussianSumQuery):

      def get_noised_result(self, sample_state, global_state):
        global_state = self._GlobalState(
            tf.maximum(global_state.l2_norm_clip - 1, 0.0),
            global_state.stddev)

        return sample_state, global_state

    query = ShrinkingSumQuery(4.0, 0.0)

    dp_aggregate_fn = dp_utils.build_dp_aggregate(query)

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn)
    global_state = initialize()

    records = [1.0, 3.0, 5.0]

    def run_and_check(global_state, expected_l2_norm_clip, expected_result):
      global_state, result = aggregate(global_state, records)
      self.assertEqual(
          getattr(global_state, 'l2_norm_clip'), expected_l2_norm_clip)
      self.assertEqual(result, expected_result)
      return global_state

    self.assertEqual(getattr(global_state, 'l2_norm_clip'), 4.0)
    global_state = run_and_check(global_state, 3.0, 8.0)
    global_state = run_and_check(global_state, 2.0, 7.0)
    global_state = run_and_check(global_state, 1.0, 5.0)
    global_state = run_and_check(global_state, 0.0, 3.0)
    global_state = run_and_check(global_state, 0.0, 0.0)


if __name__ == '__main__':
  test.main()
