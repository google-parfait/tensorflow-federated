# Lint as: python3
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

import collections

import tensorflow as tf
import privacy

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core import framework as tff_framework
from tensorflow_federated.python.core.utils import differential_privacy


def wrap_aggregate_fn(dp_aggregate_fn, sample_value):
  tff_types = tff_framework.type_from_tensors(sample_value)

  @tff.federated_computation
  def run_initialize():
    return tff.federated_value(dp_aggregate_fn.initialize(), tff.SERVER)

  @tff.federated_computation(run_initialize.type_signature.result,
                             tff.FederatedType(tff_types, tff.CLIENTS))
  def run_aggregate(global_state, client_values):
    return dp_aggregate_fn(global_state, client_values)

  return run_initialize, run_aggregate


class DpUtilsTest(test.TestCase):

  def test_dp_sum(self):
    query = privacy.GaussianSumQuery(4.0, 0.0)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(query)

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, 0.0)
    global_state = initialize()

    global_state, result = aggregate(global_state, [1.0, 3.0, 5.0])

    self.assertEqual(getattr(global_state, 'l2_norm_clip'), 4.0)
    self.assertEqual(getattr(global_state, 'stddev'), 0.0)
    self.assertEqual(result, 8.0)

  def test_dp_sum_structure_odict(self):
    query = privacy.GaussianSumQuery(5.0, 0.0)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(query)

    def datapoint(a, b):
      return collections.OrderedDict([('a', (a,)), ('b', [b])])

    data = [
        datapoint(1.0, 2.0),
        datapoint(2.0, 3.0),
        datapoint(6.0, 8.0),  # Clipped to 3.0, 4.0
    ]

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, data[0])
    global_state = initialize()

    global_state, result = aggregate(global_state, data)

    self.assertEqual(getattr(global_state, 'l2_norm_clip'), 5.0)
    self.assertEqual(getattr(global_state, 'stddev'), 0.0)

    self.assertEqual(getattr(result, 'a')[0], 6.0)
    self.assertEqual(getattr(result, 'b')[0], 9.0)

  def test_dp_sum_structure_list(self):
    query = privacy.GaussianSumQuery(5.0, 0.0)

    def _value_type_fn(value):
      del value
      return [tff.TensorType(tf.float32), tff.TensorType(tf.float32)]

    def _from_anon_tuple_fn(record):
      return list(record)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(
        query,
        value_type_fn=_value_type_fn,
        from_anon_tuple_fn=_from_anon_tuple_fn)

    def datapoint(a, b):
      return [tf.Variable(a, name='a'), tf.Variable(b, name='b')]

    data = [
        datapoint(1.0, 2.0),
        datapoint(2.0, 3.0),
        datapoint(6.0, 8.0),  # Clipped to 3.0, 4.0
    ]

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, data[0])
    global_state = initialize()

    global_state, result = aggregate(global_state, data)

    self.assertEqual(getattr(global_state, 'l2_norm_clip'), 5.0)
    self.assertEqual(getattr(global_state, 'stddev'), 0.0)

    result = list(result)
    self.assertEqual(result[0], 6.0)
    self.assertEqual(result[1], 9.0)

  def test_dp_stateful_mean(self):

    class ShrinkingSumQuery(privacy.GaussianSumQuery):

      def get_noised_result(self, sample_state, global_state):
        global_state = self._GlobalState(
            tf.maximum(global_state.l2_norm_clip - 1, 0.0), global_state.stddev)

        return sample_state, global_state

    query = ShrinkingSumQuery(4.0, 0.0)

    dp_aggregate_fn, _ = differential_privacy.build_dp_aggregate(query)

    initialize, aggregate = wrap_aggregate_fn(dp_aggregate_fn, 0.0)
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

  def test_dp_global_state_type(self):
    query = privacy.GaussianSumQuery(5.0, 0.0)

    _, dp_global_state_type = differential_privacy.build_dp_aggregate(query)

    self.assertEqual(dp_global_state_type.__class__.__name__,
                     'NamedTupleTypeWithPyContainerType')


if __name__ == '__main__':
  test.main()
