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
"""Tests for factory compositions."""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import clipping_compositions as compositions
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts


class ClippingCompositionsTest(test_case.TestCase):

  def test_adaptive_zeroing_mean(self):

    factory_ = compositions.adaptive_zeroing_mean(
        initial_quantile_estimate=1.0,
        target_quantile=0.5,
        multiplier=2.0,
        increment=1.0,
        learning_rate=np.log(4.0),
        norm_order=np.inf)
    self.assertIsInstance(factory_, factory.AggregationProcessFactory)

    process = factory_.create(computation_types.to_type(tf.float32))

    state = process.initialize()

    # Quantile estimate is 1.0, zeroing norm is 3.0.
    client_data = [1.5, 3.5]
    client_weight = [1.0, 1.0]

    output = process.next(state, client_data, client_weight)
    self.assertAllClose(1.5 / 2.0, output.result)
    self.assertAllClose(3.0, output.measurements['zeroing_norm'])
    self.assertAllClose(1.0, output.measurements['zeroed_count'])

    # New quantile estimate is 1 * exp(0.5 ln(4)) = 2, zeroing norm is 5.0.
    output = process.next(output.state, client_data, client_weight)
    self.assertAllClose(5.0 / 2.0, output.result)
    self.assertAllClose(5.0, output.measurements['zeroing_norm'])
    self.assertAllClose(0.0, output.measurements['zeroed_count'])

  def test_adaptive_zeroing_clipping_mean(self):

    factory_ = compositions.adaptive_zeroing_clipping_mean(
        initial_zeroing_quantile_estimate=1.0,
        target_zeroing_quantile=0.5,
        zeroing_multiplier=2.0,
        zeroing_increment=2.0,
        zeroing_learning_rate=np.log(4.0),
        zeroing_norm_order=np.inf,
        initial_clipping_quantile_estimate=2.0,
        target_clipping_quantile=0.0,
        clipping_learning_rate=np.log(4.0))
    self.assertIsInstance(factory_, factory.AggregationProcessFactory)

    process = factory_.create(computation_types.to_type(tf.float32))

    state = process.initialize()

    client_data = [3.0, 4.5]
    client_weight = [1.0, 1.0]

    # Zero quantile: 1.0, zero norm: 4.0, clip quantile (norm): 2.0.
    output = process.next(state, client_data, client_weight)
    self.assertAllClose(2.0 / 2.0, output.result)
    self.assertAllClose(4.0, output.measurements['zeroing_norm'])
    self.assertAllClose(1.0, output.measurements['zeroed_count'])
    clip_measurements = output.measurements['agg_process']
    self.assertAllClose(2.0, clip_measurements['clipping_norm'])
    self.assertAllClose(1.0, clip_measurements['clipped_count'])

    # New zero quantile: 1 * exp(0.5 ln(4)) = 2
    # New zero norm is 6.0
    # New clip quantile (norm) is 2 * exp(-0.5 ln(4)) = 1
    output = process.next(output.state, client_data, client_weight)
    self.assertAllClose(2.0 / 2.0, output.result)
    self.assertAllClose(6.0, output.measurements['zeroing_norm'])
    self.assertAllClose(0.0, output.measurements['zeroed_count'])
    clip_measurements = output.measurements['agg_process']
    self.assertAllClose(1.0, clip_measurements['clipping_norm'])
    self.assertAllClose(2.0, clip_measurements['clipped_count'])


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
