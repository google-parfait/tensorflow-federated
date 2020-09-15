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
"""Tests for quantile estimation."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import type_conversions


class PrivateQEComputationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('private', True), ('non_private', False))
  def test_process_type_signature(self, private):
    if private:
      quantile_estimator_query = tfp.QuantileEstimatorQuery(
          initial_estimate=1.0,
          target_quantile=0.5,
          learning_rate=1.0,
          below_estimate_stddev=0.5,
          expected_num_records=100,
          geometric_update=True)
    else:
      quantile_estimator_query = tfp.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=1.0,
          target_quantile=0.5,
          learning_rate=1.0,
          geometric_update=True)

    process = quantile_estimation.PrivateQuantileEstimatorProcess(
        quantile_estimator_query)

    query_state = quantile_estimator_query.initial_global_state()
    sum_process_state = ()

    server_state_type = computation_types.FederatedType(
        type_conversions.type_from_tensors((query_state, sum_process_state)),
        placements.SERVER)

    self.assertEqual(
        computation_types.FunctionType(
            parameter=None, result=server_state_type),
        process.initialize.type_signature)
    self.assertEqual(
        computation_types.FunctionType(
            parameter=server_state_type.member,
            result=computation_types.to_type(tf.float32)),
        process.get_current_estimate.type_signature)

    client_value_type = computation_types.FederatedType(tf.float32,
                                                        placements.CLIENTS)
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(
            computation_types.FunctionType(
                parameter=collections.OrderedDict(
                    state=server_state_type, value=client_value_type),
                result=server_state_type)))

  def test_bad_query(self):
    non_quantile_estimator_query = tfp.GaussianSumQuery(
        l2_norm_clip=1.0, stddev=1.0)

    with self.assertRaises(TypeError):
      quantile_estimation.PrivateQuantileEstimatorProcess(
          non_quantile_estimator_query)

  def test_bad_aggregation_factory(self):
    quantile_estimator_query = tfp.NoPrivacyQuantileEstimatorQuery(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0,
        geometric_update=True)

    with self.assertRaises(TypeError):
      quantile_estimation.PrivateQuantileEstimatorProcess(
          quantile_estimator_query=quantile_estimator_query,
          record_aggregation_factory="I'm not a record_aggregation_factory.")


class PrivateQEExecutionTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('arithmetic', False), ('geometric', True))
  def test_adaptation(self, geometric_update):
    initial_estimate = 3.14159
    target_quantile = 0.61803
    learning_rate = 2.71828

    quantile_estimator_query = tfp.NoPrivacyQuantileEstimatorQuery(
        initial_estimate=initial_estimate,
        target_quantile=target_quantile,
        learning_rate=learning_rate,
        geometric_update=geometric_update)

    process = quantile_estimation.PrivateQuantileEstimatorProcess(
        quantile_estimator_query)

    state = process.initialize()
    self.assertAllClose(process.get_current_estimate(state), initial_estimate)

    # Run on two records greater than estimate.
    state = process.next(state, [initial_estimate + 1, initial_estimate + 2])

    if geometric_update:
      expected_estimate = (
          initial_estimate * np.exp(learning_rate * target_quantile))
    else:
      expected_estimate = initial_estimate + learning_rate * target_quantile

    self.assertAllClose(process.get_current_estimate(state), expected_estimate)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
