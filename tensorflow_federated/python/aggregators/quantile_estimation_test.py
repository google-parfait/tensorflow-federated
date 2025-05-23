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

import collections

from absl.testing import parameterized
import federated_language
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.templates import estimation_process

QEProcess = quantile_estimation.PrivateQuantileEstimationProcess


class PrivateQEComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('private', True), ('non_private', False))
  def test_process_type_signature(self, private):
    if private:
      quantile_estimator_query = tfp.QuantileEstimatorQuery(
          initial_estimate=1.0,
          target_quantile=0.5,
          learning_rate=1.0,
          below_estimate_stddev=0.5,
          expected_num_records=100,
          geometric_update=True,
      )
      below_estimate_state = federated_language.StructType([
          (
              'numerator_state',
              federated_language.StructType(
                  [('l2_norm_clip', np.float32), ('stddev', np.float32)]
              ),
          ),
          ('denominator', np.float32),
      ])
    else:
      quantile_estimator_query = tfp.NoPrivacyQuantileEstimatorQuery(
          initial_estimate=1.0,
          target_quantile=0.5,
          learning_rate=1.0,
          geometric_update=True,
      )
      below_estimate_state = ()
    query_state_type = federated_language.StructType([
        ('current_estimate', np.float32),
        ('target_quantile', np.float32),
        ('learning_rate', np.float32),
        ('below_estimate_state', below_estimate_state),
    ])
    process = QEProcess(quantile_estimator_query)

    sum_process_state_type = ()
    state_type = federated_language.StructType(
        [query_state_type, sum_process_state_type]
    )
    server_state_type = federated_language.FederatedType(
        state_type,
        federated_language.SERVER,
    )
    expected_initialize_type_signature = federated_language.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        expected_initialize_type_signature.is_equivalent_to(
            process.initialize.type_signature
        )
    )

    estimate_type = federated_language.FederatedType(
        np.float32, federated_language.SERVER
    )
    expected_report_type_signature = federated_language.FunctionType(
        parameter=server_state_type, result=estimate_type
    )
    self.assertTrue(
        expected_report_type_signature.is_equivalent_to(
            process.report.type_signature
        )
    )

    client_value_type = federated_language.FederatedType(
        np.float32, federated_language.CLIENTS
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(
            federated_language.FunctionType(
                parameter=collections.OrderedDict(
                    state=server_state_type, value=client_value_type
                ),
                result=server_state_type,
            )
        )
    )

  def test_bad_query(self):
    non_quantile_estimator_query = tfp.GaussianSumQuery(
        l2_norm_clip=1.0, stddev=1.0
    )

    with self.assertRaises(TypeError):
      QEProcess(non_quantile_estimator_query)

  def test_bad_aggregation_factory(self):
    quantile_estimator_query = tfp.NoPrivacyQuantileEstimatorQuery(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0,
        geometric_update=True,
    )

    with self.assertRaises(TypeError):
      QEProcess(
          quantile_estimator_query=quantile_estimator_query,
          record_aggregation_factory="I'm not a record_aggregation_factory.",
      )


class PrivateQEExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('arithmetic', False), ('geometric', True))
  def test_adaptation(self, geometric_update):
    initial_estimate = 3.14159
    target_quantile = 0.61803
    learning_rate = 2.71828

    quantile_estimator_query = tfp.NoPrivacyQuantileEstimatorQuery(
        initial_estimate=initial_estimate,
        target_quantile=target_quantile,
        learning_rate=learning_rate,
        geometric_update=geometric_update,
    )

    process = QEProcess(quantile_estimator_query)

    state = process.initialize()
    self.assertAllClose(process.report(state), initial_estimate)

    # Run on two records greater than estimate.
    state = process.next(state, [initial_estimate + 1, initial_estimate + 2])

    if geometric_update:
      expected_estimate = initial_estimate * np.exp(
          learning_rate * target_quantile
      )
    else:
      expected_estimate = initial_estimate + learning_rate * target_quantile

    self.assertAllClose(process.report(state), expected_estimate)

  def test_no_noise_cls(self):
    process = QEProcess.no_noise(
        initial_estimate=1.0, target_quantile=0.5, learning_rate=1.0
    )
    self.assertIsInstance(process, QEProcess)
    state = process.initialize()
    self.assertEqual(process.report(state), 1.0)

  def test_no_noise_affine_cls(self):
    process = QEProcess.no_noise(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0,
        multiplier=2.0,
        increment=1.0,
    )
    self.assertIsInstance(process, estimation_process.EstimationProcess)
    state = process.initialize()
    self.assertEqual(process.report(state), 3.0)

  def test_no_noise_secure_true_false_equal_results(self):
    simple_process = QEProcess.no_noise(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0,
        secure_estimation=False,
    )
    secure_process = QEProcess.no_noise(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0,
        secure_estimation=True,
    )

    data = [0.5, 1.5, 2.5]  # 2 bigger than the initial estimate 1.0, 1 smaller.

    simple_state = simple_process.initialize()
    secure_state = secure_process.initialize()
    for _ in range(3):
      simple_state = simple_process.next(simple_state, data)
      secure_state = secure_process.next(secure_state, data)
      self.assertAllClose(
          simple_process.report(simple_state),
          secure_process.report(secure_state),
      )

  def test_secure_estimation_true_only_contains_secure_aggregation(self):
    secure_process = QEProcess.no_noise(
        initial_estimate=1.0,
        target_quantile=0.5,
        learning_rate=1.0,
        secure_estimation=True,
    )
    self.assertFalse(
        federated_language.framework.computation_contains_unsecure_aggregation(
            secure_process.next
        )
    )


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
