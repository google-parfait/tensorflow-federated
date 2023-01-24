# Copyright 2022, The TensorFlow Federated Authors.
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
"""Tests for differential_privacy."""
import math
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_federated.python.analytics import differential_privacy


class DifferentialPrivacyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'epsilon_0', 'epsilon': 0.0, 'regex': 'epsilon'},
      {'testcase_name': 'epsilon_neg', 'epsilon': -1.0, 'regex': 'epsilon'},
      {'testcase_name': 'epsilon_large', 'epsilon': 600, 'regex': 'epsilon'},
      {'testcase_name': 'delta_0', 'delta': 0.0, 'regex': 'delta'},
      {'testcase_name': 'delta_1', 'delta': 1.0, 'regex': 'delta'},
      {'testcase_name': 'delta_neg', 'delta': -1.0, 'regex': 'delta'},
      {'testcase_name': 'delta_large', 'delta': 10.0, 'regex': 'delta'},
      {
          'testcase_name': 'norm_bound_neg',
          'norm_bound': -1.0,
          'regex': 'norm_bound',
      },
      {'testcase_name': 'tol_0', 'tol': 0.0, 'regex': 'tol'},
      {'testcase_name': 'tol_neg', 'tol': -1.0, 'regex': 'tol'},
  )
  def test_analytic_gauss_stddev_input_validation(
      self, epsilon=1.0, delta=0.1, norm_bound=1.0, tol=1.0e-12, regex=''
  ):
    with self.assertRaisesRegex(ValueError, regex):
      differential_privacy.analytic_gauss_stddev(
          epsilon=epsilon, delta=delta, norm_bound=norm_bound, tol=tol
      )

  @parameterized.named_parameters(
      ('eps_1_delta_0.5', 1.0, 0.5, 1.0, 0.50706503),
      ('eps_log3_delta_1', math.log(3), 0.00001, 1.0, 3.42466239),
  )
  def test_analytic_gauss_stddev_as_expected(
      self, epsilon, delta, norm_bound, expected_stddev
  ):
    sigma = differential_privacy.analytic_gauss_stddev(
        epsilon=epsilon, delta=delta, norm_bound=norm_bound
    )
    self.assertAlmostEqual(sigma, expected_stddev, places=7)


if __name__ == '__main__':
  absltest.main()
