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
import math

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.analytics import differential_privacy


class DifferentialPrivacyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('epsilon_neg', {'epsilon': -1.0}, 'epsilon'),
      ('delta_neg', {'delta': -1.0}, 'delta'),
      ('delta_gt_1', {'delta': 2.0}, 'delta'),
      ('norm_bound_neg', {'norm_bound': -1.0}, 'norm_bound'),
  )
  def test_analytic_gauss_stddev_input_validation(self, args_update, regex):
    args = dict(epsilon=1.0, delta=0.1, norm_bound=1.0, tol=1.0e-12)
    args.update(args_update)
    with self.assertRaisesRegex(ValueError, regex):
      differential_privacy.analytic_gauss_stddev(**args)

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

  def test_get_epsilon_gaussian_inverts_get_noise_gaussian(self):
    for eps in np.logspace(-5, 5, 11, 10):
      for delta in [0] + np.logspace(-10, 0, 11, 10):
        noise = differential_privacy.get_noise_gaussian(eps, delta)
        recovered_eps = differential_privacy.get_epsilon_gaussian(noise, delta)
        np.testing.assert_allclose(recovered_eps, 0 if delta == 1 else eps)

  def test_get_noise_gaussian_inverts_get_epsilon_gaussian(self):
    for noise in np.logspace(-5, 5, 11, 10):
      for delta in [0] + np.logspace(-10, 0, 11, 10):
        eps = differential_privacy.get_epsilon_gaussian(noise, delta)
        recovered_noise = differential_privacy.get_noise_gaussian(eps, delta)
        if eps == 0:
          self.assertEqual(recovered_noise, 0 if delta == 1 else np.inf)
        else:
          np.testing.assert_allclose(recovered_noise, noise)


if __name__ == '__main__':
  absltest.main()
