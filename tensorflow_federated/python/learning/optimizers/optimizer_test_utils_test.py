# Copyright 2021, The TensorFlow Federated Authors.
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

import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.learning.optimizers import optimizer_test_utils


class OptimizerTestUtilsTest(test_case.TestCase):

  def test_test_problem(self):
    weights, fn, grad_fn = optimizer_test_utils.test_quadratic_problem()

    # Function value in initial point is not negligible and gradient non-zero.
    self.assertGreater(fn(weights), 5.0)
    self.assertGreater(tf.linalg.norm(grad_fn(weights)), 0.01)

    # All-zeros is optimum with function value of 0.0.
    self.assertAllClose(0.0, fn(tf.zeros_like(weights))[0, 0])
    self.assertAllClose(tf.zeros_like(weights), grad_fn(tf.zeros_like(weights)))


if __name__ == '__main__':
  test_case.main()
