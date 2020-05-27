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
"""Tests to guard against serious asymptotic performance regressions."""

import time

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()


class PerfRegressionTest(absltest.TestCase):

  def test_federated_collect_large_numbers_of_parameters(self):
    num_clients = 10
    model_size = 10**6
    client_models = [tf.ones([model_size]) for _ in range(num_clients)]
    client_data_type = tff.FederatedType((tf.float32, [model_size]),
                                         tff.CLIENTS)

    @tff.federated_computation(client_data_type)
    def comp(client_data):
      return tff.federated_collect(client_data)

    start_time_seconds = time.time()
    result = comp(client_models)
    end_time_seconds = time.time()
    runtime = end_time_seconds - start_time_seconds
    if runtime > 10:
      raise RuntimeError('comp should take much less than a second, but took ' +
                         str(runtime))
    del result


if __name__ == '__main__':
  absltest.main()
