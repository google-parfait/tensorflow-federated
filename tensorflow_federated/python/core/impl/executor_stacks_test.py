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
"""Tests for executor_stacks.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl import executor_stacks
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl import set_default_executor
from tensorflow_federated.python.core.impl import type_constructors


class ExecutorStacksTest(absltest.TestCase):

  def test_with_temperature_sensor_example(self):

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.float32)
    def count_over(ds, t):
      return ds.reduce(
          np.float32(0), lambda n, x: n + tf.cast(tf.greater(x, t), tf.float32))

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def count_total(ds):
      return ds.reduce(np.float32(0.0), lambda n, _: n + 1.0)

    @computations.federated_computation(
        type_constructors.at_clients(
            computation_types.SequenceType(tf.float32)),
        type_constructors.at_server(tf.float32))
    def comp(temperatures, threshold):
      return intrinsics.federated_mean(
          intrinsics.federated_map(
              count_over,
              intrinsics.federated_zip(
                  [temperatures,
                   intrinsics.federated_broadcast(threshold)])),
          intrinsics.federated_map(count_total, temperatures))

    num_clients = 3
    set_default_executor.set_default_executor(
        executor_stacks.create_local_executor(num_clients))
    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float)
    ]
    threshold = 15.0
    result = comp(temperatures, threshold)
    self.assertAlmostEqual(result, 8.333, places=3)
    set_default_executor.set_default_executor()

  def test_with_mnist_training_example(self):
    executor_test_utils.test_mnist_training(
        self, executor_stacks.create_local_executor(1))


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
