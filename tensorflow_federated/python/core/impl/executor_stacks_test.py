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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl import executor_stacks
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.wrappers import set_default_executor


class ExecutorStacksTest(parameterized.TestCase):

  def test_raises_with_max_fanout_1(self):
    with self.assertRaises(ValueError):
      executor_stacks.local_executor_factory(2, 1)

  @parameterized.named_parameters(
      {
          'testcase_name': 'local_executor',
          'executor_factory_fn': executor_stacks.local_executor_factory
      }, {
          'testcase_name': 'sizing_executor',
          'executor_factory_fn': executor_stacks.sizing_executor_factory
      })
  def test_with_temperature_sensor_example(self, executor_factory_fn):

    @computations.tf_computation(
        computation_types.SequenceType(tf.float32), tf.float32)
    def count_over(ds, t):
      return ds.reduce(
          np.float32(0), lambda n, x: n + tf.cast(tf.greater(x, t), tf.float32))

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def count_total(ds):
      return ds.reduce(np.float32(0.0), lambda n, _: n + 1.0)

    @computations.federated_computation(
        type_factory.at_clients(computation_types.SequenceType(tf.float32)),
        type_factory.at_server(tf.float32))
    def comp(temperatures, threshold):
      return intrinsics.federated_mean(
          intrinsics.federated_map(
              count_over,
              intrinsics.federated_zip(
                  [temperatures,
                   intrinsics.federated_broadcast(threshold)])),
          intrinsics.federated_map(count_total, temperatures))

    set_default_executor.set_default_executor(executor_factory_fn(3))

    to_float = lambda x: tf.cast(x, tf.float32)
    temperatures = [
        tf.data.Dataset.range(10).map(to_float),
        tf.data.Dataset.range(20).map(to_float),
        tf.data.Dataset.range(30).map(to_float)
    ]
    threshold = 15.0
    result = comp(temperatures, threshold)
    self.assertAlmostEqual(result, 8.333, places=3)

    set_default_executor.set_default_executor(executor_factory_fn())
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

  def test_runs_tf(self):
    executor_test_utils.test_runs_tf(
        self,
        executor_stacks.local_executor_factory(1).create_executor({}))

  def test_runs_tf_unspecified_clients(self):
    executor_test_utils.test_runs_tf(
        self,
        executor_stacks.local_executor_factory().create_executor({}))

  def test_sizing_runs_tf(self):
    executor_test_utils.test_runs_tf(
        self,
        executor_stacks.sizing_executor_factory(1).create_executor({}))

  def test_sizing_runs_tf_unspecified_clients(self):
    executor_test_utils.test_runs_tf(
        self,
        executor_stacks.sizing_executor_factory().create_executor({}))

  def test_with_no_args(self):
    set_default_executor.set_default_executor(
        executor_stacks.local_executor_factory())

  def test_executor_factory_raises_with_wrong_cardinalities(self):
    ex_factory = executor_stacks.local_executor_factory(num_clients=5)
    cardinalities = {
        placement_literals.SERVER: 1,
        None: 1,
        placement_literals.CLIENTS: 1
    }
    with self.assertRaisesRegex(ValueError,
                                'construct an executor with 1 clients'):
      ex_factory.create_executor(cardinalities)

    @computations.tf_computation
    def foo():
      return tf.constant(10)

    self.assertEqual(foo(), 10)

    set_default_executor.set_default_executor()

  def test_with_num_clients_larger_than_fanout(self):
    set_default_executor.set_default_executor(
        executor_stacks.local_executor_factory(max_fanout=3))

    @computations.federated_computation(type_factory.at_clients(tf.int32))
    def foo(x):
      return intrinsics.federated_sum(x)

    self.assertEqual(foo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 55)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
