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

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl import executor_test_utils
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_stacks

tf.compat.v1.enable_v2_behavior()


def _temperature_sensor_example_next_fn():

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

  return comp


@contextlib.contextmanager
def _execution_context(executor):
  yield execution_context.ExecutionContext(executor)


class ExecutorStacksTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_construction_with_no_args(self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn()
    self.assertIsInstance(executor_factory_impl,
                          executor_factory.ExecutorFactoryImpl)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_construction_raises_with_max_fanout_one(self, executor_factory_fn):
    with self.assertRaises(ValueError):
      executor_factory_fn(max_fanout=1)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_execution_with_none_clients_for_temperature_sensor_example(
      self, executor_factory_fn):
    comp = _temperature_sensor_example_next_fn()

    with _execution_context(executor_factory_fn()):
      to_float = lambda x: tf.cast(x, tf.float32)
      temperatures = [
          tf.data.Dataset.range(10).map(to_float),
          tf.data.Dataset.range(20).map(to_float),
          tf.data.Dataset.range(30).map(to_float)
      ]
      threshold = 15.0
      result = comp(temperatures, threshold)
      self.assertAlmostEqual(result, 8.333, places=3)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_execution_with_three_clients_for_temperature_sensor_example(
      self, executor_factory_fn):
    comp = _temperature_sensor_example_next_fn()

    with _execution_context(executor_factory_fn(3)):
      to_float = lambda x: tf.cast(x, tf.float32)
      temperatures = [
          tf.data.Dataset.range(10).map(to_float),
          tf.data.Dataset.range(20).map(to_float),
          tf.data.Dataset.range(30).map(to_float)
      ]
      threshold = 15.0
      result = comp(temperatures, threshold)
      self.assertAlmostEqual(result, 8.333, places=3)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_execution_with_inferred_clients_larger_than_fanout(
      self, executor_factory_fn):

    @computations.federated_computation(type_factory.at_clients(tf.int32))
    def foo(x):
      return intrinsics.federated_sum(x)

    with _execution_context(executor_factory_fn(max_fanout=3)):
      self.assertEqual(foo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 55)

  @parameterized.named_parameters(
      ('local_executor_with_one_client',
       executor_stacks.local_executor_factory(1)),
      ('local_executor_with_none_clients',
       executor_stacks.local_executor_factory()),
      ('sizing_executor_with_one_client',
       executor_stacks.sizing_executor_factory(1)),
      ('sizing_executor_with_none_clients',
       executor_stacks.sizing_executor_factory()),
  )
  def test_runs_tf(self, executor_factory_impl):
    executor = executor_factory_impl.create_executor({})
    executor_test_utils.test_runs_tf(self, executor)

  @parameterized.named_parameters(
      ('local_executor', executor_stacks.local_executor_factory),
      ('sizing_executor', executor_stacks.sizing_executor_factory),
  )
  def test_create_executor_raises_with_wrong_cardinalities(
      self, executor_factory_fn):
    executor_factory_impl = executor_factory_fn(num_clients=5)
    cardinalities = {
        placement_literals.SERVER: 1,
        None: 1,
        placement_literals.CLIENTS: 1,
    }
    with self.assertRaises(ValueError,):
      executor_factory_impl.create_executor(cardinalities)


if __name__ == '__main__':
  absltest.main()
