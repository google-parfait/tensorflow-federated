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

import collections

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import caching_executor
from tensorflow_federated.python.core.impl import composite_executor
from tensorflow_federated.python.core.impl import concurrent_executor
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import federated_executor
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl import set_default_executor
from tensorflow_federated.python.core.impl.compiler import type_factory


def _create_bottom_stack():
  return lambda_executor.LambdaExecutor(
      caching_executor.CachingExecutor(
          concurrent_executor.ConcurrentExecutor(
              eager_executor.EagerExecutor())))


def _create_worker_stack():
  return federated_executor.FederatedExecutor({
      placements.SERVER: _create_bottom_stack(),
      placements.CLIENTS: [_create_bottom_stack() for _ in range(2)],
      None: _create_bottom_stack()
  })


def _create_middle_stack(children):
  return lambda_executor.LambdaExecutor(
      caching_executor.CachingExecutor(
          composite_executor.CompositeExecutor(_create_bottom_stack(),
                                               children)))


class CompositeExecutorTest(absltest.TestCase):

  def setUp(self):
    super(CompositeExecutorTest, self).setUp()
    set_default_executor.set_default_executor(
        _create_middle_stack([
            _create_middle_stack([_create_worker_stack() for _ in range(3)]),
            _create_middle_stack([_create_worker_stack() for _ in range(3)])
        ]))

  def tearDown(self):
    set_default_executor.set_default_executor(None)
    super(CompositeExecutorTest, self).tearDown()

  def test_federated_value_at_server(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.SERVER)

    self.assertEqual(comp(), 10)

  def test_federated_value_at_clients(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_value(10, placements.CLIENTS)

    self.assertEqual(comp(), 10)

  def test_federated_map(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_map(
          computations.tf_computation(lambda x: x + 1, tf.int32),
          intrinsics.federated_value(10, placements.CLIENTS))

    self.assertEqual(comp(), [11] * 12)

  def test_federated_aggregate(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def add_int(x, y):
      return x + y

    @computations.tf_computation(tf.int32)
    def add_five(x):
      return x + 5

    @computations.federated_computation
    def comp():
      tens = intrinsics.federated_value(10, placements.CLIENTS)
      return intrinsics.federated_aggregate(tens, 0, add_int, add_int, add_five)

    self.assertEqual(comp(), 125)

  def test_federated_broadcast(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_map(
          computations.tf_computation(lambda x: x + 1, tf.int32),
          intrinsics.federated_broadcast(
              intrinsics.federated_value(10, placements.SERVER)))

    self.assertEqual(comp(), [11] * 12)

  def test_federated_apply(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_apply(
          computations.tf_computation(lambda x: x + 1, tf.int32),
          intrinsics.federated_value(10, placements.SERVER))

    self.assertEqual(comp(), 11)

  def test_federated_zip_at_server_unnamed(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip([
          intrinsics.federated_value(10, placements.SERVER),
          intrinsics.federated_value(20, placements.SERVER)
      ])

    self.assertEqual(str(comp.type_signature), '( -> <int32,int32>@SERVER)')
    self.assertEqual(str(comp()), '<10,20>')

  def test_federated_zip_at_server_named(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip(
          collections.OrderedDict([
              ('A', intrinsics.federated_value(10, placements.SERVER)),
              ('B', intrinsics.federated_value(20, placements.SERVER))
          ]))

    self.assertEqual(str(comp.type_signature), '( -> <A=int32,B=int32>@SERVER)')
    self.assertEqual(str(comp()), '<A=10,B=20>')

  def test_federated_zip_at_clients_named(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip(
          collections.OrderedDict([
              ('A', intrinsics.federated_value(10, placements.CLIENTS)),
              ('B', intrinsics.federated_value(20, placements.CLIENTS))
          ]))

    self.assertEqual(
        str(comp.type_signature), '( -> {<A=int32,B=int32>}@CLIENTS)')
    result = comp()
    self.assertIsInstance(result, list)
    self.assertLen(result, 12)
    for v in result:
      self.assertEqual(
          str(anonymous_tuple.map_structure(lambda x: x.numpy(), v)),
          '<A=10,B=20>')

  def test_federated_zip_at_clients_unnamed(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_zip([
          intrinsics.federated_value(10, placements.CLIENTS),
          intrinsics.federated_value(20, placements.CLIENTS)
      ])

    self.assertEqual(str(comp.type_signature), '( -> {<int32,int32>}@CLIENTS)')
    result = comp()
    self.assertIsInstance(result, list)
    self.assertLen(result, 12)
    for v in result:
      self.assertEqual(
          str(anonymous_tuple.map_structure(lambda x: x.numpy(), v)), '<10,20>')

  def test_federated_sum(self):

    @computations.federated_computation
    def comp():
      return intrinsics.federated_sum(
          intrinsics.federated_value(10, placements.CLIENTS))

    self.assertEqual(comp(), 120)

  def test_federated_mean(self):

    @computations.federated_computation(type_factory.at_clients(tf.float32))
    def comp(x):
      return intrinsics.federated_mean(x)

    self.assertEqual(comp([float(x + 1) for x in range(12)]), 6.5)

  def test_federated_weighted_mean(self):

    @computations.federated_computation(
        type_factory.at_clients(tf.float32),
        type_factory.at_clients(tf.float32))
    def comp(x, y):
      return intrinsics.federated_mean(x, y)

    result = comp([float(x + 1) for x in range(12)], [1.0, 2.0, 3.0] * 4)
    self.assertAlmostEqual(result, 6.83333333333, places=3)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
