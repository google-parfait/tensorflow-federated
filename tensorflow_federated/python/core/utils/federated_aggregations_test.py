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
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core.utils import federated_aggregations


class FederatedMinTest(absltest.TestCase):

  def test_federated_min_single_value(self):

    @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
    def call_federated_min(value):
      return federated_aggregations.federated_min(value)

    value = call_federated_min([1.0, 2.0, 5.0])
    self.assertEqual(value, 1.0)

  def test_federated_min_on_nested_scalars(self):
    tuple_type = tff.NamedTupleType([
        ('x', tf.float32),
        ('y', tf.float32),
    ])

    @tff.federated_computation(tff.FederatedType(tuple_type, tff.CLIENTS))
    def call_federated_min(value):
      return federated_aggregations.federated_min(value)

    test_type = collections.namedtuple('NestedScalars', ['x', 'y'])
    value = call_federated_min(
        [test_type(0.0, 1.0),
         test_type(-1.0, 5.0),
         test_type(2.0, -10.0)])
    self.assertDictEqual(value._asdict(), {'x': -1.0, 'y': -10.0})

  def test_federated_min_wrong_type(self):
    with self.assertRaisesRegex(TypeError,
                                r'Type must be int32 or float32. Got: .*'):

      @tff.federated_computation(tff.FederatedType(tf.bool, tff.CLIENTS))
      def call_federated_min(value):
        return federated_aggregations.federated_min(value)

      call_federated_min([False])

  def test_federated_min_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.* argument must be a tff.Value placed at CLIENTS'):

      @tff.federated_computation(tff.FederatedType(tf.int32, tff.SERVER))
      def call_federated_min(value):
        return federated_aggregations.federated_min(value)

      call_federated_min([1, 2, 3])


class FederatedMaxTest(absltest.TestCase):

  def test_federated_max_tensor_value(self):

    @tff.federated_computation(tff.FederatedType((tf.int32, [3]), tff.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    client1 = np.array([1, -2, 3], dtype=np.int32)
    client2 = np.array([0, 7, 1], dtype=np.int32)
    value = call_federated_max([client1, client2])
    self.assertCountEqual(value, [1, 7, 3])

  def test_federated_max_single_value(self):

    @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    value = call_federated_max([6.0, 4.0, 1.0, 7.0])
    self.assertEqual(value, 7.0)

  def test_federated_max_wrong_type(self):
    with self.assertRaisesRegex(TypeError,
                                r'Type must be int32 or float32. Got: .*'):

      @tff.federated_computation(tff.FederatedType(tf.bool, tff.CLIENTS))
      def call_federated_max(value):
        return federated_aggregations.federated_max(value)

      call_federated_max([True, False])

  def test_federated_max_on_nested_scalars(self):
    tuple_type = tff.NamedTupleType([
        ('a', tf.int32),
        ('b', tf.int32),
    ])

    @tff.federated_computation(tff.FederatedType(tuple_type, tff.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    test_type = collections.namedtuple('NestedScalars', ['a', 'b'])
    value = call_federated_max(
        [test_type(1, 5), test_type(2, 3),
         test_type(1, 8)])
    self.assertDictEqual(value._asdict(), {'a': 2, 'b': 8})

  def test_federated_max_nested_tensor_value(self):
    tuple_type = tff.NamedTupleType([
        ('a', (tf.int32, [2])),
        ('b', (tf.int32, [3])),
    ])

    @tff.federated_computation(tff.FederatedType(tuple_type, tff.CLIENTS))
    def call_federated_max(value):
      return federated_aggregations.federated_max(value)

    test_type = collections.namedtuple('NestedScalars', ['a', 'b'])
    client1 = test_type(
        np.array([4, 5], dtype=np.int32), np.array([1, -2, 3], dtype=np.int32))
    client2 = test_type(
        np.array([9, 0], dtype=np.int32), np.array([5, 1, -2], dtype=np.int32))
    value = call_federated_max([client1, client2])
    self.assertCountEqual(value[0], [9, 5])
    self.assertCountEqual(value[1], [5, 1, 3])

  def test_federated_max_wrong_placement(self):
    with self.assertRaisesRegex(
        TypeError, r'.*argument must be a tff.Value placed at CLIENTS.*'):

      @tff.federated_computation(tff.FederatedType(tf.float32, tff.SERVER))
      def call_federated_max(value):
        return federated_aggregations.federated_max(value)

      call_federated_max([1.0, 2.0, 3.0])


if __name__ == '__main__':
  absltest.main()
