# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

import unittest

from tensorflow_federated.python.core.api.computations import federated_computation
from tensorflow_federated.python.core.api.computations import tf_computation
from tensorflow_federated.python.core.api.intrinsics import federated_broadcast
from tensorflow_federated.python.core.api.intrinsics import federated_map
from tensorflow_federated.python.core.api.intrinsics import federated_reduce
from tensorflow_federated.python.core.api.intrinsics import federated_sum
from tensorflow_federated.python.core.api.intrinsics import federated_zip
from tensorflow_federated.python.core.api.placements import CLIENTS
from tensorflow_federated.python.core.api.placements import SERVER
from tensorflow_federated.python.core.api.types import FederatedType


class IntrinsicsTest(unittest.TestCase):

  def test_federated_broadcast_with_server_all_equal_int(self):
    @federated_computation(FederatedType(tf.int32, SERVER, True))
    def foo(x):
      return federated_broadcast(x)
    self.assertEqual(str(foo.type_signature), '(int32@SERVER -> int32@CLIENTS)')

  def test_federated_broadcast_with_server_non_all_equal_int(self):
    with self.assertRaises(TypeError):
      @federated_computation(FederatedType(tf.int32, SERVER))
      def _(x):
        return federated_broadcast(x)

  def test_federated_broadcast_with_client_int(self):
    with self.assertRaises(TypeError):
      @federated_computation(FederatedType(tf.int32, CLIENTS, True))
      def _(x):
        return federated_broadcast(x)

  def test_federated_broadcast_with_non_federated_val(self):
    with self.assertRaises(TypeError):
      @federated_computation(tf.int32)
      def _(x):
        return federated_broadcast(x)

  def test_federated_map_with_client_all_equal_int(self):
    @federated_computation(FederatedType(tf.int32, CLIENTS, True))
    def foo(x):
      return federated_map(x, tf_computation(lambda x: x > 10, tf.int32))
    self.assertEqual(
        str(foo.type_signature), '(int32@CLIENTS -> bool@CLIENTS)')

  def test_federated_map_with_client_non_all_equal_int(self):
    @federated_computation(FederatedType(tf.int32, CLIENTS))
    def foo(x):
      return federated_map(x, tf_computation(lambda x: x > 10, tf.int32))
    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {bool}@CLIENTS)')

  def test_federated_map_with_non_federated_val(self):
    with self.assertRaises(TypeError):
      @federated_computation(tf.int32)
      def _(x):
        return federated_map(x, tf_computation(lambda x: x > 10, tf.int32))

  def test_federated_sum_with_client_int(self):
    @federated_computation(FederatedType(tf.int32, CLIENTS))
    def foo(x):
      return federated_sum(x)
    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')

  def test_federated_sum_with_server_int(self):
    with self.assertRaises(TypeError):
      @federated_computation(FederatedType(tf.int32, SERVER))
      def _(x):
        return federated_sum(x)

  def test_federated_zip_with_client_non_all_equal_int_and_bool(self):
    @federated_computation([
        FederatedType(tf.int32, CLIENTS),
        FederatedType(tf.bool, CLIENTS, True)])
    def foo(x, y):
      return federated_zip([x, y])
    self.assertEqual(
        str(foo.type_signature),
        '(<{int32}@CLIENTS,bool@CLIENTS> -> {<int32,bool>}@CLIENTS)')

  def test_federated_zip_with_client_all_equal_int_and_bool(self):
    @federated_computation([
        FederatedType(tf.int32, CLIENTS, True),
        FederatedType(tf.bool, CLIENTS, True)])
    def foo(x, y):
      return federated_zip([x, y])
    self.assertEqual(
        str(foo.type_signature),
        '(<int32@CLIENTS,bool@CLIENTS> -> <int32,bool>@CLIENTS)')

  def test_federated_zip_with_server_int_and_bool(self):
    with self.assertRaises(TypeError):
      @federated_computation([
          FederatedType(tf.int32, SERVER), FederatedType(tf.bool, SERVER)])
      def _(x, y):
        return federated_zip([x, y])

  def test_federated_reduce_with_tf_add_client_int(self):
    @federated_computation(FederatedType(tf.int32, CLIENTS))
    def foo(x):
      # TODO(b/113112108): Possibly add plain constants as a building block.
      zero = tf_computation(lambda: tf.constant(0))()
      plus = tf_computation(tf.add, [tf.int32, tf.int32])
      return federated_reduce(x, zero, plus)
    self.assertEqual(
        str(foo.type_signature),
        '({int32}@CLIENTS -> int32@SERVER)')

  def test_num_over_temperature_threshold_example(self):
    @federated_computation([
        FederatedType(tf.float32, CLIENTS),
        FederatedType(tf.float32, SERVER, True)])
    def foo(temperatures, threshold):
      return federated_sum(federated_map(
          federated_zip([temperatures, federated_broadcast(threshold)]),
          tf_computation(
              lambda x, y: tf.to_int32(tf.greater(x, y)),
              [tf.float32, tf.float32])))
    self.assertEqual(
        str(foo.type_signature),
        '(<{float32}@CLIENTS,float32@SERVER> -> int32@SERVER)')


if __name__ == '__main__':
  unittest.main()
