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
"""Tests for executor_service_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import executor_service_utils


class ExecutorServiceUtilsTest(absltest.TestCase):

  def test_serialize_deserialize_tensor_value(self):
    x = tf.constant(10.0).numpy()
    value_proto, value_type = executor_service_utils.serialize_tensor_value(x)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'float32')
    y, type_spec = executor_service_utils.deserialize_tensor_value(value_proto)
    self.assertEqual(str(type_spec), 'float32')
    self.assertTrue(np.array_equal(x, y))

  def test_serialize_deserialize_tensor_value_with_type_spec(self):
    x = tf.constant(10.0).numpy()
    value_proto, value_type = (
        executor_service_utils.serialize_tensor_value(x, tf.float32))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'float32')
    y, type_spec = executor_service_utils.deserialize_tensor_value(value_proto)
    self.assertEqual(str(type_spec), 'float32')
    self.assertTrue(np.array_equal(x, y))

  def test_serialize_deserialize_tensor_value_with_different_dtype(self):
    x = tf.constant(10.0).numpy()
    value_proto, value_type = (
        executor_service_utils.serialize_tensor_value(x, tf.int32))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'int32')
    y, type_spec = executor_service_utils.deserialize_tensor_value(value_proto)
    self.assertEqual(str(type_spec), 'int32')
    self.assertEqual(y, 10)

  def test_serialize_deserialize_tensor_value_with_nontrivial_shape(self):
    x = tf.constant([10, 20, 30]).numpy()
    value_proto, value_type = executor_service_utils.serialize_tensor_value(
        x, (tf.int32, [3]))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'int32[3]')
    y, type_spec = executor_service_utils.deserialize_tensor_value(value_proto)
    self.assertEqual(str(type_spec), 'int32[3]')
    self.assertTrue(np.array_equal(x, y))

  def test_serialize_deserialize_tensor_value_with_bad_shape(self):
    x = tf.constant([10, 20, 30]).numpy()
    with self.assertRaises(TypeError):
      executor_service_utils.serialize_tensor_value(x, tf.int32)

  def test_serialize_deserialize_computation_value(self):

    @computations.tf_computation
    def comp():
      return tf.constant(10)

    value_proto, value_type = executor_service_utils.serialize_value(comp)
    self.assertEqual(value_proto.WhichOneof('value'), 'computation')
    self.assertEqual(str(value_type), '( -> int32)')
    comp, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertIsInstance(comp, computation_pb2.Computation)
    self.assertEqual(str(type_spec), '( -> int32)')

  def test_serialize_deserialize_nested_tuple_value(self):
    x = collections.OrderedDict([('a', 10), ('b', [20, 30]),
                                 ('c', collections.OrderedDict([('d', 40)]))])
    x_type = computation_types.to_type(
        collections.OrderedDict([('a', tf.int32), ('b', [tf.int32, tf.int32]),
                                 ('c', collections.OrderedDict([('d', tf.int32)
                                                               ]))]))
    value_proto, value_type = executor_service_utils.serialize_value(x, x_type)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), '<a=int32,b=<int32,int32>,c=<d=int32>>')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), str(x_type))
    self.assertTrue(str(y), '<a=10,b=<20,30>,c=<d=40>>')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
