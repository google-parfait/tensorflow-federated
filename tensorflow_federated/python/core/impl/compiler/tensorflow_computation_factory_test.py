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

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization


class CreateConstantTest(parameterized.TestCase):

  def test_returns_computation_with_tensor_int(self):
    value = 10
    type_signature = computation_types.TensorType(tf.int32, [3])
    proto = tensorflow_computation_factory.create_constant(
        value, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = [value] * 3
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertCountEqual(actual_value, expected_value)

  def test_returns_computation_with_tensor_float(self):
    value = 10.0
    type_signature = computation_types.TensorType(tf.float32, [3])
    proto = tensorflow_computation_factory.create_constant(
        value, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = [value] * 3
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertCountEqual(actual_value, expected_value)

  def test_returns_computation_with_tuple_unnamed(self):
    value = 10
    type_signature = computation_types.NamedTupleType([tf.int32] * 3)
    proto = tensorflow_computation_factory.create_constant(
        value, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = [value] * 3
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertCountEqual(actual_value, expected_value)

  def test_returns_computation_with_tuple_named(self):
    value = 10
    type_signature = computation_types.NamedTupleType([
        ('a', tf.int32),
        ('b', tf.int32),
        ('c', tf.int32),
    ])

    proto = tensorflow_computation_factory.create_constant(
        value, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = [value] * 3
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertCountEqual(actual_value, expected_value)

  def test_returns_computation_tuple_nested(self):
    value = 10
    type_signature = computation_types.NamedTupleType([[tf.int32] * 3] * 3)

    proto = tensorflow_computation_factory.create_constant(
        value, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = [[value] * 3] * 3
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    for actual_nested, expected_nested in zip(actual_value, expected_value):
      self.assertCountEqual(actual_nested, expected_nested)

  def test_raises_type_error_with_non_scalar_value(self):
    value = np.zeros([1])
    type_signature = computation_types.TensorType(tf.int32)

    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_constant(value, type_signature)

  @parameterized.named_parameters(
      ('none', None),
      ('federated_type', type_factory.at_server(tf.int32)),
  )
  def test_raises_type_error_with_type(self, type_signature):
    value = 0

    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_constant(value, type_signature)

  def test_raises_type_error_with_bad_type(self):
    value = 10.0
    type_signature = tf.int32

    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_constant(value, type_signature)


class CreateEmptyTupleTest(absltest.TestCase):

  def test_returns_coputation(self):
    proto = tensorflow_computation_factory.create_empty_tuple()

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, [])
    self.assertEqual(actual_type, expected_type)
    expected_value = anonymous_tuple.AnonymousTuple([])
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertEqual(actual_value, expected_value)


class CreateIdentityTest(parameterized.TestCase):

  def test_returns_computation_int(self):
    type_signature = computation_types.TensorType(tf.int32)

    proto = tensorflow_computation_factory.create_identity(type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = type_factory.unary_op(type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = 10
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertEqual(actual_value, expected_value)

  def test_returns_computation_tuple_unnamed(self):
    type_signature = computation_types.NamedTupleType([tf.int32, tf.float32])

    proto = tensorflow_computation_factory.create_identity(type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = type_factory.unary_op(type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = anonymous_tuple.AnonymousTuple([(None, 10), (None, 10.0)])
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertEqual(actual_value, expected_value)

  def test_returns_computation_tuple_named(self):
    type_signature = computation_types.NamedTupleType([('a', tf.int32),
                                                       ('b', tf.float32)])

    proto = tensorflow_computation_factory.create_identity(type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = type_factory.unary_op(type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = anonymous_tuple.AnonymousTuple([('a', 10), ('b', 10.0)])
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertEqual(actual_value, expected_value)

  def test_returns_computation_sequence(self):
    type_signature = computation_types.SequenceType(tf.int32)

    proto = tensorflow_computation_factory.create_identity(type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = type_factory.unary_op(type_signature)
    self.assertEqual(actual_type, expected_type)
    expected_value = [10] * 3
    actual_value = test_utils.run_tensorflow(proto, expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      ('none', None),
      ('federated_type', type_factory.at_server(tf.int32)),
  )
  def test_raises_type_error(self, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_identity(type_signature)


if __name__ == '__main__':
  absltest.main()
