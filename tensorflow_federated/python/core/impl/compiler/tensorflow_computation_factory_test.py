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
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization


class CreateBroadcastScalarToShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int_empty_shape', tf.int32, tf.TensorShape([]), 10, 10),
      ('int', tf.int32, tf.TensorShape([3]), 10, [10, 10, 10]),
      ('float', tf.float32, tf.TensorShape([3]), 10.0, [10.0, 10.0, 10.0]),
  )
  def test_returns_computation(self, type_signature, shape, value,
                               expected_result):
    proto = tensorflow_computation_factory.create_broadcast_scalar_to_shape(
        type_signature, shape)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    result_type = computation_types.TensorType(type_signature, shape=shape)
    expected_type = computation_types.FunctionType(type_signature, result_type)
    self.assertEqual(actual_type, expected_type)
    actual_result = test_utils.run_tensorflow(proto, value)
    if isinstance(expected_result, list):
      self.assertCountEqual(actual_result, expected_result)
    else:
      self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('int_type', int, tf.TensorShape([])),
      ('list_shape', tf.int32, [1, 1]),
  )
  def test_raises_type_error(self, type_signature, shape):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_broadcast_scalar_to_shape(
          type_signature, shape)

  def test_raises_value_error_with_partially_defined_shape(self):
    type_signature = tf.int32
    shape = tf.TensorShape([None, 1])

    with self.assertRaises(ValueError):
      tensorflow_computation_factory.create_broadcast_scalar_to_shape(
          type_signature, shape)


class CreateConstantTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('int', 10,
       computation_types.TensorType(tf.int32, [3]),
       [10] * 3),
      ('float', 10.0,
       computation_types.TensorType(tf.float32, [3]),
       [10.0] * 3),
      ('unnamed_tuple', 10,
       computation_types.NamedTupleType([tf.int32] * 3),
       anonymous_tuple.AnonymousTuple([(None, 10)] * 3)),
      ('named_tuple', 10,
       computation_types.NamedTupleType(
           [('a', tf.int32), ('b', tf.int32), ('c', tf.int32)]),
       anonymous_tuple.AnonymousTuple([('a', 10), ('b', 10), ('c', 10)])),
      ('nested_tuple', 10,
       computation_types.NamedTupleType([[tf.int32] * 3] * 3),
       anonymous_tuple.AnonymousTuple(
           [(None, anonymous_tuple.AnonymousTuple([(None, 10)] * 3))] * 3)),
  )
  # pyformat: enable
  def test_returns_computation(self, value, type_signature, expected_result):
    proto = tensorflow_computation_factory.create_constant(
        value, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    self.assertEqual(actual_type, expected_type)
    actual_result = test_utils.run_tensorflow(proto)
    if isinstance(expected_result, list):
      self.assertCountEqual(actual_result, expected_result)
    else:
      self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('non_scalar_value', np.zeros([1]), computation_types.TensorType(
          tf.int32)),
      ('none_type', 10, None),
      ('federated_type', 10, type_factory.at_server(tf.int32)),
      ('bad_type', 10.0, computation_types.TensorType(tf.int32)),
  )
  def test_raises_type_error(self, value, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_constant(value, type_signature)


class CreateBinaryOperatorTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('add_int', tf.math.add,
       computation_types.TensorType(tf.int32),
       [1, 2], 3),
      ('add_float', tf.math.add,
       computation_types.TensorType(tf.float32),
       [1.0, 2.25], 3.25),
      ('add_unnamed_tuple', tf.math.add,
       computation_types.NamedTupleType([tf.int32, tf.float32]),
       [[1, 1.0], [2, 2.25]],
       anonymous_tuple.AnonymousTuple([(None, 3), (None, 3.25)])),
      ('add_named_tuple', tf.math.add,
       computation_types.NamedTupleType([('a', tf.int32), ('b', tf.float32)]),
       [[1, 1.0], [2, 2.25]],
       anonymous_tuple.AnonymousTuple([('a', 3), ('b', 3.25)])),
      ('multiply_int', tf.math.multiply,
       computation_types.TensorType(tf.int32),
       [2, 2], 4),
      ('multiply_float', tf.math.multiply,
       computation_types.TensorType(tf.float32),
       [2.0, 2.25], 4.5),
      ('divide_int', tf.math.divide,
       computation_types.TensorType(tf.int32),
       [4, 2], 2.0),
      ('divide_float', tf.math.divide,
       computation_types.TensorType(tf.float32),
       [4.0, 2.0], 2.0),
      ('divide_inf', tf.math.divide,
       computation_types.TensorType(tf.int32),
       [1, 0], np.inf),
  )
  # pyformat: enable
  def test_returns_computation(self, operator, type_signature, operands,
                               expected_result):
    proto = tensorflow_computation_factory.create_binary_operator(
        operator, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    self.assertIsInstance(actual_type, computation_types.FunctionType)
    # Note: It is only useful to test the parameter type; the result type
    # depends on the `operator` used, not the implemenation
    # `create_binary_operator`.
    expected_parameter_type = computation_types.NamedTupleType(
        [type_signature, type_signature])
    self.assertEqual(actual_type.parameter, expected_parameter_type)
    actual_result = test_utils.run_tensorflow(proto, operands)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('non_callable_operator', 1, computation_types.TensorType(tf.int32)),
      ('none_type', tf.math.add, None),
      ('federated_type', tf.math.add, type_factory.at_server(tf.int32)),
      ('sequence_type', tf.math.add, computation_types.SequenceType(tf.int32)),
  )
  def test_raises_type_error(self, operator, type_signature):

    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_binary_operator(
          operator, type_signature)


class CreateEmptyTupleTest(absltest.TestCase):

  def test_returns_coputation(self):
    proto = tensorflow_computation_factory.create_empty_tuple()

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, [])
    self.assertEqual(actual_type, expected_type)
    actual_result = test_utils.run_tensorflow(proto)
    expected_result = anonymous_tuple.AnonymousTuple([])
    self.assertEqual(actual_result, expected_result)


class CreateIdentityTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('int', computation_types.TensorType(tf.int32), 10),
      ('unnamed_tuple',
       computation_types.NamedTupleType([tf.int32, tf.float32]),
       anonymous_tuple.AnonymousTuple([(None, 10), (None, 10.0)])),
      ('named_tuple',
       computation_types.NamedTupleType([('a', tf.int32), ('b', tf.float32)]),
       anonymous_tuple.AnonymousTuple([('a', 10), ('b', 10.0)])),
      ('sequence', computation_types.SequenceType(tf.int32), [10] * 3),
  )
  # pyformat: enable
  def test_returns_computation(self, type_signature, value):
    proto = tensorflow_computation_factory.create_identity(type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = type_factory.unary_op(type_signature)
    self.assertEqual(actual_type, expected_type)
    actual_result = test_utils.run_tensorflow(proto, value)
    self.assertEqual(actual_result, value)

  @parameterized.named_parameters(
      ('none', None),
      ('federated_type', type_factory.at_server(tf.int32)),
  )
  def test_raises_type_error(self, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_identity(type_signature)


class CreateReplicateInputTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int', computation_types.TensorType(tf.int32), 3, 10),
      ('float', computation_types.TensorType(tf.float32), 3, 10.0),
      ('unnamed_tuple', computation_types.NamedTupleType([
          tf.int32, tf.float32
      ]), 3, anonymous_tuple.AnonymousTuple([(None, 10), (None, 10.0)])),
      ('named_tuple',
       computation_types.NamedTupleType([
           ('a', tf.int32), ('b', tf.float32)
       ]), 3, anonymous_tuple.AnonymousTuple([('a', 10), ('b', 10.0)])),
      ('sequence', computation_types.SequenceType(tf.int32), 3, [10] * 3),
  )
  def test_returns_computation(self, type_signature, count, value):
    proto = tensorflow_computation_factory.create_replicate_input(
        type_signature, count)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(type_signature,
                                                   [type_signature] * count)
    self.assertEqual(actual_type, expected_type)
    actual_result = test_utils.run_tensorflow(proto, value)
    expected_result = anonymous_tuple.AnonymousTuple([(None, value)] * count)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('none_type', None, 3),
      ('none_count', computation_types.TensorType(tf.int32), None),
      ('federated_type', type_factory.at_server(tf.int32), 3),
  )
  def test_raises_type_error(self, type_signature, count):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_replicate_input(
          type_signature, count)


class CreateComputationForPyFnTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('const', lambda: 10, None, None, 10),
      ('identity', lambda x: x, computation_types.TensorType(tf.int32), 10, 10),
      ('add_one',
       lambda x: x + 1,
       computation_types.TensorType(tf.int32),
       10,
       11),
  )
  # pyformat: enable
  def test_returns_computation(self, py_fn, type_signature, arg,
                               expected_result):
    proto = tensorflow_computation_factory.create_computation_for_py_fn(
        py_fn, type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_result = test_utils.run_tensorflow(proto, arg)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('none_py_fn', None, computation_types.TensorType(tf.int32)),
      ('none_type', lambda x: x, None),
      ('unnecessary_type', lambda: 10, computation_types.TensorType(tf.int32)),
  )
  def test_raises_type_error_with_none(self, py_fn, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_computation_for_py_fn(
          py_fn, type_signature)


if __name__ == '__main__':
  absltest.main()
