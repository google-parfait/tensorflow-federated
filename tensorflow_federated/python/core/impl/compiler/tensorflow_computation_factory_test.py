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
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_test_utils
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization


class CreateConstantTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('scalar_int', 10, computation_types.TensorType(np.int32, [3]), [10] * 3),
      ('scalar_float', 10.0, computation_types.TensorType(np.float32, [3]), [10.0] * 3),
      ('scalar_with_unnamed_struct_type', 10,
       computation_types.StructType([np.int32] * 3),
       structure.Struct([(None, 10)] * 3)),
      ('scalar_with_named_struct_type', 10,
       computation_types.StructType([('a', np.int32), ('b', np.int32), ('c', np.int32)]),
       structure.Struct([('a', 10), ('b', 10), ('c', 10)])),
      ('scalar_with_nested_struct_type', 10,
       computation_types.StructType([[np.int32] * 3] * 3),
       structure.Struct([(None, structure.Struct([(None, 10)] * 3))] * 3)),
      ('tuple_with_struct_type', (10, 11, 12),
       computation_types.StructType([np.int32, np.int32, np.int32]),
       structure.Struct([(None, 10), (None, 11), (None, 12)])),
      ('nested_struct_with_nested_struct_type', (10, (11, 12)),
       computation_types.StructType([np.int32, [np.int32, np.int32]]),
       structure.Struct([
           (None, 10),
           (None, structure.Struct([
               (None, 11), (None, 12)]))
       ])),
      ('nested_named_struct_with_nested_struct_type',
       collections.OrderedDict(a=10, b=collections.OrderedDict(c=11, d=12)),
       computation_types.StructType(
           collections.OrderedDict(a=np.int32,
                                   b=collections.OrderedDict(
                                       c=np.int32, d=np.int32))),
       structure.Struct([
           ('a', 10),
           ('b', structure.Struct([
               ('c', 11), ('d', 12)]))
       ])),
      ('unnamed_value_named_type', (10.0,),
       computation_types.StructType([('a', np.float32)]),
       structure.Struct([('a', 10.0)])),
  )
  # pyformat: enable
  def test_returns_computation(self, value, type_signature, expected_result):
    proto, _ = tensorflow_computation_factory.create_constant(
        value, type_signature
    )

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, type_signature)
    expected_type.check_assignable_from(actual_type)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(proto)
    if isinstance(expected_result, list):
      self.assertCountEqual(actual_result, expected_result)
    else:
      self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'non_scalar_value',
          np.zeros([1]),
          computation_types.TensorType(np.int32),
      ),
      ('none_type', 10, None),
      (
          'federated_type',
          10,
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
      ('bad_type', 10.0, computation_types.TensorType(np.int32)),
      (
          'value_structure_larger_than_type_structure',
          (10.0, 11.0),
          computation_types.StructType([np.float32]),
      ),
      (
          'value_structure_smaller_than_type_structure',
          (10.0,),
          computation_types.StructType(
              [(None, np.float32), (None, np.float32)]
          ),
      ),
      (
          'named_value_unnamed_type',
          collections.OrderedDict(a=10.0),
          computation_types.StructType([(None, np.float32)]),
      ),
  )
  def test_raises_type_error(self, value, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_constant(value, type_signature)


class CreateUnaryOperatorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('abs_int', tf.math.abs, computation_types.TensorType(np.int32), [-1], 1),
      (
          'abs_float',
          tf.math.abs,
          computation_types.TensorType(np.float32),
          [-1.0],
          1.0,
      ),
      (
          'abs_unnamed_tuple',
          lambda x: structure.map_structure(tf.math.abs, x),
          computation_types.StructType([
              computation_types.TensorType(np.int32, [2]),
              computation_types.TensorType(np.float32, [2]),
          ]),
          [[-1, -2], [-3.0, -4.0]],
          structure.Struct([(None, [1, 2]), (None, [3.0, 4.0])]),
      ),
      (
          'abs_named_tuple',
          lambda x: structure.map_structure(tf.math.abs, x),
          computation_types.StructType([
              ('a', computation_types.TensorType(np.int32, [2])),
              ('b', computation_types.TensorType(np.float32, [2])),
          ]),
          [[-1, -2], [-3.0, -4.0]],
          structure.Struct([('a', [1, 2]), ('b', [3.0, 4.0])]),
      ),
      (
          'reduce_sum_int',
          tf.math.reduce_sum,
          computation_types.TensorType(np.int32, [2]),
          [2, 2],
          4,
      ),
      (
          'reduce_sum_float',
          tf.math.reduce_sum,
          computation_types.TensorType(np.float32, [2]),
          [2.0, 2.5],
          4.5,
      ),
      (
          'log_inf',
          tf.math.log,
          computation_types.TensorType(np.float32),
          [0.0],
          -np.inf,
      ),
  )
  # pyformat: enable
  def test_returns_computation(
      self, operator, operand_type, operand, expected_result
  ):
    proto, _ = tensorflow_computation_factory.create_unary_operator(
        operator, operand_type
    )

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    self.assertIsInstance(actual_type, computation_types.FunctionType)
    # Note: It is only useful to test the parameter type; the result type
    # depends on the `operator` used, not the implemenation
    # `create_unary_operator`.
    expected_parameter_type = operand_type
    self.assertEqual(actual_type.parameter, expected_parameter_type)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(
        proto, operand
    )
    self.assertAllEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('non_callable_operator', 1, computation_types.TensorType(np.int32)),
      ('none_type', tf.math.add, None),
      (
          'federated_type',
          tf.math.add,
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
      ('sequence_type', tf.math.add, computation_types.SequenceType(np.int32)),
  )
  def test_raises_type_error(self, operator, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_unary_operator(
          operator, type_signature
      )


class CreateBinaryOperatorTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('add_int', tf.math.add,
       computation_types.TensorType(np.int32), None,
       [1, 2], 3),
      ('add_float', tf.math.add,
       computation_types.TensorType(np.float32), None,
       [1.0, 2.25], 3.25),
      ('add_unnamed_tuple',
       lambda x, y: structure.map_structure(tf.math.add, x, y),
       computation_types.StructType([np.int32, np.float32]), None,
       [[1, 1.0], [2, 2.25]],
       structure.Struct([(None, 3), (None, 3.25)])),
      ('add_named_tuple',
       lambda x, y: structure.map_structure(tf.math.add, x, y),
       computation_types.StructType([('a', np.int32), ('b', np.float32)]), None,
       [[1, 1.0], [2, 2.25]],
       structure.Struct([('a', 3), ('b', 3.25)])),
      ('multiply_int', tf.math.multiply,
       computation_types.TensorType(np.int32), None,
       [2, 2], 4),
      ('multiply_float', tf.math.multiply,
       computation_types.TensorType(np.float32), None,
       [2.0, 2.25], 4.5),
      ('divide_int', tf.math.divide,
       computation_types.TensorType(np.int32), None,
       [4, 2], 2.0),
      ('divide_float', tf.math.divide,
       computation_types.TensorType(np.float32), None,
       [4.0, 2.0], 2.0),
      ('divide_inf', tf.math.divide,
       computation_types.TensorType(np.int32), None,
       [1, 0], np.inf),
      ('different_structure',
       lambda x, y: structure.map_structure(lambda v: tf.math.divide(v, y), x),
       computation_types.StructType([np.float32, np.float32]),
       computation_types.TensorType(np.float32),
       [[1, 2], 2],
       structure.Struct([(None, 0.5), (None, 1.0)])),
  )
  # pyformat: enable
  def test_returns_computation(
      self,
      operator,
      operand_type,
      second_operand_type,
      operands,
      expected_result,
  ):
    proto, _ = tensorflow_computation_factory.create_binary_operator(
        operator, operand_type, second_operand_type
    )

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    self.assertIsInstance(actual_type, computation_types.FunctionType)
    # Note: It is only useful to test the parameter type; the result type
    # depends on the `operator` used, not the implemenation
    # `create_binary_operator`.
    if second_operand_type is None:
      expected_parameter_type = computation_types.StructType(
          [operand_type, operand_type]
      )
    else:
      expected_parameter_type = computation_types.StructType(
          [operand_type, second_operand_type]
      )
    self.assertEqual(actual_type.parameter, expected_parameter_type)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(
        proto, operands
    )
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('non_callable_operator', 1, computation_types.TensorType(np.int32)),
      ('none_type', tf.math.add, None),
      (
          'federated_type',
          tf.math.add,
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
      ('sequence_type', tf.math.add, computation_types.SequenceType(np.int32)),
  )
  def test_raises_type_error(self, operator, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_binary_operator(
          operator, type_signature
      )


class CreateBinaryOperatorWithUpcastTest(parameterized.TestCase):

  # TODO: b/142795960 - arguments in parameterized are called before test main.
  # `tf.constant` will error out on GPU and TPU without proper initialization.
  # A suggested workaround is to use numpy as argument and transform to TF
  # tensor inside the function.
  # pyformat: disable
  @parameterized.named_parameters(
      ('add_int_same_shape', tf.math.add,
       computation_types.StructType([computation_types.TensorType(np.int32), computation_types.TensorType(np.int32)]),
       [1, 2], 3),
      ('add_int_different_shape', tf.math.add,
       computation_types.StructType([computation_types.TensorType(np.int64, [1]), computation_types.TensorType(np.int32)]),
       [np.array([1]), 2], 3),
      ('add_int_different_types', tf.math.add,
       computation_types.StructType([
           computation_types.StructType([
               computation_types.TensorType(np.int64, [1])]),
           computation_types.TensorType(np.int32),
       ]),
       [[np.array([1])], 2],
       structure.Struct([(None, 3)])),
      ('multiply_int_same_shape', tf.math.multiply,
       computation_types.StructType([computation_types.TensorType(np.int32), computation_types.TensorType(np.int32)]),
       [1, 2], 2),
      ('multiply_int_different_shape', tf.math.multiply,
       computation_types.StructType([computation_types.TensorType(np.int64, [1]), computation_types.TensorType(np.int32)]),
       [np.array([1]), 2], 2),
      ('multiply_int_different_types', tf.math.multiply,
       computation_types.StructType([
           computation_types.StructType([
               computation_types.TensorType(np.int64, [1])]),
           computation_types.TensorType(np.int32)
       ]),
       [[np.array([1])], 2],
       structure.Struct([(None, 2)])),
      ('divide_int_same_shape', tf.math.divide,
       computation_types.StructType([computation_types.TensorType(np.int32), computation_types.TensorType(np.int32)]),
       [1, 2], 0.5),
      ('divide_int_different_shape', tf.math.divide,
       computation_types.StructType([computation_types.TensorType(np.int64, [1]), computation_types.TensorType(np.int32)]),
       [np.array([1]), 2], 0.5),
      ('divide_int_different_types', tf.math.divide,
       computation_types.StructType([
           computation_types.StructType([
               computation_types.TensorType(np.int64, [1])]),
           computation_types.TensorType(np.int32),
       ]),
       [[np.array([1])], 2],
       structure.Struct([(None, 0.5)])),
      ('divide_int_same_structure', tf.math.divide,
       computation_types.StructType([
           computation_types.StructType([
               computation_types.TensorType(np.int64, [1]),
               computation_types.TensorType(np.int64, [1]),
           ]),
           computation_types.StructType([
               computation_types.TensorType(np.int64),
               computation_types.TensorType(np.int64),
           ]),
       ]),
       [[np.array([1]), np.array([2])], [2, 8]],
       structure.Struct([(None, 0.5), (None, 0.25)])),
      ('add_float_unknown_shape', tf.math.add,
       computation_types.StructType([
           computation_types.TensorType(np.float64, [None]),
           computation_types.TensorType(np.float64, [1])
       ]),
       [np.array([1.0]), np.array([2.25])],
       np.array([3.25])),
      ('add_float_unknown_rank', tf.math.add,
       computation_types.StructType([
           computation_types.TensorType(np.float64, None),
           computation_types.TensorType(np.float64, [1])
       ]),
       [np.array([1.0]), np.array([2.25])],
       np.array([3.25])),
      ('add_float_unknown_shape_inside_struct', tf.math.add,
       computation_types.StructType([
           computation_types.StructType([
               computation_types.TensorType(np.float64, [None])
           ]),
           computation_types.TensorType(np.float64, [1])
       ]),
       [[np.array([1.0])], np.array([2.25])],
       structure.Struct.unnamed([np.array([3.25])])),
  )
  # pyformat: enable
  def test_returns_computation(
      self, operator, type_signature, operands, expected_result
  ):
    proto, _ = (
        tensorflow_computation_factory.create_binary_operator_with_upcast(
            operator, type_signature
        )
    )

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    self.assertIsInstance(actual_type, computation_types.FunctionType)
    # Note: It is only useful to test the parameter type; the result type
    # depends on the `operator` used, not the implemenation
    # `create_binary_operator_with_upcast`.
    expected_parameter_type = computation_types.StructType(type_signature)
    self.assertEqual(actual_type.parameter, expected_parameter_type)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(
        proto, operands
    )
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'different_structures',
          tf.math.add,
          computation_types.StructType([
              computation_types.StructType([
                  computation_types.TensorType(np.int32),
              ]),
              computation_types.StructType([
                  computation_types.TensorType(np.int32),
                  computation_types.TensorType(np.int32),
              ]),
          ]),
      ),
      (
          'shape_incompatible',
          tf.math.add,
          computation_types.StructType([
              computation_types.TensorType(np.float64, [None]),
              computation_types.TensorType(np.float64, [1, 1]),
          ]),
      ),
  )
  def test_fails(self, operator, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_binary_operator_with_upcast(
          operator, type_signature
      )


class CreateEmptyTupleTest(absltest.TestCase):

  def test_returns_computation(self):
    proto, _ = tensorflow_computation_factory.create_empty_tuple()

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = computation_types.FunctionType(None, [])
    expected_type.check_assignable_from(actual_type)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(proto)
    expected_result = structure.Struct([])
    self.assertEqual(actual_result, expected_result)


class CreateIdentityTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('int', computation_types.TensorType(np.int32), 10),
      ('unnamed_tuple',
       computation_types.StructType([np.int32, np.float32]),
       structure.Struct([(None, 10), (None, 10.0)])),
      ('named_tuple',
       computation_types.StructType([('a', np.int32), ('b', np.float32)]),
       structure.Struct([('a', 10), ('b', 10.0)])),
      ('sequence', computation_types.SequenceType(np.int32), [10] * 3),
  )
  # pyformat: enable
  def test_returns_computation(self, type_signature, value):
    proto, _ = tensorflow_computation_factory.create_identity(type_signature)

    self.assertIsInstance(proto, pb.Computation)
    actual_type = type_serialization.deserialize_type(proto.type)
    expected_type = type_factory.unary_op(type_signature)
    self.assertEqual(actual_type, expected_type)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(
        proto, value
    )
    self.assertEqual(actual_result, value)

  @parameterized.named_parameters(
      ('none', None),
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.SERVER),
      ),
  )
  def test_raises_type_error(self, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_identity(type_signature)

  def test_feeds_and_fetches_different(self):
    proto, _ = tensorflow_computation_factory.create_identity(
        computation_types.TensorType(np.int32)
    )
    self.assertNotEqual(proto.tensorflow.parameter, proto.tensorflow.result)


class CreateComputationForPyFnTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('const', lambda: 10, None, None, 10),
      ('identity', lambda x: x, computation_types.TensorType(np.int32), 10, 10),
      ('add_one',
       lambda x: x + 1,
       computation_types.TensorType(np.int32),
       10,
       11),
      ('dataset_reduce',
       lambda ds: ds.reduce(np.int32(0), lambda x, y: x + y),
       computation_types.SequenceType(np.int32),
       list(range(10)),
       45),
  )
  # pyformat: enable
  def test_returns_computation(
      self, py_fn, type_signature, arg, expected_result
  ):
    proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        py_fn, type_signature
    )

    self.assertIsInstance(proto, pb.Computation)
    actual_result = tensorflow_computation_test_utils.run_tensorflow(proto, arg)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('none_py_fn', None, computation_types.TensorType(np.int32)),
      ('none_type', lambda x: x, None),
      ('unnecessary_type', lambda: 10, computation_types.TensorType(np.int32)),
  )
  def test_raises_type_error_with_none(self, py_fn, type_signature):
    with self.assertRaises(TypeError):
      tensorflow_computation_factory.create_computation_for_py_fn(
          py_fn, type_signature
      )


if __name__ == '__main__':
  absltest.main()
