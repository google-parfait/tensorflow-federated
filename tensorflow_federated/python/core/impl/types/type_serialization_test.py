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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_serialization


def _create_scalar_tensor_type(dtype):
  return pb.Type(tensor=pb.TensorType(dtype=dtype.as_datatype_enum))


def _shape_to_dims(shape):
  return [s if s is not None else -1 for s in shape]


class TypeSerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_int', tf.int32, []),
      ('tensor_int', tf.int32, [10, 20]),
      ('tensor_undefined_dim_int', tf.int32, [None, 10, 20]),
      ('scalar_string', tf.string, []),
      ('scalar_boo', tf.bool, []),
  )
  def test_serialize_tensor_type(self, dtype, shape):
    type_signature = computation_types.TensorType(dtype, shape)
    actual_proto = type_serialization.serialize_type(type_signature)
    expected_proto = pb.Type(
        tensor=pb.TensorType(
            dtype=dtype.as_datatype_enum, dims=_shape_to_dims(shape)))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_string_sequence(self):
    actual_proto = type_serialization.serialize_type(
        computation_types.SequenceType(tf.string))
    expected_proto = pb.Type(
        sequence=pb.SequenceType(element=_create_scalar_tensor_type(tf.string)))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_tensor_tuple(self):
    type_signature = computation_types.StructType([
        ('x', tf.int32),
        ('y', tf.string),
        tf.float32,
        ('z', tf.bool),
    ])
    actual_proto = type_serialization.serialize_type(type_signature)
    expected_proto = pb.Type(
        struct=pb.StructType(element=[
            pb.StructType.Element(
                name='x', value=_create_scalar_tensor_type(tf.int32)),
            pb.StructType.Element(
                name='y', value=_create_scalar_tensor_type(tf.string)),
            pb.StructType.Element(value=_create_scalar_tensor_type(tf.float32)),
            pb.StructType.Element(
                name='z', value=_create_scalar_tensor_type(tf.bool)),
        ]))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_nested_tuple(self):
    type_signature = computation_types.StructType([
        ('x', [('y', [('z', tf.bool)])]),
    ])
    actual_proto = type_serialization.serialize_type(type_signature)

    def _tuple_type_proto(elements):
      return pb.Type(struct=pb.StructType(element=elements))

    z_proto = pb.StructType.Element(
        name='z', value=_create_scalar_tensor_type(tf.bool))
    expected_proto = _tuple_type_proto([
        pb.StructType.Element(
            name='x',
            value=_tuple_type_proto([
                pb.StructType.Element(
                    name='y', value=_tuple_type_proto([z_proto]))
            ]))
    ])
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_function(self):
    actual_proto = type_serialization.serialize_type(
        computation_types.FunctionType((tf.int32, tf.int32), tf.bool))
    expected_proto = pb.Type(
        function=pb.FunctionType(
            parameter=pb.Type(
                struct=pb.StructType(element=[
                    pb.StructType.Element(
                        value=_create_scalar_tensor_type(tf.int32)),
                    pb.StructType.Element(
                        value=_create_scalar_tensor_type(tf.int32))
                ])),
            result=_create_scalar_tensor_type(tf.bool)))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_placement(self):
    actual_proto = type_serialization.serialize_type(
        computation_types.PlacementType())
    expected_proto = pb.Type(placement=pb.PlacementType())
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_federated_bool(self):
    federated_type = computation_types.FederatedType(tf.bool,
                                                     placement_literals.CLIENTS,
                                                     True)
    actual_proto = type_serialization.serialize_type(federated_type)
    expected_proto = pb.Type(
        federated=pb.FederatedType(
            placement=pb.PlacementSpec(
                value=pb.Placement(uri=placement_literals.CLIENTS.uri)),
            all_equal=True,
            member=_create_scalar_tensor_type(tf.bool)))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_deserialize_tensor_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.int32, [10]),
        computation_types.TensorType(tf.int32, [None]),
        computation_types.TensorType(tf.int32, tf.TensorShape(None)),
    ])

  def test_serialize_deserialize_sequence_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.SequenceType(tf.int32),
        computation_types.SequenceType(
            computation_types.StructType((tf.int32, tf.bool))),
    ])

  def test_serialize_deserialize_named_tuple_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.StructType([tf.int32, tf.bool]),
        computation_types.StructType([
            tf.int32,
            computation_types.StructType([('x', tf.bool)]),
        ]),
        computation_types.StructType([('x', tf.int32)]),
    ])

  def test_serialize_deserialize_named_tuple_types_py_container(self):
    # The Py container is destroyed during ser/de.
    with_container = computation_types.StructWithPythonType((tf.int32, tf.bool),
                                                            tuple)
    p1 = type_serialization.serialize_type(with_container)
    without_container = type_serialization.deserialize_type(p1)
    self.assertNotEqual(with_container, without_container)  # Not equal.
    self.assertIsInstance(without_container, computation_types.StructType)
    self.assertNotIsInstance(without_container,
                             computation_types.StructWithPythonType)
    with_container.check_equivalent_to(without_container)

  def test_serialize_deserialize_function_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.FunctionType(tf.int32, tf.bool),
        computation_types.FunctionType(None, tf.bool),
    ])

  def test_serialize_deserialize_placement_type(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.PlacementType(),
    ])

  def test_serialize_deserialize_federated_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.FederatedType(
            tf.int32, placement_literals.CLIENTS, all_equal=True),
        computation_types.FederatedType(
            tf.int32, placement_literals.CLIENTS, all_equal=False),
    ])

  def _serialize_deserialize_roundtrip_test(self, type_list):
    """Performs roundtrip serialization/deserialization of computation_types.

    Args:
      type_list: A list of instances of computation_types.Type or things
        convertible to it.
    """
    for t1 in type_list:
      p1 = type_serialization.serialize_type(t1)
      t2 = type_serialization.deserialize_type(p1)
      p2 = type_serialization.serialize_type(t2)
      self.assertEqual(repr(t1), repr(t2))
      self.assertEqual(repr(p1), repr(p2))
      self.assertTrue(t1.is_equivalent_to(t2))


if __name__ == '__main__':
  tf.test.main()
