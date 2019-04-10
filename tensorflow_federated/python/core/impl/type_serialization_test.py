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
"""Tests for type_serialization.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import type_serialization
from tensorflow_federated.python.core.impl import type_utils


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
    actual_proto = type_serialization.serialize_type((dtype, shape))
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
    actual_proto = type_serialization.serialize_type([
        ('x', tf.int32),
        ('y', tf.string),
        tf.float32,
        ('z', tf.bool),
    ])
    expected_proto = pb.Type(
        tuple=pb.NamedTupleType(element=[
            pb.NamedTupleType.Element(
                name='x', value=_create_scalar_tensor_type(tf.int32)),
            pb.NamedTupleType.Element(
                name='y', value=_create_scalar_tensor_type(tf.string)),
            pb.NamedTupleType.Element(
                value=_create_scalar_tensor_type(tf.float32)),
            pb.NamedTupleType.Element(
                name='z', value=_create_scalar_tensor_type(tf.bool)),
        ]))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_nested_tuple(self):
    actual_proto = type_serialization.serialize_type([
        ('x', [('y', [('z', tf.bool)])]),
    ])

    def _tuple_type_proto(elements):
      return pb.Type(tuple=pb.NamedTupleType(element=elements))

    z_proto = pb.NamedTupleType.Element(
        name='z', value=_create_scalar_tensor_type(tf.bool))
    expected_proto = _tuple_type_proto([
        pb.NamedTupleType.Element(
            name='x',
            value=_tuple_type_proto([
                pb.NamedTupleType.Element(
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
                tuple=pb.NamedTupleType(element=[
                    pb.NamedTupleType.Element(
                        value=_create_scalar_tensor_type(tf.int32)),
                    pb.NamedTupleType.Element(
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
    actual_proto = type_serialization.serialize_type(
        computation_types.FederatedType(tf.bool, placements.CLIENTS, True))
    expected_proto = pb.Type(
        federated=pb.FederatedType(
            placement=pb.PlacementSpec(
                value=pb.Placement(uri=placements.CLIENTS.uri)),
            all_equal=True,
            member=_create_scalar_tensor_type(tf.bool)))
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_deserialize_tensor_types(self):
    self._serialize_deserialize_roundtrip_test(
        [tf.int32, (tf.int32, [10]), (tf.int32, [None])])

  def test_serialize_deserialize_sequence_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.SequenceType(tf.int32),
        computation_types.SequenceType([tf.int32, tf.bool]),
        computation_types.SequenceType(
            [tf.int32, computation_types.SequenceType(tf.bool)])
    ])

  def test_serialize_deserialize_named_tuple_types(self):
    self._serialize_deserialize_roundtrip_test([(tf.int32, tf.bool),
                                                (tf.int32, ('x', tf.bool)),
                                                ('x', tf.int32)])

  def test_serialize_deserialize_function_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.FunctionType(tf.int32, tf.bool),
        computation_types.FunctionType(None, tf.bool)
    ])

  def test_serialize_deserialize_placement_type(self):
    self._serialize_deserialize_roundtrip_test(
        [computation_types.PlacementType()])

  def test_serialize_deserialize_federated_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
        computation_types.FederatedType(tf.int32, placements.CLIENTS, False)
    ])

  def _serialize_deserialize_roundtrip_test(self, type_list):
    """Performs roundtrip serialization/deserialization of computation_types.

    Args:
      type_list: A list of instances of computation_types.Type or things
        convertible to it.
    """
    for t in type_list:
      t1 = computation_types.to_type(t)
      p1 = type_serialization.serialize_type(t1)
      t2 = type_serialization.deserialize_type(p1)
      p2 = type_serialization.serialize_type(t2)
      self.assertEqual(repr(t1), repr(t2))
      self.assertEqual(repr(p1), repr(p2))
      self.assertTrue(type_utils.are_equivalent_types(t1, t2))


if __name__ == '__main__':
  tf.test.main()
