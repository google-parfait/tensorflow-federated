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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.proto.v0 import data_type_pb2
from tensorflow_federated.python.core.impl.types import array_shape
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import dtype_utils
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization


class TypeSerializationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_int', np.int32, []),
      ('tensor_int', np.int32, [10, 20]),
      ('tensor_undefined_dim_int', np.int32, [None, 10, 20]),
      ('scalar_string', np.str_, []),
      ('scalar_boo', np.bool_, []),
  )
  def test_serialize_tensor_type(self, dtype, shape):
    type_signature = computation_types.TensorType(dtype, shape)
    actual_proto = type_serialization.serialize_type(type_signature)
    dtype = dtype_utils.to_proto(dtype)
    shape_pb = array_shape.to_proto(shape)
    expected_proto = pb.Type(
        tensor=pb.TensorType(
            dtype=dtype,
            dims=shape_pb.dim,
            unknown_rank=shape_pb.unknown_rank,
        )
    )
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_string_sequence(self):
    actual_proto = type_serialization.serialize_type(
        computation_types.SequenceType(np.str_)
    )
    expected_proto = pb.Type(
        sequence=pb.SequenceType(
            element=pb.Type(
                tensor=pb.TensorType(dtype=data_type_pb2.DataType.DT_STRING)
            )
        )
    )
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_tensor_tuple(self):
    type_signature = computation_types.StructType([
        ('x', np.int32),
        ('y', np.str_),
        np.float32,
        ('z', np.bool_),
    ])
    actual_proto = type_serialization.serialize_type(type_signature)
    expected_proto = pb.Type(
        struct=pb.StructType(
            element=[
                pb.StructType.Element(
                    name='x',
                    value=pb.Type(
                        tensor=pb.TensorType(
                            dtype=data_type_pb2.DataType.DT_INT32
                        )
                    ),
                ),
                pb.StructType.Element(
                    name='y',
                    value=pb.Type(
                        tensor=pb.TensorType(
                            dtype=data_type_pb2.DataType.DT_STRING
                        )
                    ),
                ),
                pb.StructType.Element(
                    value=pb.Type(
                        tensor=pb.TensorType(
                            dtype=data_type_pb2.DataType.DT_FLOAT
                        )
                    )
                ),
                pb.StructType.Element(
                    name='z',
                    value=pb.Type(
                        tensor=pb.TensorType(
                            dtype=data_type_pb2.DataType.DT_BOOL
                        )
                    ),
                ),
            ]
        )
    )
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_nested_tuple(self):
    type_signature = computation_types.StructType([
        ('x', [('y', [('z', np.bool_)])]),
    ])
    actual_proto = type_serialization.serialize_type(type_signature)

    z_proto = pb.StructType.Element(
        name='z',
        value=pb.Type(
            tensor=pb.TensorType(dtype=data_type_pb2.DataType.DT_BOOL)
        ),
    )
    expected_proto = pb.Type(
        struct=pb.StructType(
            element=[
                pb.StructType.Element(
                    name='x',
                    value=pb.Type(
                        struct=pb.StructType(
                            element=[
                                pb.StructType.Element(
                                    name='y',
                                    value=pb.Type(
                                        struct=pb.StructType(element=[z_proto])
                                    ),
                                )
                            ]
                        ),
                    ),
                )
            ]
        )
    )
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_function(self):
    actual_proto = type_serialization.serialize_type(
        computation_types.FunctionType((np.int32, np.int32), np.bool_)
    )
    expected_proto = pb.Type(
        function=pb.FunctionType(
            parameter=pb.Type(
                struct=pb.StructType(
                    element=[
                        pb.StructType.Element(
                            value=pb.Type(
                                tensor=pb.TensorType(
                                    dtype=data_type_pb2.DataType.DT_INT32
                                )
                            )
                        ),
                        pb.StructType.Element(
                            value=pb.Type(
                                tensor=pb.TensorType(
                                    dtype=data_type_pb2.DataType.DT_INT32
                                )
                            )
                        ),
                    ]
                )
            ),
            result=pb.Type(
                tensor=pb.TensorType(dtype=data_type_pb2.DataType.DT_BOOL)
            ),
        )
    )
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_placement(self):
    actual_proto = type_serialization.serialize_type(
        computation_types.PlacementType()
    )
    expected_proto = pb.Type(placement=pb.PlacementType())
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_type_with_federated_bool(self):
    federated_type = computation_types.FederatedType(
        np.bool_, placements.CLIENTS, True
    )
    actual_proto = type_serialization.serialize_type(federated_type)
    expected_proto = pb.Type(
        federated=pb.FederatedType(
            placement=pb.PlacementSpec(
                value=pb.Placement(uri=placements.CLIENTS.uri)
            ),
            all_equal=True,
            member=pb.Type(
                tensor=pb.TensorType(dtype=data_type_pb2.DataType.DT_BOOL)
            ),
        )
    )
    self.assertEqual(actual_proto, expected_proto)

  def test_serialize_deserialize_tensor_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.TensorType(np.int32),
        computation_types.TensorType(np.int32, [10]),
        computation_types.TensorType(np.int32, [None]),
    ])

  def test_serialize_deserialize_sequence_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.SequenceType(np.int32),
        computation_types.SequenceType(
            computation_types.StructType((np.int32, np.bool_))
        ),
    ])

  def test_serialize_deserialize_named_tuple_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.StructType([np.int32, np.bool_]),
        computation_types.StructType([
            np.int32,
            computation_types.StructType([('x', np.bool_)]),
        ]),
        computation_types.StructType([('x', np.int32)]),
    ])

  def test_serialize_deserialize_named_tuple_types_py_container(self):
    # The Py container is destroyed during ser/de.
    with_container = computation_types.StructWithPythonType(
        (np.int32, np.bool_), tuple
    )
    p1 = type_serialization.serialize_type(with_container)
    without_container = type_serialization.deserialize_type(p1)
    self.assertNotEqual(with_container, without_container)  # Not equal.
    self.assertIsInstance(without_container, computation_types.StructType)
    self.assertNotIsInstance(
        without_container, computation_types.StructWithPythonType
    )
    with_container.check_equivalent_to(without_container)

  def test_serialize_deserialize_function_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.FunctionType(np.int32, np.bool_),
        computation_types.FunctionType(None, np.bool_),
    ])

  def test_serialize_deserialize_placement_type(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.PlacementType(),
    ])

  def test_serialize_deserialize_federated_types(self):
    self._serialize_deserialize_roundtrip_test([
        computation_types.FederatedType(
            np.int32, placements.CLIENTS, all_equal=True
        ),
        computation_types.FederatedType(
            np.int32, placements.CLIENTS, all_equal=False
        ),
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
  absltest.main()
