# Copyright 2021, The TensorFlow Federated Authors.
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
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
from federated_language.proto import computation_pb2
import numpy as np

from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.executors import value_serialization

# Convenience aliases.
TensorType = federated_language.TensorType


class _TestNamedTuple(NamedTuple):
  a: int
  b: int
  c: int


@federated_language.federated_computation(np.int32)
def _identity(x):
  return x


TENSOR_SERIALIZATION_TEST_PARAMS = [
    ('numpy_scalar', np.float32(25.0), TensorType(np.float32)),
    (
        'numpy_1d_tensor',
        np.array([1.0, 2.0], np.float32),
        TensorType(np.float32, [2]),
    ),
    ('python_scalar', 25.0, TensorType(np.float32)),
    ('python_1d_list', [1.0, 2.0], TensorType(np.float32, [2])),
    ('python_2d_list', [[1.0], [2.0]], TensorType(np.float32, [2, 1])),
]


class ValueSerializationTest(parameterized.TestCase):

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_without_hint(
      self, x, serialize_type_spec
  ):
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec
    )
    self.assertEqual(value_type, serialize_type_spec)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    self.assertEqual(type_spec, serialize_type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype)
    if isinstance(y, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(y, x)
    else:
      self.assertEqual(y, x)

  def test_serialize_deserialize_tensor_value_unknown_shape_without_hint(self):
    x = np.asarray([1.0, 2.0], np.float32)
    serialize_type_spec = TensorType(np.float32, [None])
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec
    )
    self.assertTrue(value_type.is_assignable_from(serialize_type_spec))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    self.assertTrue(serialize_type_spec.is_assignable_from(type_spec))
    self.assertEqual(y.dtype, serialize_type_spec.dtype)
    if isinstance(y, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(y, x, strict=True)
    else:
      self.assertEqual(y, x)

  @parameterized.named_parameters(TENSOR_SERIALIZATION_TEST_PARAMS)
  def test_serialize_deserialize_tensor_value_with_hint(
      self, x, serialize_type_spec
  ):
    value_proto, value_type = value_serialization.serialize_value(
        x, serialize_type_spec
    )
    self.assertEqual(value_type, serialize_type_spec)
    y, deserialize_type_spec = value_serialization.deserialize_value(
        value_proto, type_hint=serialize_type_spec
    )
    self.assertEqual(deserialize_type_spec, serialize_type_spec)
    self.assertEqual(y.dtype, serialize_type_spec.dtype)
    if isinstance(y, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(y, x)
    else:
      self.assertEqual(y, x)

  @parameterized.named_parameters(
      ('str', 'abc', TensorType(np.str_), b'abc'),
      ('bytes', b'abc', TensorType(np.str_), b'abc'),
      (
          'bytes_null_terminated',
          b'abc\x00\x00',
          TensorType(np.str_),
          b'abc\x00\x00',
      ),
      ('numpy_scalar_str', np.str_('abc'), TensorType(np.str_), b'abc'),
      ('numpy_scalar_bytes', np.bytes_(b'abc'), TensorType(np.str_), b'abc'),
      (
          'numpy_scalar_bytes_null_termianted',
          np.bytes_(b'abc\x00\x00'),
          TensorType(np.str_),
          b'abc\x00\x00',
      ),
      (
          'numpy_array_str',
          np.array(['abc', 'def'], np.str_),
          TensorType(np.str_, [2]),
          np.array([b'abc', b'def'], np.object_),
      ),
      (
          'numpy_array_bytes',
          np.array([b'abc', b'def'], np.bytes_),
          TensorType(np.str_, [2]),
          np.array([b'abc', b'def'], np.object_),
      ),
      (
          'numpy_array_bytes_null_termianted',
          np.array([b'abc\x00\x00', b'def\x00\x00'], np.object_),
          TensorType(np.str_, [2]),
          np.array([b'abc\x00\x00', b'def\x00\x00'], np.object_),
      ),
  )
  def test_serialize_deserialize_string_value(
      self, value, type_spec, expected_value
  ):
    value_proto, value_type = value_serialization.serialize_value(
        value, type_spec
    )
    self.assertEqual(value_type, type_spec)
    result, result_type = value_serialization.deserialize_value(
        value_proto, type_spec
    )
    self.assertEqual(result_type, type_spec)

    if isinstance(result, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(result, expected_value, strict=True)
    else:
      self.assertEqual(result, expected_value)

  def test_serialize_raises_on_incompatible_dtype_float_to_int(self):
    x = np.float32(10.0)
    with self.assertRaisesRegex(TypeError, 'Failed to serialize the value'):
      value_serialization.serialize_value(x, TensorType(np.int32))

  def test_serialize_deserialize_tensor_value_with_nontrivial_shape(self):
    x = np.int32([10, 20, 30])
    value_proto, value_type = value_serialization.serialize_value(
        x, TensorType(np.int32, [3])
    )
    self.assertEqual(value_type, TensorType(np.int32, [3]))
    y, type_spec = value_serialization.deserialize_value(value_proto)
    self.assertEqual(type_spec, TensorType(np.int32, [3]))
    if isinstance(y, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(y, x, strict=True)
    else:
      self.assertEqual(y, x)

  def test_serialize_struct_with_type_element_mismatch(self):
    x = {'a': 1}
    with self.assertRaisesRegex(
        TypeError,
        (
            'Cannot serialize a struct value of 1 elements to a struct '
            'type requiring 2 elements.'
        ),
    ):
      value_serialization.serialize_value(
          x, federated_language.StructType([('a', np.int32), ('b', np.int32)])
      )

  def test_serialize_sequence_raises_type_error_with_invalid_type_spec(self):
    value = [1, 2, 3]
    type_spec = federated_language.SequenceType(np.float32)
    with self.assertRaisesRegex(TypeError, 'Failed to serialize the value'):
      value_serialization.serialize_value(value, type_spec)

  @parameterized.named_parameters(
      ('scalar', [1, 2, 3], federated_language.SequenceType(np.int32)),
      (
          'tuple',
          [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
          federated_language.SequenceType([np.int32, np.int32, np.int32]),
      ),
      (
          'tuple_empty',
          [(), (), ()],
          federated_language.SequenceType([]),
      ),
      (
          'tuple_singleton',
          [(1,), (2,), (3,)],
          federated_language.SequenceType([np.int32]),
      ),
      (
          'dict',
          [
              {'a': 1, 'b': 2, 'c': 3},
              {'a': 4, 'b': 5, 'c': 6},
              {'a': 7, 'b': 8, 'c': 9},
          ],
          federated_language.SequenceType([
              ('a', np.int32),
              ('b', np.int32),
              ('c', np.int32),
          ]),
      ),
      (
          'named_tuple',
          [
              _TestNamedTuple(1, 2, 3),
              _TestNamedTuple(4, 5, 6),
              _TestNamedTuple(7, 8, 9),
          ],
          federated_language.SequenceType(
              federated_language.StructWithPythonType(
                  [
                      ('a', np.int32),
                      ('b', np.int32),
                      ('c', np.int32),
                  ],
                  container_type=_TestNamedTuple,
              )
          ),
      ),
  )
  def test_serialize_deserialize_sequence(self, value, type_spec):
    value_proto, value_type = value_serialization.serialize_value(
        value, type_spec
    )
    self.assertEqual(value_type, type_spec)
    result, result_type = value_serialization.deserialize_value(
        value_proto, type_spec
    )
    federated_language.framework.assert_types_equivalent(result_type, type_spec)
    self.assertEqual(result, value)

  def test_serialize_deserialize_tensor_value_with_bad_shape(self):
    value = np.array([10, 20, 30], np.int32)
    type_spec = federated_language.TensorType(np.int32)

    with self.assertRaises(TypeError):
      value_serialization.serialize_value(value, type_spec)

  def test_serialize_deserialize_computation_value(self):
    value_proto, value_type = value_serialization.serialize_value(_identity)
    self.assertEqual(value_proto.WhichOneof('value'), 'computation')
    self.assertEqual(
        value_type,
        federated_language.FunctionType(parameter=np.int32, result=np.int32),
    )
    _, type_spec = value_serialization.deserialize_value(value_proto)
    self.assertEqual(
        type_spec,
        federated_language.FunctionType(parameter=np.int32, result=np.int32),
    )

  def test_serialize_deserialize_nested_tuple_value_with_names(self):
    x = collections.OrderedDict(
        a=10, b=[20, 30], c=collections.OrderedDict(d=40)
    )
    x_type = federated_language.StructType(
        collections.OrderedDict(
            a=np.int32,
            b=[np.int32, np.int32],
            c=collections.OrderedDict(d=np.int32),
        )
    )
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    self.assertEqual(value_type, x_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    # Don't assert on the Python container since it is lost in serialization.
    federated_language.framework.assert_types_equivalent(type_spec, x_type)
    self.assertEqual(y, structure.from_container(x, recursive=True))

  def test_serialize_deserialize_nested_tuple_value_without_names(self):
    x = (10, 20)
    x_type = federated_language.StructType([np.int32, np.int32])
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    self.assertEqual(value_type, x_type)
    y, type_spec = value_serialization.deserialize_value(value_proto)
    federated_language.framework.assert_types_equivalent(type_spec, x_type)
    self.assertEqual(y, structure.from_container((10, 20)))

  def test_serialize_deserialize_federated_at_clients(self):
    x = [10, 20]
    x_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    self.assertEqual(
        value_type,
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    self.assertEqual(
        type_spec,
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
    )
    self.assertEqual(y, [10, 20])

  def test_deserialize_federated_value_with_unset_member_type(self):
    x = 10
    x_type = federated_language.TensorType(np.int32)
    member_proto, _ = value_serialization.serialize_value(x, x_type)
    fully_specified_type_at_clients = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    ).to_proto()

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=fully_specified_type_at_clients.federated.placement,
        all_equal=fully_specified_type_at_clients.federated.all_equal,
    )

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type, value=[member_proto]
    )
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    deserialized_federated_value, deserialized_type_spec = (
        value_serialization.deserialize_value(federated_value_proto)
    )
    self.assertEqual(
        deserialized_type_spec,
        federated_language.FederatedType(np.int32, federated_language.CLIENTS),
    )
    self.assertEqual(deserialized_federated_value, [10])

  def test_deserialize_federated_value_with_incompatible_member_types_raises(
      self,
  ):
    x = 10
    x_type = federated_language.TensorType(np.int32)
    int_member_proto, _ = value_serialization.serialize_value(x, x_type)
    y = 10.0
    y_type = federated_language.TensorType(np.float32)
    float_member_proto, _ = value_serialization.serialize_value(y, y_type)
    fully_specified_type_at_clients = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    ).to_proto()

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=fully_specified_type_at_clients.federated.placement,
        all_equal=False,
    )

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type,
        value=[int_member_proto, float_member_proto],
    )
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    with self.assertRaises(TypeError):
      value_serialization.deserialize_value(federated_value_proto)

  def test_deserialize_federated_all_equal_value_takes_first_element(self):
    tensor_value_pb, _ = value_serialization.serialize_value(
        10, TensorType(np.int32)
    )
    num_clients = 5
    value_pb = executor_pb2.Value(
        federated=executor_pb2.Value.Federated(
            value=[tensor_value_pb] * num_clients,
            type=computation_pb2.FederatedType(
                placement=computation_pb2.PlacementSpec(
                    value=computation_pb2.Placement(
                        uri=federated_language.CLIENTS.uri
                    )
                )
            ),
        )
    )
    all_equal_clients_type_hint = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS, all_equal=True
    )
    deserialized_value, deserialized_type = (
        value_serialization.deserialize_value(
            value_pb, all_equal_clients_type_hint
        )
    )
    self.assertEqual(deserialized_type, all_equal_clients_type_hint)
    self.assertEqual(deserialized_value, 10)

  def test_deserialize_federated_value_promotes_types(self):
    x = [10]
    smaller_type = federated_language.StructType([(None, np.int32)])
    smaller_type_member_proto, _ = value_serialization.serialize_value(
        x, smaller_type
    )
    larger_type = federated_language.StructType([('a', np.int32)])
    larger_type_member_proto, _ = value_serialization.serialize_value(
        x, larger_type
    )
    type_at_clients = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    ).to_proto()

    unspecified_member_federated_type = computation_pb2.FederatedType(
        placement=type_at_clients.federated.placement, all_equal=False
    )

    federated_proto = executor_pb2.Value.Federated(
        type=unspecified_member_federated_type,
        value=[larger_type_member_proto, smaller_type_member_proto],
    )
    federated_value_proto = executor_pb2.Value(federated=federated_proto)

    _, deserialized_type_spec = value_serialization.deserialize_value(
        federated_value_proto
    )
    self.assertEqual(
        deserialized_type_spec,
        federated_language.FederatedType(
            larger_type, federated_language.CLIENTS
        ),
    )

  def test_serialize_deserialize_federated_at_server(self):
    x = 10
    x_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )
    value_proto, value_type = value_serialization.serialize_value(x, x_type)
    self.assertEqual(
        value_type,
        federated_language.FederatedType(np.int32, federated_language.SERVER),
    )
    y, type_spec = value_serialization.deserialize_value(value_proto)
    self.assertEqual(type_spec, x_type)
    self.assertEqual(y, 10)


class SerializeCardinalitiesTest(absltest.TestCase):

  def test_serialize_deserialize_clients_and_server_cardinalities_roundtrip(
      self,
  ):
    client_and_server_cardinalities = {
        federated_language.CLIENTS: 10,
        federated_language.SERVER: 1,
    }
    cardinalities_list = value_serialization.serialize_cardinalities(
        client_and_server_cardinalities
    )
    for cardinality in cardinalities_list:
      self.assertIsInstance(cardinality, executor_pb2.Cardinality)
    reconstructed_cardinalities = value_serialization.deserialize_cardinalities(
        cardinalities_list
    )
    self.assertEqual(
        client_and_server_cardinalities, reconstructed_cardinalities
    )

  def test_serialize_deserialize_clients_alone(self):
    client_cardinalities = {federated_language.CLIENTS: 10}
    cardinalities_list = value_serialization.serialize_cardinalities(
        client_cardinalities
    )
    for cardinality in cardinalities_list:
      self.assertIsInstance(cardinality, executor_pb2.Cardinality)
    reconstructed_cardinalities = value_serialization.deserialize_cardinalities(
        cardinalities_list
    )
    self.assertEqual(client_cardinalities, reconstructed_cardinalities)


if __name__ == '__main__':
  absltest.main()
