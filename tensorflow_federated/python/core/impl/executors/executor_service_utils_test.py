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

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.executors import executor_service_utils
from tensorflow_federated.python.core.impl.types import type_factory


class ExecutorServiceUtilsTest(tf.test.TestCase):

  def test_serialize_deserialize_tensor_value(self):
    x = tf.constant(10.0).numpy()
    type_spec = computation_types.TensorType(tf.as_dtype(x.dtype), x.shape)
    value_proto, value_type = executor_service_utils.serialize_value(
        x, type_spec)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'float32')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), 'float32')
    self.assertTrue(np.array_equal(x, y))

  def test_serialize_deserialize_tensor_value_with_different_dtype(self):
    x = tf.constant(10.0).numpy()
    value_proto, value_type = (
        executor_service_utils.serialize_value(
            x, computation_types.TensorType(tf.int32)))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'int32')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), 'int32')
    self.assertEqual(y, 10)

  def test_serialize_deserialize_tensor_value_with_nontrivial_shape(self):
    x = tf.constant([10, 20, 30]).numpy()
    value_proto, value_type = executor_service_utils.serialize_value(
        x, computation_types.TensorType(tf.int32, [3]))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'int32[3]')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), 'int32[3]')
    self.assertTrue(np.array_equal(x, y))

  def test_serialize_sequence_bad_element_type(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    with self.assertRaisesRegex(
        TypeError, r'Cannot serialize dataset .* int64\* .* float32\*.*'):
      _ = executor_service_utils.serialize_value(
          x, computation_types.SequenceType(tf.float32))

  def test_serialize_sequence_not_a_dataset(self):
    with self.assertRaisesRegex(
        TypeError, r'Cannot serialize Python type int as .* float32\*'):
      _ = executor_service_utils.serialize_value(
          5, computation_types.SequenceType(tf.float32))

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_serialize_deserialize_sequence_of_scalars(self):
    ds = tf.data.Dataset.range(5).map(lambda x: x * 2)
    value_proto, value_type = executor_service_utils.serialize_value(
        ds, computation_types.SequenceType(tf.int64))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'int64*')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), 'int64*')
    self.assertAllEqual(list(y), [x * 2 for x in range(5)])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_serialize_deserialize_sequence_of_tuples(self):
    ds = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, tf.int32), tf.cast(x - 1, tf.float32)))

    value_proto, value_type = executor_service_utils.serialize_value(
        ds,
        computation_types.SequenceType(
            element=(tf.int64, tf.int32, tf.float32)))
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), '<int64,int32,float32>*')

    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), '<int64,int32,float32>*')
    self.assertAllEqual(
        self.evaluate(list(y)), [(x * 2, x, x - 1.) for x in range(5)])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_serialize_deserialize_sequence_of_namedtuples(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, tf.int32), c=tf.cast(x - 1, tf.float32))

    ds = tf.data.Dataset.range(5).map(make_test_tuple)

    element_type = computation_types.StructType([
        ('a', tf.int64),
        ('b', tf.int32),
        ('c', tf.float32),
    ])
    sequence_type = computation_types.SequenceType(element=element_type)
    value_proto, value_type = executor_service_utils.serialize_value(
        ds, sequence_type)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(value_type, sequence_type)

    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(type_spec, sequence_type)
    actual_values = self.evaluate(list(y))
    expected_values = [
        test_tuple_type(a=x * 2, b=x, c=x - 1.) for x in range(5)
    ]
    for actual, expected in zip(actual_values, expected_values):
      self.assertAllClose(actual, expected)

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_serialize_deserialize_sequence_of_nested_structures(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['u', 'v'])

    def _make_nested_tf_structure(x):
      return collections.OrderedDict([
          ('b', tf.cast(x, tf.int32)),
          ('a',
           tuple([
               x,
               test_tuple_type(x * 2, x * 3),
               collections.OrderedDict([('x', x**2), ('y', x**3)])
           ])),
      ])

    ds = tf.data.Dataset.range(5).map(_make_nested_tf_structure)
    element_type = computation_types.StructType([
        ('b', tf.int32),
        ('a',
         computation_types.StructType([
             (None, tf.int64),
             (None, test_tuple_type(tf.int64, tf.int64)),
             (None,
              computation_types.StructType([('x', tf.int64), ('y', tf.int64)])),
         ])),
    ])
    sequence_type = computation_types.SequenceType(element=element_type)
    value_proto, value_type = executor_service_utils.serialize_value(
        ds, sequence_type)

    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(value_type, sequence_type)

    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    # These aren't the same because ser/de destroys the PyContainer
    type_spec.check_equivalent_to(sequence_type)

    def _build_expected_structure(x):
      return collections.OrderedDict([
          ('b', x),
          ('a',
           tuple([
               x,
               test_tuple_type(x * 2, x * 3),
               collections.OrderedDict([('x', x**2), ('y', x**3)])
           ])),
      ])

    actual_values = self.evaluate(list(y))
    expected_values = [_build_expected_structure(x) for x in range(5)]
    for actual, expected in zip(actual_values, expected_values):
      self.assertEqual(type(actual), type(expected))
      self.assertAllClose(actual, expected)

  def test_serialize_deserialize_tensor_value_with_bad_shape(self):
    x = tf.constant([10, 20, 30]).numpy()
    with self.assertRaises(TypeError):
      executor_service_utils.serialize_value(x, tf.int32)

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

  def test_serialize_deserialize_nested_tuple_value_with_names(self):
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

  def test_serialize_deserialize_nested_tuple_value_without_names(self):
    x = tuple([10, 20])
    x_type = computation_types.to_type(tuple([tf.int32, tf.int32]))
    value_proto, value_type = executor_service_utils.serialize_value(x, x_type)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), '<int32,int32>')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), str(x_type))
    self.assertCountEqual(y, (10, 20))

  def test_serialize_deserialize_federated_at_clients(self):
    x = [10, 20]
    x_type = type_factory.at_clients(tf.int32)
    value_proto, value_type = executor_service_utils.serialize_value(x, x_type)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), '{int32}@CLIENTS')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), str(x_type))
    self.assertEqual(y, [10, 20])

  def test_serialize_deserialize_federated_at_server(self):
    x = 10
    x_type = type_factory.at_server(tf.int32)
    value_proto, value_type = executor_service_utils.serialize_value(x, x_type)
    self.assertIsInstance(value_proto, executor_pb2.Value)
    self.assertEqual(str(value_type), 'int32@SERVER')
    y, type_spec = executor_service_utils.deserialize_value(value_proto)
    self.assertEqual(str(type_spec), str(x_type))
    self.assertEqual(y, 10)


class DatasetSerializationTest(test.TestCase):

  def test_serialize_sequence_not_a_dataset(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*Dataset.* found int'):
      _ = executor_service_utils._serialize_dataset(5)

  def test_serialize_sequence_bytes_too_large(self):
    with self.assertRaisesRegex(ValueError,
                                r'Serialized size .* exceeds maximum allowed'):
      _ = executor_service_utils._serialize_dataset(
          tf.data.Dataset.range(5), max_serialized_size_bytes=0)

  def test_roundtrip_sequence_of_scalars(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    serialized_bytes = executor_service_utils._serialize_dataset(x)
    y = executor_service_utils._deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(list(y), [x * 2 for x in range(5)])

  def test_roundtrip_sequence_of_tuples(self):
    x = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, tf.int32), tf.cast(x - 1, tf.float32)))
    serialized_bytes = executor_service_utils._serialize_dataset(x)
    y = executor_service_utils._deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(
        self.evaluate(list(y)), [(x * 2, x, x - 1.) for x in range(5)])

  def test_roundtrip_sequence_of_singleton_tuples(self):
    x = tf.data.Dataset.range(5).map(lambda x: (x,))
    serialized_bytes = executor_service_utils._serialize_dataset(x)
    y = executor_service_utils._deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    expected_values = [(x,) for x in range(5)]
    actual_values = self.evaluate(list(y))
    self.assertAllEqual(expected_values, actual_values)

  def test_roundtrip_sequence_of_namedtuples(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, tf.int32), c=tf.cast(x - 1, tf.float32))

    x = tf.data.Dataset.range(5).map(make_test_tuple)
    serialized_bytes = executor_service_utils._serialize_dataset(x)
    y = executor_service_utils._deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(
        self.evaluate(list(y)),
        [test_tuple_type(a=x * 2, b=x, c=x - 1.) for x in range(5)])

  def test_roundtrip_sequence_of_nested_structures(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['u', 'v'])

    def _make_nested_tf_structure(x):
      return collections.OrderedDict([
          ('b', tf.cast(x, tf.int32)),
          ('a',
           tuple([
               x,
               test_tuple_type(x * 2, x * 3),
               collections.OrderedDict([('x', x**2), ('y', x**3)])
           ])),
      ])

    x = tf.data.Dataset.range(5).map(_make_nested_tf_structure)
    serialzied_bytes = executor_service_utils._serialize_dataset(x)
    y = executor_service_utils._deserialize_dataset(serialzied_bytes)

    # Note: TF loses the `OrderedDict` during serialization, so the expectation
    # here is for a `dict` in the result.
    self.assertEqual(
        y.element_spec, {
            'b':
                tf.TensorSpec([], tf.int32),
            'a':
                tuple([
                    tf.TensorSpec([], tf.int64),
                    test_tuple_type(
                        tf.TensorSpec([], tf.int64),
                        tf.TensorSpec([], tf.int64),
                    ),
                    {
                        'x': tf.TensorSpec([], tf.int64),
                        'y': tf.TensorSpec([], tf.int64),
                    },
                ]),
        })

    def _build_expected_structure(x):
      return {
          'b': x,
          'a': tuple([x,
                      test_tuple_type(x * 2, x * 3), {
                          'x': x**2,
                          'y': x**3
                      }])
      }

    actual_values = self.evaluate(list(y))
    expected_values = [_build_expected_structure(x) for x in range(5)]
    for actual, expected in zip(actual_values, expected_values):
      self.assertAllClose(actual, expected)


if __name__ == '__main__':
  tf.test.main()
