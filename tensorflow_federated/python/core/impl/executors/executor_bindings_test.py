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

from absl.testing import parameterized
import portpicker
import tensorflow as tf

from pybind11_abseil import status
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import serialization_bindings
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions

TensorType = computation_types.TensorType
StructType = computation_types.StructType
SequenceType = computation_types.SequenceType


def _test_map_integers(tensor):
  """Map an integer tensor via a lookup table."""
  # Used for testing resources.
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=list(range(5)),
          values=list(reversed(range(5))),
          key_dtype=tf.int64,
          value_dtype=tf.int64),
      default_value=-1)
  return table.lookup(tensor)


class TensorFlowExecutorBindingsTest(parameterized.TestCase,
                                     test_case.TestCase):

  def test_create(self):
    try:
      executor_bindings.create_tensorflow_executor()
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Exception: {e}')

  def test_create_value(self):
    executor = executor_bindings.create_tensorflow_executor()
    # 1. Test a simple tensor.
    expected_type_spec = TensorType(shape=[3], dtype=tf.int64)
    value_pb, _ = value_serialization.serialize_value([1, 2, 3],
                                                      expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    self.assertEqual(value.ref, 0)
    self.assertEqual(str(value), '0')
    self.assertEqual(repr(value), r'<OwnedValueId: 0>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    self.assert_types_identical(type_spec, expected_type_spec)
    self.assertAllEqual(deserialized_value, [1, 2, 3])
    # 2. Test a struct of tensors, ensure that we get a different ID.
    expected_type_spec = StructType([
        ('a', TensorType(shape=[3], dtype=tf.int64)),
        ('b', TensorType(shape=[], dtype=tf.float32))
    ])
    value_pb, _ = value_serialization.serialize_value(
        collections.OrderedDict(a=tf.constant([1, 2, 3]), b=tf.constant(42.0)),
        expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented.
    self.assertEqual(value.ref, 1)
    self.assertEqual(str(value), '1')
    self.assertEqual(repr(value), r'<OwnedValueId: 1>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    # Note: here we've lost the names `a` and `b` in the output. The output
    # is a more _strict_ type.
    self.assert_type_assignable_from(expected_type_spec, type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, expected_type_spec)
    self.assertAllClose(deserialized_value,
                        collections.OrderedDict(a=(1, 2, 3), b=42.0))
    # 3. Test creating a value from a computation.
    @computations.tf_computation(tf.int32, tf.int32)
    def foo(a, b):
      return tf.add(a, b)

    value_pb, _ = value_serialization.serialize_value(foo)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented again.
    self.assertEqual(value.ref, 2)
    self.assertEqual(str(value), '2')
    self.assertEqual(repr(value), '<OwnedValueId: 2>')
    # Note: functions are not materializable, no addition assertions.

  @parameterized.named_parameters(
      ('range', tf.data.Dataset.range(5)),
      ('shuffled_range', tf.data.Dataset.range(5).shuffle(3)),
      ('mapped_with_resource_range',
       tf.data.Dataset.range(5).map(_test_map_integers)),
      ('mapped_range', tf.data.Dataset.range(5).map(lambda x: x)),
      ('batched_range', tf.data.Dataset.range(5).batch(2,
                                                       drop_remainder=False)),
      ('tensor_slices', tf.data.Dataset.from_tensor_slices(list(range(5)))),
  )
  def test_create_value_sequence(self, dataset):
    executor = executor_bindings.create_tensorflow_executor()
    sequence_type = SequenceType(dataset.element_spec)
    arg_value_pb, _ = value_serialization.serialize_value(
        dataset, sequence_type)
    arg = executor.create_value(arg_value_pb)

    @computations.tf_computation(sequence_type)
    def sum_examples(ds):
      return ds.reduce(
          tf.constant(0, ds.element_spec.dtype),
          lambda s, x: s + tf.reduce_sum(x))

    comp_pb = serialization_bindings.Value(
        computation=sum_examples.get_proto(sum_examples))
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    output_pb = executor.materialize(result.ref)
    result, result_type_spec = value_serialization.deserialize_value(output_pb)
    self.assert_types_identical(result_type_spec,
                                TensorType(sequence_type.element.dtype))
    self.assertEqual(result, sum(range(5)))

  def test_create_tuple_of_value_sequence(self):
    self.skipTest('b/197147669')
    datasets = (tf.data.Dataset.range(5), tf.data.Dataset.range(5))
    executor = executor_bindings.create_tensorflow_executor()
    struct_of_sequence_type = StructType([
        (None, SequenceType(datasets[0].element_spec)),
        (None, SequenceType(datasets[0].element_spec))
    ])
    arg_value_pb, _ = value_serialization.serialize_value(
        datasets, struct_of_sequence_type)
    arg = executor.create_value(arg_value_pb)

    @computations.tf_computation(struct_of_sequence_type)
    def preprocess(datasets):

      def double_value(x):
        return 2 * x

      @tf.function
      def add_preprocessing(ds1, ds2):
        return ds1.map(double_value), ds2.map(double_value)

      return add_preprocessing(*datasets)

    comp_pb = serialization_bindings.Value(
        computation=preprocess.get_proto(preprocess))
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    output_pb = executor.materialize(result.ref)
    result, result_type_spec = value_serialization.deserialize_value(
        output_pb, type_hint=struct_of_sequence_type)
    self.assert_types_identical(result_type_spec, struct_of_sequence_type)

  def test_create_struct(self):
    executor = executor_bindings.create_tensorflow_executor()
    expected_type_spec = TensorType(shape=[3], dtype=tf.int64)
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec])
    self.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Create a struct from the struct and another value.
    new_struct_value = executor.create_struct([struct_value.ref, value.ref])
    materialized_value = executor.materialize(new_struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    struct_type_spec = computation_types.to_type(
        [struct_type_spec, expected_type_spec])
    self.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose([[(1, 2, 3), (1, 2, 3)], (1, 2, 3)], deserialized_value)

  def test_create_selection(self):
    executor = executor_bindings.create_tensorflow_executor()
    expected_type_spec = TensorType(shape=[3], dtype=tf.int64)
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec])
    self.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Select the first value out of the struct.
    new_value = executor.create_selection(struct_value.ref, 0)
    materialized_value = executor.materialize(new_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    self.assert_types_equivalent(type_spec, expected_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose((1, 2, 3), deserialized_value)

  def test_call_with_arg(self):
    executor = executor_bindings.create_tensorflow_executor()
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), TensorType(shape=[3], dtype=tf.int64))
    value_ref = executor.create_value(value_pb)
    arg = executor.create_struct((value_ref.ref, value_ref.ref))

    @computations.tf_computation(tf.int64, tf.int64)
    def foo(a, b):
      return tf.add(a, b)

    comp_pb = serialization_bindings.Value(computation=foo.get_proto(foo))
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertAllEqual(result_tensor, [2, 4, 6])

  def test_call_no_arg(self):
    executor = executor_bindings.create_tensorflow_executor()

    @computations.tf_computation
    def foo():
      return tf.constant(123.0)

    comp_pb = serialization_bindings.Value(computation=foo.get_proto(foo))
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, None)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertEqual(result_tensor, 123.0)

  def test_materialize_on_unkown_fails(self):
    executor = executor_bindings.create_tensorflow_executor()
    with self.assertRaisesRegex(status.StatusNotOk, 'NOT_FOUND'):
      executor.materialize(0)


class ReferenceResolvingExecutorBindingsTest(test_case.TestCase):

  def test_create(self):
    try:
      executor_bindings.create_reference_resolving_executor(
          executor_bindings.create_tensorflow_executor())
    except Exception as e:  # pylint: disable=broad-except
      self.fail(f'Exception: {e}')

  def test_create_value(self):
    executor = executor_bindings.create_reference_resolving_executor(
        executor_bindings.create_tensorflow_executor())
    # 1. Test a simple tensor.
    expected_type_spec = TensorType(shape=[3], dtype=tf.int64)
    value_pb, _ = value_serialization.serialize_value([1, 2, 3],
                                                      expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    self.assertEqual(value.ref, 0)
    self.assertEqual(str(value), '0')
    self.assertEqual(repr(value), r'<OwnedValueId: 0>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    self.assert_types_identical(type_spec, expected_type_spec)
    self.assertAllEqual(deserialized_value, [1, 2, 3])
    # 2. Test a struct of tensors, ensure that we get a different ID.
    expected_type_spec = StructType([
        ('a', TensorType(shape=[3], dtype=tf.int64)),
        ('b', TensorType(shape=[], dtype=tf.float32))
    ])
    value_pb, _ = value_serialization.serialize_value(
        collections.OrderedDict(a=tf.constant([1, 2, 3]), b=tf.constant(42.0)),
        expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented.
    self.assertEqual(value.ref, 1)
    self.assertEqual(str(value), '1')
    self.assertEqual(repr(value), r'<OwnedValueId: 1>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    # Note: here we've lost the names `a` and `b` in the output. The output
    # is a more _strict_ type.
    self.assert_type_assignable_from(expected_type_spec, type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, expected_type_spec)
    self.assertAllClose(deserialized_value,
                        collections.OrderedDict(a=(1, 2, 3), b=42.0))
    # 3. Test creating a value from a computation.
    @computations.tf_computation(tf.int32, tf.int32)
    def foo(a, b):
      return tf.add(a, b)

    value_pb, _ = value_serialization.serialize_value(foo)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented again.
    self.assertEqual(value.ref, 2)
    self.assertEqual(str(value), '2')
    self.assertEqual(repr(value), '<OwnedValueId: 2>')
    # Note: functions are not materializable, no addition assertions.

  def test_create_struct(self):
    executor = executor_bindings.create_reference_resolving_executor(
        executor_bindings.create_tensorflow_executor())
    expected_type_spec = TensorType(shape=[3], dtype=tf.int64)
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec])
    self.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Create a struct from the struct and another value.
    new_struct_value = executor.create_struct([struct_value.ref, value.ref])
    materialized_value = executor.materialize(new_struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    struct_type_spec = computation_types.to_type(
        [struct_type_spec, expected_type_spec])
    self.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose([[(1, 2, 3), (1, 2, 3)], (1, 2, 3)], deserialized_value)

  def test_create_selection(self):
    executor = executor_bindings.create_reference_resolving_executor(
        executor_bindings.create_tensorflow_executor())
    expected_type_spec = TensorType(shape=[3], dtype=tf.int64)
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), expected_type_spec)
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec])
    self.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Select the first value out of the struct.
    new_value = executor.create_selection(struct_value.ref, 0)
    materialized_value = executor.materialize(new_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value)
    self.assert_types_equivalent(type_spec, expected_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec)
    self.assertAllClose((1, 2, 3), deserialized_value)

  def test_call_with_arg(self):
    executor = executor_bindings.create_reference_resolving_executor(
        executor_bindings.create_tensorflow_executor())
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), TensorType(shape=[3], dtype=tf.int64))
    value_ref = executor.create_value(value_pb)
    arg = executor.create_struct((value_ref.ref, value_ref.ref))

    @computations.tf_computation(tf.int64, tf.int64)
    def foo(a, b):
      return tf.add(a, b)

    comp_pb = serialization_bindings.Value(computation=foo.get_proto(foo))
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertAllEqual(result_tensor, [2, 4, 6])

  def test_call_no_arg(self):
    executor = executor_bindings.create_reference_resolving_executor(
        executor_bindings.create_tensorflow_executor())

    @computations.tf_computation
    def foo():
      return tf.constant(123.0)

    comp_pb = serialization_bindings.Value(computation=foo.get_proto(foo))
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, None)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertEqual(result_tensor, 123.0)

  def test_materialize_on_unkown_fails(self):
    executor = executor_bindings.create_tensorflow_executor()
    with self.assertRaisesRegex(status.StatusNotOk, 'NOT_FOUND'):
      executor.materialize(0)


class FederatingExecutorBindingsTest(test_case.TestCase):

  def test_construction_placements_casters(self):
    with self.subTest('placement_literal_keys'):
      try:
        executor_bindings.create_federating_executor(
            executor_bindings.create_tensorflow_executor(),
            {placements.CLIENTS: 10})
      except Exception as e:  # pylint: disable=broad-except
        self.fail(f'Exception: {e}')
    with self.subTest('fails_non_dict'):
      with self.assertRaisesRegex(TypeError, 'must be a `Mapping`'):
        executor_bindings.create_federating_executor(
            executor_bindings.create_tensorflow_executor(),
            [(placements.CLIENTS, 10)])
    with self.subTest('fails_non_placement_keys'):
      with self.assertRaisesRegex(TypeError, '`PlacementLiteral`'):
        executor_bindings.create_federating_executor(
            executor_bindings.create_tensorflow_executor(), {'clients': 10})
      with self.assertRaisesRegex(TypeError, '`PlacementLiteral`'):
        executor_bindings.create_federating_executor(
            executor_bindings.create_tensorflow_executor(), {10: 10})
    with self.subTest('fails_non_int_value'):
      with self.assertRaisesRegex(TypeError, r'`int` values'):
        executor_bindings.create_federating_executor(
            executor_bindings.create_tensorflow_executor(),
            {placements.CLIENTS: 0.5})


class RemoteExecutorBindingsTest(test_case.TestCase):

  def test_insecure_channel_construction(self):
    remote_ex = executor_bindings.create_remote_executor(
        executor_bindings.create_insecure_grpc_channel('localhost:{}'.format(
            portpicker.pick_unused_port())),
        cardinalities={placements.CLIENTS: 10})
    self.assertIsInstance(remote_ex, executor_bindings.Executor)


class ComposingExecutorBindingsTest(test_case.TestCase):

  def test_construction(self):
    server = executor_bindings.create_tensorflow_executor()
    children = [
        executor_bindings.create_composing_child(
            executor_bindings.create_tensorflow_executor(),
            {placements.CLIENTS: 0})
    ]
    composing_ex = executor_bindings.create_composing_executor(server, children)
    self.assertIsInstance(composing_ex, executor_bindings.Executor)


if __name__ == '__main__':
  test_case.main()
