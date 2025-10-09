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

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import federated_language_executor
from federated_language_executor import executor_pb2
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_computation_factory
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_executor_bindings
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types


# Creating logical devices should be done only once before TF runtime startup
# Thus, perform it during setUpModule method.
def setUpModule():
  devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(
      devices[0],
      [
          tf.config.LogicalDeviceConfiguration(),
      ]
      * 8,
  )


def _test_map_integers(tensor):
  """Map an integer tensor via a lookup table."""
  # Used for testing resources.
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          keys=list(range(5)),
          values=list(reversed(range(5))),
          key_dtype=tf.int64,
          value_dtype=tf.int64,
      ),
      default_value=-1,
  )
  return table.lookup(tensor)


def get_executor():
  return tensorflow_executor_bindings.create_tensorflow_executor()


class TensorFlowExecutorBindingsTest(parameterized.TestCase, tf.test.TestCase):

  def test_construction(
      self,
  ):
    try:
      get_executor()
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')

  def test_create_value(self):
    executor = get_executor()
    # 1. Test a simple tensor.
    expected_type_spec = federated_language.TensorType(np.int64, [3])
    value_pb, _ = federated_language_executor.serialize_value(
        [1, 2, 3], expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, federated_language_executor.OwnedValueId)
    self.assertEqual(value.ref, 0)
    self.assertEqual(str(value), '0')
    self.assertEqual(repr(value), r'<OwnedValueId: 0>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = (
        federated_language_executor.deserialize_value(materialized_value)
    )
    self.assertEqual(type_spec, expected_type_spec)
    self.assertAllEqual(deserialized_value, [1, 2, 3])
    # 2. Test a struct of tensors, ensure that we get a different ID.
    expected_type_spec = federated_language.StructType([
        ('a', federated_language.TensorType(np.int64, [3])),
        ('b', federated_language.TensorType(np.float32, [])),
    ])
    value = collections.OrderedDict(
        a=np.array([1, 2, 3], np.int64),
        b=np.array(42.0, np.float32),
    )
    value_pb, _ = federated_language_executor.serialize_value(
        value, expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, federated_language_executor.OwnedValueId)
    # Assert the value ID was incremented.
    self.assertEqual(value.ref, 1)
    self.assertEqual(str(value), '1')
    self.assertEqual(repr(value), r'<OwnedValueId: 1>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = (
        federated_language_executor.deserialize_value(materialized_value)
    )
    # Note: here we've lost the names `a` and `b` in the output. The output
    # is a more _strict_ type.
    self.assertTrue(expected_type_spec.is_assignable_from(type_spec))
    deserialized_value = federated_language.framework.type_to_py_container(
        deserialized_value, expected_type_spec
    )
    self.assertAllClose(
        deserialized_value, collections.OrderedDict(a=(1, 2, 3), b=42.0)
    )

    # 3. Test creating a value from a computation.
    foo, _ = tensorflow_computation_factory.create_binary_operator(
        tf.add,
        federated_language.TensorType(np.int64),
        federated_language.TensorType(np.int64),
    )

    value_pb = executor_pb2.Value(computation=foo)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, federated_language_executor.OwnedValueId)
    # Assert the value ID was incremented again.
    self.assertEqual(value.ref, 2)
    self.assertEqual(str(value), '2')
    self.assertEqual(repr(value), '<OwnedValueId: 2>')
    # Note: functions are not materializable, no addition assertions.

  @parameterized.named_parameters(
      ('range', lambda: tf.data.Dataset.range(5), 10),
      ('shuffled_range', lambda: tf.data.Dataset.range(5).shuffle(3), 10),
      (
          'mapped_with_resource_range',
          lambda: tf.data.Dataset.range(5).map(_test_map_integers),
          10,
      ),
      ('mapped_range', lambda: tf.data.Dataset.range(5).map(lambda x: x), 10),
      (
          'batched_range',
          lambda: tf.data.Dataset.range(5).batch(2, drop_remainder=True),
          6,
      ),
      (
          'tensor_slices',
          lambda: tf.data.Dataset.from_tensor_slices(list(range(5))),
          10,
      ),
  )
  def test_create_value_sequence_with_reduce_sum(
      self, dataset_factory, expected_result
  ):
    dataset = dataset_factory()
    sequence = list(dataset.as_numpy_iterator())
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    element_type = tensorflow_types.to_type(dataset.element_spec)
    sequence_type = federated_language.SequenceType(element_type)
    arg_value_pb, _ = federated_language_executor.serialize_value(
        sequence, sequence_type
    )
    arg = executor.create_value(arg_value_pb)

    def sum_examples(ds):
      return ds.reduce(
          tf.constant(0, ds.element_spec.dtype),
          lambda s, x: s + tf.reduce_sum(x),
      )

    proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        sum_examples, sequence_type
    )

    comp_pb = executor_pb2.Value(computation=proto)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    output_pb = executor.materialize(result.ref)
    result, result_type_spec = federated_language_executor.deserialize_value(
        output_pb
    )
    self.assertEqual(
        result_type_spec,
        federated_language.TensorType(sequence_type.element.dtype),
    )
    self.assertEqual(result, expected_result)

  def test_create_tuple_of_value_sequence(self):
    sequences = ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    element_type = federated_language.TensorType(np.int32)
    struct_of_sequence_type = federated_language.StructType([
        federated_language.SequenceType(element_type),
        federated_language.SequenceType(element_type),
    ])
    arg_value_pb, _ = federated_language_executor.serialize_value(
        sequences, struct_of_sequence_type
    )
    arg = executor.create_value(arg_value_pb)

    def preprocess(datasets):
      def double_value(x):
        return 2 * x

      @tf.function
      def add_preprocessing(ds1, ds2):
        return ds1.map(double_value), ds2.map(double_value)

      return add_preprocessing(*datasets)

    proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        preprocess, struct_of_sequence_type
    )

    comp_pb = executor_pb2.Value(computation=proto)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    output_pb = executor.materialize(result.ref)
    _, result_type_spec = federated_language_executor.deserialize_value(
        output_pb, type_hint=struct_of_sequence_type
    )
    self.assertEqual(result_type_spec, struct_of_sequence_type)

  def test_create_struct(self):
    executor = get_executor()
    expected_type_spec = federated_language.TensorType(np.int64, [3])
    value_pb, _ = federated_language_executor.serialize_value(
        np.array([1, 2, 3], np.int64), expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = (
        federated_language_executor.deserialize_value(materialized_value)
    )
    struct_type_spec = federated_language.to_type(
        [expected_type_spec, expected_type_spec]
    )
    self.assertTrue(type_spec.is_equivalent_to(struct_type_spec))
    deserialized_value = federated_language.framework.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Create a struct from the struct and another value.
    new_struct_value = executor.create_struct([struct_value.ref, value.ref])
    materialized_value = executor.materialize(new_struct_value.ref)
    deserialized_value, type_spec = (
        federated_language_executor.deserialize_value(materialized_value)
    )
    struct_type_spec = federated_language.to_type(
        [struct_type_spec, expected_type_spec]
    )
    self.assertTrue(type_spec.is_equivalent_to(struct_type_spec))
    deserialized_value = federated_language.framework.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose([[(1, 2, 3), (1, 2, 3)], (1, 2, 3)], deserialized_value)

  def test_create_selection(self):
    executor = get_executor()
    expected_type_spec = federated_language.TensorType(np.int64, [3])
    value_pb, _ = federated_language_executor.serialize_value(
        np.array([1, 2, 3], np.int64), expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = (
        federated_language_executor.deserialize_value(materialized_value)
    )
    struct_type_spec = federated_language.to_type(
        [expected_type_spec, expected_type_spec]
    )
    self.assertTrue(type_spec.is_equivalent_to(struct_type_spec))
    deserialized_value = federated_language.framework.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Select the first value out of the struct.
    new_value = executor.create_selection(struct_value.ref, 0)
    materialized_value = executor.materialize(new_value.ref)
    deserialized_value, type_spec = (
        federated_language_executor.deserialize_value(materialized_value)
    )
    self.assertTrue(type_spec.is_equivalent_to(expected_type_spec))
    deserialized_value = federated_language.framework.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose((1, 2, 3), deserialized_value)

  def test_call_with_arg(self):
    executor = get_executor()
    value_pb, _ = federated_language_executor.serialize_value(
        np.array([1, 2, 3], np.int64),
        federated_language.TensorType(np.int64, [3]),
    )
    value_ref = executor.create_value(value_pb)
    arg = executor.create_struct((value_ref.ref, value_ref.ref))

    foo, _ = tensorflow_computation_factory.create_binary_operator(
        tf.add,
        federated_language.TensorType(np.int64),
        federated_language.TensorType(np.int64),
    )

    comp_pb = executor_pb2.Value(computation=foo)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = federated_language_executor.deserialize_value(
        result_value_pb
    )
    self.assertAllEqual(result_tensor, [2, 4, 6])

  def test_call_no_arg(self):
    executor = get_executor()

    foo, _ = tensorflow_computation_factory.create_constant(
        123.0, federated_language.TensorType(np.float32)
    )

    comp_pb = executor_pb2.Value(computation=foo)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, None)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = federated_language_executor.deserialize_value(
        result_value_pb
    )
    self.assertEqual(result_tensor, 123.0)

  def test_materialize_on_unkown_fails(self):
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    with self.assertRaisesRegex(Exception, 'NOT_FOUND'):
      executor.materialize(0)


if __name__ == '__main__':
  absltest.main()
