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
import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_executor_bindings
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import value_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils


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


def get_executor(use_tf_executor):
  if use_tf_executor:
    return tensorflow_executor_bindings.create_tensorflow_executor()
  else:
    # Empty string for device name and mesh
    return tensorflow_executor_bindings.create_dtensor_executor('', '', -1)


class TensorFlowExecutorBindingsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_construction(self, use_tf_executor):
    try:
      get_executor(use_tf_executor)
    except Exception:  # pylint: disable=broad-except
      self.fail('Raised `Exception` unexpectedly.')

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create_value(self, use_tf_executor):
    executor = get_executor(use_tf_executor)
    # 1. Test a simple tensor.
    expected_type_spec = computation_types.TensorType(np.int64, [3])
    value_pb, _ = value_serialization.serialize_value(
        [1, 2, 3], expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    self.assertEqual(value.ref, 0)
    self.assertEqual(str(value), '0')
    self.assertEqual(repr(value), r'<OwnedValueId: 0>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    type_test_utils.assert_types_identical(type_spec, expected_type_spec)
    self.assertAllEqual(deserialized_value, [1, 2, 3])
    # 2. Test a struct of tensors, ensure that we get a different ID.
    expected_type_spec = computation_types.StructType([
        ('a', computation_types.TensorType(np.int64, [3])),
        ('b', computation_types.TensorType(np.float32, [])),
    ])
    value_pb, _ = value_serialization.serialize_value(
        collections.OrderedDict(a=tf.constant([1, 2, 3]), b=tf.constant(42.0)),
        expected_type_spec,
    )
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented.
    self.assertEqual(value.ref, 1)
    self.assertEqual(str(value), '1')
    self.assertEqual(repr(value), r'<OwnedValueId: 1>')
    materialized_value = executor.materialize(value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    # Note: here we've lost the names `a` and `b` in the output. The output
    # is a more _strict_ type.
    self.assertTrue(expected_type_spec.is_assignable_from(type_spec))
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, expected_type_spec
    )
    self.assertAllClose(
        deserialized_value, collections.OrderedDict(a=(1, 2, 3), b=42.0)
    )

    # 3. Test creating a value from a computation.
    foo, _ = tensorflow_computation_factory.create_binary_operator(
        tf.add,
        computation_types.TensorType(np.int64),
        computation_types.TensorType(np.int64),
    )

    value_pb = executor_pb2.Value(computation=foo)
    value = executor.create_value(value_pb)
    self.assertIsInstance(value, executor_bindings.OwnedValueId)
    # Assert the value ID was incremented again.
    self.assertEqual(value.ref, 2)
    self.assertEqual(str(value), '2')
    self.assertEqual(repr(value), '<OwnedValueId: 2>')
    # Note: functions are not materializable, no addition assertions.

  @parameterized.named_parameters(
      ('range', lambda: tf.data.Dataset.range(5)),
      ('shuffled_range', lambda: tf.data.Dataset.range(5).shuffle(3)),
      (
          'mapped_with_resource_range',
          lambda: tf.data.Dataset.range(5).map(_test_map_integers),
      ),
      ('mapped_range', lambda: tf.data.Dataset.range(5).map(lambda x: x)),
      (
          'batched_range',
          lambda: tf.data.Dataset.range(5).batch(2, drop_remainder=False),
      ),
      (
          'tensor_slices',
          lambda: tf.data.Dataset.from_tensor_slices(list(range(5))),
      ),
  )
  def test_create_value_sequence(self, dataset_factory):
    dataset = dataset_factory()
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    element_type = computation_types.tensorflow_to_type(dataset.element_spec)
    sequence_type = computation_types.SequenceType(element_type)
    arg_value_pb, _ = value_serialization.serialize_value(
        dataset, sequence_type
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
    result, result_type_spec = value_serialization.deserialize_value(output_pb)
    type_test_utils.assert_types_identical(
        result_type_spec,
        computation_types.TensorType(sequence_type.element.dtype),
    )
    self.assertEqual(result, sum(range(5)))

  def test_create_tuple_of_value_sequence(self):
    datasets = (tf.data.Dataset.range(5), tf.data.Dataset.range(5))
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    element_type = computation_types.tensorflow_to_type(
        datasets[0].element_spec
    )
    struct_of_sequence_type = computation_types.StructType([
        computation_types.SequenceType(element_type),
        computation_types.SequenceType(element_type),
    ])
    arg_value_pb, _ = value_serialization.serialize_value(
        datasets, struct_of_sequence_type
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
    _, result_type_spec = value_serialization.deserialize_value(
        output_pb, type_hint=struct_of_sequence_type
    )
    type_test_utils.assert_types_identical(
        result_type_spec, struct_of_sequence_type
    )

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create_struct(self, use_tf_executor):
    executor = get_executor(use_tf_executor)
    expected_type_spec = computation_types.TensorType(np.int64, [3])
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec]
    )
    type_test_utils.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Create a struct from the struct and another value.
    new_struct_value = executor.create_struct([struct_value.ref, value.ref])
    materialized_value = executor.materialize(new_struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    struct_type_spec = computation_types.to_type(
        [struct_type_spec, expected_type_spec]
    )
    type_test_utils.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose([[(1, 2, 3), (1, 2, 3)], (1, 2, 3)], deserialized_value)

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_create_selection(self, use_tf_executor):
    executor = get_executor(use_tf_executor)
    expected_type_spec = computation_types.TensorType(np.int64, [3])
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]), expected_type_spec
    )
    value = executor.create_value(value_pb)
    self.assertEqual(value.ref, 0)
    # 1. Create a struct from duplicated values.
    struct_value = executor.create_struct([value.ref, value.ref])
    self.assertEqual(struct_value.ref, 1)
    materialized_value = executor.materialize(struct_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    struct_type_spec = computation_types.to_type(
        [expected_type_spec, expected_type_spec]
    )
    type_test_utils.assert_types_equivalent(type_spec, struct_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose([(1, 2, 3), (1, 2, 3)], deserialized_value)
    # 2. Select the first value out of the struct.
    new_value = executor.create_selection(struct_value.ref, 0)
    materialized_value = executor.materialize(new_value.ref)
    deserialized_value, type_spec = value_serialization.deserialize_value(
        materialized_value
    )
    type_test_utils.assert_types_equivalent(type_spec, expected_type_spec)
    deserialized_value = type_conversions.type_to_py_container(
        deserialized_value, struct_type_spec
    )
    self.assertAllClose((1, 2, 3), deserialized_value)

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_call_with_arg(self, use_tf_executor):
    executor = get_executor(use_tf_executor)
    value_pb, _ = value_serialization.serialize_value(
        tf.constant([1, 2, 3]),
        computation_types.TensorType(np.int64, [3]),
    )
    value_ref = executor.create_value(value_pb)
    arg = executor.create_struct((value_ref.ref, value_ref.ref))

    foo, _ = tensorflow_computation_factory.create_binary_operator(
        tf.add,
        computation_types.TensorType(np.int64),
        computation_types.TensorType(np.int64),
    )

    comp_pb = executor_pb2.Value(computation=foo)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, arg.ref)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertAllEqual(result_tensor, [2, 4, 6])

  @parameterized.named_parameters(
      ('tf_executor', True),
      ('dtensor_executor', False),
  )
  def test_call_no_arg(self, use_tf_executor=True):
    executor = get_executor(use_tf_executor)

    foo, _ = tensorflow_computation_factory.create_constant(
        123.0, computation_types.TensorType(np.float32)
    )

    comp_pb = executor_pb2.Value(computation=foo)
    comp = executor.create_value(comp_pb)
    result = executor.create_call(comp.ref, None)
    result_value_pb = executor.materialize(result.ref)
    result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
    self.assertEqual(result_tensor, 123.0)

  def test_materialize_on_unkown_fails(self):
    executor = tensorflow_executor_bindings.create_tensorflow_executor()
    with self.assertRaisesRegex(Exception, 'NOT_FOUND'):
      executor.materialize(0)


class DtensorExecutorBindingTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('sharded_input', True),
      ('replicated_input', False),
  )
  def test_call_with_arg_dtensor_executor_with_mesh(self, sharded_input):
    mesh_dim_name = 'batch'
    mesh = tf.experimental.dtensor.create_mesh(
        devices=['CPU:%d' % i for i in range(8)], mesh_dims=[(mesh_dim_name, 8)]
    )
    # dtensor.run_on method is used to set mesh for the dtensor device.
    with tf.experimental.dtensor.run_on(mesh):
      executor = tensorflow_executor_bindings.create_dtensor_executor(
          tf.experimental.dtensor.device_name(), mesh.to_string(), -1
      )
      value_pb, _ = value_serialization.serialize_value(
          tf.constant([1, 2, 3, 4, 5, 6, 7, 8]),
          computation_types.TensorType(np.int64, [8]),
      )

      value_ref = executor.create_value(value_pb)
      arg = executor.create_struct((value_ref.ref, value_ref.ref))

      mesh_dim_name = 'batch'
      spec = {}
      if sharded_input:
        spec['arg_a'] = mesh_dim_name
      else:
        spec['arg_a'] = 'unsharded'
      layout_map = computation_pb2.TensorFlow.LayoutMap(
          name_to_sharding_spec=spec
      )

      proto, _ = tensorflow_computation_factory.create_binary_operator(
          tf.add,
          computation_types.TensorType(np.int64),
          computation_types.TensorType(np.int64),
          layout_map,
      )

      comp_pb = executor_pb2.Value(computation=proto)
      comp = executor.create_value(comp_pb)
      result = executor.create_call(comp.ref, arg.ref)
      result_value_pb = executor.materialize(result.ref)
      result_tensor, _ = value_serialization.deserialize_value(result_value_pb)
      self.assertAllEqual(result_tensor, [2, 4, 6, 8, 10, 12, 14, 16])


if __name__ == '__main__':
  absltest.main()
