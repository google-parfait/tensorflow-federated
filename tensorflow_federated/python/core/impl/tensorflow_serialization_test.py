# Lint as: python3
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

import collections
from typing import List, Set

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl

tf.compat.v1.enable_v2_behavior()


class TensorFlowSerializationTest(test.TestCase):

  def test_serialize_tensorflow_with_no_parameter(self):
    comp, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        lambda: tf.constant(99), None, context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)), '( -> int32)')
    self.assertEqual(str(extra_type_spec), '( -> int32)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    results = tf.compat.v1.Session().run(
        tf.import_graph_def(
            serialization_utils.unpack_graph_def(comp.tensorflow.graph_def),
            None, [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [99])

  @test.graph_mode_test
  def test_serialize_tensorflow_with_simple_add_three_lambda(self):
    comp, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        lambda x: x + 3, tf.int32, context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)), '(int32 -> int32)')
    self.assertEqual(str(extra_type_spec), '(int32 -> int32)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    parameter = tf.constant(1000)
    results = tf.compat.v1.Session().run(
        tf.import_graph_def(
            serialization_utils.unpack_graph_def(comp.tensorflow.graph_def),
            {comp.tensorflow.parameter.tensor.tensor_name: parameter},
            [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [1003])

  @test.graph_mode_test
  def test_serialize_tensorflow_with_structured_type_signature(self):
    batch_type = collections.namedtuple('BatchType', ['x', 'y'])
    output_type = collections.namedtuple('OutputType', ['A', 'B'])
    comp, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        lambda z: output_type(2.0 * tf.cast(z.x, tf.float32), 3.0 * z.y),
        batch_type(tf.int32, (tf.float32, [2])),
        context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)),
        '(<x=int32,y=float32[2]> -> <A=float32,B=float32[2]>)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    self.assertEqual(
        str(extra_type_spec),
        '(<x=int32,y=float32[2]> -> <A=float32,B=float32[2]>)')
    self.assertIsInstance(extra_type_spec.parameter,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            extra_type_spec.parameter), batch_type)
    self.assertIsInstance(extra_type_spec.result,
                          computation_types.NamedTupleTypeWithPyContainerType)
    self.assertIs(
        computation_types.NamedTupleTypeWithPyContainerType.get_container_type(
            extra_type_spec.result), output_type)

  @test.graph_mode_test
  def test_serialize_tensorflow_with_data_set_sum_lambda(self):

    def _legacy_dataset_reducer_example(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    comp, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        _legacy_dataset_reducer_example,
        computation_types.SequenceType(tf.int64),
        context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)),
        '(int64* -> int64)')
    self.assertEqual(str(extra_type_spec), '(int64* -> int64)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    parameter = tf.data.Dataset.range(5)
    results = tf.compat.v1.Session().run(
        tf.import_graph_def(
            serialization_utils.unpack_graph_def(comp.tensorflow.graph_def), {
                comp.tensorflow.parameter.sequence.variant_tensor_name:
                    tf.data.experimental.to_variant(parameter)
            }, [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [10])

  def assertNodeDefListDoesNotContainOps(self,
                                         nodedefs: List[tf.compat.v1.NodeDef],
                                         forbidden_ops: Set[str]):
    for node in nodedefs:
      if node.op in forbidden_ops:
        self.fail('[{n}] node was found but not allowed.'.format(n=node.op))

  def assertGraphDoesNotContainOps(self, graphdef: tf.compat.v1.GraphDef,
                                   forbidden_ops: Set[str]):
    self.assertNodeDefListDoesNotContainOps(graphdef.node, forbidden_ops)
    for function in graphdef.library.function:
      self.assertNodeDefListDoesNotContainOps(function.node_def, forbidden_ops)

  @test.graph_mode_test
  def test_serialize_tensorflow_with_dataset_not_optimized(self):

    @tf.function
    def test_foo(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    def legacy_dataset_reducer_example(ds):
      return test_foo(ds)

    comp, extra_type_spec = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        legacy_dataset_reducer_example,
        computation_types.SequenceType(tf.int64),
        context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)),
        '(int64* -> int64)')
    self.assertEqual(str(extra_type_spec), '(int64* -> int64)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    parameter = tf.data.Dataset.range(5)

    graph_def = serialization_utils.unpack_graph_def(comp.tensorflow.graph_def)
    self.assertGraphDoesNotContainOps(graph_def,
                                      ['OptimizeDataset', 'ModelDataste'])
    results = tf.compat.v1.Session().run(
        tf.import_graph_def(
            graph_def, {
                comp.tensorflow.parameter.sequence.variant_tensor_name:
                    tf.data.experimental.to_variant(parameter)
            }, [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [10])


class DatasetSerializationTest(test.TestCase):

  def test_serialize_sequence_not_a_dataset(self):
    with self.assertRaisesRegex(TypeError, r'Expected .*Dataset.* found int'):
      _ = tensorflow_serialization.serialize_dataset(5)

  def test_serialize_sequence_bytes_too_large(self):
    with self.assertRaisesRegex(ValueError,
                                r'Serialized size .* exceeds maximum allowed'):
      _ = tensorflow_serialization.serialize_dataset(
          tf.data.Dataset.range(5), max_serialized_size_bytes=0)

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_roundtrip_sequence_of_scalars(self):
    x = tf.data.Dataset.range(5).map(lambda x: x * 2)
    serialized_bytes = tensorflow_serialization.serialize_dataset(x)
    y = tensorflow_serialization.deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual([y_val for y_val in y], [x * 2 for x in range(5)])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_roundtrip_sequence_of_tuples(self):
    x = tf.data.Dataset.range(5).map(
        lambda x: (x * 2, tf.cast(x, tf.int32), tf.cast(x - 1, tf.float32)))
    serialized_bytes = tensorflow_serialization.serialize_dataset(x)
    y = tensorflow_serialization.deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(
        self.evaluate([y_val for y_val in y]),
        [(x * 2, x, x - 1.) for x in range(5)])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_roundtrip_sequence_of_singleton_tuples(self):
    x = tf.data.Dataset.range(5).map(lambda x: (x,))
    serialized_bytes = tensorflow_serialization.serialize_dataset(x)
    y = tensorflow_serialization.deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    expected_values = [(x,) for x in range(5)]
    actual_values = self.evaluate([y_val for y_val in y])
    self.assertAllEqual(expected_values, actual_values)

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_roundtrip_sequence_of_namedtuples(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['a', 'b', 'c'])

    def make_test_tuple(x):
      return test_tuple_type(
          a=x * 2, b=tf.cast(x, tf.int32), c=tf.cast(x - 1, tf.float32))

    x = tf.data.Dataset.range(5).map(make_test_tuple)
    serialized_bytes = tensorflow_serialization.serialize_dataset(x)
    y = tensorflow_serialization.deserialize_dataset(serialized_bytes)

    self.assertEqual(x.element_spec, y.element_spec)
    self.assertAllEqual(
        self.evaluate([y_val for y_val in y]),
        [test_tuple_type(a=x * 2, b=x, c=x - 1.) for x in range(5)])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
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
    serialzied_bytes = tensorflow_serialization.serialize_dataset(x)
    y = tensorflow_serialization.deserialize_dataset(serialzied_bytes)

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

    actual_values = self.evaluate([y_val for y_val in y])
    expected_values = [_build_expected_structure(x) for x in range(5)]
    for actual, expected in zip(actual_values, expected_values):
      self.assertAllClose(actual, expected)


if __name__ == '__main__':
  test.main()
