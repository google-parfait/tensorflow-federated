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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


class TensorFlowSerializationTest(test_case.TestCase):

  def assert_serializes(self, fn, parameter_type, expected_fn_type_str):
    serializer = tensorflow_serialization.tf_computation_serializer(
        parameter_type, context_stack_impl.context_stack)
    arg_to_fn = next(serializer)
    result = fn(arg_to_fn)
    comp, extra_type_spec = serializer.send(result)
    deserialized_type = type_serialization.deserialize_type(comp.type)
    self.assert_types_equivalent(deserialized_type, extra_type_spec)
    self.assert_type_string(deserialized_type, expected_fn_type_str)
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    return comp.tensorflow, extra_type_spec

  def test_serialize_tensorflow_with_no_parameter(self):
    tf_proto, _ = self.assert_serializes(lambda _: tf.constant(99), None,
                                         '( -> int32)')
    results = tf.compat.v1.Session().run(
        tf.graph_util.import_graph_def(
            serialization_utils.unpack_graph_def(tf_proto.graph_def), None,
            [tf_proto.result.tensor.tensor_name]))
    self.assertEqual(results, [99])

  def test_serialize_tensorflow_with_table_no_variables(self):

    def table_lookup(word):
      table = tf.lookup.StaticVocabularyTable(
          tf.lookup.KeyValueTensorInitializer(['a', 'b', 'c'],
                                              np.arange(3, dtype=np.int64)),
          num_oov_buckets=1)
      return table.lookup(word)

    tf_proto, _ = self.assert_serializes(
        table_lookup,
        computation_types.TensorType(dtype=tf.string, shape=(None,)),
        '(string[?] -> int64[?])')

    with tf.Graph().as_default() as g:
      tf.graph_util.import_graph_def(
          serialization_utils.unpack_graph_def(tf_proto.graph_def), name='')
    with tf.compat.v1.Session(graph=g) as sess:
      sess.run(fetches=tf_proto.initialize_op)
      results = sess.run(
          fetches=tf_proto.result.tensor.tensor_name,
          feed_dict={tf_proto.parameter.tensor.tensor_name: ['b', 'c', 'a']})
    self.assertAllEqual(results, [1, 2, 0])

  @test_utils.graph_mode_test
  def test_serialize_tensorflow_with_simple_add_three_lambda(self):
    tf_proto, _ = self.assert_serializes(lambda x: x + 3,
                                         computation_types.TensorType(tf.int32),
                                         '(int32 -> int32)')
    parameter = tf.constant(1000)
    results = tf.compat.v1.Session().run(
        tf.graph_util.import_graph_def(
            serialization_utils.unpack_graph_def(tf_proto.graph_def),
            {tf_proto.parameter.tensor.tensor_name: parameter},
            [tf_proto.result.tensor.tensor_name]))
    self.assertEqual(results, [1003])

  @test_utils.graph_mode_test
  def test_serialize_tensorflow_with_structured_type_signature(self):
    batch_type = collections.namedtuple('BatchType', ['x', 'y'])
    output_type = collections.namedtuple('OutputType', ['A', 'B'])
    _, extra_type_spec = self.assert_serializes(
        lambda z: output_type(2.0 * tf.cast(z.x, tf.float32), 3.0 * z.y),
        computation_types.StructWithPythonType([('x', tf.int32),
                                                ('y', (tf.float32, [2]))],
                                               batch_type),
        '(<x=int32,y=float32[2]> -> <A=float32,B=float32[2]>)')
    self.assertIsInstance(extra_type_spec.parameter,
                          computation_types.StructWithPythonType)
    self.assertIs(extra_type_spec.parameter.python_container, batch_type)
    self.assertIsInstance(extra_type_spec.result,
                          computation_types.StructWithPythonType)
    self.assertIs(extra_type_spec.result.python_container, output_type)

  @test_utils.graph_mode_test
  def test_serialize_tensorflow_with_data_set_sum_lambda(self):

    def _legacy_dataset_reducer_example(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    tf_proto, _ = self.assert_serializes(
        _legacy_dataset_reducer_example,
        computation_types.SequenceType(tf.int64), '(int64* -> int64)')
    parameter = tf.data.Dataset.range(5)
    results = tf.compat.v1.Session().run(
        tf.graph_util.import_graph_def(
            serialization_utils.unpack_graph_def(tf_proto.graph_def), {
                tf_proto.parameter.sequence.variant_tensor_name:
                    tf.data.experimental.to_variant(parameter)
            }, [tf_proto.result.tensor.tensor_name]))
    self.assertEqual(results, [10])


if __name__ == '__main__':
  test_case.main()
