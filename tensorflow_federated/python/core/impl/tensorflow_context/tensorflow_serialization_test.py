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
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_serialization
from tensorflow_federated.python.core.impl.types import type_serialization


def _tf_computation_serializer(fn, parameter_type, context):
  serializer = tensorflow_serialization.tf_computation_serializer(
      parameter_type, context)
  arg_to_fn = next(serializer)
  result = fn(arg_to_fn)
  return serializer.send(result)


class TensorFlowSerializationTest(test.TestCase):

  def test_serialize_tensorflow_with_no_parameter(self):
    comp, extra_type_spec = _tf_computation_serializer(
        lambda _: tf.constant(99), None, context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)), '( -> int32)')
    self.assertEqual(str(extra_type_spec), '( -> int32)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    results = tf.compat.v1.Session().run(
        tf.import_graph_def(
            serialization_utils.unpack_graph_def(comp.tensorflow.graph_def),
            None, [comp.tensorflow.result.tensor.tensor_name]))
    self.assertEqual(results, [99])

  def test_serialize_tensorflow_with_table_no_variables(self):

    def table_lookup(word):
      table = tf.lookup.StaticVocabularyTable(
          tf.lookup.KeyValueTensorInitializer(['a', 'b', 'c'],
                                              np.arange(3, dtype=np.int64)),
          num_oov_buckets=1)
      return table.lookup(word)

    comp, extra_type_spec = _tf_computation_serializer(
        table_lookup,
        computation_types.TensorType(dtype=tf.string, shape=(None,)),
        context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)),
        '(string[?] -> int64[?])')
    self.assertEqual(str(extra_type_spec), '(string[?] -> int64[?])')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')

    with tf.Graph().as_default() as g:
      tf.import_graph_def(
          serialization_utils.unpack_graph_def(comp.tensorflow.graph_def),
          name='')
    with tf.compat.v1.Session(graph=g) as sess:
      sess.run(fetches=comp.tensorflow.initialize_op)
      results = sess.run(
          fetches=comp.tensorflow.result.tensor.tensor_name,
          feed_dict={
              comp.tensorflow.parameter.tensor.tensor_name: ['b', 'c', 'a']
          })
    self.assertAllEqual(results, [1, 2, 0])

  @test.graph_mode_test
  def test_serialize_tensorflow_with_simple_add_three_lambda(self):
    comp, extra_type_spec = _tf_computation_serializer(
        lambda x: x + 3, computation_types.TensorType(tf.int32),
        context_stack_impl.context_stack)
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
    comp, extra_type_spec = _tf_computation_serializer(
        lambda z: output_type(2.0 * tf.cast(z.x, tf.float32), 3.0 * z.y),
        computation_types.StructWithPythonType([('x', tf.int32),
                                                ('y', (tf.float32, [2]))],
                                               batch_type),
        context_stack_impl.context_stack)
    self.assertEqual(
        str(type_serialization.deserialize_type(comp.type)),
        '(<x=int32,y=float32[2]> -> <A=float32,B=float32[2]>)')
    self.assertEqual(comp.WhichOneof('computation'), 'tensorflow')
    self.assertEqual(
        str(extra_type_spec),
        '(<x=int32,y=float32[2]> -> <A=float32,B=float32[2]>)')
    self.assertIsInstance(extra_type_spec.parameter,
                          computation_types.StructWithPythonType)
    self.assertIs(extra_type_spec.parameter.python_container, batch_type)
    self.assertIsInstance(extra_type_spec.result,
                          computation_types.StructWithPythonType)
    self.assertIs(extra_type_spec.result.python_container, output_type)

  @test.graph_mode_test
  def test_serialize_tensorflow_with_data_set_sum_lambda(self):

    def _legacy_dataset_reducer_example(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    comp, extra_type_spec = _tf_computation_serializer(
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


if __name__ == '__main__':
  test.main()
