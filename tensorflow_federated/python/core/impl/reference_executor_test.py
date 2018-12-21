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
"""Tests for reference_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import executor_context
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import reference_executor


class ReferenceExecutorTest(absltest.TestCase):

  def test_installed_by_default(self):
    context = context_stack_impl.context_stack.current
    self.assertIsInstance(context, executor_context.ExecutorContext)
    self.assertIsInstance(
        context._executor,  # pylint: disable=protected-access
        reference_executor.ReferenceExecutor)

  def test_tensor_value(self):
    v = reference_executor.TensorValue(10, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertEqual(v.value, 10)

  def test_named_tuple_value_success(self):
    v = reference_executor.NamedTupleValue(
        anonymous_tuple.AnonymousTuple([
            ('x', reference_executor.TensorValue(10, tf.int32)),
            ('y', reference_executor.TensorValue(20, tf.int32)),
        ]), [('x', tf.int32), ('y', tf.int32)])
    self.assertEqual(str(v.type_signature), '<x=int32,y=int32>')
    self.assertCountEqual(dir(v.value), ['x', 'y'])
    self.assertEqual(str(v.value.x.type_signature), 'int32')
    self.assertEqual(v.value.x.value, 10)
    self.assertEqual(str(v.value.y.type_signature), 'int32')
    self.assertEqual(v.value.y.value, 20)

  def test_named_tuple_value_with_bad_elements(self):
    v = anonymous_tuple.AnonymousTuple([('x', 10)])
    with self.assertRaises(TypeError):
      reference_executor.NamedTupleValue(v, [('x', tf.int32)])

  def test_function_value(self):

    def foo(x):
      self.assertIsInstance(x, reference_executor.ComputedValue)
      return reference_executor.TensorValue(str(x.value), tf.string)

    v = reference_executor.FunctionValue(
        foo, computation_types.FunctionType(tf.int32, tf.string))
    self.assertEqual(str(v.type_signature), '(int32 -> string)')
    result = v.value(reference_executor.TensorValue(10, tf.int32))
    self.assertIsInstance(result, reference_executor.TensorValue)
    self.assertEqual(str(result.type_signature), 'string')
    self.assertEqual(result.value, '10')

  def test_stamp_computed_value_into_graph_with_tuples_of_tensors(self):
    v = reference_executor.NamedTupleValue(
        anonymous_tuple.AnonymousTuple(
            [('x', reference_executor.TensorValue(10, tf.int32)),
             ('y',
              reference_executor.NamedTupleValue(
                  anonymous_tuple.AnonymousTuple([
                      ('z', reference_executor.TensorValue(0.6, tf.float32)),
                  ]), [('z', tf.float32)]))]), [('x', tf.int32),
                                                ('y', [('z', tf.float32)])])
    with tf.Graph().as_default() as graph:
      stamped_v = reference_executor.stamp_computed_value_into_graph(v, graph)
      with tf.Session(graph=graph) as sess:
        v_val = graph_utils.fetch_value_in_session(stamped_v, sess)
    self.assertEqual(str(v_val), '<x=10,y=<z=0.6>>')

  def test_to_computed_value_with_tuples_of_constants(self):
    v = reference_executor.to_computed_value(
        anonymous_tuple.AnonymousTuple([
            ('x', 10),
            ('y', anonymous_tuple.AnonymousTuple([('z', 0.6)])),
        ]), [('x', tf.int32), ('y', [('z', tf.float32)])])
    with tf.Graph().as_default() as graph:
      stamped_v = reference_executor.stamp_computed_value_into_graph(v, graph)
      with tf.Session(graph=graph) as sess:
        v_val = graph_utils.fetch_value_in_session(stamped_v, sess)
    self.assertEqual(str(v_val), '<x=10,y=<z=0.6>>')

  def test_to_raw_value_with_tuples_of_constants(self):
    x = reference_executor.NamedTupleValue(
        anonymous_tuple.AnonymousTuple(
            [('x', reference_executor.TensorValue(10, tf.int32)),
             ('y',
              reference_executor.NamedTupleValue(
                  anonymous_tuple.AnonymousTuple([
                      ('z', reference_executor.TensorValue(0.6, tf.float32)),
                  ]), [('z', tf.float32)]))]), [('x', tf.int32),
                                                ('y', [('z', tf.float32)])])
    y = reference_executor.to_raw_value(x)
    self.assertEqual(str(y), '<x=10,y=<z=0.6>>')

  def test_simple_tensorflow_without_datasets(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    self.assertEqual(add_one(10), 11)


if __name__ == '__main__':
  absltest.main()
