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

  def test_computed_value(self):
    v = reference_executor.ComputedValue(10, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertEqual(v.value, 10)

  def test_check_representation_matches_type_with_tensor_value(self):
    reference_executor.check_representation_matches_type(10, tf.int32)
    with self.assertRaises(TypeError):
      reference_executor.check_representation_matches_type(0.1, tf.int32)
    with self.assertRaises(TypeError):
      reference_executor.check_representation_matches_type([], tf.int32)

  def test_check_representation_matches_type_with_named_tuple_value(self):
    reference_executor.check_representation_matches_type(
        anonymous_tuple.AnonymousTuple([('x', 10), ('y', 20)]),
        [('x', tf.int32), ('y', tf.int32)])
    reference_executor.check_representation_matches_type(
        anonymous_tuple.AnonymousTuple([
            ('x', anonymous_tuple.AnonymousTuple([(None, 10), (None, 20)])),
            ('y', 30),
        ]), [('x', [tf.int32, tf.int32]), ('y', tf.int32)])
    with self.assertRaises(TypeError):
      reference_executor.check_representation_matches_type(
          anonymous_tuple.AnonymousTuple([
              ('x', [10, 20]),
              ('y', 30),
          ]), [('x', [tf.int32, tf.int32]), ('y', tf.int32)])
    with self.assertRaises(TypeError):
      reference_executor.check_representation_matches_type(
          10, [tf.int32, tf.int32])

  def test_check_representation_matches_type_with_function_value(self):

    def foo(x):
      self.assertIsInstance(x, reference_executor.ComputedValue)
      return reference_executor.ComputedValue(str(x.value), tf.string)

    reference_executor.check_representation_matches_type(
        foo, computation_types.FunctionType(tf.int32, tf.string))
    with self.assertRaises(TypeError):
      reference_executor.check_representation_matches_type(
          10, computation_types.FunctionType(tf.int32, tf.string))

  def test_stamp_computed_value_into_graph_with_tuples_of_tensors(self):
    v = reference_executor.ComputedValue(
        anonymous_tuple.AnonymousTuple([('x', 10),
                                        ('y',
                                         anonymous_tuple.AnonymousTuple(
                                             [('z', 0.6)]))]),
        [('x', tf.int32), ('y', [('z', tf.float32)])])
    reference_executor.check_representation_matches_type(
        v.value, v.type_signature)
    with tf.Graph().as_default() as graph:
      stamped_v = reference_executor.stamp_computed_value_into_graph(v, graph)
      with tf.Session(graph=graph) as sess:
        v_val = graph_utils.fetch_value_in_session(stamped_v, sess)
    self.assertEqual(str(v_val), '<x=10,y=<z=0.6>>')

  def test_tensorflow_computation_with_one_constant(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    self.assertEqual(foo(10), 11)

  def test_tensorflow_computation_with_two_constants(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    self.assertEqual(foo(10, 20), 30)
    self.assertEqual(foo(20, 10), 30)

  def test_tensorflow_computation_with_one_empty_tuple(self):
    tuple_type = computation_types.NamedTupleType([])

    @computations.tf_computation(tuple_type)
    def foo(z):
      del z  # unused
      return tf.constant(1)

    self.assertEqual(foo(()), 1)

  def test_tensorflow_computation_with_one_tuple_of_one_constant(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
    ])

    # TODO(b/122478509): Handle single-element tuple types in decorators.
    with self.assertRaises(TypeError):

      @computations.tf_computation(tuple_type)
      def foo(x):
        return x + 1

      self.assertEqual(foo((10,)), 11)

  def test_tensorflow_computation_with_one_tuple_of_two_constants(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z.x + z.y

    self.assertEqual(foo((10, 20)), 30)
    self.assertEqual(foo((20, 10)), 30)

  def test_tensorflow_computation_with_one_tuple_of_two_tuples(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])
    tuple_group_type = computation_types.NamedTupleType([
        ('a', tuple_type),
        ('b', tuple_type),
    ])

    @computations.tf_computation(tuple_group_type)
    def foo(z):
      return z.a.x + z.a.y + z.b.x + z.b.y

    self.assertEqual(foo(((10, 20), (30, 40))), 100)
    self.assertEqual(foo(((40, 30), (20, 10))), 100)

  def test_tensorflow_computation_with_two_tuples_of_two_constants(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])

    @computations.tf_computation(tuple_type, tuple_type)
    def foo(a, b):
      return a.x + a.y + b.x + b.y

    self.assertEqual(foo((10, 20), (30, 40)), 100)
    self.assertEqual(foo((40, 30), (20, 10)), 100)


if __name__ == '__main__':
  absltest.main()
