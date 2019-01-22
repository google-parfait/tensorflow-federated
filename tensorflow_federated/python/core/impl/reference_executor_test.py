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

import collections

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import reference_executor


class ReferenceExecutorTest(absltest.TestCase):

  def test_installed_by_default(self):
    context = context_stack_impl.context_stack.current
    self.assertIsInstance(context, reference_executor.ReferenceExecutor)

  def test_computed_value(self):
    v = reference_executor.ComputedValue(10, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertEqual(v.value, 10)

  def test_to_representation_for_type_with_tensor_type(self):
    self.assertEqual(
        reference_executor.to_representation_for_type(10, tf.int32), 10)
    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type(0.1, tf.int32)
    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type([], tf.int32)

  def test_to_representation_for_type_with_named_tuple_type(self):
    foo = anonymous_tuple.AnonymousTuple([('x', 10), ('y', 20)])
    self.assertEqual(
        reference_executor.to_representation_for_type(foo, [('x', tf.int32),
                                                            ('y', tf.int32)]),
        foo)
    foo = anonymous_tuple.AnonymousTuple([
        ('x', anonymous_tuple.AnonymousTuple([(None, 10), (None, 20)])),
        ('y', 30),
    ])
    self.assertEqual(
        reference_executor.to_representation_for_type(
            foo, [('x', [tf.int32, tf.int32]), ('y', tf.int32)]), foo)
    self.assertEqual(
        reference_executor.to_representation_for_type(
            anonymous_tuple.AnonymousTuple([('x', [10, 20]), ('y', 30)]),
            [('x', [tf.int32, tf.int32]), ('y', tf.int32)]),
        anonymous_tuple.AnonymousTuple([('x',
                                         anonymous_tuple.AnonymousTuple(
                                             [(None, 10), (None, 20)])),
                                        ('y', 30)]))
    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type(10, [tf.int32, tf.int32])

  def test_to_representation_for_type_with_sequence_type(self):
    foo = [1, 2, 3]
    self.assertEqual(
        reference_executor.to_representation_for_type(
            foo, computation_types.SequenceType(tf.int32)), foo)

  def test_to_representation_for_type_with_function_type(self):

    def foo(x):
      self.assertIsInstance(x, reference_executor.ComputedValue)
      return reference_executor.ComputedValue(str(x.value), tf.string)

    self.assertIs(
        reference_executor.to_representation_for_type(
            foo, computation_types.FunctionType(
                tf.int32, tf.string), lambda x, t: x), foo)

    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type(
          foo, computation_types.FunctionType(tf.int32, tf.string))

    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type(
          10, computation_types.FunctionType(tf.int32, tf.string))

  def test_to_representation_for_type_with_abstract_type(self):
    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type(
          10, computation_types.AbstractType('T'))

  def test_to_representation_for_type_with_placement_type(self):
    self.assertIs(
        reference_executor.to_representation_for_type(
            placements.CLIENTS, computation_types.PlacementType()),
        placements.CLIENTS)

  def test_to_representation_for_type_with_federated_type(self):
    self.assertEqual(
        reference_executor.to_representation_for_type(
            10,
            computation_types.FederatedType(
                tf.int32, placements.SERVER, all_equal=True)), 10)
    x = [1, 2, 3]
    self.assertEqual(
        reference_executor.to_representation_for_type(
            x,
            computation_types.FederatedType(
                tf.int32, placements.CLIENTS, all_equal=False)), x)

  def test_stamp_computed_value_into_graph_with_tuples_of_tensors(self):
    v_val = anonymous_tuple.AnonymousTuple([('x', 10),
                                            ('y',
                                             anonymous_tuple.AnonymousTuple(
                                                 [('z', 0.6)]))])
    v_type = [('x', tf.int32), ('y', [('z', tf.float32)])]
    v = reference_executor.ComputedValue(
        reference_executor.to_representation_for_type(v_val, v_type), v_type)
    with tf.Graph().as_default() as graph:
      stamped_v = reference_executor.stamp_computed_value_into_graph(v, graph)
      with tf.Session(graph=graph) as sess:
        v_val = graph_utils.fetch_value_in_session(stamped_v, sess)
    self.assertEqual(str(v_val), '<x=10,y=<z=0.6>>')

  def test_computation_context(self):
    c1 = reference_executor.ComputationContext()
    c2 = reference_executor.ComputationContext(
        c1, {'foo': reference_executor.ComputedValue(10, tf.int32)})
    c3 = reference_executor.ComputationContext(
        c2, {'bar': reference_executor.ComputedValue(11, tf.int32)})
    c4 = reference_executor.ComputationContext(c3)
    c5 = reference_executor.ComputationContext(
        c4, {'foo': reference_executor.ComputedValue(12, tf.int32)})
    self.assertRaises(ValueError, c1.resolve_reference, 'foo')
    self.assertEqual(c2.resolve_reference('foo').value, 10)
    self.assertEqual(c3.resolve_reference('foo').value, 10)
    self.assertEqual(c4.resolve_reference('foo').value, 10)
    self.assertEqual(c5.resolve_reference('foo').value, 12)
    self.assertRaises(ValueError, c1.resolve_reference, 'bar')
    self.assertRaises(ValueError, c2.resolve_reference, 'bar')
    self.assertEqual(c3.resolve_reference('bar').value, 11)
    self.assertEqual(c4.resolve_reference('bar').value, 11)
    self.assertEqual(c5.resolve_reference('bar').value, 11)

  def test_tensorflow_computation_with_constant(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    self.assertEqual(foo(10), 11)

  def test_tensorflow_computation_with_constants(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def foo(x, y):
      return x + y

    self.assertEqual(foo(10, 20), 30)
    self.assertEqual(foo(20, 10), 30)

  def test_tensorflow_computation_with_empty_tuple(self):
    tuple_type = computation_types.NamedTupleType([])

    @computations.tf_computation(tuple_type)
    def foo(z):
      del z  # unused
      return tf.constant(1)

    self.assertEqual(foo(()), 1)

  def test_tensorflow_computation_with_tuple_of_one_constant(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
    ])

    # TODO(b/122478509): Handle single-element tuple types in decorators.
    with self.assertRaises(TypeError):

      @computations.tf_computation(tuple_type)
      def foo(x):
        return x + 1

      self.assertEqual(foo((10,)), 11)

  def test_tensorflow_computation_with_tuple_of_constants(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z.x + z.y

    self.assertEqual(foo((10, 20)), 30)
    self.assertEqual(foo((20, 10)), 30)

  def test_tensorflow_computation_with_tuple_of_empty_tuples(self):
    tuple_type = computation_types.NamedTupleType([])
    tuple_group_type = computation_types.NamedTupleType([
        ('a', tuple_type),
        ('b', tuple_type),
    ])

    @computations.tf_computation(tuple_group_type)
    def foo(z):
      del z  # unused
      return tf.constant(1)

    self.assertEqual(foo(((), ())), 1)

  def test_tensorflow_computation_with_tuple_of_tuples(self):
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

  def test_tensorflow_computation_with_tuple_of_sequences(self):
    sequence_type = computation_types.SequenceType(tf.int64)
    tuple_type = computation_types.NamedTupleType([
        ('a', sequence_type),
        ('b', sequence_type),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      value1 = z.a.reduce(np.int64(0), lambda x, y: x + y)
      value2 = z.b.reduce(np.int64(0), lambda x, y: x + y)
      return value1 + value2

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.range(5)

    @computations.federated_computation
    def baz():
      return foo((bar(), bar()))

    self.assertEqual(baz(), 20)

  def test_tensorflow_computation_with_tuples_of_constants(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])

    @computations.tf_computation(tuple_type, tuple_type)
    def foo(a, b):
      return a.x + a.y + b.x + b.y

    self.assertEqual(foo((10, 20), (30, 40)), 100)
    self.assertEqual(foo((40, 30), (20, 10)), 100)

  def test_tensorflow_computation_with_empty_sequence(self):
    sequence_type = computation_types.SequenceType(tf.int64)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      del ds  # unused
      return tf.constant(1)

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.range(0)

    @computations.federated_computation
    def baz():
      return foo(bar())

    self.assertEqual(baz(), 1)

  def test_tensorflow_computation_with_sequence_of_one_constant(self):
    sequence_type = computation_types.SequenceType(tf.int64)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y) + 1

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.range(10, 11)

    @computations.federated_computation
    def baz():
      return foo(bar())

    self.assertEqual(baz(), 11)

  def test_tensorflow_computation_with_sequence_of_constants(self):
    sequence_type = computation_types.SequenceType(tf.int64)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.range(5)

    @computations.federated_computation
    def baz():
      return foo(bar())

    self.assertEqual(baz(), 10)

  def test_tensorflow_computation_with_sequence_of_tuples(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])
    sequence_type = computation_types.SequenceType(tuple_type)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(np.int32(0), lambda x, y: x + y['x'] + y['y'])

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.from_tensor_slices(
          collections.OrderedDict([
              ('x', [10, 30]),
              ('y', [20, 40]),
          ]))

    @computations.federated_computation
    def baz():
      return foo(bar())

    self.assertEqual(baz(), 100)

  def test_tensorflow_computation_with_sequences_of_constants(self):
    sequence_type = computation_types.SequenceType(tf.int64)

    @computations.tf_computation(sequence_type, sequence_type)
    def foo(ds1, ds2):
      value1 = ds1.reduce(np.int64(0), lambda x, y: x + y)
      value2 = ds2.reduce(np.int64(0), lambda x, y: x + y)
      return value1 + value2

    @computations.tf_computation
    def bar():
      return tf.data.Dataset.range(5)

    @computations.federated_computation
    def baz():
      return foo(bar(), bar())

    self.assertEqual(baz(), 20)

  def test_tensorflow_computation_with_simple_lambda(self):

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @computations.federated_computation(tf.int32)
    def add_two(x):
      return add_one(add_one(x))

    self.assertEqual(add_two(5), 7)

  def test_tensorflow_computation_with_lambda_and_selection(self):

    @computations.federated_computation(tf.int32,
                                        computation_types.FunctionType(
                                            tf.int32, tf.int32))
    def apply_twice(x, f):
      return f(f(x))

    add_one = computations.tf_computation(lambda x: x + 1, tf.int32)

    self.assertEqual(apply_twice(5, add_one), 7)

  def test_execute_with_nested_lambda(self):
    int32_add = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(tf.add, [tf.int32, tf.int32])))

    curried_int32_add = computation_building_blocks.Lambda(
        'x', tf.int32,
        computation_building_blocks.Lambda(
            'y', tf.int32,
            computation_building_blocks.Call(
                int32_add,
                computation_building_blocks.Tuple(
                    [(None, computation_building_blocks.Reference(
                        'x', tf.int32)),
                     (None, computation_building_blocks.Reference(
                         'y', tf.int32))]))))

    make_10 = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(lambda: tf.constant(10))))

    add_10 = computation_building_blocks.Call(
        curried_int32_add, computation_building_blocks.Call(make_10))

    add_10_computation = computation_impl.ComputationImpl(
        add_10.proto, context_stack_impl.context_stack)

    self.assertEqual(add_10_computation(5), 15)

  def test_execute_with_block(self):
    add_one = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(lambda x: x + 1, tf.int32)))

    make_10 = computation_building_blocks.ComputationBuildingBlock.from_proto(
        computation_impl.ComputationImpl.get_proto(
            computations.tf_computation(lambda: tf.constant(10))))

    make_13 = computation_building_blocks.Block(
        [('x', computation_building_blocks.Call(make_10)),
         ('x',
          computation_building_blocks.Call(
              add_one, computation_building_blocks.Reference('x', tf.int32))),
         ('x',
          computation_building_blocks.Call(
              add_one, computation_building_blocks.Reference('x', tf.int32))),
         ('x',
          computation_building_blocks.Call(
              add_one, computation_building_blocks.Reference('x', tf.int32)))],
        computation_building_blocks.Reference('x', tf.int32))

    make_13_computation = computation_impl.ComputationImpl(
        make_13.proto, context_stack_impl.context_stack)

    self.assertEqual(make_13_computation(), 13)

  def test_intrinsic(self):

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def foo(x):
      return intrinsics.sequence_sum(x)

    self.assertEqual(foo([1, 2, 3]), 6)


if __name__ == '__main__':
  absltest.main()
