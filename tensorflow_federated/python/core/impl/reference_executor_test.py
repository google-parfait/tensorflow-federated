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
"""Tests for reference_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import graph_utils
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import intrinsic_utils
from tensorflow_federated.python.core.impl import reference_executor
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl import type_constructors


def _create_lambda_to_identity(parameter_name, parameter_type=tf.int32):
  ref = computation_building_blocks.Reference(parameter_name, parameter_type)
  return computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)


def _to_computation_impl(building_block):
  return computation_impl.ComputationImpl(building_block.proto,
                                          context_stack_impl.context_stack)


class ReferenceExecutorTest(test.TestCase):

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
        anonymous_tuple.AnonymousTuple([
            ('x', anonymous_tuple.AnonymousTuple([(None, 10), (None, 20)])),
            ('y', 30)
        ]))
    with self.assertRaises(TypeError):
      reference_executor.to_representation_for_type(10, [tf.int32, tf.int32])

    unordered_dict = {'a': 10, 'b': 20}
    self.assertEqual(
        str(
            reference_executor.to_representation_for_type(
                unordered_dict, [('a', tf.int32), ('b', tf.int32)])),
        '<a=10,b=20>')
    self.assertEqual(
        str(
            reference_executor.to_representation_for_type(
                unordered_dict, [('b', tf.int32), ('a', tf.int32)])),
        '<b=20,a=10>')

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
            foo, computation_types.FunctionType(tf.int32,
                                                tf.string), lambda x, t: x),
        foo)

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

  def test_stamp_computed_value_into_graph_with_undefined_tensor_dims(self):
    v_type = computation_types.TensorType(tf.int32, [None])
    v_value = np.array([1, 2, 3], dtype=np.int32)
    v = reference_executor.ComputedValue(v_value, v_type)
    with tf.Graph().as_default() as graph:
      stamped_v = reference_executor.stamp_computed_value_into_graph(v, graph)
      with tf.compat.v1.Session(graph=graph) as sess:
        v_result = graph_utils.fetch_value_in_session(sess, stamped_v)
    self.assertTrue(np.array_equal(v_result, np.array([1, 2, 3])))

  def test_stamp_computed_value_into_graph_with_tuples_of_tensors(self):
    v_val = anonymous_tuple.AnonymousTuple([
        ('x', 10), ('y', anonymous_tuple.AnonymousTuple([('z', 0.6)]))
    ])
    v_type = [('x', tf.int32), ('y', [('z', tf.float32)])]
    v = reference_executor.ComputedValue(
        reference_executor.to_representation_for_type(v_val, v_type), v_type)
    with tf.Graph().as_default() as graph:
      stamped_v = reference_executor.stamp_computed_value_into_graph(v, graph)
      with tf.compat.v1.Session(graph=graph) as sess:
        v_val = graph_utils.fetch_value_in_session(sess, stamped_v)
    self.assertEqual(str(v_val), '<x=10,y=<z=0.6>>')

  def test_computation_context_resolve_reference(self):
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

  def test_computation_context_get_cardinality(self):
    c1 = reference_executor.ComputationContext(None, None,
                                               {placements.CLIENTS: 10})
    self.assertEqual(c1.get_cardinality(placements.CLIENTS), 10)
    with self.assertRaises(ValueError):
      c1.get_cardinality(placements.SERVER)
    c2 = reference_executor.ComputationContext(c1)
    self.assertEqual(c2.get_cardinality(placements.CLIENTS), 10)

  def test_tensorflow_computation_with_no_argument(self):

    @computations.tf_computation()
    def foo():
      return 1

    self.assertEqual(foo(), 1)

  def test_tensorflow_computation_with_string(self):

    @computations.tf_computation()
    def foo():
      return 'abc'

    self.assertEqual(foo(), 'abc')

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
      self.assertEmpty(z)
      return tf.constant(1)

    self.assertEqual(foo(()), 1)

  def test_tensorflow_computation_with_tuple_of_one_constant(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z.x + 1

    self.assertEqual(foo((10,)), 11)

  # This is the same as test_tensorflow_computation_with_tuple_of_one_constant
  # above, but does not have a name on the types. This behavior may change in
  # the future, this unittest
  def test_tensorflow_computation_with_tuple_of_one_unnamed_constant(self):
    tuple_type = computation_types.NamedTupleType([
        (None, tf.int32),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      return z[0] + 1

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
      self.assertEmpty(z.a)
      self.assertEmpty(z.b)
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
    sequence_type = computation_types.SequenceType(tf.int32)
    tuple_type = computation_types.NamedTupleType([
        ('a', sequence_type),
        ('b', sequence_type),
    ])

    @computations.tf_computation(tuple_type)
    def foo(z):
      value1 = z.a.reduce(0, lambda x, y: x + y)
      value2 = z.b.reduce(0, lambda x, y: x + y)
      return value1 + value2

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40])

    # pylint: disable=too-many-function-args
    self.assertEqual(foo(ds1, ds2), 100)
    # pylint: enable=too-many-function-args
    self.assertEqual(foo([ds1, ds2]), 100)
    self.assertEqual(foo((ds1, ds2)), 100)

  def test_tensorflow_computation_output_nested_structure(self):
    test_named_tuple = collections.namedtuple('TestNamedTuple', ['a'])

    @computations.tf_computation
    def foo():
      return test_named_tuple(tf.constant(10.0))

    result = foo()
    self.assertIsInstance(result, test_named_tuple)
    self.assertEqual(result, test_named_tuple(10.0))

  def test_computation_with_batched_federated_int_sequence(self):
    ds1_shape = tf.TensorShape([None])
    sequence_type = computation_types.SequenceType(
        computation_types.TensorType(tf.int32, ds1_shape))
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placements.CLIENTS)

    @computations.tf_computation(sequence_type)
    def foo(z):
      value1 = z.reduce(0, lambda x, y: x + tf.reduce_sum(y))
      return value1

    @computations.federated_computation(federated_type)
    def bar(x):
      return intrinsics.federated_map(foo, x)

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20]).batch(1)
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40]).batch(1)

    self.assertEqual(bar([ds1, ds2]), [30, 70])

  def test_computation_with_int_sequence_raises(self):
    batch_size = 1
    ds1_shape = tf.TensorShape([batch_size])
    sequence_type = computation_types.SequenceType(
        computation_types.TensorType(tf.int32, ds1_shape))
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placements.CLIENTS)

    @computations.tf_computation(sequence_type)
    def foo(z):
      value1 = z.reduce(0, lambda x, y: x + tf.reduce_sum(y))
      return value1

    @computations.federated_computation(federated_type)
    def bar(x):
      return intrinsics.federated_map(foo, x)

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20]).batch(batch_size)
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40]).batch(batch_size)

    with self.assertRaisesRegexp(ValueError, 'Please pass a list'):
      bar(ds1)
    with self.assertRaisesRegexp(ValueError, 'Please pass a list'):
      bar(ds2)

  def test_batching_namedtuple_dataset(self):
    batch_type = collections.namedtuple('Batch', ['x', 'y'])
    federated_sequence_type = computation_types.FederatedType(
        computation_types.SequenceType(
            batch_type(
                x=computation_types.TensorType(tf.float32, [None, 2]),
                y=computation_types.TensorType(tf.float32, [None, 1]))),
        placements.CLIENTS,
        all_equal=False)

    @computations.tf_computation(federated_sequence_type.member)
    def test_batch_select_and_reduce(z):
      i = z.map(lambda x: x.y)
      return i.reduce(0., lambda x, y: x + tf.reduce_sum(y))

    @computations.federated_computation(federated_sequence_type)
    def map_y_sum(x):
      return intrinsics.federated_map(test_batch_select_and_reduce, x)

    ds = tf.data.Dataset.from_tensor_slices({
        'x': [[1., 2.], [3., 4.]],
        'y': [[5.], [6.]]
    }).batch(1)
    self.assertEqual(map_y_sum([ds] * 5), [np.array([[11.]])] * 5)

  def test_batching_ordereddict_dataset(self):
    odict_type = collections.OrderedDict([
        ('x', computation_types.TensorType(tf.float32, [None, 2])),
        ('y', computation_types.TensorType(tf.float32, [None, 1])),
    ])

    @computations.tf_computation(odict_type)
    def test_foo(z):
      return tf.reduce_sum(z['x']) + z['y']

    self.assertEqual(
        test_foo(
            collections.OrderedDict([
                ('x', np.ones(shape=[1, 2], dtype=np.float32)),
                ('y', np.ones(shape=[1, 1], dtype=np.float32)),
            ])), 3.0)

  def test_helpful_failure_federated_int_sequence(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placements.CLIENTS)

    @computations.tf_computation(sequence_type)
    def foo(z):
      value1 = z.reduce(0, lambda x, y: x + y)
      return value1

    @computations.federated_computation(federated_type)
    def bar(x):
      return intrinsics.federated_map(foo, x)

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40])

    with self.assertRaisesRegexp(TypeError,
                                 'only with a single positional argument'):
      # pylint: disable=too-many-function-args
      _ = bar(ds1, ds2)
      # pylint: enable=too-many-function-args

    with self.assertRaisesRegexp(TypeError,
                                 'argument should be placed at SERVER'):

      @computations.federated_computation(federated_type)
      def _(x):
        return intrinsics.federated_apply(foo, x)

  def test_graph_mode_dataset_fails_well(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    federated_type = computation_types.FederatedType(sequence_type,
                                                     placements.CLIENTS)

    with tf.Graph().as_default():

      @computations.tf_computation(sequence_type)
      def foo(z):
        value1 = z.reduce(0, lambda x, y: x + y)
        return value1

      @computations.federated_computation(federated_type)
      def bar(x):
        return intrinsics.federated_map(foo, x)

      ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
      ds2 = tf.data.Dataset.from_tensor_slices([30, 40])
      with self.assertRaisesRegexp(
          ValueError, 'outside of eager mode is not currently supported.'):
        bar([ds1, ds2])

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
    sequence_type = computation_types.SequenceType(tf.float32)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      del ds  # unused
      return 1

    ds = tf.data.Dataset.from_tensor_slices([])
    self.assertEqual(foo(ds), 1)

  def test_tensorflow_computation_with_sequence_of_one_constant(self):
    sequence_type = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(1, lambda x, y: x + y)

    ds = tf.data.Dataset.from_tensor_slices([10])

    self.assertEqual(foo(ds), 11)

  def test_tensorflow_computation_with_sequence_of_constants(self):
    sequence_type = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(0, lambda x, y: x + y)

    ds = tf.data.Dataset.from_tensor_slices([10, 20])
    self.assertEqual(foo(ds), 30)

  def test_tensorflow_computation_with_sequence_of_tuples(self):
    tuple_type = computation_types.NamedTupleType([
        ('x', tf.int32),
        ('y', tf.int32),
    ])
    sequence_type = computation_types.SequenceType(tuple_type)

    @computations.tf_computation(sequence_type)
    def foo(ds):
      return ds.reduce(0, lambda x, y: x + y['x'] + y['y'])

    ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict([
            ('x', [10, 30]),
            ('y', [20, 40]),
        ]))

    self.assertEqual(foo(ds), 100)

  def test_tensorflow_computation_with_sequences_of_constants(self):
    sequence_type = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(sequence_type, sequence_type)
    def foo(ds1, ds2):
      value1 = ds1.reduce(0, lambda x, y: x + y)
      value2 = ds2.reduce(0, lambda x, y: x + y)
      return value1 + value2

    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    ds2 = tf.data.Dataset.from_tensor_slices([30, 40])

    self.assertEqual(foo(ds1, ds2), 100)

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

  def test_multiply_by_scalar_with_float_and_float(self):
    self.assertEqual(
        reference_executor.multiply_by_scalar(
            reference_executor.ComputedValue(10.0, tf.float32), 0.5).value, 5.0)
    self.assertAlmostEqual(
        reference_executor.multiply_by_scalar(
            reference_executor.ComputedValue(np.float32(1.0), tf.float32),
            0.333333333333).value,
        0.3333333,
        places=3)

  def test_multiply_by_scalar_with_tuple_and_float(self):
    self.assertEqual(
        str(
            reference_executor.multiply_by_scalar(
                reference_executor.ComputedValue(
                    anonymous_tuple.AnonymousTuple([
                        ('A', 10.0),
                        ('B', anonymous_tuple.AnonymousTuple([('C', 20.0)])),
                    ]), [('A', tf.float32), ('B', [('C', tf.float32)])]),
                0.5).value), '<A=5.0,B=<C=10.0>>')

  def test_get_cardinalities_success(self):
    foo = reference_executor.get_cardinalities(
        reference_executor.ComputedValue(
            anonymous_tuple.AnonymousTuple([
                ('A', [1, 2, 3]),
                ('B',
                 anonymous_tuple.AnonymousTuple([('C', [[1, 2], [3, 4], [5,
                                                                         6]]),
                                                 ('D', [True, False, True])]))
            ]),
            [('A', computation_types.FederatedType(tf.int32,
                                                   placements.CLIENTS)),
             ('B', [('C',
                     computation_types.FederatedType(
                         computation_types.SequenceType(tf.int32),
                         placements.CLIENTS)),
                    ('D',
                     computation_types.FederatedType(tf.bool,
                                                     placements.CLIENTS))])]))
    self.assertDictEqual(foo, {placements.CLIENTS: 3})

  def test_get_cardinalities_failure(self):
    with self.assertRaises(ValueError):
      reference_executor.get_cardinalities(
          reference_executor.ComputedValue(
              anonymous_tuple.AnonymousTuple([('A', [1, 2, 3]), ('B', [1, 2])]),
              [('A',
                computation_types.FederatedType(tf.int32, placements.CLIENTS)),
               ('B',
                computation_types.FederatedType(tf.int32, placements.CLIENTS))
              ]))

  def test_fit_argument(self):
    old_arg = reference_executor.ComputedValue(
        anonymous_tuple.AnonymousTuple([('A', 10)]),
        [('A', type_constructors.at_clients(tf.int32, all_equal=True))])
    new_arg = reference_executor.fit_argument(
        old_arg, [('A', type_constructors.at_clients(tf.int32))],
        reference_executor.ComputationContext(
            cardinalities={placements.CLIENTS: 3}))
    self.assertEqual(str(new_arg.type_signature), '<A={int32}@CLIENTS>')
    self.assertEqual(new_arg.value.A, [10, 10, 10])

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
                computation_building_blocks.Tuple([
                    (None, computation_building_blocks.Reference('x',
                                                                 tf.int32)),
                    (None, computation_building_blocks.Reference('y', tf.int32))
                ]))))

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

  def test_sequence_sum_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def foo(x):
      return intrinsics.sequence_sum(x)

    self.assertEqual(str(foo.type_signature), '(int32* -> int32)')
    self.assertEqual(foo([1, 2, 3]), 6)

  def test_sequence_sum_with_list_of_tuples(self):

    @computations.federated_computation(
        computation_types.SequenceType([tf.int32, tf.int32]))
    def foo(x):
      return intrinsics.sequence_sum(x)

    self.assertEqual(
        str(foo.type_signature), '(<int32,int32>* -> <int32,int32>)')
    self.assertEqual(str(foo([[1, 2], [3, 4], [5, 6]])), '<9,12>')

  def test_federated_collect_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_collect(x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32*@SERVER)')
    self.assertEqual(foo([1, 2, 3]), [1, 2, 3])

  def test_federated_map_with_list_of_integers(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def bar(x):
      return intrinsics.federated_map(foo, x)

    self.assertEqual(
        str(bar.type_signature), '({int32}@CLIENTS -> {int32}@CLIENTS)')
    self.assertEqual(bar([1, 10, 3, 7, 2]), [2, 11, 4, 8, 3])

  def test_federated_apply_with_int(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def bar(x):
      return intrinsics.federated_apply(foo, x)

    self.assertEqual(str(bar.type_signature), '(int32@SERVER -> int32@SERVER)')
    self.assertEqual(bar(10), 11)

  def test_federated_apply_with_int_sequence(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def bar(z):
      return intrinsics.sequence_map(foo, z)

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.SERVER, True))
    def baz(x):
      return intrinsics.federated_apply(bar, x)

    self.assertEqual(
        str(baz.type_signature), '(int32*@SERVER -> int32*@SERVER)')
    ds1 = tf.data.Dataset.from_tensor_slices([10, 20])
    self.assertEqual(baz(ds1), [11, 21])

  def test_federated_sum_with_list_of_integers(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_sum(x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(foo([1, 2, 3]), 6)

  def test_federated_value_at_clients_and_at_server(self):

    @computations.federated_computation(tf.int32)
    def foo(x):
      return [
          intrinsics.federated_value(x, placements.CLIENTS),
          intrinsics.federated_value(x, placements.SERVER)
      ]

    self.assertEqual(
        str(foo.type_signature), '(int32 -> <int32@CLIENTS,int32@SERVER>)')
    self.assertEqual(str(foo(11)), '<11,11>')

  def test_generic_zero_with_scalar_int32_tensor_type(self):

    @computations.federated_computation
    def foo():
      return intrinsic_utils.zero_for(tf.int32,
                                      context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> int32)')
    self.assertEqual(foo(), 0)

  def test_generic_zero_with_two_dimensional_float32_tensor_type(self):

    @computations.federated_computation
    def foo():
      return intrinsic_utils.zero_for(
          computation_types.TensorType(tf.float32, [2, 3]),
          context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> float32[2,3])')
    foo_result = foo()
    self.assertEqual(type(foo_result), np.ndarray)
    self.assertTrue(np.array_equal(foo_result, [[0., 0., 0.], [0., 0., 0.]]))

  def test_generic_zero_with_tuple_type(self):

    @computations.federated_computation
    def foo():
      return intrinsic_utils.zero_for([('A', tf.int32), ('B', tf.float32)],
                                      context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> <A=int32,B=float32>)')
    self.assertEqual(str(foo()), '<A=0,B=0.0>')

  def test_generic_zero_with_federated_int_on_server(self):

    @computations.federated_computation
    def foo():
      return intrinsic_utils.zero_for(
          computation_types.FederatedType(tf.int32, placements.SERVER, True),
          context_stack_impl.context_stack)

    self.assertEqual(str(foo.type_signature), '( -> int32@SERVER)')
    self.assertEqual(foo(), 0)

  def test_generic_plus_with_integers(self):
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(tf.int32, tf.int32)
    def foo(x, y):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, y])

    self.assertEqual(str(foo.type_signature), '(<int32,int32> -> int32)')
    self.assertEqual(foo(2, 3), 5)

  def test_generic_plus_with_tuples(self):
    type_spec = [('A', tf.int32), ('B', tf.float32)]
    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    @computations.federated_computation(type_spec, type_spec)
    def foo(x, y):
      return bodies[intrinsic_defs.GENERIC_PLUS.uri]([x, y])

    self.assertEqual(
        str(foo.type_signature),
        '(<<A=int32,B=float32>,<A=int32,B=float32>> -> <A=int32,B=float32>)')
    foo_result = foo([2, 0.1], [3, 0.2])
    self.assertIsInstance(foo_result, anonymous_tuple.AnonymousTuple)
    self.assertSameElements(dir(foo_result), ['A', 'B'])
    self.assertEqual(foo_result.A, 5)
    self.assertAlmostEqual(foo_result.B, 0.3, places=2)

  def test_sequence_map_with_list_of_integers(self):

    @computations.tf_computation(tf.int32)
    def foo(x):
      return x + 1

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def bar(x):
      return intrinsics.sequence_map(foo, x)

    self.assertEqual(str(bar.type_signature), '(int32* -> int32*)')
    self.assertEqual(bar([1, 10, 3]), [2, 11, 4])

  def test_sequence_reduce_with_integers(self):

    @computations.tf_computation(tf.int32, tf.float32)
    def foo(x, y):
      return x + tf.cast(y > 0.5, tf.int32)

    @computations.federated_computation(
        computation_types.SequenceType(tf.float32))
    def bar(x):
      return intrinsics.sequence_reduce(x, 0, foo)

    self.assertEqual(str(bar.type_signature), '(float32* -> int32)')
    self.assertEqual(bar([0.1, 0.6, 0.2, 0.4, 0.8]), 2)

  def test_sequence_reduce_with_namedtuples(self):
    accumulator_type = collections.namedtuple('_', ['sum', 'product'])(
        sum=tf.int32, product=tf.int32)

    @computations.tf_computation(accumulator_type, tf.int32)
    def foo(accumulator, x):
      return collections.OrderedDict([('sum', accumulator.sum + x),
                                      ('product', accumulator.product * x)])

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def bar(x):
      zero = collections.OrderedDict([('sum', 0), ('product', 1)])
      return intrinsics.sequence_reduce(x, zero, foo)

    self.assertEqual(
        str(bar.type_signature), '(int32* -> <sum=int32,product=int32>)')
    self.assertEqual(str(bar([1, 2, 3, 4, 5])), '<sum=15,product=120>')

  def test_federated_reduce_with_integers(self):

    @computations.tf_computation(tf.int32, tf.float32)
    def foo(x, y):
      return x + tf.cast(y > 0.5, tf.int32)

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def bar(x):
      return intrinsics.federated_reduce(x, 0, foo)

    self.assertEqual(
        str(bar.type_signature), '({float32}@CLIENTS -> int32@SERVER)')
    self.assertEqual(bar([0.1, 0.6, 0.2, 0.4, 0.8]), 2)

  def test_federated_mean_with_floats(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x)

    self.assertEqual(
        str(foo.type_signature), '({float32}@CLIENTS -> float32@SERVER)')
    self.assertEqual(foo([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)

  def test_federated_mean_with_tuples(self):

    @computations.federated_computation(
        computation_types.FederatedType([('A', tf.float32), ('B', tf.float32)],
                                        placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_mean(x)

    self.assertEqual(
        str(foo.type_signature),
        '({<A=float32,B=float32>}@CLIENTS -> <A=float32,B=float32>@SERVER)')
    self.assertEqual(
        str(
            foo([{
                'A': 1.0,
                'B': 5.0
            }, {
                'A': 2.0,
                'B': 6.0
            }, {
                'A': 3.0,
                'B': 7.0
            }])), '<A=2.0,B=6.0>')

  def test_federated_zip_at_server(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.SERVER, True),
        computation_types.FederatedType(tf.int32, placements.SERVER, True)
    ])
    def foo(x):
      return intrinsics.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature),
        '(<int32@SERVER,int32@SERVER> -> <int32,int32>@SERVER)')

    self.assertEqual(str(foo(5, 6)), '<5,6>')  # pylint: disable=too-many-function-args

  def test_federated_zip_at_clients(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
        computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ])
    def foo(x):
      return intrinsics.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature),
        '(<{int32}@CLIENTS,{int32}@CLIENTS> -> {<int32,int32>}@CLIENTS)')
    foo_result = foo([[1, 2, 3], [4, 5, 6]])
    self.assertIsInstance(foo_result, list)
    foo_result_str = ','.join(str(x) for x in foo_result)
    self.assertEqual(foo_result_str, '<1,4>,<2,5>,<3,6>')

  def test_federated_aggregate_with_integers(self):
    test_named_tuple = collections.namedtuple('_', ['sum', 'n'])
    accu_type = computation_types.to_type(
        test_named_tuple(sum=tf.int32, n=tf.int32))

    @computations.tf_computation(accu_type, tf.int32)
    def accumulate(a, x):
      return collections.OrderedDict([('sum', a.sum + x), ('n', a.n + 1)])

    @computations.tf_computation(accu_type, accu_type)
    def merge(a, b):
      return collections.OrderedDict([('sum', a.sum + b.sum), ('n', a.n + b.n)])

    @computations.tf_computation(accu_type)
    def report(a):
      return tf.cast(a.sum, tf.float32) / tf.cast(a.n, tf.float32)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      return intrinsics.federated_aggregate(
          x, collections.OrderedDict([('sum', 0), ('n', 0)]), accumulate, merge,
          report)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> float32@SERVER)')
    self.assertEqual(foo([1, 2, 3, 4, 5, 6, 7]), 4.0)

  def test_federated_weighted_average_with_floats(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS),
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def foo(v, w):
      return intrinsics.federated_mean(v, w)

    self.assertEqual(
        str(foo.type_signature),
        '(<{float32}@CLIENTS,{float32}@CLIENTS> -> float32@SERVER)')
    self.assertEqual(foo([5.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 3.0)

  def test_federated_broadcast_without_data_on_clients(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x):
      return intrinsics.federated_broadcast(x)

    self.assertEqual(str(foo.type_signature), '(int32@SERVER -> int32@CLIENTS)')
    self.assertEqual(foo(10), 10)

  def test_federated_broadcast_zipped_with_client_data(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x, y):
      return intrinsics.federated_zip([x, intrinsics.federated_broadcast(y)])

    self.assertEqual(
        str(foo.type_signature),
        '(<{int32}@CLIENTS,int32@SERVER> -> {<int32,int32>}@CLIENTS)')

    foo_result = foo([1, 2, 3], 10)
    self.assertIsInstance(foo_result, list)
    foo_result_str = ','.join(str(x) for x in foo_result)
    self.assertEqual(foo_result_str, '<1,10>,<2,10>,<3,10>')

  def test_with_unequal_tensor_types(self):

    @computations.tf_computation
    def foo():
      return tf.data.Dataset.range(5).map(lambda _: tf.constant(10.0)).batch(1)

    self.assertEqual(str(foo.type_signature), '( -> float32[?]*)')
    foo_result = foo()
    self.assertIsInstance(foo_result, list)
    foo_result_str = ','.join(str(x) for x in foo_result).replace(' ', '')
    self.assertEqual(foo_result_str, '[10.],[10.],[10.],[10.],[10.]')

  def test_numpy_cast(self):
    self.assertEqual(
        reference_executor.numpy_cast(True, tf.bool, tf.TensorShape([])),
        np.bool_(True))
    self.assertEqual(
        reference_executor.numpy_cast(10, tf.int32, tf.TensorShape([])),
        np.int32(10))
    self.assertEqual(
        reference_executor.numpy_cast(0.3333333333333333333333333, tf.float32,
                                      tf.TensorShape([])),
        np.float32(0.3333333333333333333333333))
    self.assertTrue(
        np.array_equal(
            reference_executor.numpy_cast([[1, 2], [3, 4]], tf.int32,
                                          tf.TensorShape([2, 2])),
            np.array([[1, 2], [3, 4]])))

  def test_sum_of_squares(self):
    int32_sequence = computation_types.SequenceType(tf.int32)

    @computations.tf_computation(tf.int32, tf.int32)
    def square_error(x, y):
      return tf.pow(x - y, 2)

    @computations.federated_computation(tf.int32, int32_sequence)
    def sum_of_square_errors(x, y):

      @computations.federated_computation(tf.int32)
      def mapping_fn(v):
        return square_error(x, v)

      return intrinsics.sequence_sum(intrinsics.sequence_map(mapping_fn, y))

    self.assertEqual(sum_of_square_errors(10, [11, 8, 13]), 14)

  def test_variable_input(self):
    assert tf.executing_eagerly()
    v = tf.Variable(10.0)

    @computations.tf_computation
    def add_one(x):
      return x + 1.0

    self.assertEqual(add_one(v), 11.0)


class UnwrapPlacementIntegrationTest(test.TestCase):

  def test_unwrap_placement_with_federated_map_executes_correctly(self):
    int_ref = computation_building_blocks.Reference('x', tf.int32)
    int_id = computation_building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = computation_building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32, placements.CLIENTS))
    applied_id = computation_constructing_utils.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = computation_constructing_utils.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)
    lambda_wrapping_id = computation_building_blocks.Lambda(
        'x', fed_ref.type_signature, second_applied_id)
    lambda_wrapping_placement_unwrapped = computation_building_blocks.Lambda(
        'x', fed_ref.type_signature, placement_unwrapped)
    executable_identity = _to_computation_impl(lambda_wrapping_id)
    executable_unwrapped = _to_computation_impl(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(executable_identity([k]), executable_unwrapped([k]))

  def test_unwrap_placement_with_federated_apply_executes_correctly(self):
    int_ref = computation_building_blocks.Reference('x', tf.int32)
    int_id = computation_building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = computation_building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32, placements.SERVER))
    applied_id = computation_constructing_utils.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = computation_constructing_utils.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)
    lambda_wrapping_id = computation_building_blocks.Lambda(
        'x', fed_ref.type_signature, second_applied_id)
    lambda_wrapping_placement_unwrapped = computation_building_blocks.Lambda(
        'x', fed_ref.type_signature, placement_unwrapped)
    executable_identity = _to_computation_impl(lambda_wrapping_id)
    executable_unwrapped = _to_computation_impl(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(executable_identity(k), executable_unwrapped(k))

  def test_unwrap_placement_with_federated_zip_at_server_executes_correctly(
      self):
    fed_tuple = computation_building_blocks.Reference(
        'tup',
        computation_types.FederatedType([tf.int32, tf.float32] * 2,
                                        placements.SERVER))
    unzipped = computation_constructing_utils.create_federated_unzip(fed_tuple)
    zipped = computation_constructing_utils.create_federated_zip(unzipped)
    placement_unwrapped, modified = transformations.unwrap_placement(zipped)
    self.assertTrue(modified)

    lambda_wrapping_zip = computation_building_blocks.Lambda(
        'tup', fed_tuple.type_signature, zipped)
    lambda_wrapping_placement_unwrapped = computation_building_blocks.Lambda(
        'tup', fed_tuple.type_signature, placement_unwrapped)
    executable_zip = _to_computation_impl(lambda_wrapping_zip)
    executable_unwrapped = _to_computation_impl(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(
          executable_zip([k, k * 1., k, k * 1.]),
          executable_unwrapped([k, k * 1., k, k * 1.]))

  def test_unwrap_placement_with_federated_zip_at_clients_executes_correctly(
      self):
    fed_tuple = computation_building_blocks.Reference(
        'tup',
        computation_types.FederatedType([tf.int32, tf.float32] * 2,
                                        placements.CLIENTS))
    unzipped = computation_constructing_utils.create_federated_unzip(fed_tuple)
    zipped = computation_constructing_utils.create_federated_zip(unzipped)
    placement_unwrapped, modified = transformations.unwrap_placement(zipped)
    self.assertTrue(modified)
    lambda_wrapping_zip = computation_building_blocks.Lambda(
        'tup', fed_tuple.type_signature, zipped)
    lambda_wrapping_placement_unwrapped = computation_building_blocks.Lambda(
        'tup', fed_tuple.type_signature, placement_unwrapped)
    executable_zip = _to_computation_impl(lambda_wrapping_zip)
    executable_unwrapped = _to_computation_impl(
        lambda_wrapping_placement_unwrapped)

    for k in range(10):
      self.assertEqual(
          executable_zip([[k, k * 1., k, k * 1.]]),
          executable_unwrapped([[k, k * 1., k, k * 1.]]))


class MergeTupleIntrinsicsIntegrationTest(test.TestCase):

  def test_merge_tuple_intrinsics_executes_with_federated_aggregate(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref_type = computation_types.NamedTupleType(
        (value_type, tf.float32, tf.float32, tf.float32, tf.bool))
    ref = computation_building_blocks.Reference('a', ref_type)
    value = computation_building_blocks.Selection(ref, index=0)
    zero = computation_building_blocks.Selection(ref, index=1)
    accumulate_type = computation_types.NamedTupleType((tf.float32, tf.int32))
    accumulate_result = computation_building_blocks.Selection(ref, index=2)
    accumulate = computation_building_blocks.Lambda('b', accumulate_type,
                                                    accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.float32, tf.float32))
    merge_result = computation_building_blocks.Selection(ref, index=3)
    merge = computation_building_blocks.Lambda('c', merge_type, merge_result)
    report_result = computation_building_blocks.Selection(ref, index=4)
    report = computation_building_blocks.Lambda('d', tf.float32, report_result)
    called_intrinsic = computation_constructing_utils.create_federated_aggregate(
        value, zero, accumulate, merge, report)
    tup = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    comp_impl = _to_computation_impl(comp)
    transformed_comp_impl = _to_computation_impl(transformed_comp)

    self.assertEqual(
        comp_impl(((1,), 1.0, 2.0, 3.0, True)),
        transformed_comp_impl(((2,), 4.0, 5.0, 6.0, True)))

  def test_merge_tuple_intrinsics_executes_with_federated_apply(self):
    ref_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    ref = computation_building_blocks.Reference('a', ref_type)
    fn = _create_lambda_to_identity('b')
    arg = ref
    called_intrinsic = computation_constructing_utils.create_federated_apply(
        fn, arg)
    tup = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_APPLY.uri)

    comp_impl = _to_computation_impl(comp)
    transformed_comp_impl = _to_computation_impl(transformed_comp)

    self.assertEqual(comp_impl(1), transformed_comp_impl(1))

  def test_merge_tuple_intrinsics_executes_with_federated_broadcast(self):
    self.skipTest('b/135279151')
    ref_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    ref = computation_building_blocks.Reference('a', ref_type)
    called_intrinsic = computation_constructing_utils.create_federated_broadcast(
        ref)
    tup = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_BROADCAST.uri)

    comp_impl = _to_computation_impl(comp)
    transformed_comp_impl = _to_computation_impl(transformed_comp)

    self.assertEqual(comp_impl(10), transformed_comp_impl(10))

  def test_merge_tuple_intrinsics_executes_with_federated_map(self):
    ref_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref = computation_building_blocks.Reference('a', ref_type)
    fn = _create_lambda_to_identity('b')
    arg = ref
    called_intrinsic = computation_constructing_utils.create_federated_map(
        fn, arg)
    tup = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, tup)
    transformed_comp, _ = transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    comp_impl = _to_computation_impl(comp)
    transformed_comp_impl = _to_computation_impl(transformed_comp)

    self.assertEqual(comp_impl((1,)), transformed_comp_impl((1,)))


if __name__ == '__main__':
  # We need to be able to individually test all components of the executor, and
  # the compiler pipeline will potentially interfere with this process by
  # performing reductions. Since this executor is intended to be the standards
  # to compare against, it is the compiler pipeline that should get tested
  # against this implementation, not the other way round.
  executor_without_compiler = reference_executor.ReferenceExecutor()
  with context_stack_impl.context_stack.install(executor_without_compiler):
    test.main()
