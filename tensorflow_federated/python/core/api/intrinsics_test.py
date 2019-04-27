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
"""Tests for tff."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf


from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core import api as tff


class IntrinsicsTest(parameterized.TestCase):

  def test_federated_broadcast_with_server_all_equal_int(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.SERVER, True))
    def foo(x):
      return tff.federated_broadcast(x)

    self.assertEqual(str(foo.type_signature), '(int32@SERVER -> int32@CLIENTS)')

  def test_federated_broadcast_with_server_non_all_equal_int(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tff.FederatedType(tf.int32, tff.SERVER))
      def _(x):
        return tff.federated_broadcast(x)

  def test_federated_broadcast_with_client_int(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS, True))
      def _(x):
        return tff.federated_broadcast(x)

  def test_federated_broadcast_with_non_federated_val(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tf.int32)
      def _(x):
        return tff.federated_broadcast(x)

  def test_federated_map_with_client_all_equal_int(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS, True))
    def foo(x):
      return tff.federated_map(
          tff.tf_computation(lambda x: x > 10, tf.int32), x)

    self.assertEqual(str(foo.type_signature), '(int32@CLIENTS -> bool@CLIENTS)')

  def test_federated_map_with_client_non_all_equal_int(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def foo(x):
      return tff.federated_map(
          tff.tf_computation(lambda x: x > 10, tf.int32), x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> {bool}@CLIENTS)')

  def test_federated_map_with_non_federated_val(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tf.int32)
      def _(x):
        return tff.federated_map(
            tff.tf_computation(lambda x: x > 10, tf.int32), x)

  def test_federated_sum_with_client_int(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def foo(x):
      return tff.federated_sum(x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')

  def test_federated_sum_with_client_string(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tff.FederatedType(tf.string, tff.CLIENTS))
      def _(x):
        return tff.federated_sum(x)

  def test_federated_sum_with_server_int(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tff.FederatedType(tf.int32, tff.SERVER))
      def _(x):
        return tff.federated_sum(x)

  def test_federated_zip_with_client_non_all_equal_int_and_bool(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.CLIENTS),
        tff.FederatedType(tf.bool, tff.CLIENTS, True)
    ])
    def foo(x, y):
      return tff.federated_zip([x, y])

    self.assertEqual(
        str(foo.type_signature),
        '(<{int32}@CLIENTS,bool@CLIENTS> -> {<int32,bool>}@CLIENTS)')

  def test_federated_zip_with_single_unnamed_int_client(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.CLIENTS),
    ])
    def foo(x):
      return tff.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature), '(<{int32}@CLIENTS> -> {<int32>}@CLIENTS)')

  def test_federated_zip_with_single_unnamed_int_server(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.SERVER, all_equal=True),
    ])
    def foo(x):
      return tff.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature), '(<int32@SERVER> -> <int32>@SERVER)')

  def test_federated_zip_with_single_named_bool_clients(self):

    @tff.federated_computation([
        ('a', tff.FederatedType(tf.bool, tff.CLIENTS)),
    ])
    def foo(x):
      return tff.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature), '(<a={bool}@CLIENTS> -> {<a=bool>}@CLIENTS)')

  def test_federated_zip_with_single_named_bool_server(self):

    @tff.federated_computation([
        ('a', tff.FederatedType(tf.bool, tff.SERVER, all_equal=True)),
    ])
    def foo(x):
      return tff.federated_zip(x)

    self.assertEqual(
        str(foo.type_signature), '(<a=bool@SERVER> -> <a=bool>@SERVER)')

  def test_federated_zip_with_names_client_non_all_equal_int_and_bool(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.CLIENTS),
        tff.FederatedType(tf.bool, tff.CLIENTS, True)
    ])
    def foo(x, y):
      a = {'x': x, 'y': y}
      return tff.federated_zip(a)

    self.assertEqual(
        str(foo.type_signature),
        '(<{int32}@CLIENTS,bool@CLIENTS> -> {<x=int32,y=bool>}@CLIENTS)')

  def test_federated_zip_with_client_all_equal_int_and_bool(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.CLIENTS, True),
        tff.FederatedType(tf.bool, tff.CLIENTS, True)
    ])
    def foo(x, y):
      return tff.federated_zip([x, y])

    self.assertEqual(
        str(foo.type_signature),
        '(<int32@CLIENTS,bool@CLIENTS> -> {<int32,bool>}@CLIENTS)')

  def test_federated_zip_with_names_client_all_equal_int_and_bool(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.CLIENTS, True),
        tff.FederatedType(tf.bool, tff.CLIENTS, True)
    ])
    def foo(arg):
      a = {'x': arg[0], 'y': arg[1]}
      return tff.federated_zip(a)

    self.assertEqual(
        str(foo.type_signature),
        '(<int32@CLIENTS,bool@CLIENTS> -> {<x=int32,y=bool>}@CLIENTS)')

  def test_federated_zip_with_server_int_and_bool(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.SERVER, True),
        tff.FederatedType(tf.bool, tff.SERVER, True)
    ])
    def foo(x, y):
      return tff.federated_zip([x, y])

    self.assertEqual(
        str(foo.type_signature),
        '(<int32@SERVER,bool@SERVER> -> <int32,bool>@SERVER)')

  def test_federated_zip_with_names_server_int_and_bool(self):

    @tff.federated_computation([
        ('a', tff.FederatedType(tf.int32, tff.SERVER, True)),
        ('b', tff.FederatedType(tf.bool, tff.SERVER, True))
    ])
    def foo(arg):
      return tff.federated_zip(arg)

    self.assertEqual(
        str(foo.type_signature),
        '(<a=int32@SERVER,b=bool@SERVER> -> <a=int32,b=bool>@SERVER)')

  def test_federated_collect_with_client_int(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def foo(x):
      return tff.federated_collect(x)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32*@SERVER)')

  def test_federated_collect_with_server_int_fails(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tff.FederatedType(tf.int32, tff.SERVER))
      def _(x):
        return tff.federated_collect(x)

  def test_federated_mean_with_client_float32_without_weight(self):

    @tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
    def foo(x):
      return tff.federated_mean(x)

    self.assertEqual(
        str(foo.type_signature), '({float32}@CLIENTS -> float32@SERVER)')

  def test_federated_mean_with_client_tuple_with_int32_weight(self):

    @tff.federated_computation([
        tff.FederatedType([('x', tf.float64), ('y', tf.float64)], tff.CLIENTS),
        tff.FederatedType(tf.int32, tff.CLIENTS)
    ])
    def foo(x, y):
      return tff.federated_mean(x, y)

    self.assertEqual(
        str(foo.type_signature),
        '(<{<x=float64,y=float64>}@CLIENTS,{int32}@CLIENTS> '
        '-> <x=float64,y=float64>@SERVER)')

  def test_federated_mean_with_client_int32_fails(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
      def _(x):
        return tff.federated_mean(x)

  def test_federated_mean_with_string_weight_fails(self):
    with self.assertRaises(TypeError):

      @tff.federated_computation([
          tff.FederatedType(tf.float32, tff.CLIENTS),
          tff.FederatedType(tf.string, tff.CLIENTS)
      ])
      def _(x, y):
        return tff.federated_mean(x, y)

  def test_federated_aggregate_with_client_int(self):
    # The representation used during the aggregation process will be a named
    # tuple with 2 elements - the integer 'total' that represents the sum of
    # elements encountered, and the integer element 'count'.
    # pylint: disable=invalid-name
    Accumulator = collections.namedtuple('Accumulator', 'total count')
    # pylint: enable=invalid-name
    accumulator_type = tff.NamedTupleType(Accumulator(tf.int32, tf.int32))

    # The operator to use during the first stage simply adds an element to the
    # total and updates the count.
    @tff.tf_computation([accumulator_type, tf.int32])
    def accumulate(accu, elem):
      return Accumulator(accu.total + elem, accu.count + 1)

    # The operator to use during the second stage simply adds total and count.
    @tff.tf_computation([accumulator_type, accumulator_type])
    def merge(x, y):
      return Accumulator(x.total + y.total, x.count + y.count)

    # The operator to use during the final stage simply computes the ratio.
    @tff.tf_computation(accumulator_type)
    def report(accu):
      return tf.to_float(accu.total) / tf.to_float(accu.count)

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def foo(x):
      return tff.federated_aggregate(x, Accumulator(0, 0), accumulate, merge,
                                     report)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> float32@SERVER)')

  def test_federated_reduce_with_tf_add_raw_constant(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.CLIENTS))
    def foo(x):
      plus = tff.tf_computation(tf.add, [tf.int32, tf.int32])
      return tff.federated_reduce(x, 0, plus)

    self.assertEqual(
        str(foo.type_signature), '({int32}@CLIENTS -> int32@SERVER)')

  def test_num_over_temperature_threshold_example(self):

    @tff.federated_computation([
        tff.FederatedType(tf.float32, tff.CLIENTS),
        tff.FederatedType(tf.float32, tff.SERVER, True)
    ])
    def foo(temperatures, threshold):
      return tff.federated_sum(
          tff.federated_map(
              tff.tf_computation(lambda x, y: tf.to_int32(tf.greater(x, y)),
                                 [tf.float32, tf.float32]),
              [temperatures, tff.federated_broadcast(threshold)]))

    self.assertEqual(
        str(foo.type_signature),
        '(<{float32}@CLIENTS,float32@SERVER> -> int32@SERVER)')

  @parameterized.named_parameters(('test_n_2', 2), ('test_n_3', 3),
                                  ('test_n_5', 5))
  def test_n_tuple_federated_zip_tensor_args(self, n):
    fed_type = tff.FederatedType(tf.int32, tff.CLIENTS)
    initial_tuple_type = tff.NamedTupleType([fed_type] * n)
    final_fed_type = tff.FederatedType([tf.int32] * n, tff.CLIENTS)
    function_type = tff.FunctionType(initial_tuple_type, final_fed_type)
    type_string = str(function_type)

    @tff.federated_computation([tff.FederatedType(tf.int32, tff.CLIENTS)] * n)
    def foo(x):
      return tff.federated_zip(x)

    self.assertEqual(str(foo.type_signature), type_string)

  @parameterized.named_parameters(
      ('test_n_2_int', 2, tff.FederatedType(tf.int32, tff.CLIENTS)),
      ('test_n_3_int', 3, tff.FederatedType(tf.int32, tff.CLIENTS)),
      ('test_n_5_int', 5, tff.FederatedType(tf.int32, tff.CLIENTS)),
      ('test_n_2_tuple', 2, tff.FederatedType([tf.int32, tf.int32],
                                              tff.CLIENTS)),
      ('test_n_3_tuple', 3, tff.FederatedType([tf.int32, tf.int32],
                                              tff.CLIENTS)),
      ('test_n_5_tuple', 5, tff.FederatedType([tf.int32, tf.int32],
                                              tff.CLIENTS)))
  def test_named_n_tuple_federated_zip(self, n, fed_type):
    initial_tuple_type = tff.NamedTupleType([fed_type] * n)
    named_fed_type = tff.FederatedType(
        [(str(k), fed_type.member) for k in range(n)], tff.CLIENTS)
    mixed_fed_type = tff.FederatedType(
        [(str(k), fed_type.member) if k % 2 == 0 else fed_type.member
         for k in range(n)], tff.CLIENTS)
    named_function_type = tff.FunctionType(initial_tuple_type, named_fed_type)
    mixed_function_type = tff.FunctionType(initial_tuple_type, mixed_fed_type)
    named_type_string = str(named_function_type)
    mixed_type_string = str(mixed_function_type)

    @tff.federated_computation([fed_type] * n)
    def foo(x):
      arg = {str(k): x[k] for k in range(n)}
      return tff.federated_zip(arg)

    self.assertEqual(str(foo.type_signature), named_type_string)

    @tff.federated_computation([fed_type] * n)
    def bar(x):
      arg = anonymous_tuple.AnonymousTuple([
          (str(k), x[k]) if k % 2 == 0 else (None, x[k]) for k in range(n)
      ])
      return tff.federated_zip(arg)

    self.assertEqual(str(bar.type_signature), mixed_type_string)

  @parameterized.named_parameters([
      ('test_n_' + str(n) + '_m_' + str(m), n, m)
      for n, m in itertools.product([1, 2, 3], [1, 2, 3])
  ])
  def test_n_tuple_federated_zip_mixed_args(self, n, m):
    tuple_fed_type = tff.FederatedType([tf.int32, tf.int32], tff.CLIENTS)
    single_fed_type = tff.FederatedType(tf.int32, tff.CLIENTS)
    initial_tuple_type = tff.NamedTupleType([tuple_fed_type] * n +
                                            [single_fed_type] * m)
    final_fed_type = tff.FederatedType([[tf.int32, tf.int32]] * n +
                                       [tf.int32] * m, tff.CLIENTS)
    function_type = tff.FunctionType(initial_tuple_type, final_fed_type)
    type_string = str(function_type)

    @tff.federated_computation([
        tff.FederatedType(
            tff.NamedTupleType([tf.int32, tf.int32]), tff.CLIENTS)
    ] * n + [tff.FederatedType(tf.int32, tff.CLIENTS)] * m)
    def baz(x):
      return tff.federated_zip(x)

    self.assertEqual(str(baz.type_signature), type_string)

  def test_federated_apply_with_int(self):

    @tff.federated_computation(tff.FederatedType(tf.int32, tff.SERVER, True))
    def foo(x):
      return tff.federated_apply(
          tff.tf_computation(lambda x: x > 10, tf.int32), x)

    self.assertEqual(str(foo.type_signature), '(int32@SERVER -> bool@SERVER)')

  def test_federated_apply_injected_zip_int(self):

    @tff.federated_computation([
        tff.FederatedType(tf.int32, tff.SERVER, True),
        tff.FederatedType(tf.int32, tff.SERVER, True)
    ])
    def foo(x, y):
      return tff.federated_apply(
          tff.tf_computation(lambda x, y: x > 10, [tf.int32, tf.int32]), [x, y])

    self.assertEqual(
        str(foo.type_signature), '(<int32@SERVER,int32@SERVER> -> bool@SERVER)')

  def test_federated_value_with_bool_on_clients(self):

    @tff.federated_computation(tf.bool)
    def foo(x):
      return tff.federated_value(x, tff.CLIENTS)

    self.assertEqual(str(foo.type_signature), '(bool -> bool@CLIENTS)')

  def test_federated_value_raw_np_scalar(self):

    @tff.federated_computation
    def test_np_values():
      floatv = np.float64(0)
      tff_float = tff.federated_value(floatv, tff.SERVER)
      self.assertEqual(str(tff_float.type_signature), 'float64@SERVER')
      intv = np.int64(0)
      tff_int = tff.federated_value(intv, tff.SERVER)
      self.assertEqual(str(tff_int.type_signature), 'int64@SERVER')
      return (tff_float, tff_int)

    floatv, intv = test_np_values()
    self.assertEqual(floatv, 0.0)
    self.assertEqual(intv, 0)

  def test_federated_value_raw_tf_scalar_variable(self):
    v = tf.Variable(initial_value=0., name='test_var')
    with self.assertRaisesRegex(
        TypeError, 'TensorFlow construct (.*) has been '
        'encountered in a federated context.'):
      _ = tff.federated_value(v, tff.SERVER)

  def test_federated_value_with_bool_on_server(self):

    @tff.federated_computation(tf.bool)
    def foo(x):
      return tff.federated_value(x, tff.SERVER)

    self.assertEqual(str(foo.type_signature), '(bool -> bool@SERVER)')

  def test_sequence_sum(self):

    @tff.federated_computation(tff.SequenceType(tf.int32))
    def foo1(x):
      return tff.sequence_sum(x)

    self.assertEqual(str(foo1.type_signature), '(int32* -> int32)')

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(tf.int32), tff.SERVER, True))
    def foo2(x):
      return tff.sequence_sum(x)

    self.assertEqual(
        str(foo2.type_signature), '(int32*@SERVER -> int32@SERVER)')

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(tf.int32), tff.CLIENTS))
    def foo3(x):
      return tff.sequence_sum(x)

    self.assertEqual(
        str(foo3.type_signature), '({int32*}@CLIENTS -> {int32}@CLIENTS)')

  def test_sequence_map(self):

    @tff.tf_computation(tf.int32)
    def over_threshold(x):
      return x > 10

    @tff.federated_computation(tff.SequenceType(tf.int32))
    def foo1(x):
      return tff.sequence_map(over_threshold, x)

    self.assertEqual(str(foo1.type_signature), '(int32* -> bool*)')

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(tf.int32), tff.SERVER, True))
    def foo2(x):
      return tff.sequence_map(over_threshold, x)

    self.assertEqual(
        str(foo2.type_signature), '(int32*@SERVER -> bool*@SERVER)')

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(tf.int32), tff.CLIENTS))
    def foo3(x):
      return tff.sequence_map(over_threshold, x)

    self.assertEqual(
        str(foo3.type_signature), '({int32*}@CLIENTS -> {bool*}@CLIENTS)')

  def test_sequence_reduce(self):
    add_numbers = tff.tf_computation(tf.add, [tf.int32, tf.int32])

    @tff.federated_computation(tff.SequenceType(tf.int32))
    def foo1(x):
      return tff.sequence_reduce(x, 0, add_numbers)

    self.assertEqual(str(foo1.type_signature), '(int32* -> int32)')

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(tf.int32), tff.SERVER, True))
    def foo2(x):
      return tff.sequence_reduce(x, 0, add_numbers)

    self.assertEqual(
        str(foo2.type_signature), '(int32*@SERVER -> int32@SERVER)')

    @tff.federated_computation(
        tff.FederatedType(tff.SequenceType(tf.int32), tff.CLIENTS))
    def foo3(x):
      return tff.sequence_reduce(x, 0, add_numbers)

    self.assertEqual(
        str(foo3.type_signature), '({int32*}@CLIENTS -> {int32}@CLIENTS)')


if __name__ == '__main__':
  absltest.main()
