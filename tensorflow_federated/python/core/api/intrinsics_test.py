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
import itertools
import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import value_base
from tensorflow_federated.python.core.impl.context_stack import context_base


class IntrinsicsTest(parameterized.TestCase):

  def assert_type(self, value, type_string):
    self.assertEqual(value.type_signature.compact_representation(), type_string)

  def test_constant_to_value_raises_outside_decorator(self):

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_value(2, placements.SERVER)

  def test_intrinsic_construction_raises_context_error_outside_decorator(self):

    @computations.tf_computation()
    def return_2():
      return 2

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_eval(return_2, placements.SERVER)

  def test_federated_broadcast_with_server_all_equal_int(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER))
    def foo(x):
      val = intrinsics.federated_broadcast(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(int32@SERVER -> int32@CLIENTS)')

  def test_federated_broadcast_with_server_non_all_equal_int(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(
          computation_types.FederatedType(
              tf.int32, placements.SERVER, all_equal=False))
      def _(x):
        return intrinsics.federated_broadcast(x)

  def test_federated_broadcast_with_client_int(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.CLIENTS, True))
      def _(x):
        return intrinsics.federated_broadcast(x)

  def test_federated_broadcast_with_non_federated_val(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(tf.int32)
      def _(x):
        return intrinsics.federated_broadcast(x)

  def test_federated_eval_rand_on_clients(self):

    @computations.federated_computation
    def rand_on_clients():

      @computations.tf_computation
      def rand():
        return tf.random.normal([])

      val = intrinsics.federated_eval(rand, placements.CLIENTS)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(rand_on_clients, '( -> {float32}@CLIENTS)')

  def test_federated_eval_rand_on_server(self):

    @computations.federated_computation
    def rand_on_server():

      @computations.tf_computation
      def rand():
        return tf.random.normal([])

      val = intrinsics.federated_eval(rand, placements.SERVER)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(rand_on_server, '( -> float32@SERVER)')

  def test_federated_map_with_client_all_equal_int(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS, True))
    def foo(x):
      val = intrinsics.federated_map(
          computations.tf_computation(lambda x: x > 10), x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(int32@CLIENTS -> {bool}@CLIENTS)')

  def test_federated_map_with_client_non_all_equal_int(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      val = intrinsics.federated_map(
          computations.tf_computation(lambda x: x > 10), x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({int32}@CLIENTS -> {bool}@CLIENTS)')

  def test_federated_map_with_server_int(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER))
    def foo(x):
      val = intrinsics.federated_map(
          computations.tf_computation(lambda x: x > 10), x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(int32@SERVER -> bool@SERVER)')

  def test_federated_map_injected_zip_with_server_int(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.SERVER),
        computation_types.FederatedType(tf.int32, placements.SERVER)
    ])
    def foo(x, y):
      val = intrinsics.federated_map(
          computations.tf_computation(lambda x, y: x > 10,
                                     ), [x, y])
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(<x=int32@SERVER,y=int32@SERVER> -> bool@SERVER)')

  def test_federated_map_injected_zip_fails_different_placements(self):

    def foo(x, y):
      val = intrinsics.federated_map(
          computations.tf_computation(lambda x, y: x > 10,
                                     ), [x, y])
      self.assertIsInstance(val, value_base.Value)
      return val

    with self.assertRaisesRegex(
        TypeError,
        'The value to be mapped must be a FederatedType or implicitly '
        'convertible to a FederatedType.'):

      computations.federated_computation(foo, [
          computation_types.FederatedType(tf.int32, placements.SERVER),
          computation_types.FederatedType(tf.int32, placements.CLIENTS)
      ])

  def test_federated_map_with_non_federated_val(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(tf.int32)
      def _(x):
        return intrinsics.federated_map(
            computations.tf_computation(lambda x: x > 10), x)

  def test_federated_sum_with_client_int(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      val = intrinsics.federated_sum(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({int32}@CLIENTS -> int32@SERVER)')

  def test_federated_sum_with_client_string(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.string, placements.CLIENTS))
      def _(x):
        return intrinsics.federated_sum(x)

  def test_federated_sum_with_server_int(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def _(x):
        return intrinsics.federated_sum(x)

  def test_federated_zip_with_client_non_all_equal_int_and_bool(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
        computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
    ])
    def foo(x, y):
      val = intrinsics.federated_zip([x, y])
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo, '(<x={int32}@CLIENTS,y=bool@CLIENTS> -> {<int32,bool>}@CLIENTS)')

  def test_federated_zip_with_single_unnamed_int_client(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
    ])
    def foo(x):
      val = intrinsics.federated_zip(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(<{int32}@CLIENTS> -> {<int32>}@CLIENTS)')

  def test_federated_zip_with_single_unnamed_int_server(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.SERVER),
    ])
    def foo(x):
      val = intrinsics.federated_zip(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(<int32@SERVER> -> <int32>@SERVER)')

  def test_federated_zip_with_single_named_bool_clients(self):

    @computations.federated_computation([
        ('a', computation_types.FederatedType(tf.bool, placements.CLIENTS)),
    ])
    def foo(x):
      val = intrinsics.federated_zip(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(<a={bool}@CLIENTS> -> {<a=bool>}@CLIENTS)')

  def test_federated_zip_with_single_named_bool_server(self):

    @computations.federated_computation([
        ('a', computation_types.FederatedType(tf.bool, placements.SERVER)),
    ])
    def foo(x):
      val = intrinsics.federated_zip(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(<a=bool@SERVER> -> <a=bool>@SERVER)')

  def test_federated_zip_with_names_client_non_all_equal_int_and_bool(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.CLIENTS),
        computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
    ])
    def foo(x, y):
      a = {'x': x, 'y': y}
      val = intrinsics.federated_zip(a)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo,
        '(<x={int32}@CLIENTS,y=bool@CLIENTS> -> {<x=int32,y=bool>}@CLIENTS)')

  def test_federated_zip_with_client_all_equal_int_and_bool(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
        computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
    ])
    def foo(x, y):
      val = intrinsics.federated_zip([x, y])
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo, '(<x=int32@CLIENTS,y=bool@CLIENTS> -> {<int32,bool>}@CLIENTS)')

  def test_federated_zip_with_names_client_all_equal_int_and_bool(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.CLIENTS, True),
        computation_types.FederatedType(tf.bool, placements.CLIENTS, True)
    ])
    def foo(arg):
      a = {'x': arg[0], 'y': arg[1]}
      val = intrinsics.federated_zip(a)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo, '(<int32@CLIENTS,bool@CLIENTS> -> {<x=int32,y=bool>}@CLIENTS)')

  def test_federated_zip_with_server_int_and_bool(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.int32, placements.SERVER),
        computation_types.FederatedType(tf.bool, placements.SERVER)
    ])
    def foo(x, y):
      val = intrinsics.federated_zip([x, y])
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo,
                     '(<x=int32@SERVER,y=bool@SERVER> -> <int32,bool>@SERVER)')

  def test_federated_zip_with_names_server_int_and_bool(self):

    @computations.federated_computation([
        ('a', computation_types.FederatedType(tf.int32, placements.SERVER)),
        ('b', computation_types.FederatedType(tf.bool, placements.SERVER)),
    ])
    def foo(arg):
      val = intrinsics.federated_zip(arg)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo, '(<a=int32@SERVER,b=bool@SERVER> -> <a=int32,b=bool>@SERVER)')

  def test_federated_zip_error_different_placements(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation([
          ('a', computation_types.FederatedType(tf.int32, placements.SERVER)),
          ('b', computation_types.FederatedType(tf.bool, placements.CLIENTS)),
      ])
      def _(arg):
        return intrinsics.federated_zip(arg)

  def test_federated_collect_with_client_int(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      val = intrinsics.federated_collect(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({int32}@CLIENTS -> int32*@SERVER)')

  def test_federated_collect_with_server_int_fails(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def _(x):
        return intrinsics.federated_collect(x)

  def test_federated_mean_with_client_float32_without_weight(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.float32, placements.CLIENTS))
    def foo(x):
      val = intrinsics.federated_mean(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({float32}@CLIENTS -> float32@SERVER)')

  def test_federated_mean_with_all_equal_client_float32_without_weight(self):
    federated_all_equal_float = computation_types.FederatedType(
        tf.float32, placements.CLIENTS, all_equal=True)

    @computations.federated_computation(federated_all_equal_float)
    def foo(x):
      val = intrinsics.federated_mean(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(float32@CLIENTS -> float32@SERVER)')

  def test_federated_mean_with_all_equal_client_float32_with_weight(self):
    federated_all_equal_float = computation_types.FederatedType(
        tf.float32, placements.CLIENTS, all_equal=True)

    @computations.federated_computation(federated_all_equal_float)
    def foo(x):
      val = intrinsics.federated_mean(x, x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(float32@CLIENTS -> float32@SERVER)')

  def test_federated_mean_with_client_tuple_with_int32_weight(self):

    @computations.federated_computation([
        computation_types.FederatedType([('x', tf.float64), ('y', tf.float64)],
                                        placements.CLIENTS),
        computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ])
    def foo(x, y):
      val = intrinsics.federated_mean(x, y)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo, '(<x={<x=float64,y=float64>}@CLIENTS,y={int32}@CLIENTS> '
        '-> <x=float64,y=float64>@SERVER)')

  def test_federated_mean_with_client_int32_fails(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.CLIENTS))
      def _(x):
        return intrinsics.federated_mean(x)

  def test_federated_mean_with_string_weight_fails(self):
    with self.assertRaises(TypeError):

      @computations.federated_computation([
          computation_types.FederatedType(tf.float32, placements.CLIENTS),
          computation_types.FederatedType(tf.string, placements.CLIENTS)
      ])
      def _(x, y):
        return intrinsics.federated_mean(x, y)

  def test_federated_aggregate_with_client_int(self):
    # The representation used during the aggregation process will be a named
    # tuple with 2 elements - the integer 'total' that represents the sum of
    # elements encountered, and the integer element 'count'.
    # pylint: disable=invalid-name
    Accumulator = collections.namedtuple('Accumulator', 'total count')
    # pylint: enable=invalid-name

    # The operator to use during the first stage simply adds an element to the
    # total and updates the count.
    @computations.tf_computation
    def accumulate(accu, elem):
      return Accumulator(accu.total + elem, accu.count + 1)

    # The operator to use during the second stage simply adds total and count.
    @computations.tf_computation
    def merge(x, y):
      return Accumulator(x.total + y.total, x.count + y.count)

    # The operator to use during the final stage simply computes the ratio.
    @computations.tf_computation
    def report(accu):
      return tf.cast(accu.total, tf.float32) / tf.cast(accu.count, tf.float32)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      val = intrinsics.federated_aggregate(x, Accumulator(0, 0), accumulate,
                                           merge, report)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({int32}@CLIENTS -> float32@SERVER)')

  def test_federated_aggregate_with_federated_zero_fails(self):

    @computations.federated_computation()
    def build_federated_zero():
      val = intrinsics.federated_value(0, placements.SERVER)
      self.assertIsInstance(val, value_base.Value)
      return val

    @computations.tf_computation([tf.int32, tf.int32])
    def accumulate(accu, elem):
      return accu + elem

    # The operator to use during the second stage simply adds total and count.
    @computations.tf_computation([tf.int32, tf.int32])
    def merge(x, y):
      return x + y

    # The operator to use during the final stage simply computes the ratio.
    @computations.tf_computation(tf.int32)
    def report(accu):
      return accu

    def foo(x):
      return intrinsics.federated_aggregate(x, build_federated_zero(),
                                            accumulate, merge, report)

    with self.assertRaisesRegex(
        TypeError, 'Expected `zero` to be assignable to type int32, '
        'but was of incompatible type int32@SERVER'):
      computations.federated_computation(
          foo, computation_types.FederatedType(tf.int32, placements.CLIENTS))

  def test_federated_aggregate_with_unknown_dimension(self):
    Accumulator = collections.namedtuple('Accumulator', ['samples'])  # pylint: disable=invalid-name
    accumulator_type = computation_types.StructType(
        Accumulator(
            samples=computation_types.TensorType(dtype=tf.int32, shape=[None])))

    @computations.tf_computation()
    def build_empty_accumulator():
      return Accumulator(samples=tf.zeros(shape=[0], dtype=tf.int32))

    # The operator to use during the first stage simply adds an element to the
    # tensor, increasing its size.
    @computations.tf_computation([accumulator_type, tf.int32])
    def accumulate(accu, elem):
      return Accumulator(
          samples=tf.concat(
              [accu.samples, tf.expand_dims(elem, axis=0)], axis=0))

    # The operator to use during the second stage simply adds total and count.
    @computations.tf_computation([accumulator_type, accumulator_type])
    def merge(x, y):
      return Accumulator(samples=tf.concat([x.samples, y.samples], axis=0))

    # The operator to use during the final stage simply computes the ratio.
    @computations.tf_computation(accumulator_type)
    def report(accu):
      return accu

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      val = intrinsics.federated_aggregate(x, build_empty_accumulator(),
                                           accumulate, merge, report)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({int32}@CLIENTS -> <samples=int32[?]>@SERVER)')

  def test_federated_reduce_with_tf_add_raw_constant(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def foo(x):
      plus = computations.tf_computation(lambda a, b: tf.add(a, b))  # pylint: disable=unnecessary-lambda
      val = intrinsics.federated_reduce(x, 0, plus)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '({int32}@CLIENTS -> int32@SERVER)')

  def test_num_over_temperature_threshold_example(self):

    @computations.federated_computation([
        computation_types.FederatedType(tf.float32, placements.CLIENTS),
        computation_types.FederatedType(tf.float32, placements.SERVER)
    ])
    def foo(temperatures, threshold):
      val = intrinsics.federated_sum(
          intrinsics.federated_map(
              computations.tf_computation(
                  lambda x, y: tf.cast(tf.greater(x, y), tf.int32)),
              [temperatures,
               intrinsics.federated_broadcast(threshold)]))
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(
        foo,
        '(<temperatures={float32}@CLIENTS,threshold=float32@SERVER> -> int32@SERVER)'
    )

  @parameterized.named_parameters(('test_n_2', 2), ('test_n_3', 3),
                                  ('test_n_5', 5))
  def test_n_tuple_federated_zip_tensor_args(self, n):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    initial_tuple_type = computation_types.StructType([fed_type] * n)
    final_fed_type = computation_types.FederatedType([tf.int32] * n,
                                                     placements.CLIENTS)
    function_type = computation_types.FunctionType(initial_tuple_type,
                                                   final_fed_type)

    @computations.federated_computation(
        [computation_types.FederatedType(tf.int32, placements.CLIENTS)] * n)
    def foo(x):
      val = intrinsics.federated_zip(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, function_type.compact_representation())

  @parameterized.named_parameters(
      ('test_n_2_int', 2,
       computation_types.FederatedType(tf.int32, placements.CLIENTS)),
      ('test_n_3_int', 3,
       computation_types.FederatedType(tf.int32, placements.CLIENTS)),
      ('test_n_5_int', 5,
       computation_types.FederatedType(tf.int32, placements.CLIENTS)),
      ('test_n_2_tuple', 2,
       computation_types.FederatedType([tf.int32, tf.int32],
                                       placements.CLIENTS)),
      ('test_n_3_tuple', 3,
       computation_types.FederatedType([tf.int32, tf.int32],
                                       placements.CLIENTS)),
      ('test_n_5_tuple', 5,
       computation_types.FederatedType([tf.int32, tf.int32],
                                       placements.CLIENTS)))
  def test_named_n_tuple_federated_zip(self, n, fed_type):
    initial_tuple_type = computation_types.StructType([fed_type] * n)
    named_fed_type = computation_types.FederatedType(
        [(str(k), fed_type.member) for k in range(n)], placements.CLIENTS)
    mixed_fed_type = computation_types.FederatedType(
        [(str(k), fed_type.member) if k % 2 == 0 else fed_type.member
         for k in range(n)], placements.CLIENTS)
    named_function_type = computation_types.FunctionType(
        initial_tuple_type, named_fed_type)
    mixed_function_type = computation_types.FunctionType(
        initial_tuple_type, mixed_fed_type)

    @computations.federated_computation([fed_type] * n)
    def foo(x):
      arg = {str(k): x[k] for k in range(n)}
      val = intrinsics.federated_zip(arg)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, named_function_type.compact_representation())

    def _make_test_tuple(x, k):
      """Make a test tuple with a name if k is even, otherwise unnamed."""
      if k % 2 == 0:
        return str(k), x[k]
      else:
        return None, x[k]

    @computations.federated_computation([fed_type] * n)
    def bar(x):
      arg = structure.Struct(_make_test_tuple(x, k) for k in range(n))
      val = intrinsics.federated_zip(arg)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(bar, mixed_function_type.compact_representation())

  @parameterized.named_parameters([
      ('test_n_' + str(n) + '_m_' + str(m), n, m)
      for n, m in itertools.product([1, 2, 3], [1, 2, 3])
  ])
  def test_n_tuple_federated_zip_mixed_args(self, n, m):
    tuple_fed_type = computation_types.FederatedType([tf.int32, tf.int32],
                                                     placements.CLIENTS)
    single_fed_type = computation_types.FederatedType(tf.int32,
                                                      placements.CLIENTS)
    initial_tuple_type = computation_types.StructType([tuple_fed_type] * n +
                                                      [single_fed_type] * m)
    final_fed_type = computation_types.FederatedType(
        [[tf.int32, tf.int32]] * n + [tf.int32] * m, placements.CLIENTS)
    function_type = computation_types.FunctionType(initial_tuple_type,
                                                   final_fed_type)

    @computations.federated_computation([
        computation_types.FederatedType(
            computation_types.StructType([tf.int32, tf.int32]),
            placements.CLIENTS)
    ] * n + [computation_types.FederatedType(tf.int32, placements.CLIENTS)] * m)
    def baz(x):
      val = intrinsics.federated_zip(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(baz, function_type.compact_representation())

  def test_federated_apply_raises_warning(self):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')

      @computations.federated_computation(
          computation_types.FederatedType(tf.int32, placements.SERVER))
      def foo(x):
        val = intrinsics.federated_apply(
            computations.tf_computation(lambda x: x * x), x)
        self.assertIsInstance(val, value_base.Value)
        return val

      self.assertLen(w, 1)
      self.assertIsInstance(w[0].category(), DeprecationWarning)
      self.assertIn('tff.federated_apply() is deprecated', str(w[0].message))
      self.assert_type(foo, '(int32@SERVER -> int32@SERVER)')

  def test_federated_value_with_bool_on_clients(self):

    @computations.federated_computation(tf.bool)
    def foo(x):
      val = intrinsics.federated_value(x, placements.CLIENTS)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(bool -> bool@CLIENTS)')

  def test_federated_value_raw_np_scalar(self):

    @computations.federated_computation
    def foo():
      floatv = np.float64(0)
      tff_float = intrinsics.federated_value(floatv, placements.SERVER)
      self.assertIsInstance(tff_float, value_base.Value)
      self.assert_type(tff_float, 'float64@SERVER')
      intv = np.int64(0)
      tff_int = intrinsics.federated_value(intv, placements.SERVER)
      self.assertIsInstance(tff_int, value_base.Value)
      self.assert_type(tff_int, 'int64@SERVER')
      return (tff_float, tff_int)

    self.assert_type(foo, '( -> <float64@SERVER,int64@SERVER>)')

  def test_federated_value_raw_tf_scalar_variable(self):
    v = tf.Variable(initial_value=0., name='test_var')
    with self.assertRaisesRegex(
        TypeError, 'TensorFlow construct (.*) has been '
        'encountered in a federated context.'):

      @computations.federated_computation()
      def _():
        return intrinsics.federated_value(v, placements.SERVER)

  def test_federated_value_with_bool_on_server(self):

    @computations.federated_computation(tf.bool)
    def foo(x):
      val = intrinsics.federated_value(x, placements.SERVER)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo, '(bool -> bool@SERVER)')

  def test_sequence_sum(self):

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def foo1(x):
      val = intrinsics.sequence_sum(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo1, '(int32* -> int32)')

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.SERVER))
    def foo2(x):
      val = intrinsics.sequence_sum(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo2, '(int32*@SERVER -> int32@SERVER)')

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.CLIENTS))
    def foo3(x):
      val = intrinsics.sequence_sum(x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo3, '({int32*}@CLIENTS -> {int32}@CLIENTS)')

  def test_sequence_map(self):

    @computations.tf_computation(tf.int32)
    def over_threshold(x):
      return x > 10

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def foo1(x):
      val = intrinsics.sequence_map(over_threshold, x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo1, '(int32* -> bool*)')

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.SERVER))
    def foo2(x):
      val = intrinsics.sequence_map(over_threshold, x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo2, '(int32*@SERVER -> bool*@SERVER)')

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.CLIENTS))
    def foo3(x):
      val = intrinsics.sequence_map(over_threshold, x)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo3, '({int32*}@CLIENTS -> {bool*}@CLIENTS)')

  def test_sequence_reduce(self):
    add_numbers = computations.tf_computation(
        lambda a, b: tf.add(a, b),  # pylint: disable=unnecessary-lambda
        [tf.int32, tf.int32])

    @computations.federated_computation(
        computation_types.SequenceType(tf.int32))
    def foo1(x):
      val = intrinsics.sequence_reduce(x, 0, add_numbers)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo1, '(int32* -> int32)')

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.SERVER))
    def foo2(x):
      val = intrinsics.sequence_reduce(x, 0, add_numbers)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo2, '(int32*@SERVER -> int32@SERVER)')

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(tf.int32), placements.CLIENTS))
    def foo3(x):
      val = intrinsics.sequence_reduce(x, 0, add_numbers)
      self.assertIsInstance(val, value_base.Value)
      return val

    self.assert_type(foo3, '({int32*}@CLIENTS -> {int32}@CLIENTS)')


if __name__ == '__main__':
  common_test.main()
