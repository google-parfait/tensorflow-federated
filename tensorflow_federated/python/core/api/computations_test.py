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
"""Integration tests for federated and tensorflow computations.

These tests test the public TFF core API surface by defining and executing
computations; tests are grouped into `TestCase`s based on the kind of
computation. Many of these tests are parameterized to test different parts of
the  TFF implementation, for example different executor stacks.
"""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test as common_test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import do_not_use_compiler
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_test_utils


class TensorFlowComputationsTest(parameterized.TestCase):

  def test_py_and_tf_args(self):

    @tf.function(autograph=False)
    def foo(x, y, add=True):
      return x + y if add else x - y

      # Note: tf.Functions support mixing tensorflow and Python arguments,
      # usually with the semantics you would expect. Currently, TFF does not
      # support this kind of mixing, even for Polymorphic TFF functions.
      # However, you can work around this by explicitly binding any Python
      # arguments on a tf.Function:

    tf_poly_add = computations.tf_computation(lambda x, y: foo(x, y, True))
    tf_poly_sub = computations.tf_computation(lambda x, y: foo(x, y, False))
    self.assertEqual(tf_poly_add(2, 1), 3)
    self.assertEqual(tf_poly_add(2., 1.), 3.)
    self.assertEqual(tf_poly_sub(2, 1), 1)

  def test_with_variable(self):

    v_slot = []

    @tf.function(autograph=False)
    def foo(x):
      if not v_slot:
        v_slot.append(tf.Variable(0))
      v = v_slot[0]
      v.assign(1)
      return v + x

    tf_comp = computations.tf_computation(foo, tf.int32)
    self.assertEqual(tf_comp(1), 2)

  def test_one_param(self):

    @tf.function
    def foo(x):
      return x + 1

    tf_comp = computations.tf_computation(foo, tf.int32)
    self.assertEqual(tf_comp(1), 2)

  def test_no_params_structured_outputs(self):
    # We also test that the correct Python containers are returned.
    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo():
      d = collections.OrderedDict([('foo', 3.0), ('bar', 5.0)])
      return (1, 2, d, MyType(True, False), [1.5, 3.0], (1,))

    tf_comp = computations.tf_computation(foo, None)
    result = tf_comp()
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 2)
    self.assertEqual(result[2], {'foo': 3.0, 'bar': 5.0})
    self.assertEqual(type(result[2]), collections.OrderedDict)
    self.assertEqual(result[3], MyType(True, False))
    self.assertEqual(type(result[3]), MyType)
    self.assertEqual(result[4], [1.5, 3.0])
    self.assertEqual(type(result[4]), list)
    self.assertEqual(result[5], (1,))
    self.assertEqual(type(result[5]), tuple)

  def test_explicit_tuple_param(self):
    # See also test_polymorphic_tuple_input
    @tf.function
    def foo(t):
      return t[0] + t[1]

    tf_comp = computations.tf_computation(foo, (tf.int32, tf.int32))
    self.assertEqual(tf_comp((1, 2)), 3)

  def test_polymorphic_tuple_input(self):

    def foo(t):
      return t[0] + t[1]

    tf_poly = computations.tf_computation(foo)
    self.assertEqual(tf_poly((1, 2)), 3)

  def test_nested_tuple_input_polymorphic(self):

    @tf.function(autograph=False)
    def foo(tuple1, tuple2):
      return tuple1[0] + tuple1[1][0] + tuple1[1][1] + tuple2[0] + tuple2[1]

    # Polymorphic
    tf_poly = computations.tf_computation(foo)
    self.assertEqual(tf_poly((1, (2, 3)), (1, 2)), 9)
    self.assertEqual(tf_poly((1, (2, 3)), (0, 0)), 6)

  def test_nested_tuple_input_explicit_types(self):

    @tf.function(autograph=False)
    def foo(tuple1, tuple2):
      return tuple1[0] + tuple1[1][0] + tuple1[1][1] + tuple2[0] + tuple2[1]

    tff_type = [(tf.int32, (tf.int32, tf.int32)), (tf.int32, tf.int32)]
    tf_comp = computations.tf_computation(foo, tff_type)
    self.assertEqual(tf_comp((1, (2, 3)), (0, 0)), 6)

  def test_namedtuple_param(self):

    MyType = collections.namedtuple('MyType', ['x', 'y'])  # pylint: disable=invalid-name

    @tf.function
    def foo(t):
      self.assertIsInstance(t, MyType)
      return t.x + t.y

    # Explicit type
    tf_comp = computations.tf_computation(foo, MyType(tf.int32, tf.int32))
    self.assertEqual(tf_comp(MyType(1, 2)), 3)

    # Polymorphic
    tf_comp = computations.tf_computation(foo)
    self.assertEqual(tf_comp(MyType(1, 2)), 3)

  @executor_test_utils.executors
  def test_complex_param(self):
    # See also test_nested_tuple_input

    MyType = collections.namedtuple('MyType', ['x', 'd'])  # pylint: disable=invalid-name

    @tf.function
    def foo(t, odict, unnamed_tuple):
      self.assertIsInstance(t, MyType)
      self.assertIsInstance(t.d, dict)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(unnamed_tuple, tuple)
      return t.x + t.d['y'] + t.d['z'] + odict['o'] + unnamed_tuple[0]

    args = [
        MyType(1, dict(y=2, z=3)),
        collections.OrderedDict([('o', 0)]),
        (0,),
    ]
    arg_type = [
        MyType(tf.int32, collections.OrderedDict(y=tf.int32, z=tf.int32)),
        collections.OrderedDict([('o', tf.int32)]),
        (tf.int32,),
    ]

    # Explicit type
    tf_comp = computations.tf_computation(foo, arg_type)
    self.assertEqual(tf_comp(*args), 6)

    # Polymorphic
    tf_comp = computations.tf_computation(foo)
    self.assertEqual(tf_comp(*args), 6)

  @executor_test_utils.executors
  def test_with_none_tensorshape_param(self):

    @computations.tf_computation(
        computation_types.TensorType(
            dtype=tf.int32, shape=tf.TensorShape(None)))
    def add_one(x):
      return x + 1

    self.assertEqual(add_one(0), 1)


class TensorFlowComputationsWithDatasetsTest(parameterized.TestCase):

  @executor_test_utils.executors
  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @common_test.skip_test_for_gpu
  def test_with_tf_datasets(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def consume(ds):
      return ds.reduce(np.int64(0), lambda x, y: x + y)

    self.assertEqual(str(consume.type_signature), '(int64* -> int64)')

    @computations.tf_computation
    def produce():
      return tf.data.Dataset.range(10)

    self.assertEqual(str(produce.type_signature), '( -> int64*)')

    self.assertEqual(consume(produce()), 45)

  # TODO(b/131363314): The reference executor should support generating and
  # returning infinite datasets
  @executor_test_utils.executors(
      ('local', executor_stacks.local_executor_factory(1)),)
  def test_consume_infinite_tf_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def consume(ds):
      # Consume the first 10 elements of the dataset.
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    self.assertEqual(consume(tf.data.Dataset.range(10).repeat()), 45)

  # TODO(b/131363314): The reference executor should support generating and
  # returning infinite datasets
  @executor_test_utils.executors(
      ('local', executor_stacks.local_executor_factory(1)),)
  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @common_test.skip_test_for_gpu
  def test_produce_and_consume_infinite_tf_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def consume(ds):
      # Consume the first 10 elements of the dataset.
      return ds.take(10).reduce(np.int64(0), lambda x, y: x + y)

    @computations.tf_computation
    def produce():
      # Produce an infinite dataset.
      return tf.data.Dataset.range(10).repeat()

    self.assertEqual(consume(produce()), 45)

  @executor_test_utils.executors
  def test_with_sequence_of_pairs(self):
    pairs = tf.data.Dataset.from_tensor_slices(
        (list(range(5)), list(range(5, 10))))

    @computations.tf_computation
    def process_pairs(ds):
      return ds.reduce(0, lambda state, pair: state + pair[0] + pair[1])

    self.assertEqual(process_pairs(pairs), 45)

  @executor_test_utils.executors
  def test_tf_comp_with_sequence_inputs_and_outputs_does_not_fail(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def _(x):
      return x

  @executor_test_utils.executors
  def test_with_four_element_dataset_pipeline(self):

    @computations.tf_computation
    def comp1():
      return tf.data.Dataset.range(5)

    @computations.tf_computation(computation_types.SequenceType(tf.int64))
    def comp2(ds):
      return ds.map(lambda x: tf.cast(x + 1, tf.float32))

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def comp3(ds):
      return ds.repeat(5)

    @computations.tf_computation(computation_types.SequenceType(tf.float32))
    def comp4(ds):
      return ds.reduce(0.0, lambda x, y: x + y)

    @computations.tf_computation
    def comp5():
      return comp4(comp3(comp2(comp1())))

    self.assertEqual(comp5(), 75.0)


if __name__ == '__main__':
  do_not_use_compiler._do_not_use_set_local_execution_context()
  common_test.main()
