# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for eager_executor.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import absltest

import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import eager_executor


class EagerExecutorTest(absltest.TestCase):

  # TODO(b/134764569): Potentially take advantage of something similar to the
  # `tf.test.TestCase.evaluate()` to avoid having to call `.numpy()` everywhere.

  def test_get_available_devices(self):
    devices = eager_executor.get_available_devices()
    self.assertIsInstance(devices, list)
    self.assertIn('/CPU:0', devices)

  def test_embed_tensorflow_computation_with_int_arg_and_result(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return x + 1

    fn = eager_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(10)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 11)

  def test_embed_tensorflow_computation_with_no_arg_and_int_result(self):

    @computations.tf_computation
    def comp():
      return 1000

    fn = eager_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn()
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 1000)

  def test_embed_tensorflow_computation_with_tuple_arg_and_result(self):

    @computations.tf_computation([('a', tf.int32), ('b', tf.int32)])
    def comp(a, b):
      return {'sum': a + b}

    fn = eager_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    p = tf.constant(10)
    q = tf.constant(20)
    result = fn(anonymous_tuple.AnonymousTuple([('a', p), ('b', q)]))
    self.assertIsInstance(result, anonymous_tuple.AnonymousTuple)
    self.assertCountEqual(dir(result), ['sum'])
    self.assertIsInstance(result.sum, tf.Tensor)
    self.assertEqual(result.sum.numpy(), 30)

  def test_to_representation_for_type_with_int(self):
    v = eager_executor.to_representation_for_type(10, tf.int32)
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v.numpy(), 10)
    self.assertEqual(v.dtype, tf.int32)

  def test_to_representation_for_type_with_int_on_specific_device(self):
    v = eager_executor.to_representation_for_type(10, tf.int32, '/CPU:0')
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v.numpy(), 10)
    self.assertEqual(v.dtype, tf.int32)
    self.assertTrue(v.device.endswith('CPU:0'))

  def test_eager_value_constructor_with_int_constant(self):
    v = eager_executor.EagerValue(10, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertIsInstance(v.internal_representation, tf.Tensor)
    self.assertEqual(v.internal_representation.numpy(), 10)

  def test_executor_constructor_fails_if_not_in_eager_mode(self):
    with tf.Graph().as_default():
      with self.assertRaises(RuntimeError):
        eager_executor.EagerExecutor()

  def test_executor_constructor_fails_if_bogus_device_name(self):
    with self.assertRaises(TypeError):
      eager_executor.EagerExecutor(10)
    with self.assertRaises(ValueError):
      eager_executor.EagerExecutor('Mary had a little lamb.')

  def test_executor_construction_with_correct_device_name(self):
    eager_executor.EagerExecutor('/CPU:0')

  def test_executor_construction_with_no_device_name(self):
    eager_executor.EagerExecutor()

  def test_executor_ingest_int(self):
    ex = eager_executor.EagerExecutor()
    val = ex.ingest(10, tf.int32)
    self.assertIsInstance(val, eager_executor.EagerValue)
    self.assertIsInstance(val.internal_representation, tf.Tensor)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertEqual(val.internal_representation.numpy(), 10)

  def test_executor_ingest_unnamed_int_pair(self):
    ex = eager_executor.EagerExecutor()
    val = ex.ingest([10, {
        'a': 20
    }], [tf.int32, collections.OrderedDict([('a', tf.int32)])])
    self.assertIsInstance(val, eager_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '<int32,<a=int32>>')
    self.assertIsInstance(val.internal_representation,
                          anonymous_tuple.AnonymousTuple)
    self.assertLen(val.internal_representation, 2)
    self.assertIsInstance(val.internal_representation[0], tf.Tensor)
    self.assertIsInstance(val.internal_representation[1],
                          anonymous_tuple.AnonymousTuple)
    self.assertLen(val.internal_representation[1], 1)
    self.assertEqual(dir(val.internal_representation[1]), ['a'])
    self.assertIsInstance(val.internal_representation[1][0], tf.Tensor)
    self.assertEqual(val.internal_representation[0].numpy(), 10)
    self.assertEqual(val.internal_representation[1][0].numpy(), 20)

  def test_executor_ingest_no_arg_computation(self):
    ex = eager_executor.EagerExecutor()

    @computations.tf_computation
    def comp():
      return 1000

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)
    val = ex.ingest(comp_proto, computation_types.FunctionType(None, tf.int32))
    self.assertIsInstance(val, eager_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '( -> int32)')
    self.assertTrue(callable(val.internal_representation))
    result = val.internal_representation()
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 1000)

  def test_executor_ingest_two_arg_computation(self):
    ex = eager_executor.EagerExecutor()

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)
    val = ex.ingest(
        comp_proto,
        computation_types.FunctionType([tf.int32, tf.int32], tf.int32))
    self.assertIsInstance(val, eager_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '(<int32,int32> -> int32)')
    self.assertTrue(callable(val.internal_representation))
    arg = anonymous_tuple.AnonymousTuple([('a', tf.constant(10)),
                                          ('b', tf.constant(10))])
    result = val.internal_representation(arg)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 20)

  def test_executor_invoke_add_numbers(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    result = eager_executor.EagerExecutor().invoke(comp, [10, 20])
    self.assertIsInstance(result, eager_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32')
    self.assertIsInstance(result.internal_representation, tf.Tensor)
    self.assertEqual(result.internal_representation.numpy(), 30)

  def test_executor_invoke_take_two_int_from_finite_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.take(2)

    ds = tf.data.Dataset.from_tensor_slices([10, 20, 30, 40, 50])
    result = eager_executor.EagerExecutor().invoke(comp, ds)
    self.assertIsInstance(result, eager_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32*')
    self.assertIn('Dataset', type(result.internal_representation).__name__)
    self.assertCountEqual([x.numpy() for x in result.internal_representation],
                          [10, 20])

  def test_executor_as_context_with_int_args(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    with context_stack_impl.context_stack.install(
        eager_executor.EagerExecutor()):
      a = comp(10, 20)
      b = comp(30, 40)
      result = comp(a, b)

    self.assertIsInstance(result, eager_executor.EagerValue)
    self.assertEqual(result.internal_representation.numpy(), 100)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
