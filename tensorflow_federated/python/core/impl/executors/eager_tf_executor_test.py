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

import asyncio
import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_test_utils


def _get_logical_devices_for_testing(device_list=('CPU', 'GPU', 'TPU')):
  result = []
  for dev in tf.config.list_logical_devices():
    parts = dev.name.split(':')
    if ((len(parts) == 3) and (parts[0] == '/device') and
        (parts[1] in device_list)):
      result.append(':'.join(parts[1:]))
  return result


def create_test_executor_factory():
  executor = eager_tf_executor.EagerTFExecutor()
  return executor_factory.ExecutorFactoryImpl(lambda _: executor)


class EagerTFExecutorTest(tf.test.TestCase):

  # TODO(b/134764569): Potentially take advantage of something similar to the
  # `tf.test.TestCase.evaluate()` to avoid having to call `.numpy()` everywhere.

  def setUp(self):
    super().setUp()
    self._logical_devices_cpu = _get_logical_devices_for_testing(['CPU'])
    self._logical_devices_gpu = _get_logical_devices_for_testing(['GPU'])
    self._logical_devices_tpu = _get_logical_devices_for_testing(['TPU'])

  def test_embed_tensorflow_computation_with_int_arg_and_result(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return x + 1

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(10)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 11)

  def test_embed_tensorflow_computation_with_float(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      return x + 0.5

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(10.0)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 10.5)

  def test_embed_tensorflow_computation_with_no_arg_and_int_result(self):

    @computations.tf_computation
    def comp():
      return 1000

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn()
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 1000)

  def test_embed_tensorflow_computation_with_dataset_arg_and_int_result(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.reduce(np.int32(0), lambda p, q: p + q)

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(tf.data.Dataset.from_tensor_slices([10, 20]))
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 30)

  def test_embed_tensorflow_computation_with_tuple_arg_and_result(self):

    @computations.tf_computation([('a', tf.int32), ('b', tf.int32)])
    def comp(a, b):
      return {'sum': a + b}

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    p = tf.constant(10)
    q = tf.constant(20)
    result = fn(anonymous_tuple.AnonymousTuple([('a', p), ('b', q)]))
    self.assertIsInstance(result, anonymous_tuple.AnonymousTuple)
    self.assertCountEqual(dir(result), ['sum'])
    self.assertIsInstance(result.sum, tf.Tensor)
    self.assertEqual(result.sum.numpy(), 30)

  def test_embed_tensorflow_computation_with_variable_v1(self):

    @computations.tf_computation
    def comp():
      x = tf.Variable(10)
      with tf.control_dependencies([x.initializer]):
        return tf.add(x, 20)

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn()
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 30)

  def test_embed_tensorflow_computation_with_variable_v2(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      v = tf.Variable(10)
      with tf.control_dependencies([v.initializer]):
        with tf.control_dependencies([v.assign_add(20)]):
          return tf.add(x, v)

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(30)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 60)

  def test_embed_tensorflow_computation_with_float_variables_same_name(self):

    @computations.tf_computation
    def comp1():
      x = tf.Variable(0.5, name='bob')
      with tf.control_dependencies([x.initializer]):
        return tf.add(x, 0.6)

    @computations.tf_computation
    def comp2():
      x = tf.Variable(0.5, name='bob')
      with tf.control_dependencies([x.initializer]):
        return tf.add(x, 0.7)

    fns = [
        eager_tf_executor.embed_tensorflow_computation(
            computation_impl.ComputationImpl.get_proto(x))
        for x in [comp1, comp2]
    ]
    results = [f() for f in fns]
    for res in results:
      self.assertIsInstance(res, tf.Tensor)
    self.assertAlmostEqual(results[0].numpy(), 1.1)
    self.assertAlmostEqual(results[1].numpy(), 1.2)

  def test_to_representation_for_type_with_int(self):
    value = 10
    type_signature = computation_types.TensorType(tf.int32)
    v = eager_tf_executor.to_representation_for_type(value, {}, type_signature)
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v.numpy(), 10)
    self.assertEqual(v.dtype, tf.int32)

  def test_to_representation_for_tf_variable(self):
    value = tf.Variable(10, dtype=tf.int32)
    type_signature = computation_types.TensorType(tf.int32)
    v = eager_tf_executor.to_representation_for_type(value, {}, type_signature)
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v.numpy(), 10)
    self.assertEqual(v.dtype, tf.int32)

  def test_to_representation_for_type_with_int_on_specific_device(self):
    value = 10
    type_signature = computation_types.TensorType(tf.int32)
    v = eager_tf_executor.to_representation_for_type(value, {}, type_signature,
                                                     '/CPU:0')
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v.numpy(), 10)
    self.assertEqual(v.dtype, tf.int32)
    self.assertTrue(v.device.endswith('CPU:0'))

  def test_eager_value_constructor_with_int_constant(self):
    v = eager_tf_executor.EagerValue(10, {}, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertIsInstance(v.internal_representation, tf.Tensor)
    self.assertEqual(v.internal_representation.numpy(), 10)

  def test_executor_constructor_fails_if_not_in_eager_mode(self):
    with tf.Graph().as_default():
      with self.assertRaises(RuntimeError):
        eager_tf_executor.EagerTFExecutor()

  def test_executor_construction_with_correct_device_name(self):
    eager_tf_executor.EagerTFExecutor('/CPU:0')

  def test_executor_construction_with_no_device_name(self):
    eager_tf_executor.EagerTFExecutor()

  def test_executor_create_value_int(self):
    ex = eager_tf_executor.EagerTFExecutor()
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(10, tf.int32))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertIsInstance(val.internal_representation, tf.Tensor)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertEqual(val.internal_representation.numpy(), 10)

  def test_executor_create_value_unnamed_int_pair(self):
    ex = eager_tf_executor.EagerTFExecutor()
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value([10, {
            'a': 20
        }], [tf.int32, collections.OrderedDict([('a', tf.int32)])]))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
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

  def test_executor_create_value_no_arg_computation(self):
    ex = eager_tf_executor.EagerTFExecutor()

    @computations.tf_computation
    def comp():
      return 1000

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(comp_proto,
                        computation_types.FunctionType(None, tf.int32)))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '( -> int32)')
    self.assertTrue(callable(val.internal_representation))
    result = val.internal_representation()
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 1000)

  def test_executor_create_value_two_arg_computation(self):
    ex = eager_tf_executor.EagerTFExecutor()

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(
            comp_proto,
            computation_types.FunctionType([tf.int32, tf.int32], tf.int32)))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '(<int32,int32> -> int32)')
    self.assertTrue(callable(val.internal_representation))
    arg = anonymous_tuple.AnonymousTuple([('a', tf.constant(10)),
                                          ('b', tf.constant(10))])
    result = val.internal_representation(arg)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result.numpy(), 20)

  def test_executor_create_call_add_numbers(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value([10, 20], comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32')
    self.assertIsInstance(result.internal_representation, tf.Tensor)
    self.assertEqual(result.internal_representation.numpy(), 30)

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_executor_create_call_take_two_int_from_finite_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.take(2)

    ds = tf.data.Dataset.from_tensor_slices([10, 20, 30, 40, 50])
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value(ds, comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32*')
    self.assertIn('Dataset', type(result.internal_representation).__name__)
    self.assertCountEqual([x.numpy() for x in result.internal_representation],
                          [10, 20])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_executor_create_call_take_three_int_from_infinite_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.take(3)

    ds = tf.data.Dataset.from_tensor_slices([10]).repeat()
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value(ds, comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32*')
    self.assertIn('Dataset', type(result.internal_representation).__name__)
    self.assertCountEqual([x.numpy() for x in result.internal_representation],
                          [10, 10, 10])

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_executor_create_call_reduce_first_five_from_infinite_dataset(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.take(5).reduce(np.int32(0), lambda p, q: p + q)

    ds = tf.data.Dataset.from_tensor_slices([10, 20, 30]).repeat()
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value(ds, comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32')
    self.assertIsInstance(result.internal_representation, tf.Tensor)
    self.assertEqual(result.internal_representation.numpy(), 90)

  # TODO(b/137602785): bring GPU test back after the fix for `wrap_function`.
  @test.skip_test_for_gpu
  def test_executor_create_call_with_dataset_of_tuples(self):

    element = collections.namedtuple('_', 'a b')

    @computations.tf_computation(
        computation_types.SequenceType(element(tf.int32, tf.int32)))
    def comp(ds):
      return ds.reduce(
          element(np.int32(0), np.int32(0)),
          lambda p, q: element(p.a + q.a, p.b + q.b))

    ds = tf.data.Dataset.from_tensor_slices(element([10, 20, 30], [4, 5, 6]))
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value(ds, comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), '<a=int32,b=int32>')
    self.assertIsInstance(result.internal_representation,
                          anonymous_tuple.AnonymousTuple)
    self.assertCountEqual(dir(result.internal_representation), ['a', 'b'])
    self.assertIsInstance(result.internal_representation.a, tf.Tensor)
    self.assertIsInstance(result.internal_representation.b, tf.Tensor)
    self.assertEqual(result.internal_representation.a.numpy(), 60)
    self.assertEqual(result.internal_representation.b.numpy(), 15)

  def test_executor_create_tuple_and_selection(self):
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    v1, v2 = tuple(
        loop.run_until_complete(
            asyncio.gather(*[ex.create_value(x, tf.int32) for x in [10, 20]])))
    v3 = loop.run_until_complete(
        ex.create_tuple(collections.OrderedDict([('a', v1), ('b', v2)])))
    self.assertIsInstance(v3, eager_tf_executor.EagerValue)
    self.assertIsInstance(v3.internal_representation,
                          anonymous_tuple.AnonymousTuple)
    self.assertLen(v3.internal_representation, 2)
    self.assertCountEqual(dir(v3.internal_representation), ['a', 'b'])
    self.assertIsInstance(v3.internal_representation[0], tf.Tensor)
    self.assertIsInstance(v3.internal_representation[1], tf.Tensor)
    self.assertEqual(str(v3.type_signature), '<a=int32,b=int32>')
    self.assertEqual(v3.internal_representation[0].numpy(), 10)
    self.assertEqual(v3.internal_representation[1].numpy(), 20)
    v4 = loop.run_until_complete(ex.create_selection(v3, name='a'))
    self.assertIsInstance(v4, eager_tf_executor.EagerValue)
    self.assertIsInstance(v4.internal_representation, tf.Tensor)
    self.assertEqual(str(v4.type_signature), 'int32')
    self.assertEqual(v4.internal_representation.numpy(), 10)
    v5 = loop.run_until_complete(ex.create_selection(v3, index=1))
    self.assertIsInstance(v5, eager_tf_executor.EagerValue)
    self.assertIsInstance(v5.internal_representation, tf.Tensor)
    self.assertEqual(str(v5.type_signature), 'int32')
    self.assertEqual(v5.internal_representation.numpy(), 20)
    with self.assertRaises(ValueError):
      loop.run_until_complete(ex.create_selection(v3, name='a', index=1))
    with self.assertRaises(ValueError):
      loop.run_until_complete(ex.create_selection(v3))

  def test_executor_compute(self):
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    val = loop.run_until_complete(ex.create_value(10, tf.int32))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    val = loop.run_until_complete(val.compute())
    self.assertIsInstance(val, tf.Tensor)
    self.assertEqual(val.numpy(), 10)

  def test_with_repeated_variable_assignment(self):
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()

    @computations.tf_computation(tf.int32)
    def comp(x):
      v = tf.Variable(10)
      with tf.control_dependencies([v.initializer]):
        with tf.control_dependencies([v.assign(x)]):
          with tf.control_dependencies([v.assign_add(10)]):
            return tf.identity(v)

    fn = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(ex.create_value(10, tf.int32))
    for n in range(10):
      arg = loop.run_until_complete(ex.create_call(fn, arg))
      val = loop.run_until_complete(arg.compute())
      self.assertEqual(val.numpy(), 10 * (n + 2))

  def test_execution_of_tensorflow(self):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    executor = create_test_executor_factory()
    with executor_test_utils.install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)

  # TODO(b/155239129): used list_logical_device in `setUp` for GPU tests.
  def _get_wrap_function_on_device(self, device):
    with tf.Graph().as_default() as graph:
      x = tf.compat.v1.placeholder(tf.int32, shape=[])
      y = tf.add(x, tf.constant(1))

    arg_for_tf_device = '/{}'.format(device)

    def _function_to_wrap(arg):
      with tf.device(arg_for_tf_device):
        return tf.import_graph_def(
            graph.as_graph_def(),
            input_map={x.name: arg},
            return_elements=[y.name])[0]

    signature = [tf.TensorSpec([], tf.int32)]
    wrapped_fn = tf.compat.v1.wrap_function(_function_to_wrap, signature)

    def fn(arg):
      with tf.device(arg_for_tf_device):
        return wrapped_fn(arg)

    result = fn(tf.constant(10))
    return result

  def test_wrap_function_on_all_available_logical_devices_cpu(self):
    for device in self._logical_devices_cpu:
      self.assertTrue(
          self._get_wrap_function_on_device(device).device.endswith(device))

  def test_wrap_function_on_all_available_logical_devices_gpu(self):
    for device in self._logical_devices_gpu:
      self.assertTrue(
          self._get_wrap_function_on_device(device).device.endswith(device))

  def test_wrap_function_on_all_available_logical_devices_tpu(self):
    for device in self._logical_devices_tpu:
      self.assertTrue(
          self._get_wrap_function_on_device(device).device.endswith(device))

  def test_embed_tensorflow_computation_fails_with_bogus_device(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 1)

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)

    with self.assertRaises(ValueError):
      eager_tf_executor.embed_tensorflow_computation(
          comp_proto, comp.type_signature, device='/there_is_no_such_device')

  # TODO(b/155239129): used list_logical_device in `setUp` for GPU tests.
  def _get_embed_tensorflow_computation_succeeds_with_device(self, device):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 1)

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)

    fn = eager_tf_executor.embed_tensorflow_computation(
        comp_proto, comp.type_signature, device='/{}'.format(device))
    result = fn(tf.constant(20))
    return result

  def test_embed_tensorflow_computation_succeeds_with_cpu(self):
    for device in self._logical_devices_cpu:
      self.assertTrue(
          self._get_embed_tensorflow_computation_succeeds_with_device(
              device).device.endswith(device))

  def test_embed_tensorflow_computation_succeeds_with_gpu(self):
    for device in self._logical_devices_gpu:
      self.assertTrue(
          self._get_embed_tensorflow_computation_succeeds_with_device(
              device).device.endswith(device))

  def test_embed_tensorflow_computation_succeeds_with_tpu(self):
    for device in self._logical_devices_tpu:
      self.assertTrue(
          self._get_embed_tensorflow_computation_succeeds_with_device(
              device).device.endswith(device))

  # TODO(b/155239129): used list_logical_device in `setUp` for GPU tests.
  def _get_to_representation_for_type_succeeds_on_device(self, device):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 1)

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)

    fn = eager_tf_executor.to_representation_for_type(
        comp_proto, {}, comp.type_signature, device='/{}'.format(device))
    result = fn(tf.constant(20))
    return result

  def test_to_representation_for_type_succeeds_on_devices_cpu(self):
    for device in self._logical_devices_cpu:
      self.assertTrue(
          self._get_to_representation_for_type_succeeds_on_device(
              device).device.endswith(device))

  def test_to_representation_for_type_succeeds_on_devices_gpu(self):
    for device in self._logical_devices_gpu:
      self.assertTrue(
          self._get_to_representation_for_type_succeeds_on_device(
              device).device.endswith(device))

  def test_to_representation_for_type_succeeds_on_devices_tpu(self):
    for device in self._logical_devices_tpu:
      self.assertTrue(
          self._get_to_representation_for_type_succeeds_on_device(
              device).device.endswith(device))


if __name__ == '__main__':
  tf.test.main()
