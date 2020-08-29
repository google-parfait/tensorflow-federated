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
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_test_utils


class EmbedTfCompTest(tf.test.TestCase, parameterized.TestCase):

  def test_embed_tensorflow_computation_with_int_arg_and_result(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return x + 1

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(tf.constant(10))
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result, 11)

  def test_embed_tensorflow_computation_with_float(self):

    @computations.tf_computation(tf.float32)
    def comp(x):
      return x + 0.5

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(tf.constant(10.0))
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result, 10.5)

  def test_embed_tensorflow_computation_with_no_arg_and_int_result(self):

    @computations.tf_computation
    def comp():
      return 1000

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn()
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result, 1000)

  def test_embed_tensorflow_computation_with_dataset_arg_and_int_result(self):

    @computations.tf_computation(computation_types.SequenceType(tf.int32))
    def comp(ds):
      return ds.reduce(np.int32(0), lambda p, q: p + q)

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(tf.data.Dataset.from_tensor_slices([10, 20]))
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result, 30)

  def test_embed_tensorflow_computation_with_tuple_arg_and_result(self):

    @computations.tf_computation([('a', tf.int32), ('b', tf.int32)])
    def comp(a, b):
      return {'sum': a + b}

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    p = tf.constant(10)
    q = tf.constant(20)
    result = fn(structure.Struct([('a', p), ('b', q)]))
    self.assertIsInstance(result, structure.Struct)
    self.assertCountEqual(dir(result), ['sum'])
    self.assertIsInstance(result.sum, tf.Tensor)
    self.assertEqual(result.sum, 30)

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
    self.assertEqual(result, 30)

  def test_embed_tensorflow_computation_with_variable_v2(self):

    @computations.tf_computation(tf.int32)
    def comp(x):
      v = tf.Variable(10)
      with tf.control_dependencies([v.initializer]):
        with tf.control_dependencies([v.assign_add(20)]):
          return tf.add(x, v)

    fn = eager_tf_executor.embed_tensorflow_computation(
        computation_impl.ComputationImpl.get_proto(comp))
    result = fn(tf.constant(30))
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result, 60)

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
    self.assertAlmostEqual(results[0], 1.1)
    self.assertAlmostEqual(results[1], 1.2)

  def _get_wrap_function_on_device(self, device):
    with tf.Graph().as_default() as graph:
      x = tf.compat.v1.placeholder(tf.int32, shape=[])
      y = tf.add(x, tf.constant(1))

    def _function_to_wrap(arg):
      with tf.device(device.name):
        return tf.import_graph_def(
            graph.as_graph_def(),
            input_map={x.name: arg},
            return_elements=[y.name])[0]

    signature = [tf.TensorSpec([], tf.int32)]
    wrapped_fn = tf.compat.v1.wrap_function(_function_to_wrap, signature)

    def fn(arg):
      with tf.device(device.name):
        return wrapped_fn(arg)

    result = fn(tf.constant(10))
    return result

  @parameterized.named_parameters(('CPU', 'CPU'), ('GPU', 'GPU'),
                                  ('TPU', 'TPU'))
  def test_wrap_function_on_all_available_logical_devices(self, device_str):
    for device in tf.config.list_logical_devices(device_str):
      self.assertTrue(
          self._get_wrap_function_on_device(device).device.endswith(
              device.name))

  def _get_embed_tensorflow_computation_succeeds_with_device(self, device):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 1)

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)

    fn = eager_tf_executor.embed_tensorflow_computation(
        comp_proto, comp.type_signature, device=device)
    result = fn(tf.constant(20))
    return result

  @parameterized.named_parameters(('CPU', 'CPU'), ('GPU', 'GPU'),
                                  ('TPU', 'TPU'))
  def test_embed_tensorflow_computation_succeeds_with_cpu(self, device_str):
    for device in tf.config.list_logical_devices(device_str):
      self.assertTrue(
          self._get_embed_tensorflow_computation_succeeds_with_device(
              device).device.endswith(device.name))

  def _get_to_representation_for_type_succeeds_on_device(self, device):

    @computations.tf_computation(tf.int32)
    def comp(x):
      return tf.add(x, 1)

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)

    fn = eager_tf_executor.to_representation_for_type(
        comp_proto, {}, comp.type_signature, device=device)
    result = fn(tf.constant(20))
    return result

  @parameterized.named_parameters(('CPU', 'CPU'), ('GPU', 'GPU'),
                                  ('TPU', 'TPU'))
  def test_to_representation_for_type_succeeds_on_devices_cpu(self, device_str):
    for device in tf.config.list_logical_devices(device_str):
      self.assertTrue(
          self._get_to_representation_for_type_succeeds_on_device(
              device).device.endswith(device.name))

  def _skip_in_multi_gpus(self):
    logical_gpus = tf.config.list_logical_devices('GPU')
    if len(logical_gpus) > 1:
      self.skipTest('Skip the test if multi-GPUs, checkout the MultiGPUTests')

  def test_get_no_arg_wrapped_function_from_comp_with_dataset_reduce(self):

    self._skip_in_multi_gpus()

    @computations.tf_computation
    def comp():
      return tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)

    wrapped_fn = eager_tf_executor._get_wrapped_function_from_comp(
        computation_impl.ComputationImpl.get_proto(comp),
        must_pin_function_to_cpu=False,
        param_type=None,
        device=None)
    self.assertEqual(wrapped_fn(), np.int64(45))

  def test_get_wrapped_function_from_comp_raises_with_incorrect_binding(self):

    self._skip_in_multi_gpus()

    with tf.Graph().as_default() as graph:
      var = tf.Variable(initial_value=0.0, name='var1', import_scope='')
      assign_op = var.assign_add(tf.constant(1.0))
      tf.add(1.0, assign_op)

    result_binding = pb.TensorFlow.Binding(
        tensor=pb.TensorFlow.TensorBinding(tensor_name='Invalid'))
    comp = pb.Computation(
        tensorflow=pb.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            result=result_binding))
    with self.assertRaisesRegex(TypeError,
                                'Caught exception trying to prune graph.*'):
      eager_tf_executor._get_wrapped_function_from_comp(
          comp, must_pin_function_to_cpu=False, param_type=None, device=None)

  def test_check_dataset_reduce_in_multi_gpu_no_mgpu_no_raise(self):
    self._skip_in_multi_gpus()
    with tf.Graph().as_default() as graph:
      tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)
    eager_tf_executor._check_dataset_reduce_in_multi_gpu(graph.as_graph_def())


class MultiGPUTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Multiple logical GPU devices will be created for tests in this clss. And
    # logical deviceds have to be created before listed in each indivisual test.
    self._create_logical_multi_gpus()

  def _create_logical_multi_gpus(self):
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
      self.skipTest('Skip GPU tests when no GPU is provided')
    if len(gpu_devices) == 1:
      tf.config.set_logical_device_configuration(gpu_devices[0], [
          tf.config.LogicalDeviceConfiguration(memory_limit=128),
          tf.config.LogicalDeviceConfiguration(memory_limit=128)
      ])

  def test_check_dataset_reduce_in_multi_gpu_no_reduce_no_raise(self):
    with tf.Graph().as_default() as graph:
      tf.data.Dataset.range(10).map(lambda x: x + 1)
    eager_tf_executor._check_dataset_reduce_in_multi_gpu(graph.graph_def())

  def test_check_dataset_reduce_in_multi_gpu(self):
    with tf.Graph().as_default() as graph:
      tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)
    with self.assertRaisesRegex(
        ValueError, 'Detected dataset reduce op in multi-GPU TFF simulation.*'):
      eager_tf_executor._check_dataset_reduce_in_multi_gpu(graph.as_graph_def())

  def test_check_dataset_reduce_in_multi_gpu_tf_device_no_raise(self):
    logical_gpus = tf.config.list_logical_devices('GPU')
    with tf.Graph().as_default() as graph:
      with tf.device(logical_gpus[0].name):
        tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)
    eager_tf_executor._check_dataset_reduce_in_multi_gpu(graph.graph_def())

  def test_get_no_arg_wrapped_function_check_dataset_reduce_in_multi_gpu(self):

    @computations.tf_computation
    def comp():
      return tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)

    with self.assertRaisesRegex(
        ValueError, 'Detected dataset reduce op in multi-GPU TFF simulation.*'):
      eager_tf_executor._get_wrapped_function_from_comp(
          computation_impl.ComputationImpl.get_proto(comp),
          must_pin_function_to_cpu=False,
          param_type=None,
          device=None)

  def test_get_no_arg_wrapped_function_multi_gpu_no_reduce(self):

    @computations.tf_computation
    @tf.function
    def comp():
      value = tf.constant(0, dtype=tf.int64)
      for d in iter(tf.data.Dataset.range(10)):
        value += d
      return value

    wrapped_fn = eager_tf_executor._get_wrapped_function_from_comp(
        computation_impl.ComputationImpl.get_proto(comp),
        must_pin_function_to_cpu=False,
        param_type=None,
        device=None)
    self.assertEqual(wrapped_fn(), np.int64(45))

  def test_get_no_arg_wrapped_function_multi_gpu_tf_device(self):

    logical_gpus = tf.config.list_logical_devices('GPU')

    @computations.tf_computation
    def comp():
      with tf.device(logical_gpus[0].name):
        return tf.data.Dataset.range(10).reduce(np.int64(0), lambda p, q: p + q)

    wrapped_fn = eager_tf_executor._get_wrapped_function_from_comp(
        computation_impl.ComputationImpl.get_proto(comp),
        must_pin_function_to_cpu=False,
        param_type=None,
        device=None)
    self.assertEqual(wrapped_fn(), np.int64(45))


def _create_test_executor_factory():
  executor = eager_tf_executor.EagerTFExecutor()
  return executor_factory.ExecutorFactoryImpl(lambda _: executor)


class EagerTFExecutorTest(tf.test.TestCase, parameterized.TestCase):

  def test_to_representation_for_type_with_int(self):
    value = 10
    type_signature = computation_types.TensorType(tf.int32)
    v = eager_tf_executor.to_representation_for_type(value, {}, type_signature)
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v, 10)
    self.assertEqual(v.dtype, tf.int32)

  def test_to_representation_for_tf_variable(self):
    value = tf.Variable(10, dtype=tf.int32)
    type_signature = computation_types.TensorType(tf.int32)
    v = eager_tf_executor.to_representation_for_type(value, {}, type_signature)
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v, 10)
    self.assertEqual(v.dtype, tf.int32)

  def test_to_representation_for_type_with_int_on_specific_device(self):
    value = 10
    type_signature = computation_types.TensorType(tf.int32)
    v = eager_tf_executor.to_representation_for_type(
        value, {}, type_signature,
        tf.config.list_logical_devices('CPU')[0])
    self.assertIsInstance(v, tf.Tensor)
    self.assertEqual(v, 10)
    self.assertEqual(v.dtype, tf.int32)
    self.assertTrue(v.device.endswith('CPU:0'))

  def test_eager_value_constructor_with_int_constant(self):
    v = eager_tf_executor.EagerValue(10, {}, tf.int32)
    self.assertEqual(str(v.type_signature), 'int32')
    self.assertIsInstance(v.internal_representation, tf.Tensor)
    self.assertEqual(v.internal_representation, 10)

  def test_executor_constructor_fails_if_not_in_eager_mode(self):
    with tf.Graph().as_default():
      with self.assertRaises(RuntimeError):
        eager_tf_executor.EagerTFExecutor()

  def test_executor_construction_with_correct_device_name(self):
    eager_tf_executor.EagerTFExecutor(tf.config.list_logical_devices('CPU')[0])

  def test_executor_construction_with_no_device_name(self):
    eager_tf_executor.EagerTFExecutor()

  def test_executor_create_value_int(self):
    ex = eager_tf_executor.EagerTFExecutor()
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(10, tf.int32))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertIsInstance(val.internal_representation, tf.Tensor)
    self.assertEqual(str(val.type_signature), 'int32')
    self.assertEqual(val.internal_representation, 10)

  def test_executor_create_value_unnamed_int_pair(self):
    ex = eager_tf_executor.EagerTFExecutor()
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value([10, {
            'a': 20
        }], [tf.int32, collections.OrderedDict([('a', tf.int32)])]))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '<int32,<a=int32>>')
    self.assertIsInstance(val.internal_representation, structure.Struct)
    self.assertLen(val.internal_representation, 2)
    self.assertIsInstance(val.internal_representation[0], tf.Tensor)
    self.assertIsInstance(val.internal_representation[1], structure.Struct)
    self.assertLen(val.internal_representation[1], 1)
    self.assertEqual(dir(val.internal_representation[1]), ['a'])
    self.assertIsInstance(val.internal_representation[1][0], tf.Tensor)
    self.assertEqual(val.internal_representation[0], 10)
    self.assertEqual(val.internal_representation[1][0], 20)

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
    self.assertEqual(result, 1000)

  def test_executor_create_value_two_arg_computation(self):
    ex = eager_tf_executor.EagerTFExecutor()

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    comp_proto = computation_impl.ComputationImpl.get_proto(comp)
    val = asyncio.get_event_loop().run_until_complete(
        ex.create_value(
            comp_proto,
            computation_types.FunctionType(
                computation_types.StructType([('a', tf.int32),
                                              ('b', tf.int32)]), tf.int32)))
    self.assertIsInstance(val, eager_tf_executor.EagerValue)
    self.assertEqual(str(val.type_signature), '(<a=int32,b=int32> -> int32)')
    self.assertTrue(callable(val.internal_representation))
    arg = structure.Struct([('a', tf.constant(10)), ('b', tf.constant(10))])
    result = val.internal_representation(arg)
    self.assertIsInstance(result, tf.Tensor)
    self.assertEqual(result, 20)

  def test_executor_create_call_add_numbers(self):

    @computations.tf_computation(tf.int32, tf.int32)
    def comp(a, b):
      return a + b

    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value(
            structure.Struct([('a', 10), ('b', 20)]),
            comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int32')
    self.assertIsInstance(result.internal_representation, tf.Tensor)
    self.assertEqual(result.internal_representation, 30)

  def test_dynamic_lookup_table_usage(self):

    @computations.tf_computation(
        computation_types.TensorType(shape=[None], dtype=tf.string),
        computation_types.TensorType(shape=[], dtype=tf.string))
    def comp(table_args, to_lookup):
      values = tf.range(tf.shape(table_args)[0])
      initializer = tf.lookup.KeyValueTensorInitializer(table_args, values)
      table = tf.lookup.StaticHashTable(initializer, 100)
      return table.lookup(to_lookup)

    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg_1 = loop.run_until_complete(
        ex.create_value(
            structure.Struct([('table_args', tf.constant(['a', 'b', 'c'])),
                              ('to_lookup', tf.constant('a'))]),
            comp.type_signature.parameter))
    arg_2 = loop.run_until_complete(
        ex.create_value(
            structure.Struct([('table_args', tf.constant(['a', 'b', 'c', 'd'])),
                              ('to_lookup', tf.constant('d'))]),
            comp.type_signature.parameter))
    result_1 = loop.run_until_complete(ex.create_call(comp, arg_1))
    result_2 = loop.run_until_complete(ex.create_call(comp, arg_2))

    self.assertEqual(self.evaluate(result_1.internal_representation), 0)
    self.assertEqual(self.evaluate(result_2.internal_representation), 3)

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
  def test_executor_create_call_take_two_from_stateful_dataset(self):

    vocab = ['a', 'b', 'c', 'd', 'e', 'f']

    @computations.tf_computation(computation_types.SequenceType(tf.string))
    def comp(ds):
      table = tf.lookup.StaticVocabularyTable(
          tf.lookup.KeyValueTensorInitializer(
              vocab, tf.range(len(vocab), dtype=tf.int64)),
          num_oov_buckets=1)
      ds = ds.map(table.lookup)
      return ds.take(2)

    ds = tf.data.Dataset.from_tensor_slices(vocab)
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    comp = loop.run_until_complete(ex.create_value(comp))
    arg = loop.run_until_complete(
        ex.create_value(ds, comp.type_signature.parameter))
    result = loop.run_until_complete(ex.create_call(comp, arg))
    self.assertIsInstance(result, eager_tf_executor.EagerValue)
    self.assertEqual(str(result.type_signature), 'int64*')
    self.assertIn('Dataset', type(result.internal_representation).__name__)
    self.assertCountEqual([x.numpy() for x in result.internal_representation],
                          [0, 1])

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
    self.assertEqual(result.internal_representation, 90)

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
    self.assertIsInstance(result.internal_representation, structure.Struct)
    self.assertCountEqual(dir(result.internal_representation), ['a', 'b'])
    self.assertIsInstance(result.internal_representation.a, tf.Tensor)
    self.assertIsInstance(result.internal_representation.b, tf.Tensor)
    self.assertEqual(result.internal_representation.a, 60)
    self.assertEqual(result.internal_representation.b, 15)

  def test_executor_create_struct_and_selection(self):
    ex = eager_tf_executor.EagerTFExecutor()
    loop = asyncio.get_event_loop()
    v1, v2 = tuple(
        loop.run_until_complete(
            asyncio.gather(*[ex.create_value(x, tf.int32) for x in [10, 20]])))
    v3 = loop.run_until_complete(
        ex.create_struct(collections.OrderedDict([('a', v1), ('b', v2)])))
    self.assertIsInstance(v3, eager_tf_executor.EagerValue)
    self.assertIsInstance(v3.internal_representation, structure.Struct)
    self.assertLen(v3.internal_representation, 2)
    self.assertCountEqual(dir(v3.internal_representation), ['a', 'b'])
    self.assertIsInstance(v3.internal_representation[0], tf.Tensor)
    self.assertIsInstance(v3.internal_representation[1], tf.Tensor)
    self.assertEqual(str(v3.type_signature), '<a=int32,b=int32>')
    self.assertEqual(v3.internal_representation[0], 10)
    self.assertEqual(v3.internal_representation[1], 20)
    v4 = loop.run_until_complete(ex.create_selection(v3, name='a'))
    self.assertIsInstance(v4, eager_tf_executor.EagerValue)
    self.assertIsInstance(v4.internal_representation, tf.Tensor)
    self.assertEqual(str(v4.type_signature), 'int32')
    self.assertEqual(v4.internal_representation, 10)
    v5 = loop.run_until_complete(ex.create_selection(v3, index=1))
    self.assertIsInstance(v5, eager_tf_executor.EagerValue)
    self.assertIsInstance(v5.internal_representation, tf.Tensor)
    self.assertEqual(str(v5.type_signature), 'int32')
    self.assertEqual(v5.internal_representation, 20)
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
    self.assertEqual(val, 10)

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
      self.assertEqual(val, 10 * (n + 2))

  def test_execution_of_tensorflow(self):

    @computations.tf_computation
    def comp():
      return tf.math.add(5, 5)

    executor = _create_test_executor_factory()
    with executor_test_utils.install_executor(executor):
      result = comp()

    self.assertEqual(result, 10)


if __name__ == '__main__':
  tf.test.main()
