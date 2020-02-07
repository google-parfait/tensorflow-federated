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

import asyncio
import collections

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl import eager_executor
from tensorflow_federated.python.core.impl import lambda_executor
from tensorflow_federated.python.core.impl.executors import caching_executor
from tensorflow_federated.python.core.impl.executors import sizing_executor


class SizingExecutorTest(parameterized.TestCase):

  def test_simple(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())

    tensor_type = computation_types.TensorType(tf.int32, 10)

    @computations.tf_computation(tensor_type)
    def add_one(x):
      return tf.add(x, 1)

    async def _make():
      v1 = await ex.create_value(add_one)
      v2 = await ex.create_value(
          tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], tf.int32), tensor_type)
      v3 = await ex.create_call(v1, v2)
      v4 = await ex.create_tuple(anonymous_tuple.AnonymousTuple([('foo', v3)]))
      v5 = await ex.create_selection(v4, name='foo')
      return await v5.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history, [[10, tf.int32]])
    self.assertCountEqual(ex.output_history, [[10, tf.int32]])

  def test_string(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())
    tensor_type = computation_types.TensorType(tf.string, [4])
    strings = ['hi', 'bye', 'tensor', 'federated']
    total_string_length = sum([len(s) for s in strings])

    async def _make():
      v1 = await ex.create_value(strings, tensor_type)
      return await v1.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history, [[total_string_length, tf.string]])
    self.assertCountEqual(ex.output_history, [[total_string_length, tf.string]])

  def test_different_input_output(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())

    tensor_type = computation_types.TensorType(tf.int32, 10)

    @computations.tf_computation(tensor_type)
    def return_constant(x):
      del x
      return tf.constant(0, tf.int32)

    async def _make():
      v1 = await ex.create_value(return_constant)
      v2 = await ex.create_value([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], tensor_type)
      v3 = await ex.create_call(v1, v2)
      return await v3.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history, [[10, tf.int32]])
    self.assertCountEqual(ex.output_history, [[1, tf.int32]])

  def test_multiple_inputs(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())

    int_type = computation_types.TensorType(tf.int32, 10)
    float_type = computation_types.TensorType(tf.float64, 10)

    @computations.tf_computation(float_type, int_type)
    def add(x, y):
      x = tf.cast(x, tf.int64)
      y = tf.cast(y, tf.int64)
      return x + y

    async def _make():
      v1 = await ex.create_value(add)
      v2 = await ex.create_value([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], float_type)
      v3 = await ex.create_value([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], int_type)
      v4 = await ex.create_tuple(
          anonymous_tuple.AnonymousTuple([(None, v2), (None, v3)]))
      v5 = await ex.create_call(v1, v4)
      return await v5.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history, [[10, tf.int32], [10, tf.float64]])
    self.assertCountEqual(ex.output_history, [[10, tf.int64]])

  def test_nested_tuple(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())
    a = computation_types.TensorType(tf.int32, [4])
    b = computation_types.TensorType(tf.bool, [2])
    c = computation_types.TensorType(tf.int64, [2, 3])
    inner_type = computation_types.NamedTupleType([('a', a), ('b', b),
                                                   ('c', c)])
    outer_type = computation_types.NamedTupleType([('a', inner_type),
                                                   ('b', inner_type)])
    inner_type_val = collections.OrderedDict()
    inner_type_val['a'] = [0, 1, 2, 3]
    inner_type_val['b'] = [True, False]
    inner_type_val['c'] = [[1, 2, 3], [4, 5, 6]]
    outer_type_val = collections.OrderedDict()
    outer_type_val['a'] = inner_type_val
    outer_type_val['b'] = inner_type_val

    async def _make():
      v1 = await ex.create_value(outer_type_val, outer_type)
      return await v1.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history,
                          [[4, tf.int32], [2, tf.bool], [6, tf.int64],
                           [4, tf.int32], [2, tf.bool], [6, tf.int64]])

  def test_empty_tuple(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())
    tup = computation_types.NamedTupleType([])
    empty_dict = collections.OrderedDict()

    async def _make():
      v1 = await ex.create_value(empty_dict, tup)
      return await v1.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history, [])

  def test_ordered_dict(self):
    a = computation_types.TensorType(tf.string, [4])
    b = computation_types.TensorType(tf.int64, [2, 3])
    tup = computation_types.NamedTupleType([('a', a), ('b', b)])
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())
    od = collections.OrderedDict()
    od['a'] = ['some', 'arbitrary', 'string', 'here']
    od['b'] = [[3, 4, 1], [6, 8, -5]]
    total_string_length = sum([len(s) for s in od['a']])

    async def _make():
      v1 = await ex.create_value(od, tup)
      return await v1.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history,
                          [[total_string_length, tf.string], [6, tf.int64]])

  def test_unnamed_tuple(self):
    ex = sizing_executor.SizingExecutor(eager_executor.EagerExecutor())
    type_spec = computation_types.NamedTupleType([tf.int32, tf.int32])
    value = anonymous_tuple.AnonymousTuple([(None, 0), (None, 1)])
    async def _make():
      v1 = await ex.create_value(value, type_spec)
      return await v1.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history, [[1, tf.int32], [1, tf.int32]])
    self.assertCountEqual(ex.output_history, [[1, tf.int32], [1, tf.int32]])

  @parameterized.named_parameters(
      {
          'testcase_name': 'basic',
          'executor_stack': [sizing_executor.SizingExecutor]
      }, {
          'testcase_name': 'big_stack',
          'executor_stack': [
              sizing_executor.SizingExecutor,
              lambda_executor.LambdaExecutor,
              caching_executor.CachingExecutor,
              lambda_executor.LambdaExecutor
          ]
      }, {
          'testcase_name': 'big_caching_stack',
          'executor_stack': [
              sizing_executor.SizingExecutor,
              caching_executor.CachingExecutor,
              lambda_executor.LambdaExecutor,
              caching_executor.CachingExecutor,
              caching_executor.CachingExecutor,
              lambda_executor.LambdaExecutor,
              caching_executor.CachingExecutor,
          ]
      }, {
          'testcase_name': 'lambda_executor',
          'executor_stack': [
              sizing_executor.SizingExecutor,
              lambda_executor.LambdaExecutor
          ]
      }, {
          'testcase_name': 'caching_executor',
          'executor_stack': [
              sizing_executor.SizingExecutor,
              caching_executor.CachingExecutor,
          ]
      })
  def test_executor_stacks(self, executor_stack):
    assert executor_stack
    ex = eager_executor.EagerExecutor()
    for wrapping_executor in reversed(executor_stack):
      ex = wrapping_executor(ex)
    a = computation_types.TensorType(tf.int32, [4])
    b = computation_types.TensorType(tf.bool, [2])
    c = computation_types.TensorType(tf.int64, [2, 3])
    inner_type = computation_types.NamedTupleType([('a', a), ('b', b),
                                                   ('c', c)])
    outer_type = computation_types.NamedTupleType([('a', inner_type),
                                                   ('b', inner_type)])
    inner_type_val = collections.OrderedDict()
    inner_type_val['a'] = [0, 1, 2, 3]
    inner_type_val['b'] = [True, False]
    inner_type_val['c'] = [[1, 2, 3], [4, 5, 6]]
    outer_type_val = collections.OrderedDict()
    outer_type_val['a'] = inner_type_val
    outer_type_val['b'] = inner_type_val

    async def _make():
      v1 = await ex.create_value(outer_type_val, outer_type)
      return await v1.compute()

    asyncio.get_event_loop().run_until_complete(_make())
    self.assertCountEqual(ex.input_history,
                          [[4, tf.int32], [2, tf.bool], [6, tf.int64],
                           [4, tf.int32], [2, tf.bool], [6, tf.int64]])


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
