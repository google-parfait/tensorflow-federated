# Copyright 2021, The TensorFlow Federated Authors.
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

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import executor_factory
from tensorflow_federated.python.core.impl.executors import executors_errors
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class RetryableErrorTest(tf.test.TestCase):

  def test_is_retryable_error(self):
    retryable_error = executors_errors.RetryableError()
    self.assertTrue(
        async_execution_context._is_retryable_error(retryable_error)
    )
    self.assertFalse(async_execution_context._is_retryable_error(TypeError()))
    self.assertFalse(async_execution_context._is_retryable_error(1))
    self.assertFalse(async_execution_context._is_retryable_error('a'))
    self.assertFalse(async_execution_context._is_retryable_error(None))


class UnwrapValueTest(tf.test.TestCase):

  def test_tensor(self):
    result = async_execution_context._unwrap(tf.constant(1))
    self.assertIsInstance(result, np.int32)
    result = async_execution_context._unwrap(tf.constant([1, 2]))
    self.assertIsInstance(result, np.ndarray)
    self.assertAllEqual(result, [1, 2])

  def test_structure_of_tensors(self):
    result = async_execution_context._unwrap([tf.constant(x) for x in range(5)])
    self.assertIsInstance(result, list)
    for x in range(5):
      self.assertIsInstance(result[x], np.int32)
      self.assertEqual(result[x], x)


class AsyncContextInstallationTest(tf.test.TestCase):

  def test_install_and_execute_in_context(self):
    factory = executor_factory.local_cpp_executor_factory()
    context = async_execution_context.AsyncExecutionContext(factory)

    @tensorflow_computation.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    with get_context_stack.get_context_stack().install(context):
      val_coro = add_one(1)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 2)

  def test_install_and_execute_computations_with_different_cardinalities(self):
    factory = executor_factory.local_cpp_executor_factory()
    context = async_execution_context.AsyncExecutionContext(factory)

    @federated_computation.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS)
    )
    def repackage_arg(x):
      return [x, x]

    with get_context_stack.get_context_stack().install(context):
      single_val_coro = repackage_arg([1])
      second_val_coro = repackage_arg([1, 2])
      self.assertTrue(asyncio.iscoroutine(single_val_coro))
      self.assertTrue(asyncio.iscoroutine(second_val_coro))
      self.assertEqual(
          [asyncio.run(single_val_coro), asyncio.run(second_val_coro)],
          [[[1], [1]], [[1, 2], [1, 2]]],
      )

  def test_runs_cardinality_free(self):
    factory = executor_factory.local_cpp_executor_factory()
    context = async_execution_context.AsyncExecutionContext(
        factory, cardinality_inference_fn=(lambda x, y: {})
    )

    @federated_computation.federated_computation(tf.int32)
    def identity(x):
      return x

    with get_context_stack.get_context_stack().install(context):
      data = 0
      # This computation is independent of cardinalities
      val_coro = identity(data)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 0)

  def test_raises_exception(self):
    factory = executor_factory.local_cpp_executor_factory()

    def _cardinality_fn(x, y):
      del x, y  # Unused
      return {placements.CLIENTS: 1}

    context = async_execution_context.AsyncExecutionContext(
        factory, cardinality_inference_fn=_cardinality_fn
    )

    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)

    @federated_computation.federated_computation(arg_type)
    def identity(x):
      return x

    with get_context_stack.get_context_stack().install(context):
      # This argument conflicts with the value returned by the
      # cardinality-inference function; we should get an error surfaced.
      data = [0, 1]
      val_coro = identity(data)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      with self.assertRaises(Exception) as e:
        self.assertTrue(hasattr(e, 'status'))
        self.assertTrue(hasattr(e.status, 'code_int'))
        asyncio.run(val_coro)


if __name__ == '__main__':
  tf.test.main()
