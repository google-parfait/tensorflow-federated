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
import tensorflow as tf

from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.execution_contexts import cpp_async_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class AsyncContextInstallationTest(tf.test.TestCase):

  def test_install_and_execute_in_context(self):
    factory = cpp_executor_factory.local_cpp_executor_factory()
    context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
        factory, compiler_fn=lambda x: x)

    @computations.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    with get_context_stack.get_context_stack().install(context):
      val_coro = add_one(1)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 2)

  def test_install_and_execute_unpacked_structure_arg_in_context(self):
    factory = cpp_executor_factory.local_cpp_executor_factory()
    context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
        factory, compiler_fn=lambda x: x)

    @computations.tf_computation(tf.int32, tf.int32)
    def add_one(x):
      return x[0] + 1

    with get_context_stack.get_context_stack().install(context):
      val_coro = add_one(1, 2)  # pylint: disable=too-many-function-args
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 2)

  def test_install_and_execute_packed_structure_arg_in_context(self):
    factory = cpp_executor_factory.local_cpp_executor_factory()
    context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
        factory, compiler_fn=lambda x: x)

    @computations.tf_computation(tf.int32, tf.int32)
    def add_one(x, y):
      del y  # Unused
      return x + 1

    with get_context_stack.get_context_stack().install(context):
      val_coro = add_one(1, 2)
      self.assertTrue(asyncio.iscoroutine(val_coro))
      self.assertEqual(asyncio.run(val_coro), 2)

  def test_install_and_execute_computations_with_different_cardinalities(self):
    factory = cpp_executor_factory.local_cpp_executor_factory()
    context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
        factory, compiler_fn=lambda x: x)

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.CLIENTS))
    def repackage_arg(x):
      return [x, x]

    with get_context_stack.get_context_stack().install(context):
      single_val_coro = repackage_arg([1])
      second_val_coro = repackage_arg([1, 2])
      self.assertTrue(asyncio.iscoroutine(single_val_coro))
      self.assertTrue(asyncio.iscoroutine(second_val_coro))
      self.assertEqual(
          [asyncio.run(single_val_coro),
           asyncio.run(second_val_coro)], [[[1], [1]], [[1, 2], [1, 2]]])


if __name__ == '__main__':
  tf.test.main()
