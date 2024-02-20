# Copyright 2020, The TensorFlow Federated Authors.
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

import unittest

from absl.testing import absltest
import numpy as np

from tensorflow_federated.python.core.backends.xla import cpp_execution_contexts
from tensorflow_federated.python.core.environments.jax_frontend import jax_computation
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class AsyncLocalCppExecutionContextTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_create_async_local_cpp_execution_context_returns_async_context(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()
    self.assertIsInstance(context, context_base.AsyncContext)

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_async_local_cpp_execution_context
  )
  async def test_jax_computation_returns_result(self):
    @jax_computation.jax_computation(np.int32, np.int32)
    def _comp(a, b):
      return a + b

    actual_result = await _comp(1, 2)

    self.assertEqual(actual_result, 3)


class SyncLocalCppExecutionContextTest(absltest.TestCase):

  def test_create_sync_local_cpp_execution_context_returns_sync_context(self):
    context = cpp_execution_contexts.create_sync_local_cpp_execution_context()
    self.assertIsInstance(context, context_base.SyncContext)

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_jax_computation_returns_result(self):
    @jax_computation.jax_computation(np.int32, np.int32)
    def _comp(a, b):
      return a + b

    actual_result = _comp(1, 2)

    self.assertEqual(actual_result, 3)

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_federated_aggergate(self):
    @jax_computation.jax_computation(np.float32, np.float32)
    def _add(a, b):
      return a + b

    @jax_computation.jax_computation(np.float32)
    def _identity(a):
      return a

    # IMPORTANT: we must wrap the zero literal in a `jax_computation` because
    # TFF by default wraps Python literals as `tf.constant` which will fail in
    # this execution context.
    @jax_computation.jax_computation
    def zeros():
      return np.float32(0)

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.float32, placements.CLIENTS)
    )
    def aggregate(client_values):
      return intrinsics.federated_aggregate(
          client_values,
          zero=zeros(),
          accumulate=_add,
          merge=_add,
          report=_identity,
      )

    self.assertEqual(
        aggregate([np.float32(1), np.float32(2), np.float32(3)]), np.float32(6)
    )

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_sequence_reduce(self):
    sequence = list(range(10))

    @federated_computation.federated_computation(
        computation_types.SequenceType(np.int32)
    )
    def comp(x):
      @jax_computation.jax_computation
      def _zero():
        return np.int32(0)

      @jax_computation.jax_computation(np.int32, np.int32)
      def _add(a, b):
        return a + b

      return intrinsics.sequence_reduce(x, _zero(), _add)

    self.assertEqual(comp(sequence), sum(range(10)))

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_federated_sequence_reduce(self):
    sequence = list(range(10))

    @federated_computation.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(np.int32), placements.SERVER
        )
    )
    def comp(x):
      @jax_computation.jax_computation
      def _zero():
        return np.int32(0)

      @jax_computation.jax_computation(np.int32, np.int32)
      def _add(a, b):
        return a + b

      @federated_computation.federated_computation(
          computation_types.SequenceType(np.int32)
      )
      def _sum(sequence):
        return intrinsics.sequence_reduce(sequence, _zero(), _add)

      return intrinsics.federated_map(_sum, x)

    self.assertEqual(comp(sequence), sum(range(10)))

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_federated_sum(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    def comp(x):
      return intrinsics.federated_sum(x)

    # TODO: b/27340091 - use a TFF specific error message after converting the
    # result coming out of the execution stack.
    with self.assertRaisesRegex(Exception, 'Cannot embed a federated value'):
      # TODO: b/255978089 - implement intrinsic lowering using JAX computations,
      # the compiler currently generates TF logic which will fail.
      # self.assertEqual(comp([1, 2, 3]), 6)
      comp([1, 2, 3])

  @context_stack_test_utils.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_unweighted_federated_mean(self):

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.float32, placements.CLIENTS)
    )
    def comp(x):
      return intrinsics.federated_mean(x)

    # TODO: b/27340091 - use a TFF specific error message after converting the
    # result coming out of the execution stack.
    with self.assertRaisesRegex(Exception, 'Cannot embed a federated value'):
      # TODO: b/255978089 - implement intrinsic lowering using JAX computations,
      # the compiler currently generates TF logic which will fail.
      # self.assertEqual(comp([1.0, 2.0, 3.0]), 2.0)
      comp([1.0, 2.0, 3.0])


if __name__ == '__main__':
  absltest.main()
