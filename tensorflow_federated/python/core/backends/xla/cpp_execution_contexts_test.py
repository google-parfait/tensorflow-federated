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
import federated_language
import numpy as np

from tensorflow_federated.python.core.backends.xla import cpp_execution_contexts
from tensorflow_federated.python.core.environments.jax_frontend import jax_computation


class AsyncLocalCppExecutionContextTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  def test_create_async_local_cpp_execution_context_returns_async_context(self):
    context = cpp_execution_contexts.create_async_local_cpp_execution_context()
    self.assertIsInstance(context, federated_language.framework.AsyncContext)

  @federated_language.framework.with_context(
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
    self.assertIsInstance(context, federated_language.framework.SyncContext)

  @federated_language.framework.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_jax_computation_returns_result(self):
    @jax_computation.jax_computation(np.int32, np.int32)
    def _comp(a, b):
      return a + b

    actual_result = _comp(1, 2)

    self.assertEqual(actual_result, 3)

  @federated_language.framework.with_context(
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

    @federated_language.federated_computation(
        federated_language.FederatedType(np.float32, federated_language.CLIENTS)
    )
    def aggregate(client_values):
      return federated_language.federated_aggregate(
          client_values,
          zero=zeros(),
          accumulate=_add,
          merge=_add,
          report=_identity,
      )

    self.assertEqual(
        aggregate([np.float32(1), np.float32(2), np.float32(3)]), np.float32(6)
    )

  @federated_language.framework.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_sequence_reduce(self):
    sequence = list(range(10))

    @federated_language.federated_computation(
        federated_language.SequenceType(np.int32)
    )
    def comp(x):
      @jax_computation.jax_computation
      def _zero():
        return np.int32(0)

      @jax_computation.jax_computation(np.int32, np.int32)
      def _add(a, b):
        return a + b

      return federated_language.sequence_reduce(x, _zero(), _add)

    self.assertEqual(comp(sequence), sum(range(10)))

  @federated_language.framework.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_federated_sequence_reduce(self):
    sequence = list(range(10))

    @federated_language.federated_computation(
        federated_language.FederatedType(
            federated_language.SequenceType(np.int32), federated_language.SERVER
        )
    )
    def comp(x):
      @jax_computation.jax_computation
      def _zero():
        return np.int32(0)

      @jax_computation.jax_computation(np.int32, np.int32)
      def _add(a, b):
        return a + b

      @federated_language.federated_computation(
          federated_language.SequenceType(np.int32)
      )
      def _sum(sequence):
        return federated_language.sequence_reduce(sequence, _zero(), _add)

      return federated_language.federated_map(_sum, x)

    self.assertEqual(comp(sequence), sum(range(10)))

  @federated_language.framework.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_federated_sum(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def comp(x):
      return federated_language.federated_sum(x)

    # TODO: b/27340091 - use a TFF specific error message after converting the
    # result coming out of the execution stack.
    with self.assertRaisesRegex(Exception, 'Cannot embed a federated value'):
      # TODO: b/255978089 - implement intrinsic lowering using JAX computations,
      # the compiler currently generates TF logic which will fail.
      # self.assertEqual(comp([1, 2, 3]), 6)
      comp([1, 2, 3])

  @federated_language.framework.with_context(
      cpp_execution_contexts.create_sync_local_cpp_execution_context
  )
  def test_unweighted_federated_mean(self):

    @federated_language.federated_computation(
        federated_language.FederatedType(np.float32, federated_language.CLIENTS)
    )
    def comp(x):
      return federated_language.federated_mean(x)

    # TODO: b/27340091 - use a TFF specific error message after converting the
    # result coming out of the execution stack.
    with self.assertRaisesRegex(Exception, 'Cannot embed a federated value'):
      # TODO: b/255978089 - implement intrinsic lowering using JAX computations,
      # the compiler currently generates TF logic which will fail.
      # self.assertEqual(comp([1.0, 2.0, 3.0]), 2.0)
      comp([1.0, 2.0, 3.0])


if __name__ == '__main__':
  absltest.main()
