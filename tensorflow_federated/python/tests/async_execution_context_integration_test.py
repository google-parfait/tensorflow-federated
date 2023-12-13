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

import unittest

from absl.testing import absltest
import numpy as np
import tensorflow_federated as tff


class AsyncContextInstallationTest(
    absltest.TestCase, unittest.IsolatedAsyncioTestCase
):

  async def test_install_and_execute_in_context(self):
    factory = tff.framework.local_cpp_executor_factory()
    context = tff.framework.AsyncExecutionContext(factory)

    @tff.federated_computation(np.int32)
    def identity(x):
      return x

    with tff.framework.get_context_stack().install(context):
      value = await identity(1)
      self.assertEqual(value, 1)

  async def test_install_and_execute_computations_with_different_cardinalities(
      self,
  ):
    factory = tff.framework.local_cpp_executor_factory()
    context = tff.framework.AsyncExecutionContext(factory)

    @tff.federated_computation(tff.FederatedType(np.int32, tff.CLIENTS))
    def repackage_arg(x):
      return [x, x]

    with tff.framework.get_context_stack().install(context):
      single_value = await repackage_arg([1])
      self.assertEqual(single_value, [[1], [1]])
      second_value = await repackage_arg([1, 2])
      self.assertEqual(second_value, [[1, 2], [1, 2]])

  async def test_runs_cardinality_free(self):
    factory = tff.framework.local_cpp_executor_factory()
    context = tff.framework.AsyncExecutionContext(
        factory, cardinality_inference_fn=(lambda x, y: {})
    )

    @tff.federated_computation(np.int32)
    def identity(x):
      return x

    with tff.framework.get_context_stack().install(context):
      data = 0
      # This computation is independent of cardinalities
      value = await identity(data)
      self.assertEqual(value, 0)

  async def test_raises_exception(self):
    factory = tff.framework.local_cpp_executor_factory()

    def _cardinality_fn(x, y):
      del x, y  # Unused
      return {tff.CLIENTS: 1}

    context = tff.framework.AsyncExecutionContext(
        factory, cardinality_inference_fn=_cardinality_fn
    )

    arg_type = tff.FederatedType(np.int32, tff.CLIENTS)

    @tff.federated_computation(arg_type)
    def identity(x):
      return x

    with tff.framework.get_context_stack().install(context):
      # This argument conflicts with the value returned by the
      # cardinality-inference function; we should get an error surfaced.
      data = [0, 1]
      with self.assertRaises(Exception):
        await identity(data)


if __name__ == '__main__':
  absltest.main()
