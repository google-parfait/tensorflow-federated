# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import federated_language

from tensorflow_federated.python.tests import test_contexts


class _TestContext(federated_language.framework.SyncContext):
  """A test context."""

  def invoke(self, comp, arg):
    return NotImplementedError


class WithContextsTest(parameterized.TestCase):

  def test_installs_contexts_test_case(self):
    def _context_fn():
      return _TestContext()

    named_contexts = [
        ('1', _context_fn),
        ('2', _context_fn),
        ('3', _context_fn),
    ]

    class _FooTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

      @test_contexts.with_contexts(*named_contexts)
      async def test_async(self):
        pass

      @test_contexts.with_contexts(*named_contexts)
      def test_sync(self):
        pass

      def test_undecorated(self):
        pass

    # Assert that a sync function is returned.
    for name, _ in named_contexts:
      test_name = f'test_sync_{name}'
      self.assertTrue(hasattr(_FooTest, test_name))
      test_fn = getattr(_FooTest, test_name)
      self.assertFalse(asyncio.iscoroutinefunction(test_fn))

    # Assert that an async function is returned.
    for name, _ in named_contexts:
      test_name = f'test_async_{name}'
      self.assertTrue(hasattr(_FooTest, test_name))
      test_fn = getattr(_FooTest, test_name)
      self.assertTrue(asyncio.iscoroutinefunction(test_fn))

    async_values = [lambda _: mock.AsyncMock()] * 3
    sync_values = [lambda _: mock.MagicMock()] * 3
    with mock.patch.object(
        federated_language.framework,
        'with_context',
        side_effect=async_values + sync_values,
    ) as mock_with_context:
      # Assert that the test passes with the expected number of test cases.
      suite = unittest.defaultTestLoader.loadTestsFromTestCase(_FooTest)
      self.assertEqual(suite.countTestCases(), len(named_contexts) * 2 + 1)
      runner = unittest.TextTestRunner()
      result = runner.run(suite)
      self.assertEqual(result.testsRun, len(named_contexts) * 2 + 1)
      self.assertTrue(result.wasSuccessful())

      # Assert that `with_context` is called with the expected parameters.
      calls = [mock.call(a) for _, a in named_contexts] * 2
      self.assertEqual(mock_with_context.mock_calls, calls)

  def test_raises_value_error(self):
    with self.assertRaises(ValueError):
      test_contexts.with_contexts()


if __name__ == '__main__':
  absltest.main()
