# Copyright 2022, The TensorFlow Federated Authors.
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
import contextlib
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils


class WithContextTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('context_and_no_environment',
       context_stack_test_utils.TestContext(),
       None),
      ('context_and_empty_environment',
       context_stack_test_utils.TestContext(),
       []),
      ('context_and_1_environment',
       context_stack_test_utils.TestContext(),
       [context_stack_test_utils.test_environment()]),
      ('context_and_3_environment',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  def test_installs_context_fn_sync_no_arg(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      _foo()

      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertEqual(mock_enter_context.mock_calls, calls)

  # pyformat: disable
  @parameterized.named_parameters(
      ('context_and_no_environment',
       context_stack_test_utils.TestContext(),
       None),
      ('context_and_empty_environment',
       context_stack_test_utils.TestContext(),
       []),
      ('context_and_1_environment',
       context_stack_test_utils.TestContext(),
       [context_stack_test_utils.test_environment()]),
      ('context_and_3_environment',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  def test_installs_context_fn_sync_args(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo(x):
      del x  # Unused.
      self.assertEqual(context_stack_impl.context_stack.current, context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      _foo(1)

      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertEqual(mock_enter_context.mock_calls, calls)

  # pyformat: disable
  @parameterized.named_parameters(
      ('context_and_no_environment',
       context_stack_test_utils.TestContext(),
       None),
      ('context_and_empty_environment',
       context_stack_test_utils.TestContext(),
       []),
      ('context_and_1_environment',
       context_stack_test_utils.TestContext(),
       [context_stack_test_utils.test_environment()]),
      ('context_and_3_environment',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  def test_installs_context_fn_sync_kwargs(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo(*, x):
      del x  # Unused.
      self.assertEqual(context_stack_impl.context_stack.current, context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      _foo(x=1)

      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertEqual(mock_enter_context.mock_calls, calls)

  # pyformat: disable
  @parameterized.named_parameters(
      ('context_and_no_environment',
       context_stack_test_utils.TestContext(),
       None),
      ('context_and_empty_environment',
       context_stack_test_utils.TestContext(),
       []),
      ('context_and_1_environment',
       context_stack_test_utils.TestContext(),
       [context_stack_test_utils.test_environment()]),
      ('context_and_3_environment',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  def test_installs_context_fn_sync_return(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, context)
      return 1

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      x = _foo()

      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertEqual(mock_enter_context.mock_calls, calls)

      # Assert that the return value is returned by the decorator.
      self.assertEqual(x, 1)

  # pyformat: disable
  @parameterized.named_parameters(
      ('context_and_no_environment',
       context_stack_test_utils.TestContext(),
       None),
      ('context_and_empty_environment',
       context_stack_test_utils.TestContext(),
       []),
      ('context_and_1_environment',
       context_stack_test_utils.TestContext(),
       [context_stack_test_utils.test_environment()]),
      ('context_and_3_environment',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  async def test_installs_context_fn_async(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    async def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, context)

    # Assert that an async function is returned.
    self.assertTrue(asyncio.iscoroutinefunction(_foo))

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      await _foo()

      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertEqual(mock_enter_context.mock_calls, calls)

  # pyformat: disable
  @parameterized.named_parameters(
      ('context_and_no_environment',
       context_stack_test_utils.TestContext(),
       None),
      ('context_and_empty_environment',
       context_stack_test_utils.TestContext(),
       []),
      ('context_and_1_environment',
       context_stack_test_utils.TestContext(),
       [context_stack_test_utils.test_environment()]),
      ('context_and_3_environments',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  def test_installs_context_test_case(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    class _FooTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

      @context_stack_test_utils.with_context(context_fn, environment_fn)
      async def test_async(self):
        self.assertEqual(context_stack_impl.context_stack.current, context)

      @context_stack_test_utils.with_context(context_fn, environment_fn)
      def test_sync(self):
        self.assertEqual(context_stack_impl.context_stack.current, context)

      def test_undecorated(self):
        self.assertNotEqual(context_stack_impl.context_stack.current, context)

    # Assert that a sync function is returned.
    self.assertFalse(asyncio.iscoroutinefunction(_FooTest.test_sync))

    # Assert that an async function is returned.
    self.assertTrue(asyncio.iscoroutinefunction(_FooTest.test_async))

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that the test passes with the expected number of test cases.
      suite = unittest.defaultTestLoader.loadTestsFromTestCase(_FooTest)
      self.assertEqual(suite.countTestCases(), 3)
      runner = unittest.TextTestRunner()
      result = runner.run(suite)
      self.assertEqual(result.testsRun, 3)
      self.assertTrue(result.wasSuccessful())

      # Assert that the context is not installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments] * 2
        self.assertEqual(mock_enter_context.mock_calls, calls)


class WithContextsTest(parameterized.TestCase):

  def test_installs_contexts_test_case(self):
    def _context_fn():
      return context_stack_test_utils.TestContext()

    def _environment_fn():
      return [
          context_stack_test_utils.test_environment(),
          context_stack_test_utils.test_environment(),
          context_stack_test_utils.test_environment(),
      ]

    named_contexts = [
        ('1', _context_fn, _environment_fn),
        ('2', _context_fn, _environment_fn),
        ('3', _context_fn, _environment_fn),
    ]

    class _FooTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):

      @context_stack_test_utils.with_contexts(*named_contexts)
      async def test_async(self):
        pass

      @context_stack_test_utils.with_contexts(*named_contexts)
      def test_sync(self):
        pass

      def test_undecorated(self):
        pass

    # Assert that a sync function is returned.
    for name, _, _ in named_contexts:
      test_name = f'test_sync_{name}'
      self.assertTrue(hasattr(_FooTest, test_name))
      test_fn = getattr(_FooTest, test_name)
      self.assertFalse(asyncio.iscoroutinefunction(test_fn))

    # Assert that an async function is returned.
    for name, _, _ in named_contexts:
      test_name = f'test_async_{name}'
      self.assertTrue(hasattr(_FooTest, test_name))
      test_fn = getattr(_FooTest, test_name)
      self.assertTrue(asyncio.iscoroutinefunction(test_fn))

    async_values = [lambda _: mock.AsyncMock()] * 3
    sync_values = [lambda _: mock.MagicMock()] * 3
    with mock.patch.object(
        context_stack_test_utils,
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
      calls = [mock.call(a, b) for _, a, b in named_contexts] * 2
      self.assertEqual(mock_with_context.mock_calls, calls)

  def test_raises_value_error(self):
    with self.assertRaises(ValueError):
      context_stack_test_utils.with_contexts()


if __name__ == '__main__':
  absltest.main()
