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

import contextlib
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils


class WithContextTest(parameterized.TestCase):

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
  def test_with_context_fn_no_arg(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo():
      self.assertEqual(context_stack_impl.context_stack.current, context)

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)
      _foo()
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertLen(mock_enter_context.mock_calls, len(calls))
        mock_enter_context.assert_has_calls(calls)

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
  def test_with_context_fn_args(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo(x):
      del x  # Unused.
      self.assertEqual(context_stack_impl.context_stack.current, context)

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)
      _foo(1)
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertLen(mock_enter_context.mock_calls, len(calls))
        mock_enter_context.assert_has_calls(calls)

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
  def test_with_context_fn_kwargs(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo(x):
      del x  # Unused.
      self.assertEqual(context_stack_impl.context_stack.current, context)

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)
      _foo(x=1)
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertLen(mock_enter_context.mock_calls, len(calls))
        mock_enter_context.assert_has_calls(calls)

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
  def test_with_context_fn_return(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    @context_stack_test_utils.with_context(context_fn, environment_fn)
    def _foo(x):
      self.assertEqual(context_stack_impl.context_stack.current, context)
      return x

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)
      x = _foo(1)
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertLen(mock_enter_context.mock_calls, len(calls))
        mock_enter_context.assert_has_calls(calls)

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
      ('context_and_3_environments',
       context_stack_test_utils.TestContext(),
       [
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
           context_stack_test_utils.test_environment(),
       ]),
  )
  # pyformat: enable
  def test_with_context_test(self, context, environments):
    context_fn = lambda: context
    environment_fn = None if environments is None else lambda: environments

    class _FooTest(absltest.TestCase):

      @context_stack_test_utils.with_context(context_fn, environment_fn)
      def test_foo(self):
        self.assertEqual(context_stack_impl.context_stack.current, context)

      def test_bar(self):
        self.assertNotEqual(context_stack_impl.context_stack.current, context)

    with mock.patch.object(
        contextlib.ExitStack, 'enter_context'
    ) as mock_enter_context:
      # Assert that the context is installed.
      self.assertNotEqual(context_stack_impl.context_stack.current, context)
      suite = unittest.defaultTestLoader.loadTestsFromTestCase(_FooTest)
      runner = unittest.TextTestRunner()
      result = runner.run(suite)
      self.assertNotEqual(context_stack_impl.context_stack.current, context)

      # Assert that `enter_context` is called with the expected environment.
      if environments is not None:
        calls = [mock.call(e) for e in environments]
        self.assertLen(mock_enter_context.mock_calls, len(calls))
        mock_enter_context.assert_has_calls(calls)

      # Assert that the test passes with the expected number of test cases.
      self.assertEqual(suite.countTestCases(), 2)
      self.assertEqual(result.testsRun, 2)
      self.assertTrue(result.wasSuccessful())


class WithContextsTest(parameterized.TestCase):

  def test_with_contexts(self):
    # context = context_stack_test_utils.TestContext()
    # context_fn = lambda: context
    # environments = [
    #     context_stack_test_utils.test_environment(),
    #     context_stack_test_utils.test_environment(),
    #     context_stack_test_utils.test_environment(),
    # ]
    # environment_fn = lambda: environments

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

    class _FooTest(parameterized.TestCase):

      @context_stack_test_utils.with_contexts(*named_contexts)
      def test_foo(self):
        pass

      def test_bar(self):
        pass

    with mock.patch.object(
        context_stack_test_utils, 'with_context'
    ) as mock_with_context:
      suite = unittest.defaultTestLoader.loadTestsFromTestCase(_FooTest)
      runner = unittest.TextTestRunner()
      result = runner.run(suite)

      # Assert that `with_context` is called with the expected parameters.
      calls = []
      for named_context in named_contexts:
        _, a, b = named_context
        # It is not entirely clear why this mock is being called three times for
        # each named_context instead of one. It appears that the two additional
        # calls are for the decorator and for the wrapper; however, the
        # `with_context` function is only invoked once.
        calls.extend([
            mock.call(a, b),
            mock.call()(mock.ANY),
            mock.call()()(mock.ANY),
        ])
      self.assertLen(mock_with_context.mock_calls, len(calls))
      mock_with_context.assert_has_calls(calls)

      # Assert that the test passes with the expected number of test cases.
      self.assertEqual(suite.countTestCases(), len(named_contexts) + 1)
      self.assertEqual(result.testsRun, len(named_contexts) + 1)
      self.assertTrue(result.wasSuccessful())

  def test_with_contexts_raises_value_error(self):
    with self.assertRaises(ValueError):
      context_stack_test_utils.with_contexts()


if __name__ == '__main__':
  absltest.main()
