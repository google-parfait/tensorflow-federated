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
import functools
import io
import logging as std_logging
import threading
import time

from absl import logging
from absl.testing import absltest

from tensorflow_federated.python.common_libs import tracing

# Traces may not run in _exactly_ one second, but we can assert it was at least
# one second; and most importantly the time should be logged.
ELAPSED_ONE_REGEX = r'Elapsed time [1-9][0-9]*\.[0-9]+'


class DebugLoggingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.log = io.StringIO()
    self.handler = std_logging.StreamHandler(self.log)
    std_logging.root.addHandler(self.handler)

  def tearDown(self):
    std_logging.root.removeHandler(self.handler)
    self.handler.close()
    super().tearDown()

  def _test_debug_logging_with_async_function(self, async_fn, test_regex, *args,
                                              **kwargs):
    loop = asyncio.get_event_loop()
    try:
      logging.set_verbosity(1)
      retval = loop.run_until_complete(async_fn(*args, **kwargs))
    finally:
      logging.set_verbosity(0)
    self.assertRegexMatch(''.join(self.log.getvalue()), [test_regex])
    self.log.truncate(0)
    loop.run_until_complete(async_fn(*args, **kwargs))
    self.assertEmpty(''.join(self.log.getvalue()))
    return retval

  def _test_debug_logging_with_sync_function(self, sync_fn, test_regex, *args,
                                             **kwargs):
    try:
      logging.set_verbosity(1)
      retval = sync_fn(*args, **kwargs)
    finally:
      logging.set_verbosity(0)
    self.assertRegexMatch(''.join(self.log.getvalue()), [test_regex])
    self.log.truncate(0)
    self.assertEmpty(''.join(self.log.getvalue()))
    return retval

  def test_logging_enter_exit(self):

    @tracing.trace
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(
        foo, '.*Entering .*foo.*\n.*Exiting .*foo.*')

  def test_logging_timing_captured(self):

    @tracing.trace
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(foo, 'Elapsed time')

  def test_logging_timing_captures_value_around_async_call(self):

    @tracing.trace
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(
        foo, r'<locals>\.foo\. ' + ELAPSED_ONE_REGEX)

  def test_logging_non_blocking_function(self):

    @tracing.trace(span=True)
    async def foo():
      return await asyncio.gather(
          asyncio.sleep(1), asyncio.sleep(1), asyncio.sleep(1))

    self._test_debug_logging_with_async_function(
        foo, r'<locals>\.foo\. ' + ELAPSED_ONE_REGEX)

  def test_logging_non_blocking_method(self):

    class AClass(absltest.TestCase):

      @tracing.trace(span=True)
      async def async_method(self, foo_arg, bar_arg, arg3=None, arg4=None):
        self.assertEqual('foo', foo_arg)
        self.assertEqual('bar', bar_arg)
        self.assertIsNotNone(arg3)
        self.assertIsNotNone(arg4)
        await asyncio.sleep(1)
        return 3

    a_class = AClass()

    result = self._test_debug_logging_with_async_function(
        a_class.async_method,
        # Non-blocking may not run exactly one second, but we can assert it was
        # at least one second; and most importantly it should be logged.
        r'AClass\.async_method\. ' + ELAPSED_ONE_REGEX,
        'foo',
        'bar',
        arg3='baz',
        arg4=True)
    self.assertEqual(3, result)

  def test_logging_blocking_method(self):

    class AClass(absltest.TestCase):

      @tracing.trace(span=True)
      def sync_method(self, foo_arg, bar_arg, arg3=None, arg4=None):
        self.assertEqual('foo', foo_arg)
        self.assertEqual('bar', bar_arg)
        self.assertIsNotNone(arg3)
        self.assertIsNotNone(arg4)
        # Sleep for 1s is used to test that we measured runtime correctly
        time.sleep(1)
        return 3

    a_class = AClass()

    result = self._test_debug_logging_with_sync_function(
        a_class.sync_method,
        r'AClass\.sync_method\. ' + ELAPSED_ONE_REGEX,
        'foo',
        'bar',
        arg3='baz',
        arg4=True)
    self.assertEqual(3, result)

  def test_logging_blocking_function(self):

    @tracing.trace(span=True)
    def foo(foo_arg, bar_arg, arg3=None, arg4=None):
      self.assertEqual('foo', foo_arg)
      self.assertEqual('bar', bar_arg)
      self.assertIsNotNone(arg3)
      self.assertIsNotNone(arg4)
      # Sleep for 1s is used to test that we measured runtime correctly
      time.sleep(1)
      return 3

    result = self._test_debug_logging_with_sync_function(
        foo,
        r'<locals>\.foo\. ' + ELAPSED_ONE_REGEX,
        'foo',
        'bar',
        arg3='baz',
        arg4=True)
    self.assertEqual(3, result)


class MockTracingProvider(tracing.TracingProvider):

  def __init__(self):
    self.scopes = []
    self.sub_scopes = []
    self.nonces = []
    self.parent_span_yields = []
    self.fn_argss = []
    self.fn_kwargss = []
    self.trace_optss = []
    self.trace_results = []

  def span(self, scope, sub_scope, nonce, parent_span_yield, fn_args, fn_kwargs,
           trace_opts):
    self.scopes.append(scope)
    self.sub_scopes.append(sub_scope)
    self.nonces.append(nonce)
    self.parent_span_yields.append(parent_span_yield)
    self.fn_argss.append(fn_args)
    self.fn_kwargss.append(fn_kwargs)
    self.trace_optss.append(trace_opts)
    if parent_span_yield is None:
      new_yield = 0
    else:
      new_yield = parent_span_yield + 1
    result = yield new_yield
    self.trace_results.append(result)


def set_mock_trace() -> MockTracingProvider:
  mock = MockTracingProvider()
  tracing.set_tracing_providers([mock])
  return mock


class TracingProviderInterfaceTest(absltest.TestCase):

  def test_basic_span(self):
    mock = set_mock_trace()
    with tracing.span('scope', 'sub_scope', options='some_option'):
      pass
    self.assertEqual(mock.scopes[0], 'scope')
    self.assertEqual(mock.sub_scopes[0], 'sub_scope')
    self.assertIsNone(mock.parent_span_yields[0])
    self.assertIsNone(mock.fn_argss[0])
    self.assertIsNone(mock.fn_kwargss[0])
    self.assertEqual(mock.trace_optss[0], {'options': 'some_option'})
    self.assertIsInstance(mock.trace_results[0], tracing.TracedSpan)

  def test_sibling_spans(self):
    mock = set_mock_trace()
    with tracing.span('parent', ''):
      with tracing.span('child1', ''):
        pass
      with tracing.span('child2', ''):
        pass
    with tracing.span('parentless', ''):
      pass

    self.assertEqual(mock.scopes, ['parent', 'child1', 'child2', 'parentless'])
    self.assertEqual(mock.parent_span_yields, [None, 0, 0, None])

  def test_nested_non_async_span(self):
    mock = set_mock_trace()
    with tracing.span('outer', 'osub'):
      with tracing.span('middle', 'msub'):
        with tracing.span('inner', 'isub'):
          pass
    self.assertEqual(mock.scopes, ['outer', 'middle', 'inner'])
    self.assertEqual(mock.sub_scopes, ['osub', 'msub', 'isub'])
    self.assertEqual(mock.parent_span_yields, [None, 0, 1])

  def test_basic_trace(self):
    mock = set_mock_trace()

    class MyClass:

      @tracing.trace(options='some_option')
      def my_func(a, b, kw=None):  # pylint: disable=no-self-argument
        del a, b, kw
        return 5

    MyClass.my_func(1, 2, kw=3)
    self.assertEqual(mock.scopes[0], 'MyClass')
    self.assertEqual(mock.sub_scopes[0], 'my_func')
    self.assertIsNone(mock.parent_span_yields[0])
    self.assertEqual(mock.fn_argss[0], (1, 2))
    self.assertEqual(mock.fn_kwargss[0], {'kw': 3})
    self.assertEqual(mock.trace_optss[0], {'options': 'some_option'})
    self.assertIsInstance(mock.trace_results[0], tracing.TracedFunctionReturned)
    self.assertEqual(mock.trace_results[0].value, 5)

  def test_trace_throws(self):
    mock = set_mock_trace()

    class MyClass:

      @tracing.trace
      def my_func():  # pylint: disable=no-method-argument
        raise ValueError(5)

    try:
      MyClass.my_func()
      raise AssertionError('should have thrown')
    except ValueError:
      pass

    self.assertIsInstance(mock.trace_results[0], tracing.TracedFunctionThrew)
    self.assertEqual(mock.trace_results[0].error_type, ValueError)
    self.assertIsInstance(mock.trace_results[0].error_value, ValueError)

  def test_parenting_non_async_to_async_to_nested_async(self):
    mock = set_mock_trace()
    loop = asyncio.new_event_loop()
    loop.set_task_factory(tracing.propagate_trace_context_task_factory)

    def run_loop():
      loop.run_forever()
      loop.close()

    thread = threading.Thread(target=functools.partial(run_loop), daemon=True)
    thread.start()

    @tracing.trace
    async def middle():
      with tracing.span('inner', ''):
        pass

    with tracing.span('outer', ''):
      # This sends the coroutine over to another thread,
      # keeping the current trace context.
      coro_with_trace_ctx = tracing.wrap_coroutine_in_current_trace_context(
          middle())
      asyncio.run_coroutine_threadsafe(coro_with_trace_ctx, loop).result()

    loop.call_soon_threadsafe(loop.stop)
    thread.join()

    self.assertEqual(mock.parent_span_yields, [None, 0, 1])
    self.assertEqual(mock.scopes, ['outer', '<locals>', 'inner'])
    self.assertEqual(mock.sub_scopes, ['', 'middle', ''])


if __name__ == '__main__':
  absltest.main()
