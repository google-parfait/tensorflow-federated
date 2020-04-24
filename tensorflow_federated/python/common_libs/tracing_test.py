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
import io
import logging as std_logging
import time

from absl import logging
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.common_libs import tracing

tf.compat.v1.enable_v2_behavior()

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

    self._test_debug_logging_with_async_function(foo, '1.0')

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


if __name__ == '__main__':
  absltest.main()
