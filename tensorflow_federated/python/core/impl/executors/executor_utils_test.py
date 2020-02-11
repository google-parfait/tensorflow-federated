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
"""Tests for executor_utils."""

import asyncio
import io
import logging as std_logging
import time

from absl import logging
from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.python.core.impl.executors import executor_utils


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

  def _test_debug_logging_with_async_function(self, async_fn, test_regex):
    loop = asyncio.get_event_loop()
    try:
      logging.set_verbosity(1)
      loop.run_until_complete(async_fn())
    finally:
      logging.set_verbosity(0)
    self.assertRegexMatch(''.join(self.log.getvalue()), [test_regex])
    self.log.truncate(0)
    loop.run_until_complete(async_fn())
    self.assertEmpty(''.join(self.log.getvalue()))

  def test_log_async_fails_non_async_fn(self):

    with self.assertRaises(TypeError):

      @executor_utils.log_async
      def _():
        return time.sleep(1)

  def test_logging_enter_exit(self):

    @executor_utils.log_async
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(
        foo, 'Entering .*foo.*\nExiting .*foo.*')

  def test_logging_provenance(self):

    @executor_utils.log_async
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(foo, 'DebugLoggingTest')

  def test_logging_timing_captured(self):

    @executor_utils.log_async
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(foo, 'Elapsed time')

  def test_logging_timing_captures_value_around_async_call(self):

    @executor_utils.log_async
    async def foo():
      return await asyncio.sleep(1)

    self._test_debug_logging_with_async_function(foo, '1.0')

  def test_logging_non_blocking(self):

    @executor_utils.log_async
    async def foo():
      return await asyncio.gather(
          asyncio.sleep(1), asyncio.sleep(1), asyncio.sleep(1))

    self._test_debug_logging_with_async_function(foo, '1.0')


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  absltest.main()
