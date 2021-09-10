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
from typing import Any
from unittest import mock

from absl.testing import absltest

from tensorflow_federated.python.common_libs import retrying


class RetryingArgValidationtest(absltest.TestCase):

  def test_raises_non_function(self):
    with self.assertRaises(TypeError):
      retrying.retry(fn=0)

  def test_raises_non_function_exception_filter(self):
    with self.assertRaises(TypeError):
      retrying.retry(fn=lambda x: x, retry_on_exception_filter=0)

  def test_raises_complex_wait_multiplier(self):
    with self.assertRaises(TypeError):
      retrying.retry(fn=lambda x: x, wait_multiplier=1j)

  def test_raises_complex_max_wait_ms(self):
    with self.assertRaises(TypeError):
      retrying.retry(fn=lambda x: x, wait_max_ms=1j)

  def test_raises_zero_wait_multiplier(self):
    with self.assertRaises(ValueError):
      retrying.retry(fn=lambda x: x, wait_multiplier=0)

  def test_raises_zero_max_wait_ms(self):
    with self.assertRaises(ValueError):
      retrying.retry(fn=lambda x: x, wait_max_ms=0)


class CountInvocations():

  def __init__(self, n_invocations_to_raise: int, error_to_raise: Exception,
               return_value: Any):
    self._n_invocations_to_raise = n_invocations_to_raise
    self._error_to_raise = error_to_raise
    self._return_value = return_value
    self._n_invocations = 0

  @property
  def n_invocations(self):
    return self._n_invocations

  def __call__(self, *args, **kwargs):
    del args, kwargs  # Unused
    self._n_invocations += 1
    if self._n_invocations <= self._n_invocations_to_raise:
      raise self._error_to_raise
    return self._return_value


class RetryingFunctionTest(absltest.TestCase):

  def test_standalone_decorator_always_retries(self):

    expected_return_val = 0
    expected_num_invocations = 3
    count_invocations_callable = CountInvocations(expected_num_invocations,
                                                  TypeError('Error'),
                                                  expected_return_val)

    @retrying.retry
    def invoke_callable(*args, **kwargs):
      return count_invocations_callable(*args, **kwargs)

    return_val = invoke_callable()

    self.assertEqual(return_val, expected_return_val)
    # Final call succeeds
    self.assertEqual(count_invocations_callable.n_invocations,
                     expected_num_invocations + 1)

  def test_error_filter_raises_wrong_error_type(self):

    count_invocations_callable = CountInvocations(1, TypeError('Error'), 0)

    @retrying.retry(
        retry_on_exception_filter=lambda e: isinstance(e, ValueError))
    def invoke_callable(*args, **kwargs):
      return count_invocations_callable(*args, **kwargs)

    with self.assertRaises(TypeError):
      invoke_callable()

  def test_error_filter_called_with_raised_err(self):

    error = TypeError('error')
    expected_result = 1

    count_invocations_callable = CountInvocations(1, error, 1)
    mock_callable = mock.MagicMock(return_value=True)

    def err_filter(*args):
      return mock_callable(*args)

    @retrying.retry(retry_on_exception_filter=err_filter)
    def invoke_callable(*args, **kwargs):
      return count_invocations_callable(*args, **kwargs)

    result = invoke_callable()
    self.assertEqual(result, expected_result)
    mock_callable.assert_called_once_with(error)


class RetryingCoroFunctionTest(absltest.TestCase):

  def setUp(self):
    self._loop = asyncio.new_event_loop()
    super().setUp()

  def _run_sync(self, fn, args=None):
    return self._loop.run_until_complete(fn(args))

  def test_standalone_decorator_always_retries(self):

    expected_return_val = 0
    expected_num_invocations = 3
    count_invocations_callable = CountInvocations(expected_num_invocations,
                                                  TypeError('Error'),
                                                  expected_return_val)

    @retrying.retry
    async def invoke_callable(*args, **kwargs):
      return count_invocations_callable(*args, **kwargs)

    return_val = self._run_sync(invoke_callable)

    self.assertEqual(return_val, expected_return_val)
    # Final call succeeds
    self.assertEqual(count_invocations_callable.n_invocations,
                     expected_num_invocations + 1)

  def test_error_filter_raises_wrong_error_type(self):

    count_invocations_callable = CountInvocations(1, TypeError('Error'), 0)

    @retrying.retry(
        retry_on_exception_filter=lambda e: isinstance(e, ValueError))
    async def invoke_callable(*args, **kwargs):
      return count_invocations_callable(*args, **kwargs)

    with self.assertRaises(TypeError):
      self._run_sync(invoke_callable)

  def test_error_filter_called_with_raised_err(self):

    error = TypeError('error')
    expected_result = 1

    count_invocations_callable = CountInvocations(1, error, 1)
    mock_callable = mock.MagicMock(return_value=True)

    def err_filter(*args):
      return mock_callable(*args)

    @retrying.retry(retry_on_exception_filter=err_filter)
    async def invoke_callable(*args, **kwargs):
      return count_invocations_callable(*args, **kwargs)

    result = self._run_sync(invoke_callable)
    self.assertEqual(result, expected_result)
    mock_callable.assert_called_once_with(error)


if __name__ == '__main__':
  absltest.main()
