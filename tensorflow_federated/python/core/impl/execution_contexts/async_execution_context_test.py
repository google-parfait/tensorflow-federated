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

from absl.testing import absltest

from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.executors import executors_errors


class RetryableErrorTest(absltest.TestCase):

  def test_is_retryable_error(self):
    retryable_error = executors_errors.RetryableError()
    self.assertTrue(
        async_execution_context._is_retryable_error(retryable_error))
    self.assertFalse(async_execution_context._is_retryable_error(TypeError()))
    self.assertFalse(async_execution_context._is_retryable_error(1))
    self.assertFalse(async_execution_context._is_retryable_error('a'))
    self.assertFalse(async_execution_context._is_retryable_error(None))


if __name__ == '__main__':
  absltest.main()
