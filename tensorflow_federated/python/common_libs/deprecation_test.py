## Copyright 2022, The TensorFlow Federated Authors.
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
"""Tests for deprecation."""

from unittest import mock

from absl.testing import absltest

from tensorflow_federated.python.common_libs import deprecation


class DeprecationTest(absltest.TestCase):

  def test_wraps_method(self):
    mock_method = mock.Mock()
    test_message = 'test warning'
    wrapped_mock_method = deprecation.deprecated(mock_method, test_message)
    with self.assertWarnsRegex(DeprecationWarning, test_message):
      wrapped_mock_method(1, 2, c=3)
    self.assertSequenceEqual(mock_method.call_args_list, [mock.call(1, 2, c=3)])

  def test_decorates_method(self):
    test_message = 'test warning'
    mock_method = mock.Mock()

    @deprecation.deprecated(test_message)
    def foo():
      mock_method()

    with self.assertWarnsRegex(DeprecationWarning, test_message):
      foo()
    mock_method.assert_called_once()


if __name__ == '__main__':
  absltest.main()
