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

  def test_deprecated_class(self):
    message = 'test warning'
    mock_fn = mock.Mock()

    @deprecation.deprecated(message)
    class Foo:

      def __init__(self):
        mock_fn()

    with self.assertWarnsRegex(DeprecationWarning, message):
      foo = Foo()
    mock_fn.assert_called_once()
    self.assertIsNotNone(foo)

  def test_deprecated_method(self):
    message = 'test warning'
    mock_fn = mock.Mock()

    class Foo:

      @deprecation.deprecated(message)
      def bar(self):
        mock_fn()

    foo = Foo()
    with self.assertWarnsRegex(DeprecationWarning, message):
      foo.bar()
    mock_fn.assert_called_once()

  def test_deprecated_function(self):
    message = 'test warning'
    mock_fn = mock.Mock()

    @deprecation.deprecated(message)
    def foo():
      mock_fn()

    with self.assertWarnsRegex(DeprecationWarning, message):
      foo()
    mock_fn.assert_called_once()


if __name__ == '__main__':
  absltest.main()
