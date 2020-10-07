# Copyright 2018, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.api import test_case


class TestUtilsTest(test_case.TestCase):

  def test_nested_structures_are_same_where_they_are_same(self):
    self.assert_nested_struct_eq({'a': 10}, {'a': 10})

  def test_nested_structures_are_same_where_nesting_differs(self):
    with self.assertRaises(self.failureException):  # pylint: disable=g-error-prone-assert-raises
      self.assert_nested_struct_eq({'a': 10}, 10)

  def test_nested_structures_are_same_where_values_differ(self):
    with self.assertRaises(self.failureException):  # pylint: disable=g-error-prone-assert-raises
      self.assert_nested_struct_eq({'a': 10}, {'a': False})


if __name__ == '__main__':
  test_case.main()
