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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case


empty_struct = computation_types.StructType([])
container_mismatch = (empty_struct,
                      computation_types.StructWithPythonType([], tuple))
named_field = computation_types.StructType([('a', empty_struct)])
unnamed_field = computation_types.StructType([(None, empty_struct)])
naming_mismatch = (named_field, unnamed_field)


class TestUtilsTest(test_case.TestCase):

  def test_type_assignable_from_passes_add_name(self):
    self.assert_type_assignable_from(named_field, unnamed_field)

  def test_type_assignable_from_fails_remove_name(self):
    with self.assertRaises(self.failureException):  # pylint: disable=g-error-prone-assert-raises
      self.assert_type_assignable_from(unnamed_field, named_field)

  def test_types_equivalent_passes_container(self):
    self.assert_types_equivalent(*container_mismatch)

  def test_types_equivalent_fails_naming(self):
    with self.assertRaises(self.failureException):  # pylint: disable=g-error-prone-assert-raises
      self.assert_types_equivalent(*naming_mismatch)

  def test_types_identical_passes_exact(self):
    self.assert_types_identical(empty_struct, empty_struct)

  def test_types_identical_fails_container(self):
    with self.assertRaises(self.failureException):  # pylint: disable=g-error-prone-assert-raises
      self.assert_types_identical(*container_mismatch)

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
