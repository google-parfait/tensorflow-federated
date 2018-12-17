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
"""Tests for test_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils as common_test_utils

from tensorflow_federated.python.core.impl import test_utils


class TestUtilsTest(common_test_utils.TffTestCase):

  def test_nested_structures_are_same_where_they_are_same(self):
    test_utils.assert_nested_struct_eq({'a': 10}, {'a': 10})

  def test_nested_structures_are_same_where_nesting_differs(self):
    with self.assertRaises(ValueError):
      test_utils.assert_nested_struct_eq({'a': 10}, 10)

  def test_nested_structures_are_same_where_values_differ(self):
    with self.assertRaises(ValueError):
      test_utils.assert_nested_struct_eq({'a': 10}, {'a': False})


if __name__ == '__main__':
  tf.test.main()
