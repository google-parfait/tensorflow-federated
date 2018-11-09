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
"""Tests for py_typecheck.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from six import string_types

import unittest

from tensorflow_federated.python.common_libs import py_typecheck


class PyTypeCheckTest(unittest.TestCase):

  def test_check_type(self):
    try:
      py_typecheck.check_type('foo', str)
      py_typecheck.check_type('foo', string_types)
      py_typecheck.check_type(10, int)
      py_typecheck.check_type(10, (str, int))
      py_typecheck.check_type(10, (str, int, bool, float))
    except TypeError:
      self.fail('Function {} raised TypeError unexpectedly.'.format(
          py_typecheck.check_type.__name__))
    self.assertRaisesRegexp(
        TypeError,
        'Expected .*TestCase, found int.',
        py_typecheck.check_type,
        10,
        unittest.TestCase)
    self.assertRaisesRegexp(
        TypeError,
        'Expected foo to be of type int, found __main__.PyTypeCheckTest.',
        py_typecheck.check_type,
        self,
        int,
        label='foo')
    self.assertRaisesRegexp(
        TypeError,
        'Expected int or bool, found str.',
        py_typecheck.check_type,
        'a',
        (int, bool))
    self.assertRaisesRegexp(
        TypeError,
        'Expected int, bool, or float, found str.',
        py_typecheck.check_type,
        'a',
        (int, bool, float))


if __name__ == '__main__':
  unittest.main()
