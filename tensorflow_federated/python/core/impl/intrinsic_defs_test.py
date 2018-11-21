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
"""Tests for intrinsic_defs.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_utils


class IntrinsicDefsTest(unittest.TestCase):

  def setUp(self):
    self._defs = [
        value for value in [
            getattr(intrinsic_defs, name) for name in dir(intrinsic_defs)]
        if isinstance(value, intrinsic_defs.IntrinsicDef)]

  def test_names_are_unique(self):
    found = set()
    for d in self._defs:
      self.assertNotIn(d.name, found)
      found.add(d.name)

  def test_uris_are_unique(self):
    found = set()
    for d in self._defs:
      self.assertNotIn(d.uri, found)
      found.add(d.uri)

  def test_types_are_well_formed(self):
    for d in self._defs:
      type_utils.check_well_formed(d.type_signature)


if __name__ == '__main__':
  unittest.main()
