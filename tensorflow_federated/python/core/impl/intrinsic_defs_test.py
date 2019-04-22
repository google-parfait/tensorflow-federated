# Lint as: python3
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

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import type_utils


def _get_intrinsic_names():
  return [
      name for name in dir(intrinsic_defs)
      if isinstance(getattr(intrinsic_defs, name), intrinsic_defs.IntrinsicDef)
  ]


class IntrinsicDefsTest(parameterized.TestCase):

  @parameterized.parameters(*[(name,) for name in _get_intrinsic_names()])
  def test_names_match_those_in_module(self, name):
    self.assertEqual(getattr(intrinsic_defs, name).name, name)

  def test_uris_are_unique(self):
    uris_found = set()
    for name in _get_intrinsic_names():
      uri = getattr(intrinsic_defs, name).uri
      self.assertNotIn(uri, uris_found)
      uris_found.add(uri)

  @parameterized.parameters(*[(name,) for name in _get_intrinsic_names()])
  def test_types_are_well_formed(self, name):
    type_utils.check_well_formed(getattr(intrinsic_defs, name).type_signature)

  @parameterized.parameters(
      ('FEDERATED_BROADCAST', '(T@SERVER -> T@CLIENTS)'),
      ('FEDERATED_MAP', '(<(T -> U),{T}@CLIENTS> -> {U}@CLIENTS)'),
      ('FEDERATED_SUM', '({T}@CLIENTS -> T@SERVER)'),
      ('FEDERATED_ZIP_AT_CLIENTS',
       '(<{T}@CLIENTS,{U}@CLIENTS> -> {<T,U>}@CLIENTS)'),
      ('FEDERATED_ZIP_AT_SERVER', '(<T@SERVER,U@SERVER> -> <T,U>@SERVER)'))
  def test_type_signature_strings(self, name, type_str):
    self.assertEqual(
        str(getattr(intrinsic_defs, name).type_signature), type_str)


if __name__ == '__main__':
  absltest.main()
