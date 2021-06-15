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

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow_federated.python.core.impl.compiler import intrinsic_defs


def _get_intrinsic_names():
  return [
      name for name in dir(intrinsic_defs)
      if isinstance(getattr(intrinsic_defs, name), intrinsic_defs.IntrinsicDef)
  ]


class IntrinsicDefsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *[(name.lower(), name) for name in _get_intrinsic_names()])
  def test_names_match_those_in_module(self, name):
    self.assertEqual(getattr(intrinsic_defs, name).name, name)

  def test_uris_are_unique(self):
    uris_found = set()
    for name in _get_intrinsic_names():
      uri = getattr(intrinsic_defs, name).uri
      self.assertNotIn(uri, uris_found)
      uris_found.add(uri)

  @parameterized.named_parameters(
      ('federated_broadcast', 'FEDERATED_BROADCAST', '(T@SERVER -> T@CLIENTS)'),
      ('federated_eval_at_clients', 'FEDERATED_EVAL_AT_CLIENTS',
       '(( -> T) -> {T}@CLIENTS)'),
      ('federated_eval_at_server', 'FEDERATED_EVAL_AT_SERVER',
       '(( -> T) -> T@SERVER)'),
      ('federated_map', 'FEDERATED_MAP',
       '(<(T -> U),{T}@CLIENTS> -> {U}@CLIENTS)'),
      ('federated_secure_sum_bitwidth', 'FEDERATED_SECURE_SUM',
       '(<{V}@CLIENTS,B> -> V@SERVER)'),
      ('federated_secure_select', 'FEDERATED_SECURE_SELECT',
       '(<{Ks}@CLIENTS,int32@SERVER,T@SERVER,(<T,int32> -> U)> -> {U*}@CLIENTS)'
      ),
      ('federated_select', 'FEDERATED_SELECT',
       '(<{Ks}@CLIENTS,int32@SERVER,T@SERVER,(<T,int32> -> U)> -> {U*}@CLIENTS)'
      ),
      ('federated_sum', 'FEDERATED_SUM', '({T}@CLIENTS -> T@SERVER)'),
      ('federated_zip_at_clients', 'FEDERATED_ZIP_AT_CLIENTS',
       '(<{T}@CLIENTS,{U}@CLIENTS> -> {<T,U>}@CLIENTS)'),
      ('federated_zip_at_server', 'FEDERATED_ZIP_AT_SERVER',
       '(<T@SERVER,U@SERVER> -> <T,U>@SERVER)'),
  )
  def test_type_signature_strings(self, name, type_str):
    intrinsic = getattr(intrinsic_defs, name)
    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     type_str)


if __name__ == '__main__':
  absltest.main()
