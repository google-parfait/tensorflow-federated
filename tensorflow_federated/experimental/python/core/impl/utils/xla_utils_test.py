# Copyright 2020, The TensorFlow Federated Authors.
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
import jax

from google.protobuf import any_pb2
from tensorflow_federated.experimental.python.core.impl.utils import xla_utils


class XlaUtilsTest(absltest.TestCase):

  def test_pack_xla_computation(self):
    xla_comp = jax.xla_computation(lambda: 10)()
    any_pb = xla_utils.pack_xla_computation(xla_comp)
    self.assertEqual(type(any_pb), any_pb2.Any)

  def test_pack_unpack_xla_computation_roundtrip(self):
    xla_comp = jax.xla_computation(lambda: 10)()
    any_pb = xla_utils.pack_xla_computation(xla_comp)
    new_comp = xla_utils.unpack_xla_computation(any_pb)
    self.assertEqual(new_comp.as_hlo_text(), xla_comp.as_hlo_text())


if __name__ == '__main__':
  absltest.main()
