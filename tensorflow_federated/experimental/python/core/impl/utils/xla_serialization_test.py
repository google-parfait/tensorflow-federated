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
from jax.lib.xla_bridge import xla_client
import numpy as np

from google.protobuf import any_pb2
from tensorflow_federated.experimental.python.core.impl.utils import xla_serialization
from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization


def _make_test_xla_comp():
  builder = xla_client.XlaBuilder('comp')
  xla_client.ops.Constant(builder, np.int32(10))
  return builder.build()


class XlaUtilsTest(absltest.TestCase):

  def test_pack_xla_computation(self):
    xla_comp = _make_test_xla_comp()
    any_pb = xla_serialization.pack_xla_computation(xla_comp)
    self.assertEqual(type(any_pb), any_pb2.Any)

  def test_pack_unpack_xla_computation_roundtrip(self):
    xla_comp = _make_test_xla_comp()
    any_pb = xla_serialization.pack_xla_computation(xla_comp)
    new_comp = xla_serialization.unpack_xla_computation(any_pb)
    self.assertEqual(new_comp.as_hlo_text(), xla_comp.as_hlo_text())

  def test_create_xla_tff_computation(self):
    xla_comp = _make_test_xla_comp()
    comp_pb = xla_serialization.create_xla_tff_computation(
        xla_comp, computation_types.FunctionType(None, np.int32))
    self.assertIsInstance(comp_pb, pb.Computation)
    self.assertEqual(comp_pb.WhichOneof('computation'), 'xla')
    type_spec = type_serialization.deserialize_type(comp_pb.type)
    self.assertEqual(str(type_spec), '( -> int32)')
    xla_comp = xla_serialization.unpack_xla_computation(comp_pb.xla.hlo_module)
    self.assertIn('ROOT constant.1 = s32[] constant(10)',
                  xla_comp.as_hlo_text())
    self.assertEqual(str(comp_pb.xla.parameter), '')
    self.assertEqual(str(comp_pb.xla.result), 'tensor {\n' '  index: 0\n' '}\n')


if __name__ == '__main__':
  absltest.main()
