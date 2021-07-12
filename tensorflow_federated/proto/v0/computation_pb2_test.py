# Copyright 2021, The TensorFlow Federated Authors.
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

from tensorflow.core.framework import types_pb2
from tensorflow_federated.proto.v0 import computation_pb2


class ComputationPb2Test(absltest.TestCase):

  def test_tensor_datatype_enum_subset_of_tensorflow(self):
    self.assertContainsSubset(
        list(computation_pb2.TensorType.DataType.items()),
        list(types_pb2.DataType.items()))

  def test_tensor_datatype_enum_no_reference_types(self):
    tff_enum_items = frozenset(
        list(computation_pb2.TensorType.DataType.items()))
    for name, value in types_pb2.DataType.items():
      if not name.endswith('_REF'):
        continue
      self.assertNotIn((name, value), tff_enum_items)

  def test_tensor_datatype_enum_no_variant_type(self):
    self.assertNotIn(types_pb2.DT_VARIANT,
                     list(computation_pb2.TensorType.DataType.values()))

  def test_tensor_datatype_enum_no_resource_type(self):
    self.assertNotIn(types_pb2.DT_RESOURCE,
                     list(computation_pb2.TensorType.DataType.values()))


if __name__ == '__main__':
  absltest.main()
