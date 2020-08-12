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
import numpy as np

from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.types import placement_literals


class FederatedSecureSumTest(absltest.TestCase):

  def run(self, result=None):
    fc_context = federated_computation_context.FederatedComputationContext(
        context_stack_impl.context_stack)
    with context_stack_impl.context_stack.install(fc_context):
      super(FederatedSecureSumTest, self).run(result)

  def test_type_signature_with_int(self):
    value = intrinsics.federated_value(1, placement_literals.CLIENTS)
    bitwidth = 8

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     'int32@SERVER')

  def test_type_signature_with_structure_of_ints(self):
    value = intrinsics.federated_value([1, [1, 1]], placement_literals.CLIENTS)
    bitwidth = [8, [4, 2]]

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_one_tensor_and_bitwidth(self):
    value = intrinsics.federated_value(
        np.ndarray(shape=(5, 37), dtype=np.int16), placement_literals.CLIENTS)
    bitwidth = 2

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     'int16[5,37]@SERVER')

  def test_type_signature_with_structure_of_tensors_and_bitwidths(self):
    np_array = np.ndarray(shape=(5, 37), dtype=np.int16)
    value = intrinsics.federated_value((np_array, np_array),
                                       placement_literals.CLIENTS)
    bitwidth = (2, 2)

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     '<int16[5,37],int16[5,37]>@SERVER')

  def test_raises_type_error_with_value_float(self):
    value = intrinsics.federated_value(1.0, placement_literals.CLIENTS)
    bitwidth = intrinsics.federated_value(1, placement_literals.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, bitwidth)

  def test_raises_type_error_with_bitwith_int_at_server(self):
    value = intrinsics.federated_value(1, placement_literals.CLIENTS)
    bitwidth = intrinsics.federated_value(1, placement_literals.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, bitwidth)

  def test_raises_type_error_with_different_structures(self):
    value = intrinsics.federated_value([1, [1, 1]], placement_literals.CLIENTS)
    bitwidth = 8

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, bitwidth)


if __name__ == '__main__':
  absltest.main()
