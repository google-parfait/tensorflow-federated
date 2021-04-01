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

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import intrinsic_factory
from tensorflow_federated.python.core.impl.types import placements


class FederatedSecureSumTest(absltest.TestCase):

  def test_type_signature_with_int(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    bitwidth = 8

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     'int32@SERVER')

  def test_type_signature_with_structure_of_ints(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    bitwidth = [8, [4, 2]]

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_structure_of_ints_scalar_bitwidth(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    bitwidth = 8

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_one_tensor_and_bitwidth(self):
    value = intrinsics.federated_value(
        np.ndarray(shape=(5, 37), dtype=np.int16), placements.CLIENTS)
    bitwidth = 2

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     'int16[5,37]@SERVER')

  def test_type_signature_with_structure_of_tensors_and_bitwidths(self):
    np_array = np.ndarray(shape=(5, 37), dtype=np.int16)
    value = intrinsics.federated_value((np_array, np_array), placements.CLIENTS)
    bitwidth = (2, 2)

    intrinsic = intrinsics.federated_secure_sum(value, bitwidth)

    self.assertEqual(intrinsic.type_signature.compact_representation(),
                     '<int16[5,37],int16[5,37]>@SERVER')

  def test_raises_type_error_with_value_float(self):
    value = intrinsics.federated_value(1.0, placements.CLIENTS)
    bitwidth = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, bitwidth)

  def test_raises_type_error_with_bitwith_int_at_server(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    bitwidth = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, bitwidth)

  def test_raises_type_error_with_different_structures(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    bitwidth = [8, 4, 2]

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, bitwidth)


class SequenceReduceTest(absltest.TestCase):

  def test_type_signature_with_non_federated_type(self):
    factory = intrinsic_factory.IntrinsicFactory(
        context_stack_impl.context_stack)

    @computations.tf_computation(np.int32, np.int32)
    def add(x, y):
      return x + y

    @computations.federated_computation(
        computation_types.SequenceType(np.int32))
    def foo(value):
      return factory.sequence_reduce(value, 0, add)

    self.assertEqual(foo.type_signature.compact_representation(),
                     '(int32* -> int32)')

  def test_type_signature_with_federated_type(self):
    factory = intrinsic_factory.IntrinsicFactory(
        context_stack_impl.context_stack)

    @computations.tf_computation(np.int32, np.int32)
    def add(x, y):
      return x + y

    @computations.federated_computation(
        computation_types.FederatedType(
            computation_types.SequenceType(np.int32), placements.CLIENTS))
    def foo(value):
      zero = intrinsics.federated_value(0, placements.CLIENTS)
      return factory.sequence_reduce(value, zero, add)

    self.assertEqual(foo.type_signature.compact_representation(),
                     '({int32*}@CLIENTS -> {int32}@CLIENTS)')


if __name__ == '__main__':
  context = federated_computation_context.FederatedComputationContext(
      context_stack_impl.context_stack)
  with context_stack_impl.context_stack.install(context):
    absltest.main()
