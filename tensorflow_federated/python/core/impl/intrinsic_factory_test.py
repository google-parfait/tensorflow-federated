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
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.impl import intrinsic_factory
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.types import placement_literals


class FederatedReduceTest(absltest.TestCase):

  def test_allows_assignable_but_not_equal_zero_and_reduction_types(self):
    factory = intrinsic_factory.IntrinsicFactory(
        context_stack_impl.context_stack)

    element_type = tf.string
    zero_type = computation_types.TensorType(tf.string, [1])
    reduced_type = computation_types.TensorType(tf.string, [None])

    @computations.tf_computation(reduced_type, element_type)
    @computations.check_returns_type(reduced_type)
    def append(accumulator, element):
      return tf.concat([accumulator, [element]], 0)

    @computations.tf_computation
    @computations.check_returns_type(zero_type)
    def zero():
      return tf.convert_to_tensor(['The beginning'])

    @computations.federated_computation(
        computation_types.at_clients(element_type))
    @computations.check_returns_type(computation_types.at_server(reduced_type))
    def collect(client_values):
      return factory.federated_reduce(client_values, zero(), append)

    self.assertEqual(collect.type_signature.compact_representation(),
                     '({string}@CLIENTS -> string[?]@SERVER)')


class FederatedSecureSumTest(absltest.TestCase):

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

  def test_type_signature_with_structure_of_ints_scalar_bitwidth(self):
    value = intrinsics.federated_value([1, [1, 1]], placement_literals.CLIENTS)
    bitwidth = 8

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
            computation_types.SequenceType(np.int32),
            placement_literals.CLIENTS))
    def foo(value):
      zero = intrinsics.federated_value(0, placement_literals.CLIENTS)
      return factory.sequence_reduce(value, zero, add)

    self.assertEqual(foo.type_signature.compact_representation(),
                     '({int32*}@CLIENTS -> {int32}@CLIENTS)')


if __name__ == '__main__':
  context = federated_computation_context.FederatedComputationContext(
      context_stack_impl.context_stack)
  with context_stack_impl.context_stack.install(context):
    absltest.main()
