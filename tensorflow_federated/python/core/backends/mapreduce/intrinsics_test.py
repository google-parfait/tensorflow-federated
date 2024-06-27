# Copyright 2018 Google LLC
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


import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.core.backends.mapreduce import intrinsics
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


def _create_context() -> (
    federated_computation_context.FederatedComputationContext
):
  return federated_computation_context.FederatedComputationContext(
      context_stack_impl.context_stack
  )


def _create_fake_value(type_spec: computation_types.Type) -> value_impl.Value:
  value = building_blocks.Reference('value', type_spec)
  return value_impl.Value(value)


class IntrinsicDefsTest(absltest.TestCase):

  def test_type_signature_strings(self):
    name = 'FEDERATED_SECURE_MODULAR_SUM'
    type_str = '(<{V}@CLIENTS,M> -> V@SERVER)'
    intrinsic = getattr(intrinsics, name)
    self.assertEqual(
        intrinsic.type_signature.compact_representation(), type_str
    )


class CreateFederatedSecureModularSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    modulus = mock.create_autospec(
        building_blocks.CompiledComputation, spec_set=True, instance=True
    )

    with self.assertRaises(TypeError):
      intrinsics.create_federated_secure_modular_sum(None, modulus)

  def test_raises_type_error_with_none_modulus(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )

    with self.assertRaises(TypeError):
      intrinsics.create_federated_secure_modular_sum(value, None)

  def test_returns_federated_sum(self):
    value = building_block_factory.create_federated_value(
        building_blocks.Literal(1, computation_types.TensorType(np.int32)),
        placement=placements.CLIENTS,
    )
    modulus_type = computation_types.TensorType(np.int32)
    modulus = building_blocks.Literal(2, modulus_type)
    comp = intrinsics.create_federated_secure_modular_sum(value, modulus)
    # Regex replaces compiled computations such as `comp#b03f` to ensure a
    # consistent output.
    golden.check_string(
        'federated_secure_modular_sum.expected',
        re.sub(
            r'comp\#\w*', 'some_compiled_comp', comp.formatted_representation()
        ),
    )
    self.assertEqual(
        comp.type_signature.compact_representation(), 'int32@SERVER'
    )


class FederatedSecureModularSumTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'value_int_clients_and_modulus_int',
          _create_fake_value(
              computation_types.FederatedType(np.int32, placements.CLIENTS)
          ),
          _create_fake_value(computation_types.TensorType(np.int32)),
      ),
      (
          'value_struct_int_clients_and_modulus_int',
          _create_fake_value(
              computation_types.FederatedType(
                  [np.int32, np.int32, np.int32], placements.CLIENTS
              )
          ),
          _create_fake_value(computation_types.TensorType(np.int32)),
      ),
      (
          'value_struct_int_clients_and_modulus_struct',
          _create_fake_value(
              computation_types.FederatedType(
                  [np.int32, np.int32, np.int32], placements.CLIENTS
              )
          ),
          _create_fake_value(
              computation_types.StructWithPythonType(
                  [np.int32, np.int32, np.int32], list
              )
          ),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, modulus):
    result = intrinsics.federated_secure_modular_sum(value, modulus)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'value_int_unplaced',
          _create_fake_value(computation_types.TensorType(np.int32)),
          _create_fake_value(computation_types.TensorType(np.int32)),
      ),
      (
          'value_float_clients',
          _create_fake_value(
              computation_types.FederatedType(np.float32, placements.CLIENTS)
          ),
          _create_fake_value(computation_types.TensorType(np.int32)),
      ),
      (
          'value_int_server',
          _create_fake_value(
              computation_types.FederatedType(np.int32, placements.SERVER)
          ),
          _create_fake_value(computation_types.TensorType(np.int32)),
      ),
      (
          'modulus_int_clients',
          _create_fake_value(
              computation_types.FederatedType(np.int32, placements.CLIENTS)
          ),
          _create_fake_value(
              computation_types.FederatedType(np.int32, placements.CLIENTS)
          ),
      ),
      (
          'modulus_int_server',
          _create_fake_value(
              computation_types.FederatedType(np.int32, placements.CLIENTS)
          ),
          _create_fake_value(
              computation_types.FederatedType(np.int32, placements.SERVER)
          ),
      ),
      (
          'mismatched_structures',
          _create_fake_value(
              computation_types.FederatedType(
                  [np.int32] * 2, placements.CLIENTS
              ),
          ),
          _create_fake_value(computation_types.StructType([np.int32] * 3)),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value, modulus):
    with self.assertRaises(TypeError):
      intrinsics.federated_secure_modular_sum(value, modulus)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(
        computation_types.FederatedType(np.int32, placements.CLIENTS)
    )
    modulus = _create_fake_value(computation_types.TensorType(np.int32))

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_secure_modular_sum(value, modulus)


if __name__ == '__main__':
  absltest.main()
