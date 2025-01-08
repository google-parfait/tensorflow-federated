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
import federated_language
import numpy as np

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.core.backends.mapreduce import intrinsics


def _create_context() -> (
    federated_language.framework.FederatedComputationContext
):
  return federated_language.framework.FederatedComputationContext(
      federated_language.framework.get_context_stack()
  )


def _create_fake_value(
    type_spec: federated_language.Type,
) -> federated_language.Value:
  value = federated_language.framework.Reference('value', type_spec)
  return federated_language.Value(value)


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
        federated_language.framework.CompiledComputation,
        spec_set=True,
        instance=True,
    )

    with self.assertRaises(TypeError):
      intrinsics.create_federated_secure_modular_sum(None, modulus)

  def test_raises_type_error_with_none_modulus(self):
    value = federated_language.framework.create_federated_value(
        federated_language.framework.Literal(
            1, federated_language.TensorType(np.int32)
        ),
        placement=federated_language.CLIENTS,
    )

    with self.assertRaises(TypeError):
      intrinsics.create_federated_secure_modular_sum(value, None)

  def test_returns_federated_sum(self):
    value = federated_language.framework.create_federated_value(
        federated_language.framework.Literal(
            1, federated_language.TensorType(np.int32)
        ),
        placement=federated_language.CLIENTS,
    )
    modulus_type = federated_language.TensorType(np.int32)
    modulus = federated_language.framework.Literal(2, modulus_type)
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
              federated_language.FederatedType(
                  np.int32, federated_language.CLIENTS
              )
          ),
          _create_fake_value(federated_language.TensorType(np.int32)),
      ),
      (
          'value_struct_int_clients_and_modulus_int',
          _create_fake_value(
              federated_language.FederatedType(
                  [np.int32, np.int32, np.int32], federated_language.CLIENTS
              )
          ),
          _create_fake_value(federated_language.TensorType(np.int32)),
      ),
      (
          'value_struct_int_clients_and_modulus_struct',
          _create_fake_value(
              federated_language.FederatedType(
                  [np.int32, np.int32, np.int32], federated_language.CLIENTS
              )
          ),
          _create_fake_value(
              federated_language.StructWithPythonType(
                  [np.int32, np.int32, np.int32], list
              )
          ),
      ),
  )
  @federated_language.framework.with_context(_create_context)
  def test_returns_result(self, value, modulus):
    result = intrinsics.federated_secure_modular_sum(value, modulus)

    expected_type = federated_language.FederatedType(
        value.type_signature.member, federated_language.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'value_int_unplaced',
          _create_fake_value(federated_language.TensorType(np.int32)),
          _create_fake_value(federated_language.TensorType(np.int32)),
      ),
      (
          'value_float_clients',
          _create_fake_value(
              federated_language.FederatedType(
                  np.float32, federated_language.CLIENTS
              )
          ),
          _create_fake_value(federated_language.TensorType(np.int32)),
      ),
      (
          'value_int_server',
          _create_fake_value(
              federated_language.FederatedType(
                  np.int32, federated_language.SERVER
              )
          ),
          _create_fake_value(federated_language.TensorType(np.int32)),
      ),
      (
          'modulus_int_clients',
          _create_fake_value(
              federated_language.FederatedType(
                  np.int32, federated_language.CLIENTS
              )
          ),
          _create_fake_value(
              federated_language.FederatedType(
                  np.int32, federated_language.CLIENTS
              )
          ),
      ),
      (
          'modulus_int_server',
          _create_fake_value(
              federated_language.FederatedType(
                  np.int32, federated_language.CLIENTS
              )
          ),
          _create_fake_value(
              federated_language.FederatedType(
                  np.int32, federated_language.SERVER
              )
          ),
      ),
      (
          'mismatched_structures',
          _create_fake_value(
              federated_language.FederatedType(
                  [np.int32] * 2, federated_language.CLIENTS
              ),
          ),
          _create_fake_value(federated_language.StructType([np.int32] * 3)),
      ),
  )
  @federated_language.framework.with_context(_create_context)
  def test_raises_type_error(self, value, modulus):
    with self.assertRaises(TypeError):
      intrinsics.federated_secure_modular_sum(value, modulus)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    modulus = _create_fake_value(federated_language.TensorType(np.int32))

    with self.assertRaises(federated_language.framework.ContextError):
      intrinsics.federated_secure_modular_sum(value, modulus)


if __name__ == '__main__':
  absltest.main()
