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

from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_test_utils
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


_INT = computation_types.TensorType(np.int32)
_INT_CLIENTS = computation_types.FederatedType(_INT, placements.CLIENTS)
_INT_SERVER = computation_types.FederatedType(_INT, placements.SERVER)
_FLOAT = computation_types.TensorType(np.float32)
_FLOAT_CLIENTS = computation_types.FederatedType(_FLOAT, placements.CLIENTS)
_FLOAT_SERVER = computation_types.FederatedType(_FLOAT, placements.SERVER)
_STR = computation_types.TensorType(np.str_)
_STR_CLIENTS = computation_types.FederatedType(_STR, placements.CLIENTS)
_ARRAY_INT = computation_types.TensorType(np.int32, shape=[3])
_ARRAY_INT_CLIENTS = computation_types.FederatedType(
    _ARRAY_INT, placements.CLIENTS
)
_ARRAY_INT_SERVER = computation_types.FederatedType(
    _ARRAY_INT, placements.SERVER
)
_SEQUENCE_INT = computation_types.SequenceType(np.int32)
_SEQUENCE_INT_CLIENTS = computation_types.FederatedType(
    _SEQUENCE_INT, placements.CLIENTS
)
_SEQUENCE_INT_SERVER = computation_types.FederatedType(
    _SEQUENCE_INT, placements.SERVER
)
_SEQUENCE_FLOAT = computation_types.SequenceType(np.float32)
_SEQUENCE_FLOAT_CLIENTS = computation_types.FederatedType(
    _SEQUENCE_FLOAT, placements.CLIENTS
)
_SEQUENCE_FLOAT_SERVER = computation_types.FederatedType(
    _SEQUENCE_FLOAT, placements.SERVER
)
_STRUCT_INT = computation_types.StructWithPythonType(
    [np.int32, np.int32, np.int32], list
)
_STRUCT_INT_CLIENTS = computation_types.FederatedType(
    _STRUCT_INT, placements.CLIENTS
)
_STRUCT_INT_SERVER = computation_types.FederatedType(
    _STRUCT_INT, placements.SERVER
)
_STRUCT_FLOAT = computation_types.StructWithPythonType(
    [np.float32, np.float32, np.float32], list
)
_STRUCT_FLOAT_CLIENTS = computation_types.FederatedType(
    _STRUCT_FLOAT, placements.CLIENTS
)


def _create_context() -> (
    federated_computation_context.FederatedComputationContext
):
  return federated_computation_context.FederatedComputationContext(
      context_stack_impl.context_stack
  )


def _create_fake_fn(
    parameter_type: Optional[computation_types.TensorType],
    result_type: computation_types.TensorType,
) -> value_impl.Value:
  result = building_blocks.Reference('result', result_type)
  parameter_name = None if parameter_type is None else 'arg'
  fn = building_blocks.Lambda(parameter_name, parameter_type, result)
  return value_impl.Value(fn)


def _create_fake_value(type_spec: computation_types.Type) -> value_impl.Value:
  value = building_blocks.Reference('value', type_spec)
  return value_impl.Value(value)


class FederatedBroadcastTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int_server', _create_fake_value(_INT_SERVER)),
      ('sequence_server', _create_fake_value(_SEQUENCE_INT_SERVER)),
      ('struct_server', _create_fake_value(_STRUCT_INT_SERVER)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value):
    result = intrinsics.federated_broadcast(value)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.CLIENTS, all_equal=True
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int_unplaced', _create_fake_value(_INT)),
      ('int_clients', _create_fake_value(_INT_CLIENTS)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value):
    with self.assertRaises(TypeError):
      intrinsics.federated_broadcast(value)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_SERVER)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_broadcast(value)


class FederatedEvalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'int_and_clients',
          _create_fake_fn(None, _INT),
          placements.CLIENTS,
      ),
      (
          'int_and_server',
          _create_fake_fn(None, _INT),
          placements.SERVER,
      ),
      (
          'sequence_and_clieints',
          _create_fake_fn(None, _SEQUENCE_INT),
          placements.CLIENTS,
      ),
      (
          'struct_and_clients',
          _create_fake_fn(None, _STRUCT_INT),
          placements.CLIENTS,
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, fn, placement):
    result = intrinsics.federated_eval(fn, placement)

    expected_type = computation_types.FederatedType(
        fn.type_signature.result, placement
    )
    self.assertEqual(result.type_signature, expected_type)

  def test_raises_context_error_with_no_federated_context(self):
    fn = _create_fake_fn(None, _INT)
    placement = placements.CLIENTS

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_eval(fn, placement)


class FederatedMapTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'int_clients',
          _create_fake_fn(_INT, _FLOAT),
          _create_fake_value(_INT_CLIENTS),
          _FLOAT_CLIENTS,
      ),
      (
          'int_server',
          _create_fake_fn(_INT, _FLOAT),
          _create_fake_value(_INT_SERVER),
          _FLOAT_SERVER,
      ),
      (
          'sequence_clients',
          _create_fake_fn(_SEQUENCE_INT, _FLOAT),
          _create_fake_value(_SEQUENCE_INT_CLIENTS),
          _FLOAT_CLIENTS,
      ),
      (
          'struct_clients',
          _create_fake_fn(_STRUCT_INT, _FLOAT),
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _FLOAT_CLIENTS,
      ),
      (
          'struct_injected_zip',
          _create_fake_fn(_STRUCT_INT, _FLOAT),
          [
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_INT_CLIENTS),
          ],
          _FLOAT_CLIENTS,
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, fn, arg, expected_type):
    result = intrinsics.federated_map(fn, arg)

    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'int_unplaced',
          _create_fake_fn(_INT, _FLOAT),
          _create_fake_value(_INT),
      ),
      (
          'struct_different_placements',
          _create_fake_fn(_STRUCT_INT, _FLOAT),
          [
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_FLOAT_SERVER),
          ],
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, fn, arg):
    with self.assertRaises(TypeError):
      intrinsics.federated_map(fn, arg)

  def test_raises_context_error_with_no_federated_context(self):
    fn = _create_fake_fn(_INT, _FLOAT)
    arg = _create_fake_value(_INT_CLIENTS)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_map(fn, arg)


class FederatedSecureModularSumTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'value_int_clients_and_modulus_int',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_struct_int_clients_and_modulus_int',
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_struct_int_clients_and_modulus_struct',
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_value(_STRUCT_INT),
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
          _create_fake_value(_INT),
          _create_fake_value(_INT),
      ),
      (
          'value_float_clients',
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_int_server',
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_INT),
      ),
      (
          'modulus_int_clients',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT_CLIENTS),
      ),
      (
          'modulus_int_server',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
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
    value = _create_fake_value(_INT_CLIENTS)
    modulus = _create_fake_value(_INT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_secure_modular_sum(value, modulus)


class FederatedSecureSumTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'value_int_clients_and_max_input_int',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_struct_int_clients_and_max_input_int',
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_struct_int_clients_and_max_input_struct',
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_value(_STRUCT_INT),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, max_input):
    result = intrinsics.federated_secure_sum(value, max_input)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'value_int_unplaced',
          _create_fake_value(_INT),
          _create_fake_value(_INT),
      ),
      (
          'value_float_clients',
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_int_server',
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_INT),
      ),
      (
          'max_input_int_clients',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT_CLIENTS),
      ),
      (
          'max_input_int_server',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
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
  def test_raises_type_error(self, value, max_input):
    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, max_input)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_CLIENTS)
    max_input = _create_fake_value(_INT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_secure_sum(value, max_input)


class FederatedSecureSumBitwidthTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'value_int_clients_and_bitwidth_int',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_struct_int_clients_and_bitwidth_int',
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_struct_int_clients_and_bitwidth_struct',
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_value(_STRUCT_INT),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, bitwidth):
    result = intrinsics.federated_secure_sum_bitwidth(value, bitwidth)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'value_int_unplaced',
          _create_fake_value(_INT),
          _create_fake_value(_INT),
      ),
      (
          'value_float_clients',
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_value(_INT),
      ),
      (
          'value_int_server',
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_INT),
      ),
      (
          'bitwidth_int_clients',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT_CLIENTS),
      ),
      (
          'bitwidth_int_server',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
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
  def test_raises_type_error(self, value, bitwidth):
    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum_bitwidth(value, bitwidth)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_CLIENTS)
    bitwidth = _create_fake_value(_INT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_secure_sum_bitwidth(value, bitwidth)


class FederatedSelectTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'struct_int',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
          _SEQUENCE_FLOAT_CLIENTS,
      ),
      (
          'struct_injected_zip',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
          [
              _create_fake_value(_INT_SERVER),
              _create_fake_value(_INT_SERVER),
              _create_fake_value(_INT_SERVER),
          ],
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
          _SEQUENCE_FLOAT_CLIENTS,
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(
      self,
      client_keys,
      max_key,
      server_value,
      select_fn,
      expected_type,
  ):
    result = intrinsics.federated_select(
        client_keys, max_key, server_value, select_fn
    )

    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'client_keys_array_unplaced',
          _create_fake_value(_ARRAY_INT),
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'client_keys_array_server',
          _create_fake_value(_ARRAY_INT_SERVER),
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'max_key_int_unplaced',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'max_key_int_clients',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'max_key_float_server',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_FLOAT_SERVER),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'server_value_struct_unplaced',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_STRUCT_INT),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'server_value_struct_clients',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_STRUCT_INT_CLIENTS),
          _create_fake_fn([_STRUCT_INT, _INT], _FLOAT),
      ),
      (
          'select_fn_second_parameter_float',
          _create_fake_value(_ARRAY_INT_CLIENTS),
          _create_fake_value(_INT_SERVER),
          _create_fake_value(_STRUCT_INT_SERVER),
          _create_fake_fn([_STRUCT_INT, _FLOAT], _FLOAT),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(
      self, client_keys, max_key, server_value, select_fn
  ):
    with self.assertRaises(TypeError):
      intrinsics.federated_select(client_keys, max_key, server_value, select_fn)

  def test_raises_context_error_with_no_federated_context(self):
    client_keys = _create_fake_value(_ARRAY_INT_CLIENTS)
    max_key = _create_fake_value(_INT_SERVER)
    server_value = _create_fake_value(_STRUCT_INT_SERVER)
    select_fn = _create_fake_fn([_STRUCT_INT, _INT], _FLOAT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_select(client_keys, max_key, server_value, select_fn)


class FederatedSumTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int_clients', _create_fake_value(_INT_CLIENTS)),
      ('struct_clients', _create_fake_value(_STRUCT_INT_CLIENTS)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value):
    result = intrinsics.federated_sum(value)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int_unplaced', _create_fake_value(_INT)),
      ('str_clients', _create_fake_value(_STR_CLIENTS)),
      ('int_server', _create_fake_value(_INT_SERVER)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value):
    with self.assertRaises(TypeError):
      intrinsics.federated_sum(value)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_CLIENTS)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_sum(value)


class FederatedZipTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'struct_clients',
          [
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_INT_CLIENTS),
          ],
          _STRUCT_INT_CLIENTS,
      ),
      (
          'struct_server',
          [
              _create_fake_value(_INT_SERVER),
              _create_fake_value(_INT_SERVER),
              _create_fake_value(_INT_SERVER),
          ],
          _STRUCT_INT_SERVER,
      ),
      (
          'struct_different_dtypes',
          [
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_FLOAT_CLIENTS),
              _create_fake_value(_STR_CLIENTS),
          ],
          computation_types.FederatedType(
              [np.int32, np.float32, np.str_], placements.CLIENTS
          ),
      ),
      (
          'struct_one_element',
          [
              _create_fake_value(_INT_CLIENTS),
          ],
          computation_types.FederatedType([np.int32], placements.CLIENTS),
      ),
      (
          'struct_nested',
          [
              _create_fake_value(_INT_CLIENTS),
              [
                  _create_fake_value(_INT_CLIENTS),
                  _create_fake_value(_INT_CLIENTS),
              ],
          ],
          computation_types.FederatedType(
              [np.int32, [np.int32, np.int32]], placements.CLIENTS
          ),
      ),
      (
          'struct_named',
          {
              'a': _create_fake_value(_INT_CLIENTS),
              'b': _create_fake_value(_INT_CLIENTS),
              'c': _create_fake_value(_INT_CLIENTS),
          },
          computation_types.FederatedType(
              {'a': np.int32, 'b': np.int32, 'c': np.int32}, placements.CLIENTS
          ),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, expected_type):
    result = intrinsics.federated_zip(value)

    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int_unplaced', _create_fake_value(_INT)),
      ('int_clients', _create_fake_value(_INT_CLIENTS)),
      ('int_server', _create_fake_value(_INT_SERVER)),
      (
          'struct_different_placements',
          [
              _create_fake_value(_INT_CLIENTS),
              _create_fake_value(_INT_SERVER),
          ],
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value):
    with self.assertRaises(TypeError):
      intrinsics.federated_zip(value)

  def test_raises_context_error_with_no_federated_context(self):
    value = [
        _create_fake_value(_INT_CLIENTS),
        _create_fake_value(_INT_CLIENTS),
        _create_fake_value(_INT_CLIENTS),
    ]

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_zip(value)


class FederatedMeanTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'value_float_clients_and_weight_none',
          _create_fake_value(_FLOAT_CLIENTS),
          None,
      ),
      (
          'value_float_clients_and_weight_float_clients',
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_value(_FLOAT_CLIENTS),
      ),
      (
          'value_struct_int_clients_and_weight_none',
          _create_fake_value(_STRUCT_FLOAT_CLIENTS),
          None,
      ),
      (
          'value_struct_int_clients_and_weight_float_clients',
          _create_fake_value(_STRUCT_FLOAT_CLIENTS),
          _create_fake_value(_FLOAT_CLIENTS),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, weight):
    result = intrinsics.federated_mean(value, weight)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'value_int_clients',
          _create_fake_value(_INT_CLIENTS),
          None,
      ),
      (
          'value_float_server',
          _create_fake_value(_FLOAT_SERVER),
          None,
      ),
      (
          'weight_str_clients',
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_value(_STR_CLIENTS),
      ),
      (
          'weight_float_server',
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_value(_FLOAT_SERVER),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value, weight):
    with self.assertRaises(TypeError):
      intrinsics.federated_mean(value, weight)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_FLOAT_CLIENTS)
    weight = None

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_mean(value, weight)


class FederatedMinTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int_clients', _create_fake_value(_INT_CLIENTS)),
      ('struct_clients', _create_fake_value(_STRUCT_INT_CLIENTS)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value):
    result = intrinsics.federated_min(value)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int_unplaced', _create_fake_value(_INT)),
      ('int_server', _create_fake_value(_INT_SERVER)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value):
    with self.assertRaises(TypeError):
      intrinsics.federated_min(value)

  @parameterized.named_parameters(
      ('str_clients', _create_fake_value(_STR_CLIENTS)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_value_error(self, value):
    with self.assertRaises(ValueError):
      intrinsics.federated_min(value)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_CLIENTS)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_min(value)


class FederatedMaxTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('int_clients', _create_fake_value(_INT_CLIENTS)),
      ('struct_clients', _create_fake_value(_STRUCT_INT_CLIENTS)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value):
    result = intrinsics.federated_max(value)

    expected_type = computation_types.FederatedType(
        value.type_signature.member, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      ('int_unplaced', _create_fake_value(_INT)),
      ('int_server', _create_fake_value(_INT_SERVER)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value):
    with self.assertRaises(TypeError):
      intrinsics.federated_max(value)

  @parameterized.named_parameters(
      ('str_clients', _create_fake_value(_STR_CLIENTS)),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_value_error(self, value):
    with self.assertRaises(ValueError):
      intrinsics.federated_max(value)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_CLIENTS)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_min(value)


class FederatedAggregateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'int_clients',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_FLOAT),
          _create_fake_fn([_FLOAT, _INT], _FLOAT),
          _create_fake_fn([_FLOAT, _FLOAT], _FLOAT),
          _create_fake_fn(_FLOAT, _STR),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, zero, accumulate, merge, report):
    result = intrinsics.federated_aggregate(
        value, zero, accumulate, merge, report
    )

    expected_type = computation_types.FederatedType(
        report.type_signature.result, placements.SERVER
    )
    self.assertEqual(result.type_signature, expected_type)

  @parameterized.named_parameters(
      (
          'zero_mismatched_type',
          _create_fake_value(_INT_CLIENTS),
          _create_fake_value(_INT),
          _create_fake_fn([_FLOAT, _INT], _FLOAT),
          _create_fake_fn([_FLOAT, _FLOAT], _FLOAT),
          _create_fake_fn(_FLOAT, _STR),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_raises_type_error(self, value, zero, accumulate, merge, report):
    with self.assertRaises(TypeError):
      intrinsics.federated_aggregate(value, zero, accumulate, merge, report)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT_CLIENTS)
    zero = _create_fake_value(_FLOAT)
    accumulate = _create_fake_fn([_FLOAT, _INT], _FLOAT)
    merge = _create_fake_fn([_FLOAT, _FLOAT], _FLOAT)
    report = _create_fake_fn(_FLOAT, _STR)

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_aggregate(value, zero, accumulate, merge, report)


class FederatedValueTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'int_and_clients',
          _create_fake_value(_INT),
          placements.CLIENTS,
      ),
      (
          'int_and_server',
          _create_fake_value(_INT),
          placements.SERVER,
      ),
      (
          'sequence_and_clients',
          _create_fake_value(_SEQUENCE_INT),
          placements.CLIENTS,
      ),
      (
          'struct_and_clients',
          _create_fake_value(_STRUCT_INT),
          placements.CLIENTS,
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, placement):
    result = intrinsics.federated_value(value, placement)

    expected_type = computation_types.FederatedType(
        value.type_signature, placement, all_equal=True
    )
    self.assertEqual(result.type_signature, expected_type)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_INT)
    placement = placements.CLIENTS

    with self.assertRaises(context_base.ContextError):
      intrinsics.federated_value(value, placement)


class SequenceMapTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'sequence_unplaced',
          _create_fake_fn(_INT, _FLOAT),
          _create_fake_value(_SEQUENCE_INT),
          _SEQUENCE_FLOAT,
      ),
      (
          'sequence_clients',
          _create_fake_fn(_INT, _FLOAT),
          _create_fake_value(_SEQUENCE_INT_CLIENTS),
          _SEQUENCE_FLOAT_CLIENTS,
      ),
      (
          'sequence_server',
          _create_fake_fn(_INT, _FLOAT),
          _create_fake_value(_SEQUENCE_INT_SERVER),
          _SEQUENCE_FLOAT_SERVER,
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, fn, arg, expected_type):
    result = intrinsics.sequence_map(fn, arg)

    self.assertEqual(result.type_signature, expected_type)

  def test_raises_context_error_with_no_federated_context(self):
    fn = _create_fake_fn(_INT, _FLOAT)
    arg = _create_fake_value(_SEQUENCE_INT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.sequence_map(fn, arg)


class SequenceReduceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'sequence_unplaced',
          _create_fake_value(_SEQUENCE_INT),
          _create_fake_value(_FLOAT),
          _create_fake_fn([_FLOAT, _INT], _FLOAT),
      ),
      (
          'sequence_clients',
          _create_fake_value(_SEQUENCE_INT_CLIENTS),
          _create_fake_value(_FLOAT_CLIENTS),
          _create_fake_fn([_FLOAT, _INT], _FLOAT),
      ),
      (
          'sequence_server',
          _create_fake_value(_SEQUENCE_INT_SERVER),
          _create_fake_value(_FLOAT_SERVER),
          _create_fake_fn([_FLOAT, _INT], _FLOAT),
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, zero, op):
    result = intrinsics.sequence_reduce(value, zero, op)

    expected_type = zero.type_signature
    self.assertEqual(result.type_signature, expected_type)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_SEQUENCE_INT)
    zero = _create_fake_value(_FLOAT)
    op = _create_fake_fn([_FLOAT, _INT], _FLOAT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.sequence_reduce(value, zero, op)


class SequenceSumTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'sequence_unplaced',
          _create_fake_value(_SEQUENCE_INT),
          _INT,
      ),
      (
          'sequence_clients',
          _create_fake_value(_SEQUENCE_INT_CLIENTS),
          _INT_CLIENTS,
      ),
      (
          'sequence_server',
          _create_fake_value(_SEQUENCE_INT_SERVER),
          _INT_SERVER,
      ),
  )
  @context_stack_test_utils.with_context(_create_context)
  def test_returns_result(self, value, expected_type):
    result = intrinsics.sequence_sum(value)

    self.assertEqual(result.type_signature, expected_type)

  def test_raises_context_error_with_no_federated_context(self):
    value = _create_fake_value(_SEQUENCE_INT)

    with self.assertRaises(context_base.ContextError):
      intrinsics.sequence_sum(value)


if __name__ == '__main__':
  absltest.main()
