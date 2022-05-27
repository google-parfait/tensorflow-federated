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

import collections
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils


def _create_computation_add() -> computation_base.Computation:
  operand_type = computation_types.TensorType(tf.int32)
  computation_proto, _ = tensorflow_computation_factory.create_binary_operator(
      tf.add, operand_type, operand_type)
  return computation_impl.ConcreteComputation(computation_proto,
                                              context_stack_impl.context_stack)


def _create_computation_random() -> computation_base.Computation:
  computation_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
      lambda: tf.random.normal([]), None)
  return computation_impl.ConcreteComputation(computation_proto,
                                              context_stack_impl.context_stack)


def _create_computation_greater_than_10() -> computation_base.Computation:
  parameter_type = computation_types.TensorType(tf.int32)
  computation_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
      lambda x: x > 10, parameter_type)
  return computation_impl.ConcreteComputation(computation_proto,
                                              context_stack_impl.context_stack)


def _create_computation_greater_than_10_with_unused_parameter(
) -> computation_base.Computation:
  parameter_type = computation_types.StructType([
      computation_types.TensorType(tf.int32),
      computation_types.TensorType(tf.int32),
  ])
  computation_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
      lambda arg: arg[0] > 10, parameter_type)
  return computation_impl.ConcreteComputation(computation_proto,
                                              context_stack_impl.context_stack)


def _create_computation_reduce() -> computation_base.Computation:
  parameter_type = computation_types.SequenceType(tf.int32)
  computation_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
      lambda ds: ds.reduce(np.int32(0), lambda x, y: x + y), parameter_type)
  return computation_impl.ConcreteComputation(computation_proto,
                                              context_stack_impl.context_stack)


def _mock_data_of_type(type_spec, name='mock'):
  type_spec = computation_types.to_type(type_spec)
  return value_impl.Value(building_blocks.Data(name, type_spec))


class OutsideFederatedComputationTest(absltest.TestCase):

  def test_constant_to_value_raises_outside_symbol_binding_context(self):
    with context_stack_impl.context_stack.install(
        runtime_error_context.RuntimeErrorContext()):
      with self.assertRaises(context_base.ContextError):
        intrinsics.federated_value(2, placements.SERVER)

  def test_intrinsic_construction_raises_outside_symbol_binding_context(self):
    type_signature = computation_types.TensorType(tf.int32)
    computation_proto, _ = tensorflow_computation_factory.create_constant(
        2, type_signature)
    return_2 = computation_impl.ConcreteComputation(
        computation_proto, context_stack_impl.context_stack)

    with context_stack_impl.context_stack.install(
        runtime_error_context.RuntimeErrorContext()):
      with self.assertRaises(context_base.ContextError):
        intrinsics.federated_eval(return_2, placements.SERVER)


class IntrinsicTestBase(absltest.TestCase):

  def assert_value(self, value, type_string):
    self.assertIsInstance(value, value_impl.Value)
    self.assertEqual(value.type_signature.compact_representation(), type_string)


class FederatedBroadcastTest(IntrinsicTestBase):

  def test_accepts_server_all_equal_int(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    val = intrinsics.federated_broadcast(x)
    self.assert_value(val, 'int32@CLIENTS')

  def test_errors_on_client_int(self):
    with self.assertRaises(TypeError):
      x = _mock_data_of_type(
          computation_types.at_clients(tf.int32, all_equal=True))
      intrinsics.federated_broadcast(x)

  def test_federated_broadcast_with_non_federated_val(self):
    with self.assertRaises(TypeError):
      x = _mock_data_of_type(tf.int32)
      intrinsics.federated_broadcast(x)


class FederatedEvalTest(IntrinsicTestBase):

  def test_federated_eval_rand_on_clients(self):
    random = _create_computation_random()
    value = intrinsics.federated_eval(random, placements.CLIENTS)
    self.assert_value(value, '{float32}@CLIENTS')

  def test_federated_eval_rand_on_server(self):
    random = _create_computation_random()
    value = intrinsics.federated_eval(random, placements.SERVER)
    self.assert_value(value, 'float32@SERVER')


class FederatedMapTest(IntrinsicTestBase):

  def test_federated_map_with_client_all_equal_int(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=True))
    value = intrinsics.federated_map(computation, x)
    self.assert_value(value, '{bool}@CLIENTS')

  def test_federated_map_with_client_non_all_equal_int(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=False))
    value = intrinsics.federated_map(computation, x)
    self.assert_value(value, '{bool}@CLIENTS')

  def test_federated_map_with_server_int(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    value = intrinsics.federated_map(computation, x)
    self.assert_value(value, 'bool@SERVER')

  def test_federated_map_with_client_dataset_reduce(self):
    computation = _create_computation_reduce()
    ds = _mock_data_of_type(
        computation_types.at_clients(
            computation_types.SequenceType(tf.int32), all_equal=True))
    value = intrinsics.federated_map(computation, ds)
    self.assert_value(value, '{int32}@CLIENTS')

  def test_federated_map_injected_zip_with_server_int(self):
    computation = _create_computation_greater_than_10_with_unused_parameter()
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_server(tf.int32))
    value = intrinsics.federated_map(computation, [x, y])
    self.assert_value(value, 'bool@SERVER')

  def test_federated_map_injected_zip_fails_different_placements(self):
    computation = _create_computation_greater_than_10_with_unused_parameter()
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_clients(tf.int32))

    with self.assertRaisesRegex(
        TypeError,
        'The value to be mapped must be a FederatedType or implicitly '
        'convertible to a FederatedType.'):
      intrinsics.federated_map(computation, [x, y])

  def test_federated_map_with_non_federated_val(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(tf.int32)

    with self.assertRaises(TypeError):
      intrinsics.federated_map(computation, x)


class FederatedSecureModularSumTest(IntrinsicTestBase):

  def test_type_signature_with_int(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    modulus = 1
    result = intrinsics.federated_secure_modular_sum(value, modulus)
    self.assert_value(result, 'int32@SERVER')

  def test_type_signature_with_structure_of_ints(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    modulus = [8, [4, 2]]
    result = intrinsics.federated_secure_modular_sum(value, modulus)
    self.assert_value(result, '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_structure_of_ints_scalar_modulus(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    modulus = 8
    result = intrinsics.federated_secure_modular_sum(value, modulus)
    self.assert_value(result, '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_one_tensor_and_modulus(self):
    value = intrinsics.federated_value(
        np.ndarray(shape=(5, 37), dtype=np.int16), placements.CLIENTS)
    modulus = 2
    result = intrinsics.federated_secure_modular_sum(value, modulus)
    self.assert_value(result, 'int16[5,37]@SERVER')

  def test_type_signature_with_structure_of_tensors_and_moduli(self):
    np_array = np.ndarray(shape=(5, 37), dtype=np.int16)
    value = intrinsics.federated_value((np_array, np_array), placements.CLIENTS)
    modulus = (2, 2)
    result = intrinsics.federated_secure_modular_sum(value, modulus)
    self.assert_value(result, '<int16[5,37],int16[5,37]>@SERVER')

  def test_raises_type_error_with_value_float(self):
    value = intrinsics.federated_value(1.0, placements.CLIENTS)
    modulus = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_modular_sum(value, modulus)

  def test_raises_type_error_with_modulus_int_at_server(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    modulus = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_modular_sum(value, modulus)

  def test_raises_type_error_with_different_structures(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    modulus = [8, 4, 2]

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_modular_sum(value, modulus)


class FederatedSecureSumTest(IntrinsicTestBase):

  def test_type_signature_with_int(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    max_value = 1
    result = intrinsics.federated_secure_sum(value, max_value)
    self.assert_value(result, 'int32@SERVER')

  def test_type_signature_with_structure_of_ints(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    max_value = [8, [4, 2]]
    result = intrinsics.federated_secure_sum(value, max_value)
    self.assert_value(result, '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_structure_of_ints_scalar_max_value(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    max_value = 8
    result = intrinsics.federated_secure_sum(value, max_value)
    self.assert_value(result, '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_one_tensor_and_max_value(self):
    value = intrinsics.federated_value(
        np.ndarray(shape=(5, 37), dtype=np.int16), placements.CLIENTS)
    max_value = 2
    result = intrinsics.federated_secure_sum(value, max_value)
    self.assert_value(result, 'int16[5,37]@SERVER')

  def test_type_signature_with_structure_of_tensors_and_max_values(self):
    np_array = np.ndarray(shape=(5, 37), dtype=np.int16)
    value = intrinsics.federated_value((np_array, np_array), placements.CLIENTS)
    max_value = (2, 2)
    result = intrinsics.federated_secure_sum(value, max_value)
    self.assert_value(result, '<int16[5,37],int16[5,37]>@SERVER')

  def test_raises_type_error_with_value_float(self):
    value = intrinsics.federated_value(1.0, placements.CLIENTS)
    max_value = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, max_value)

  def test_raises_type_error_with_max_value_int_at_server(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    max_value = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, max_value)

  def test_raises_type_error_with_different_structures(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    max_value = [8, 4, 2]

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum(value, max_value)


class FederatedSecureSumBitwidthTest(IntrinsicTestBase):

  def test_type_signature_with_int(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    bitwidth = 8
    result = intrinsics.federated_secure_sum_bitwidth(value, bitwidth)
    self.assert_value(result, 'int32@SERVER')

  def test_type_signature_with_structure_of_ints(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    bitwidth = [8, [4, 2]]
    result = intrinsics.federated_secure_sum_bitwidth(value, bitwidth)
    self.assert_value(result, '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_structure_of_ints_scalar_bitwidth(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    bitwidth = 8
    result = intrinsics.federated_secure_sum_bitwidth(value, bitwidth)
    self.assert_value(result, '<int32,<int32,int32>>@SERVER')

  def test_type_signature_with_one_tensor_and_bitwidth(self):
    value = intrinsics.federated_value(
        np.ndarray(shape=(5, 37), dtype=np.int16), placements.CLIENTS)
    bitwidth = 2
    result = intrinsics.federated_secure_sum_bitwidth(value, bitwidth)
    self.assert_value(result, 'int16[5,37]@SERVER')

  def test_type_signature_with_structure_of_tensors_and_bitwidths(self):
    np_array = np.ndarray(shape=(5, 37), dtype=np.int16)
    value = intrinsics.federated_value((np_array, np_array), placements.CLIENTS)
    bitwidth = (2, 2)
    result = intrinsics.federated_secure_sum_bitwidth(value, bitwidth)
    self.assert_value(result, '<int16[5,37],int16[5,37]>@SERVER')

  def test_raises_type_error_with_value_float(self):
    value = intrinsics.federated_value(1.0, placements.CLIENTS)
    bitwidth = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum_bitwidth(value, bitwidth)

  def test_raises_type_error_with_bitwidth_int_at_server(self):
    value = intrinsics.federated_value(1, placements.CLIENTS)
    bitwidth = intrinsics.federated_value(1, placements.SERVER)

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum_bitwidth(value, bitwidth)

  def test_raises_type_error_with_different_structures(self):
    value = intrinsics.federated_value([1, [1, 1]], placements.CLIENTS)
    bitwidth = [8, 4, 2]

    with self.assertRaises(TypeError):
      intrinsics.federated_secure_sum_bitwidth(value, bitwidth)


class FederatedSelectTest(parameterized.TestCase, IntrinsicTestBase):

  def basic_federated_select_args(self):
    values = ['first', 'second', 'third']
    server_val = intrinsics.federated_value(values, placements.SERVER)

    max_key_py = len(values)
    max_key = intrinsics.federated_value(max_key_py, placements.SERVER)

    def get_three_random_keys_fn():
      return tf.random.uniform(
          shape=[3], minval=0, maxval=max_key_py, dtype=tf.int32)

    get_three_random_keys_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        get_three_random_keys_fn, None)
    get_three_random_keys = computation_impl.ConcreteComputation(
        get_three_random_keys_proto, context_stack_impl.context_stack)
    client_keys = intrinsics.federated_eval(get_three_random_keys,
                                            placements.CLIENTS)

    state_type = server_val.type_signature.member

    def _select_fn(arg):
      state = type_conversions.type_to_py_container(arg[0], state_type)
      key = arg[1]
      return tf.gather(state, key)

    select_fn_type = computation_types.StructType([state_type, tf.int32])
    select_fn_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _select_fn, select_fn_type)
    select_fn = computation_impl.ConcreteComputation(
        select_fn_proto, context_stack_impl.context_stack)

    return (client_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select),
  )
  def test_federated_select_succeeds(self, federated_select):
    result = federated_select(*self.basic_federated_select_args())
    self.assert_value(result, '{string*}@CLIENTS')

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_server_val_must_be_server_placed(
      self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del server_val

    bad_server_val_proto, _ = tensorflow_computation_factory.create_constant(
        tf.constant(['first', 'second', 'third']),
        computation_types.TensorType(dtype=tf.string, shape=[3]))
    bad_server_val = computation_impl.ConcreteComputation(
        bad_server_val_proto, context_stack_impl.context_stack)
    bad_server_val = bad_server_val()

    with self.assertRaises(TypeError):
      federated_select(client_keys, max_key, bad_server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_autozips_server_val(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del server_val, select_fn

    values = ['first', 'second', 'third']
    server_val_element = intrinsics.federated_value(values, placements.SERVER)
    server_val_dict = collections.OrderedDict(
        e1=server_val_element, e2=server_val_element)

    state_type = computation_types.StructType([
        ('e1', server_val_element.type_signature.member),
        ('e2', server_val_element.type_signature.member),
    ])

    def _select_fn_dict(arg):
      state = type_conversions.type_to_py_container(arg[0], state_type)
      key = arg[1]
      return (tf.gather(state['e1'], key), tf.gather(state['e2'], key))

    select_fn_dict_type = computation_types.StructType([state_type, tf.int32])
    select_fn_dict_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _select_fn_dict, select_fn_dict_type)
    select_fn_dict = computation_impl.ConcreteComputation(
        select_fn_dict_proto, context_stack_impl.context_stack)

    result = federated_select(client_keys, max_key, server_val_dict,
                              select_fn_dict)
    self.assert_value(result, '{<string,string>*}@CLIENTS')

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_must_be_client_placed(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del client_keys

    def get_three_random_keys_fn():
      return tf.random.uniform(shape=[3], minval=0, maxval=3, dtype=tf.int32)

    get_three_random_keys_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        get_three_random_keys_fn, None)
    get_three_random_keys = computation_impl.ConcreteComputation(
        get_three_random_keys_proto, context_stack_impl.context_stack)
    bad_client_keys = intrinsics.federated_eval(get_three_random_keys,
                                                placements.SERVER)

    with self.assertRaises(TypeError):
      federated_select(bad_client_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_must_be_int32(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del client_keys

    def get_three_random_keys_fn():
      return tf.random.uniform(shape=[3], minval=0, maxval=3, dtype=tf.int64)

    get_three_random_keys_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        get_three_random_keys_fn, None)
    get_three_random_keys = computation_impl.ConcreteComputation(
        get_three_random_keys_proto, context_stack_impl.context_stack)
    bad_client_keys = intrinsics.federated_eval(get_three_random_keys,
                                                placements.CLIENTS)

    with self.assertRaises(TypeError):
      federated_select(bad_client_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_cannot_be_scalar(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del client_keys

    bad_client_keys = intrinsics.federated_value(1, placements.CLIENTS)

    with self.assertRaises(TypeError):
      federated_select(bad_client_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_must_be_fixed_length(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())

    unshape_type = computation_types.TensorType(tf.int32, [None])
    unshape_proto, _ = tensorflow_computation_factory.create_identity(
        unshape_type)
    unshape = computation_impl.ConcreteComputation(
        unshape_proto, context_stack_impl.context_stack)
    bad_client_keys = intrinsics.federated_map(unshape, client_keys)

    with self.assertRaises(TypeError):
      federated_select(bad_client_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_fn_must_take_int32_keys(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del select_fn

    state_type = server_val.type_signature.member

    def _bad_select_fn(arg):
      state = type_conversions.type_to_py_container(arg[0], state_type)
      key = arg[1]
      return tf.gather(state, key)

    bad_select_fn_type = computation_types.StructType([state_type, tf.int64])
    bad_select_fn_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _bad_select_fn, bad_select_fn_type)
    bad_select_fn = computation_impl.ConcreteComputation(
        bad_select_fn_proto, context_stack_impl.context_stack)

    with self.assertRaises(TypeError):
      federated_select(client_keys, max_key, server_val, bad_select_fn)


class FederatedSumTest(IntrinsicTestBase):

  def test_federated_sum_with_client_int(self):
    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    val = intrinsics.federated_sum(x)
    self.assert_value(val, 'int32@SERVER')

  def test_federated_sum_with_client_string(self):
    x = _mock_data_of_type(computation_types.at_clients(tf.string))
    with self.assertRaises(TypeError):
      intrinsics.federated_sum(x)

  def test_federated_sum_with_server_int(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    with self.assertRaises(TypeError):
      intrinsics.federated_sum(x)


class FederatedZipTest(parameterized.TestCase, IntrinsicTestBase):

  def test_federated_zip_with_client_non_all_equal_int_and_bool(self):
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=False))
    y = _mock_data_of_type(
        computation_types.at_clients(tf.bool, all_equal=True))
    val = intrinsics.federated_zip([x, y])
    self.assert_value(val, '{<int32,bool>}@CLIENTS')

  def test_federated_zip_with_single_unnamed_int_client(self):
    x = _mock_data_of_type([computation_types.at_clients(tf.int32)])
    val = intrinsics.federated_zip(x)
    self.assert_value(val, '{<int32>}@CLIENTS')

  def test_federated_zip_with_single_unnamed_int_server(self):
    x = _mock_data_of_type([computation_types.at_server(tf.int32)])
    val = intrinsics.federated_zip(x)
    self.assert_value(val, '<int32>@SERVER')

  def test_federated_zip_with_single_named_bool_clients(self):
    x = _mock_data_of_type(
        computation_types.StructType([('a',
                                       computation_types.at_clients(tf.bool))]))
    val = intrinsics.federated_zip(x)
    self.assert_value(val, '{<a=bool>}@CLIENTS')

  def test_federated_zip_with_single_named_bool_server(self):
    x = _mock_data_of_type(
        computation_types.StructType([('a',
                                       computation_types.at_server(tf.bool))]))
    val = intrinsics.federated_zip(x)
    self.assert_value(val, '<a=bool>@SERVER')

  def test_federated_zip_with_names_client_non_all_equal_int_and_bool(self):
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=False))
    y = _mock_data_of_type(
        computation_types.at_clients(tf.bool, all_equal=True))
    val = intrinsics.federated_zip(collections.OrderedDict(x=x, y=y))
    self.assert_value(val, '{<x=int32,y=bool>}@CLIENTS')

  def test_federated_zip_with_client_all_equal_int_and_bool(self):
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=True))
    y = _mock_data_of_type(
        computation_types.at_clients(tf.bool, all_equal=True))
    val = intrinsics.federated_zip([x, y])
    self.assert_value(val, '{<int32,bool>}@CLIENTS')

  def test_federated_zip_with_names_client_all_equal_int_and_bool(self):
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=True))
    y = _mock_data_of_type(
        computation_types.at_clients(tf.bool, all_equal=True))
    val = intrinsics.federated_zip(collections.OrderedDict(x=x, y=y))
    self.assert_value(val, '{<x=int32,y=bool>}@CLIENTS')

  def test_federated_zip_with_server_int_and_bool(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_server(tf.bool))
    val = intrinsics.federated_zip([x, y])
    self.assert_value(val, '<int32,bool>@SERVER')

  def test_federated_zip_with_names_server_int_and_bool(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_server(tf.bool))
    val = intrinsics.federated_zip(collections.OrderedDict(x=x, y=y))
    self.assert_value(val, '<x=int32,y=bool>@SERVER')

  def test_federated_zip_error_different_placements(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_clients(tf.bool))
    with self.assertRaises(TypeError):
      intrinsics.federated_zip([x, y])

  @parameterized.named_parameters(('test_n_2', 2), ('test_n_3', 3),
                                  ('test_n_5', 5))
  def test_federated_zip_n_tuple(self, n):
    fed_type = computation_types.at_clients(tf.int32)
    x = _mock_data_of_type([fed_type] * n)
    val = intrinsics.federated_zip(x)
    self.assertIsInstance(val, value_impl.Value)
    expected = computation_types.at_clients([tf.int32] * n)
    type_test_utils.assert_types_identical(val.type_signature, expected)

  @parameterized.named_parameters(('test_n_2_int', 2, tf.int32),
                                  ('test_n_3_int', 3, tf.int32),
                                  ('test_n_5_int', 5, tf.int32),
                                  ('test_n_2_tuple', 2, [tf.int32, tf.int32]),
                                  ('test_n_3_tuple', 3, [tf.int32, tf.int32]),
                                  ('test_n_5_tuple', 5, [tf.int32, tf.int32]))
  def test_federated_zip_named_n_tuple(self, n, element_type):
    fed_type = computation_types.at_clients(element_type)
    initial_tuple_type = computation_types.to_type([fed_type] * n)
    initial_tuple = _mock_data_of_type(initial_tuple_type)

    naming_fn = str
    named_result = intrinsics.federated_zip(
        collections.OrderedDict(
            (naming_fn(i), initial_tuple[i]) for i in range(n)))
    self.assertIsInstance(named_result, value_impl.Value)
    expected = computation_types.at_clients(
        collections.OrderedDict((naming_fn(i), element_type) for i in range(n)))
    type_test_utils.assert_types_identical(named_result.type_signature,
                                           expected)

    naming_fn = lambda i: str(i) if i % 2 == 0 else None
    mixed_result = intrinsics.federated_zip(
        structure.Struct((naming_fn(i), initial_tuple[i]) for i in range(n)))
    self.assertIsInstance(mixed_result, value_impl.Value)
    expected = computation_types.at_clients(
        computation_types.StructType([
            (naming_fn(i), element_type) for i in range(n)
        ]))
    type_test_utils.assert_types_identical(mixed_result.type_signature,
                                           expected)

  @parameterized.named_parameters(
      ('n_1_m_1', 1, 1),
      ('n_1_m_2', 1, 2),
      ('n_1_m_3', 1, 3),
      ('n_2_m_1', 2, 1),
      ('n_2_m_2', 2, 2),
      ('n_2_m_3', 2, 3),
      ('n_3_m_1', 3, 1),
      ('n_3_m_2', 3, 2),
      ('n_3_m_3', 3, 3),
  )
  def test_federated_zip_n_tuple_mixed_args(self, n, m):
    tuple_fed_type = computation_types.at_clients((tf.int32, tf.int32))
    single_fed_type = computation_types.at_clients(tf.int32)
    tuple_repeat = lambda v, n: (v,) * n
    initial_tuple_type = computation_types.to_type(
        tuple_repeat(tuple_fed_type, n) + tuple_repeat(single_fed_type, m))
    final_fed_type = computation_types.at_clients(
        tuple_repeat((tf.int32, tf.int32), n) + tuple_repeat(tf.int32, m))

    x = _mock_data_of_type(initial_tuple_type)
    val = intrinsics.federated_zip(x)
    self.assertIsInstance(val, value_impl.Value)
    type_test_utils.assert_types_identical(val.type_signature, final_fed_type)


class FederatedMeanTest(IntrinsicTestBase):

  def test_federated_mean_with_client_float32_without_weight(self):
    x = _mock_data_of_type(computation_types.at_clients(tf.float32))
    val = intrinsics.federated_mean(x)
    self.assert_value(val, 'float32@SERVER')

  def test_federated_mean_with_all_equal_client_float32_without_weight(self):
    federated_all_equal_float = computation_types.FederatedType(
        tf.float32, placements.CLIENTS, all_equal=True)
    x = _mock_data_of_type(federated_all_equal_float)
    val = intrinsics.federated_mean(x)
    self.assert_value(val, 'float32@SERVER')

  def test_federated_mean_with_all_equal_client_float32_with_weight(self):
    federated_all_equal_float = computation_types.FederatedType(
        tf.float32, placements.CLIENTS, all_equal=True)
    x = _mock_data_of_type(federated_all_equal_float)
    val = intrinsics.federated_mean(x, x)
    self.assert_value(val, 'float32@SERVER')

  def test_federated_mean_with_client_tuple_with_int32_weight(self):
    values = _mock_data_of_type(
        computation_types.at_clients(
            collections.OrderedDict(
                x=tf.float64,
                y=tf.float64,
            )))
    weights = _mock_data_of_type(computation_types.at_clients(tf.int32))
    val = intrinsics.federated_mean(values, weights)
    self.assert_value(val, '<x=float64,y=float64>@SERVER')

  def test_federated_mean_with_client_int32_fails(self):
    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    with self.assertRaises(TypeError):
      intrinsics.federated_mean(x)

  def test_federated_mean_with_string_weight_fails(self):
    values = _mock_data_of_type(computation_types.at_clients(tf.float32))
    weights = _mock_data_of_type(computation_types.at_clients(tf.string))
    with self.assertRaises(TypeError):
      intrinsics.federated_mean(values, weights)


class FederatedAggregateTest(IntrinsicTestBase):

  # def _create_federated_aggregate_computaitons(self):
  #   return None

  def test_federated_aggregate_with_client_int(self):
    # The representation used during the aggregation process will be a named
    # tuple with 2 elements - the integer 'total' that represents the sum of
    # elements encountered, and the integer element 'count'.
    Accumulator = collections.namedtuple('Accumulator', 'total count')  # pylint: disable=invalid-name
    accumulator_type = computation_types.to_type(
        Accumulator(
            total=computation_types.TensorType(dtype=tf.int32),
            count=computation_types.TensorType(dtype=tf.int32)))

    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    zero = Accumulator(0, 0)

    # The operator to use during the first stage simply adds an element to the
    # total and updates the count.
    def _accumulate(arg):
      return Accumulator(arg[0].total + arg[1], arg[0].count + 1)

    accumulate_type = computation_types.StructType([accumulator_type, tf.int32])
    accumulate_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _accumulate, accumulate_type)
    accumulate = computation_impl.ConcreteComputation(
        accumulate_proto, context_stack_impl.context_stack)

    # The operator to use during the second stage simply adds total and count.
    def _merge(arg):
      return Accumulator(arg[0].total + arg[1].total,
                         arg[0].count + arg[1].count)

    merge_type = computation_types.StructType(
        [accumulator_type, accumulator_type])
    merge_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _merge, merge_type)
    merge = computation_impl.ConcreteComputation(
        merge_proto, context_stack_impl.context_stack)

    # The operator to use during the final stage simply computes the ratio.
    def _report(arg):
      return tf.cast(arg.total, tf.float32) / tf.cast(arg.count, tf.float32)

    report_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _report, accumulator_type)
    report = computation_impl.ConcreteComputation(
        report_proto, context_stack_impl.context_stack)

    value = intrinsics.federated_aggregate(x, zero, accumulate, merge, report)
    self.assert_value(value, 'float32@SERVER')

  def test_federated_aggregate_with_federated_zero_fails(self):
    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    zero = intrinsics.federated_value(0, placements.SERVER)
    accumulate = _create_computation_add()

    # The operator to use during the second stage simply adds total and count.
    merge = _create_computation_add()

    # The operator to use during the final stage simply computes the ratio.
    report_type = computation_types.TensorType(tf.int32)
    report_proto, _ = tensorflow_computation_factory.create_identity(
        report_type)
    report = computation_impl.ConcreteComputation(
        report_proto, context_stack_impl.context_stack)

    with self.assertRaisesRegex(
        TypeError, 'Expected `zero` to be assignable to type int32, '
        'but was of incompatible type int32@SERVER'):
      intrinsics.federated_aggregate(x, zero, accumulate, merge, report)

  def test_federated_aggregate_with_unknown_dimension(self):
    Accumulator = collections.namedtuple('Accumulator', ['samples'])  # pylint: disable=invalid-name
    accumulator_type = computation_types.to_type(
        Accumulator(
            samples=computation_types.TensorType(dtype=tf.int32, shape=[None])))

    x = _mock_data_of_type(computation_types.at_clients(tf.int32))

    def initialize_fn():
      return Accumulator(samples=tf.zeros(shape=[0], dtype=tf.int32))

    initialize_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        initialize_fn, None)
    initialize = computation_impl.ConcreteComputation(
        initialize_proto, context_stack_impl.context_stack)
    zero = initialize()

    # The operator to use during the first stage simply adds an element to the
    # tensor, increasing its size.
    def _accumulate(arg):
      return Accumulator(
          samples=tf.concat(
              [arg[0].samples, tf.expand_dims(arg[1], axis=0)], axis=0))

    accumulate_type = computation_types.StructType([accumulator_type, tf.int32])
    accumulate_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _accumulate, accumulate_type)
    accumulate = computation_impl.ConcreteComputation(
        accumulate_proto, context_stack_impl.context_stack)

    # The operator to use during the second stage simply adds total and count.
    def _merge(arg):
      return Accumulator(
          samples=tf.concat([arg[0].samples, arg[1].samples], axis=0))

    merge_type = computation_types.StructType(
        [accumulator_type, accumulator_type])
    merge_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _merge, merge_type)
    merge = computation_impl.ConcreteComputation(
        merge_proto, context_stack_impl.context_stack)

    # The operator to use during the final stage simply computes the ratio.
    report_proto, _ = tensorflow_computation_factory.create_identity(
        accumulator_type)
    report = computation_impl.ConcreteComputation(
        report_proto, context_stack_impl.context_stack)

    value = intrinsics.federated_aggregate(x, zero, accumulate, merge, report)
    self.assert_value(value, '<samples=int32[?]>@SERVER')

  def test_infers_accumulate_return_as_merge_arg_merge_return_as_report_arg(
      self):
    type_spec = computation_types.TensorType(dtype=tf.int64, shape=[None])
    x = _mock_data_of_type(computation_types.at_clients(tf.int64))

    def initialize_fn():
      return tf.constant([], dtype=tf.int64, shape=[0])

    initialize_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        initialize_fn, None)
    initialize = computation_impl.ConcreteComputation(
        initialize_proto, context_stack_impl.context_stack)
    zero = initialize()

    def _accumulate(arg):
      return tf.concat([arg[0], [arg[1]]], 0)

    accumulate_type = computation_types.StructType([type_spec, tf.int64])
    accumulate_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _accumulate, accumulate_type)
    accumulate = computation_impl.ConcreteComputation(
        accumulate_proto, context_stack_impl.context_stack)

    def _merge(arg):
      return tf.concat([arg[0], arg[1]], 0)

    merge_type = computation_types.StructType([type_spec, type_spec])
    merge_proto, _ = tensorflow_computation_factory.create_computation_for_py_fn(
        _merge, merge_type)
    merge = computation_impl.ConcreteComputation(
        merge_proto, context_stack_impl.context_stack)

    report_proto, _ = tensorflow_computation_factory.create_identity(type_spec)
    report = computation_impl.ConcreteComputation(
        report_proto, context_stack_impl.context_stack)

    value = intrinsics.federated_aggregate(x, zero, accumulate, merge, report)
    self.assert_value(value, 'int64[?]@SERVER')


class FederatedValueTest(IntrinsicTestBase):

  def test_federated_value_with_bool_on_clients(self):
    x = _mock_data_of_type(tf.bool)
    val = intrinsics.federated_value(x, placements.CLIENTS)
    self.assert_value(val, 'bool@CLIENTS')

  def test_federated_value_raw_np_scalar(self):
    floatv = np.float64(0)
    tff_float = intrinsics.federated_value(floatv, placements.SERVER)
    self.assert_value(tff_float, 'float64@SERVER')
    intv = np.int64(0)
    tff_int = intrinsics.federated_value(intv, placements.SERVER)
    self.assert_value(tff_int, 'int64@SERVER')

  def test_federated_value_raw_tf_scalar_variable(self):
    v = tf.Variable(initial_value=0., name='test_var')
    with self.assertRaisesRegex(
        TypeError, 'TensorFlow construct (.*) has been '
        'encountered in a federated context.'):
      intrinsics.federated_value(v, placements.SERVER)

  def test_federated_value_with_bool_on_server(self):
    x = _mock_data_of_type(tf.bool)
    val = intrinsics.federated_value(x, placements.SERVER)
    self.assert_value(val, 'bool@SERVER')

  def test_federated_value_raises_deprecation_warning(self):
    with warnings.catch_warnings(record=True) as warning:
      warnings.simplefilter('always')

      x = _mock_data_of_type(tf.bool)
      intrinsics.federated_value(x, placements.CLIENTS)
      self.assertLen(warning, 1)
      self.assertIsInstance(warning[0].category(), DeprecationWarning)


class SequenceMapTest(IntrinsicTestBase):

  def test_unplaced(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(computation_types.SequenceType(tf.int32))
    value = intrinsics.sequence_map(computation, x)
    self.assert_value(value, 'bool*')

  def test_server_placed(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(
        computation_types.at_server(computation_types.SequenceType(tf.int32)))
    value = intrinsics.sequence_map(computation, x)
    self.assert_value(value, 'bool*@SERVER')

  def test_clients_placed(self):
    computation = _create_computation_greater_than_10()
    x = _mock_data_of_type(
        computation_types.at_clients(computation_types.SequenceType(tf.int32)))
    value = intrinsics.sequence_map(computation, x)
    self.assert_value(value, '{bool*}@CLIENTS')


class SequenceReduceTest(IntrinsicTestBase):

  def test_with_non_federated_type(self):
    add = _create_computation_add()
    value = _mock_data_of_type(computation_types.SequenceType(np.int32))
    result = intrinsics.sequence_reduce(value, 0, add)
    self.assert_value(result, 'int32')

  def test_with_federated_type(self):
    add = _create_computation_add()
    value = _mock_data_of_type(
        computation_types.at_clients(computation_types.SequenceType(np.int32)))
    zero = intrinsics.federated_value(0, placements.CLIENTS)
    result = intrinsics.sequence_reduce(value, zero, add)
    self.assert_value(result, '{int32}@CLIENTS')


class SequenceSumTest(IntrinsicTestBase):

  def test_unplaced(self):
    x = _mock_data_of_type(computation_types.SequenceType(tf.int32))
    val = intrinsics.sequence_sum(x)
    self.assert_value(val, 'int32')

  def test_server_placed(self):
    x = _mock_data_of_type(
        computation_types.at_server(computation_types.SequenceType(tf.int32)))
    val = intrinsics.sequence_sum(x)
    self.assert_value(val, 'int32@SERVER')

  def test_clients_placed(self):
    x = _mock_data_of_type(
        computation_types.at_clients(computation_types.SequenceType(tf.int32)))
    val = intrinsics.sequence_sum(x)
    self.assert_value(val, '{int32}@CLIENTS')


if __name__ == '__main__':
  context = federated_computation_context.FederatedComputationContext(
      context_stack_impl.context_stack)
  with context_stack_impl.context_stack.install(context):
    absltest.main()
