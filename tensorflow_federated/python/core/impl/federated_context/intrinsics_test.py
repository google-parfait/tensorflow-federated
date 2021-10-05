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
import itertools
import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context
from tensorflow_federated.python.core.impl.federated_context import federated_computation_context
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.federated_context import value_impl
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class OutsideFederatedComputationTest(test_case.TestCase):

  def test_constant_to_value_raises_outside_symbol_binding_context(self):
    with context_stack_impl.context_stack.install(
        runtime_error_context.RuntimeErrorContext()):
      with self.assertRaises(context_base.ContextError):
        intrinsics.federated_value(2, placements.SERVER)

  def test_intrinsic_construction_raises_outside_symbol_binding_context(self):

    @computations.tf_computation
    def return_2():
      return 2

    with context_stack_impl.context_stack.install(
        runtime_error_context.RuntimeErrorContext()):
      with self.assertRaises(context_base.ContextError):
        intrinsics.federated_eval(return_2, placements.SERVER)


def _mock_data_of_type(type_spec, name='mock'):
  type_spec = computation_types.to_type(type_spec)
  return value_impl.Value(building_blocks.Data(name, type_spec))


class IntrinsicTestBase(test_case.TestCase):

  def assert_value(self, value, type_string):
    self.assertIsInstance(value, value_impl.Value)
    self.assert_type_string(value.type_signature, type_string)


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

    @computations.tf_computation
    def rand():
      return tf.random.normal([])

    val = intrinsics.federated_eval(rand, placements.CLIENTS)
    self.assert_value(val, '{float32}@CLIENTS')

  def test_federated_eval_rand_on_server(self):

    @computations.tf_computation
    def rand():
      return tf.random.normal([])

    val = intrinsics.federated_eval(rand, placements.SERVER)
    self.assert_value(val, 'float32@SERVER')


class FederatedMapTest(IntrinsicTestBase):

  def test_federated_map_with_client_all_equal_int(self):
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=True))
    val = intrinsics.federated_map(
        computations.tf_computation(lambda x: x > 10), x)
    self.assert_value(val, '{bool}@CLIENTS')

  def test_federated_map_with_client_non_all_equal_int(self):
    x = _mock_data_of_type(
        computation_types.at_clients(tf.int32, all_equal=False))
    val = intrinsics.federated_map(
        computations.tf_computation(lambda x: x > 10), x)
    self.assert_value(val, '{bool}@CLIENTS')

  def test_federated_map_with_server_int(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    val = intrinsics.federated_map(
        computations.tf_computation(lambda x: x > 10), x)
    self.assert_value(val, 'bool@SERVER')

  def test_federated_map_with_client_dataset_reduce(self):
    ds = _mock_data_of_type(
        computation_types.at_clients(
            computation_types.SequenceType(tf.int32), all_equal=True))
    val = intrinsics.federated_map(
        computations.tf_computation(
            lambda ds: ds.reduce(np.int32(0), lambda x, y: x + y)), ds)
    self.assert_value(val, '{int32}@CLIENTS')

  def test_federated_map_injected_zip_with_server_int(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_server(tf.int32))
    val = intrinsics.federated_map(
        computations.tf_computation(lambda x, y: x > 10), [x, y])
    self.assert_value(val, 'bool@SERVER')

  def test_federated_map_injected_zip_fails_different_placements(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_clients(tf.int32))

    with self.assertRaisesRegex(
        TypeError,
        'The value to be mapped must be a FederatedType or implicitly '
        'convertible to a FederatedType.'):
      intrinsics.federated_map(
          computations.tf_computation(lambda x, y: x > 10), [x, y])

  def test_federated_map_with_non_federated_val(self):
    x = _mock_data_of_type(tf.int32)
    with self.assertRaises(TypeError):
      intrinsics.federated_map(computations.tf_computation(lambda x: x > 10), x)


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

    @computations.tf_computation
    def get_three_random_keys():
      return tf.random.uniform(
          shape=[3], minval=0, maxval=max_key_py, dtype=tf.int32)

    client_keys = intrinsics.federated_eval(get_three_random_keys,
                                            placements.CLIENTS)

    @computations.tf_computation
    def select_fn(state, key):
      return tf.gather(state, key)

    return (client_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_succeeds(self, federated_select):
    result = federated_select(*self.basic_federated_select_args())
    self.assert_value(result, '{string*}@CLIENTS')

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_must_be_client_placed(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del client_keys

    @computations.tf_computation
    def get_three_random_keys():
      return tf.random.uniform(shape=[3], minval=0, maxval=3, dtype=tf.int32)

    bad_keys = intrinsics.federated_eval(get_three_random_keys,
                                         placements.SERVER)
    with self.assertRaises(TypeError):
      federated_select(bad_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_must_be_int32(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del client_keys

    @computations.tf_computation
    def get_three_random_int64_keys():
      return tf.random.uniform(shape=[3], minval=0, maxval=3, dtype=tf.int64)

    bad_keys = intrinsics.federated_eval(get_three_random_int64_keys,
                                         placements.CLIENTS)
    with self.assertRaises(TypeError):
      federated_select(bad_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_cannot_be_scalar(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del client_keys
    bad_keys = intrinsics.federated_value(1, placements.CLIENTS)
    with self.assertRaises(TypeError):
      federated_select(bad_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_keys_must_be_fixed_length(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())

    @computations.tf_computation(computation_types.TensorType(tf.int32, [None]))
    def unshape(x):
      return x

    bad_keys = intrinsics.federated_map(unshape, client_keys)
    with self.assertRaises(TypeError):
      federated_select(bad_keys, max_key, server_val, select_fn)

  @parameterized.named_parameters(
      ('non_secure', intrinsics.federated_select),
      ('secure', intrinsics.federated_secure_select))
  def test_federated_select_fn_must_take_int32_keys(self, federated_select):
    client_keys, max_key, server_val, select_fn = (
        self.basic_federated_select_args())
    del select_fn

    @computations.tf_computation(server_val.type_signature.member, tf.int64)
    def bad_select_fn(server_value, key):
      return tf.gather(server_value, key)

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
    val = intrinsics.federated_zip({'x': x, 'y': y})
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
    val = intrinsics.federated_zip({'x': x, 'y': y})
    self.assert_value(val, '{<x=int32,y=bool>}@CLIENTS')

  def test_federated_zip_with_server_int_and_bool(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_server(tf.bool))
    val = intrinsics.federated_zip([x, y])
    self.assert_value(val, '<int32,bool>@SERVER')

  def test_federated_zip_with_names_server_int_and_bool(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    y = _mock_data_of_type(computation_types.at_server(tf.bool))
    val = intrinsics.federated_zip({'x': x, 'y': y})
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
    self.assert_types_identical(val.type_signature, expected)

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
    self.assert_types_identical(named_result.type_signature, expected)

    naming_fn = lambda i: str(i) if i % 2 == 0 else None
    mixed_result = intrinsics.federated_zip(
        structure.Struct((naming_fn(i), initial_tuple[i]) for i in range(n)))
    self.assertIsInstance(mixed_result, value_impl.Value)
    expected = computation_types.at_clients(
        computation_types.StructType([
            (naming_fn(i), element_type) for i in range(n)
        ]))
    self.assert_types_identical(mixed_result.type_signature, expected)

  @parameterized.named_parameters([
      ('test_n_' + str(n) + '_m_' + str(m), n, m)
      for n, m in itertools.product([1, 2, 3], [1, 2, 3])
  ])
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
    self.assert_types_identical(val.type_signature, final_fed_type)


class FederatedCollectTest(IntrinsicTestBase):

  def test_federated_collect_with_client_int(self):
    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    val = intrinsics.federated_collect(x)
    self.assert_value(val, 'int32*@SERVER')

  def test_federated_collect_with_server_int_fails(self):
    x = _mock_data_of_type(computation_types.at_server(tf.int32))
    with self.assertRaises(TypeError):
      intrinsics.federated_collect(x)


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

  def test_federated_aggregate_with_client_int(self):
    # The representation used during the aggregation process will be a named
    # tuple with 2 elements - the integer 'total' that represents the sum of
    # elements encountered, and the integer element 'count'.
    # pylint: disable=invalid-name
    Accumulator = collections.namedtuple('Accumulator', 'total count')
    # pylint: enable=invalid-name

    # The operator to use during the first stage simply adds an element to the
    # total and updates the count.
    @computations.tf_computation
    def accumulate(accu, elem):
      return Accumulator(accu.total + elem, accu.count + 1)

    # The operator to use during the second stage simply adds total and count.
    @computations.tf_computation
    def merge(x, y):
      return Accumulator(x.total + y.total, x.count + y.count)

    # The operator to use during the final stage simply computes the ratio.
    @computations.tf_computation
    def report(accu):
      return tf.cast(accu.total, tf.float32) / tf.cast(accu.count, tf.float32)

    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    val = intrinsics.federated_aggregate(x, Accumulator(0, 0), accumulate,
                                         merge, report)
    self.assert_value(val, 'float32@SERVER')

  def test_federated_aggregate_with_federated_zero_fails(self):
    zero = intrinsics.federated_value(0, placements.SERVER)

    @computations.tf_computation([tf.int32, tf.int32])
    def accumulate(accu, elem):
      return accu + elem

    # The operator to use during the second stage simply adds total and count.
    @computations.tf_computation([tf.int32, tf.int32])
    def merge(x, y):
      return x + y

    # The operator to use during the final stage simply computes the ratio.
    @computations.tf_computation(tf.int32)
    def report(accu):
      return accu

    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    with self.assertRaisesRegex(
        TypeError, 'Expected `zero` to be assignable to type int32, '
        'but was of incompatible type int32@SERVER'):
      intrinsics.federated_aggregate(x, zero, accumulate, merge, report)

  def test_federated_aggregate_with_unknown_dimension(self):
    Accumulator = collections.namedtuple('Accumulator', ['samples'])  # pylint: disable=invalid-name
    accumulator_type = computation_types.StructType(
        Accumulator(
            samples=computation_types.TensorType(dtype=tf.int32, shape=[None])))

    @computations.tf_computation()
    def build_empty_accumulator():
      return Accumulator(samples=tf.zeros(shape=[0], dtype=tf.int32))

    # The operator to use during the first stage simply adds an element to the
    # tensor, increasing its size.
    @computations.tf_computation([accumulator_type, tf.int32])
    def accumulate(accu, elem):
      return Accumulator(
          samples=tf.concat(
              [accu.samples, tf.expand_dims(elem, axis=0)], axis=0))

    # The operator to use during the second stage simply adds total and count.
    @computations.tf_computation([accumulator_type, accumulator_type])
    def merge(x, y):
      return Accumulator(samples=tf.concat([x.samples, y.samples], axis=0))

    # The operator to use during the final stage simply computes the ratio.
    @computations.tf_computation(accumulator_type)
    def report(accu):
      return accu

    x = _mock_data_of_type(computation_types.at_clients(tf.int32))
    val = intrinsics.federated_aggregate(x, build_empty_accumulator(),
                                         accumulate, merge, report)
    self.assert_value(val, '<samples=int32[?]>@SERVER')

  def test_infers_accumulate_return_as_merge_arg_merge_return_as_report_arg(
      self):
    x = _mock_data_of_type(computation_types.at_clients(tf.int64))

    @computations.tf_computation
    def nil_stack():
      return tf.constant([], dtype=tf.int64, shape=[0])

    @computations.tf_computation(
        computation_types.TensorType(tf.int64, [None]), tf.int64)
    def append(stack, num):
      return tf.concat([stack, [num]], 0)

    @computations.tf_computation
    def merge(stack_a, stack_b):
      return tf.concat([stack_a, stack_b], 0)

    report = computations.tf_computation(lambda a: a)

    val = intrinsics.federated_aggregate(x, nil_stack(), append, merge, report)
    self.assert_value(val, 'int64[?]@SERVER')


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

  def over_ten_fn(self):
    return computations.tf_computation(tf.int32)(lambda x: x > 10)

  def test_unplaced(self):
    x = _mock_data_of_type(computation_types.SequenceType(tf.int32))
    val = intrinsics.sequence_map(self.over_ten_fn(), x)
    self.assert_value(val, 'bool*')

  def test_server_placed(self):
    x = _mock_data_of_type(
        computation_types.at_server(computation_types.SequenceType(tf.int32)))
    val = intrinsics.sequence_map(self.over_ten_fn(), x)
    self.assert_value(val, 'bool*@SERVER')

  def test_clients_placed(self):
    x = _mock_data_of_type(
        computation_types.at_clients(computation_types.SequenceType(tf.int32)))
    val = intrinsics.sequence_map(self.over_ten_fn(), x)
    self.assert_value(val, '{bool*}@CLIENTS')


class SequenceReduceTest(IntrinsicTestBase):

  def test_with_non_federated_type(self):

    @computations.tf_computation(np.int32, np.int32)
    def add(x, y):
      return x + y

    value = _mock_data_of_type(computation_types.SequenceType(np.int32))
    result = intrinsics.sequence_reduce(value, 0, add)
    self.assert_value(result, 'int32')

  def test_with_federated_type(self):

    @computations.tf_computation(np.int32, np.int32)
    def add(x, y):
      return x + y

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
    test_case.main()
